from __future__ import annotations

"""
Generic dataset loader / registry for TIRBench.

This module provides:
  - DatasetConfig: per-dataset static metadata (category, default path, prompt & filter config)
  - DatasetLoader: a thin wrapper around existing adapter modules that:
        * knows the dataset name / category
        * can load data via the adapter's `load(args)` function
        * can build prompts via adapter-specific `build_prompt` or a generic fallback
        * exposes filtering rules loaded from a JSON schema file
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import importlib


# Default to the per-dataset schema folder; fall back to single YAML/JSON file if needed.
SCHEMA_PATH = Path(__file__).with_name("dataset")


@dataclass
class DatasetConfig:
    """Static configuration for a single dataset."""

    name: str
    key: str
    category: str           # e.g. "commonsense" | "math" | "other"
    task_type: str          # e.g. "extractive" | "math" | "other"
    default_source: Optional[str] = None  # "hf" | "json" | None
    default_input: Optional[str] = None   # local path or HF name
    adapter: Optional[str] = None         # dotted module path for adapter, e.g. "dataformat.adapters.gsm8k_adapter"
    prompt: Dict[str, Any] = None         # prompt-related config
    filter: Dict[str, Any] = None         # filtering rule config


def _load_schema(path: Path = SCHEMA_PATH) -> Dict[str, Any]:
    """
    Load all dataset configs.

    - If `path` is a YAML/JSON file, expect the monolithic format:
        { "datasets": { "gsm8k": { ... }, ... } }
    - If `path` is a directory, load each `*.yaml` / `*.yml` (or legacy `*.json`) and merge:
        the dataset key is taken from `obj['key']` if present, otherwise from the filename stem.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset schema not found: {path}")

    # Single-file format (YAML or JSON)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore[import]
            except ImportError as e:
                raise ImportError("PyYAML is required to load YAML schemas, please `pip install pyyaml`.") from e
            with open(path, "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        return obj.get("datasets", {})

    # Multi-file directory format
    if path.is_dir():
        datasets: Dict[str, Any] = {}

        def _load_one(p: Path) -> Optional[Dict[str, Any]]:
            suffix = p.suffix.lower()
            if suffix in {".yml", ".yaml"}:
                try:
                    import yaml  # type: ignore[import]
                except ImportError as e:
                    raise ImportError("PyYAML is required to load YAML schemas, please `pip install pyyaml`.") from e
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            elif suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        for p in sorted(path.glob("*.*")):
            if p.suffix.lower() not in {".yml", ".yaml", ".json"}:
                continue
            obj = _load_one(p)
            if not isinstance(obj, dict):
                continue
            key = obj.get("key") or p.stem
            datasets[key] = obj
        return datasets

    raise FileNotFoundError(f"Unsupported schema path (neither file nor directory): {path}")


def get_dataset_config(dataset_key: str, schema_path: Path = SCHEMA_PATH) -> DatasetConfig:
    """
    Load DatasetConfig for a given dataset key (e.g. 'gsm8k', 'math', ...).
    """
    datasets = _load_schema(schema_path)
    if dataset_key not in datasets:
        raise KeyError(f"Dataset '{dataset_key}' not found in schema: {schema_path}")
    cfg = datasets[dataset_key]
    return DatasetConfig(
        name=cfg.get("name", dataset_key),
        key=dataset_key,
        category=cfg.get("category", "other"),
        task_type=cfg.get("task_type", "other"),
        default_source=cfg.get("default_source"),
        default_input=cfg.get("default_input"),
        adapter=cfg.get("adapter"),
        prompt=cfg.get("prompt", {}) or {},
        filter=cfg.get("filter", {}) or {},
    )


class DatasetLoader:
    """
    Unified façade around dataset-specific loading / prompts driven by schema.
    
    设计目标：后续可以删除各个 adapter 文件，仅依赖本类完成：
      - 读取数据（统一成 {id, question, context, answers, is_unanswerable, raw_rec} 格式）
      - 构造 prompt（从 per-dataset schema 中读取模板）
      - 提供过滤规则（schema 中的 filter 字段）
    """
    
    def __init__(
        self,
        dataset_key: str,
        adapter_module_path: Optional[str] = None,
        schema_path: Path = SCHEMA_PATH,
    ) -> None:
        self.config = get_dataset_config(dataset_key, schema_path)
        self.key = self.config.key
        # 如果 schema 中指定了 adapter，则优先使用；否则默认 dataformat.adapters.<dataset>_adapter
        self.adapter_module_path = (
            adapter_module_path
            or self.config.adapter
            or f"dataformat.adapters.{dataset_key}_adapter"
        )
        try:
            self._adapter_mod = importlib.import_module(self.adapter_module_path)
        except Exception:
            self._adapter_mod = None

        # adapter 提供的自定义 prompt 构造函数（可选）
        self._build_prompt_fn: Optional[Callable] = (
            getattr(self._adapter_mod, "build_prompt", None) if self._adapter_mod else None
        )

        # 过滤规则：schema.filter 中可以包含 rule 名和可选的函数路径 fn（module:function）
        self._filter_conf: Dict[str, Any] = self.config.filter or {}
        self._filter_fn: Optional[Callable] = None
        fn_path = self._filter_conf.get("fn") or self._filter_conf.get("function")
        if fn_path:
            try:
                # 支持 "pkg.mod:func" 或 "pkg.mod.func" 两种写法
                if ":" in fn_path:
                    mod_name, func_name = fn_path.split(":", 1)
                else:
                    parts = fn_path.split(".")
                    mod_name, func_name = ".".join(parts[:-1]), parts[-1]
                mod = importlib.import_module(mod_name)
                self._filter_fn = getattr(mod, func_name)
            except Exception:
                # 静默失败：保留 _filter_fn=None，由上层 fallback 到内置逻辑
                self._filter_fn = None

    # ---------- Core loading ----------
    def load(self, args) -> Any:
        """
        加载指定数据集，并统一为 data_cleaning 期望的列表[dict]格式：
          {id, question, context, answers, is_unanswerable, raw_rec}

        逻辑（简化版）：
          - 完全依赖 schema 中指定的 adapter 模块，例如：
                adapter: \"dataformat.adapters.gsm8k_adapter\"
          - 该 adapter 必须实现 `load(args)`，并返回标准化后的记录。
        """
        # 填充 schema 中的默认 source / input（若 CLI 未显式指定）
        if getattr(args, "source", None) is None and self.config.default_source:
            args.source = self.config.default_source
        if getattr(args, "input", None) is None and self.config.default_input:
            args.input = os.path.expandvars(self.config.default_input)

        # ---------- 使用 adapter 模块的 load(args) ----------
        if self._adapter_mod is not None and hasattr(self._adapter_mod, "load"):
            return self._adapter_mod.load(args)

        # ---------- 若 adapter 不存在或未实现 load，则直接报错 ----------
        raise AttributeError(
            f"No loader available for dataset '{self.key}'. "
            f"Expected adapter module '{self.adapter_module_path}' to define load(args)->List[dict]."
        )

    

    # ---------- Filter rule access ----------
    @property
    def filter_rule(self) -> Dict[str, Any]:
        """
        Return the filter rule configuration for this dataset, as defined in schema.
        Example (for GSM8K):
            {
              "rule": "math_numeric",
              "relative_tolerance": 1e-6
            }
        """
        return self.config.filter or {}

    @property
    def filter_fn(self) -> Optional[Callable]:
        """
        Optional callable loaded from schema.filter.fn.
        签名约定：fn(pred: str, ex: Any, args: Any, rule_conf: Dict[str, Any]) -> (hit, score1, score2)
        """
        return self._filter_fn

    @property
    def is_math(self) -> bool:
        return self.config.category == "math" or self.config.task_type == "math"

    @property
    def is_extractive(self) -> bool:
        return self.config.task_type == "extractive"
    


__all__ = [
    "DatasetConfig",
    "DatasetLoader",
    "get_dataset_config",
]


