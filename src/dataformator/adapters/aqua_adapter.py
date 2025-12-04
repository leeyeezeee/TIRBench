"""
AQuA 数据集适配器（仅负责数据加载和标准化，不包含 prompt 逻辑）。
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd  # type: ignore[import]


def load(args) -> List[Dict[str, Any]]:
    """
    加载 AQuA 数据集，生成 data_cleaning.py 需要的标准格式：
        {id, question, context, answers, is_unanswerable, raw_rec}
    """
    # 根据 source 类型加载数据
    if args.source == "json":
        suffix = Path(args.input).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            with open(args.input, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(args.input)
            raw_data = df.to_dict("records")
        else:
            raise ValueError("Unsupported input format")
    elif args.source == "hf":
        # 从 HuggingFace 加载
        from datasets import load_dataset  # type: ignore[import]

        dataset = load_dataset("aqua_rat", "raw", split=args.split)
        raw_data = list(dataset)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    # 转换为统一格式
    examples: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw_data):
        qid = ex.get("id", f"aqua_{i}")
        question = ex["question"]
        # 确保 options 是列表（从 numpy array 转换）
        options = list(ex.get("options", []))
        correct_label = ex.get("correct", "")

        # 构建 context（包含所有选项）
        context = "Options:\n" + "\n".join(options)

        examples.append(
            {
                "id": qid,
                "question": question,
                "context": context,
                "answers": [correct_label],
                "is_unanswerable": False,
                "raw_rec": ex,
            }
        )

    return examples


__all__ = ["load"]


