"""
OMINI 数据集适配器（仅负责数据加载和标准化，不包含 prompt 逻辑）。
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd  # type: ignore[import]


def load(args) -> List[Dict[str, Any]]:
    """
    加载 OMINI 数据集，生成 data_cleaning.py 需要的标准格式：
        {id, question, context, answers, is_unanswerable, raw_rec}
    """
    if args.source == "json":
        suffix = Path(args.input).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            with open(args.input, "r", encoding="utf-8") as f:
                if suffix == ".jsonl":
                    raw_data = [json.loads(line) for line in f]
                else:
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict):
                        raw_data = [raw_data]
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(args.input)
            raw_data = df.to_dict("records")
        else:
            raise ValueError(
                f"Unsupported input format: {suffix}. Expected .json, .jsonl, .parquet, or .pq"
            )
    elif args.source == "hf":
        from datasets import load_dataset  # type: ignore[import]

        dataset = load_dataset("omini", split=args.split)
        raw_data = list(dataset)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    examples: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw_data):
        qid = ex.get("id", f"omini_{i}")
        question = ex.get("question", "")
        thinking = ex.get("thinking", "")
        answer = ex.get("answer", "").strip()
        context = thinking if thinking else ""
        examples.append(
            {
                "id": qid,
                "question": question,
                "context": context,
                "answers": [answer],
                "is_unanswerable": False,
                "raw_rec": ex,
            }
        )
    return examples


__all__ = ["load"]


