"""
GSM8K 数据集适配器（仅负责数据加载和标准化，不包含 prompt 逻辑）。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd  # type: ignore[import]


def load(args) -> List[Dict[str, Any]]:
    """
    加载 GSM8K 数据集，生成 data_cleaning.py 需要的标准格式：
        {id, question, context, answers, is_unanswerable, raw_rec}
    """
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
        from datasets import load_dataset  # type: ignore[import]

        dataset = load_dataset("gsm8k", "main", split=args.split)
        raw_data = list(dataset)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    examples: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw_data):
        qid = ex.get("id", f"gsm8k_{i}")
        question = ex["question"]
        raw_answer = ex["answer"].strip()
        final_answer = extract_final_answer(raw_answer)
        examples.append(
            {
                "id": qid,
                "question": question,
                "context": "",
                "answers": [final_answer],
                "is_unanswerable": False,
                "raw_rec": ex,
            }
        )

    return examples


def extract_final_answer(answer_text: str) -> str:
    """
    从 GSM8K 的 answer 字段中提取最终答案
    格式: "推理过程... ### 最终答案"
    """
    text = (answer_text or "").strip()
    if "####" in text:
        final_answer = text.split("####")[-1].strip()
    elif "###" in text:
        final_answer = text.split("###")[-1].strip()
    else:
        numbers = re.findall(r"-?\d+\.?\d*", text)
        final_answer = numbers[-1] if numbers else text
    return final_answer


__all__ = ["load", "extract_final_answer"]


