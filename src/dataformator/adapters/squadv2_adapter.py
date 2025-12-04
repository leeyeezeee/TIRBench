# -*- coding: utf-8 -*-
"""
SQuAD v2.0 适配器（仅负责数据加载和标准化，不包含 prompt 逻辑）。
返回字段：{id, question, context, answers(list[str]), is_unanswerable(bool), raw_rec}
"""

import json
from typing import Any, Dict, List


def load(args) -> List[Dict[str, Any]]:
    """
    Args
    ----
    args.source : "json" | "hf"
    args.input  : 当 source="json" 时，本地文件路径（.json）
    args.split  : 当 source="hf" 时，HF split（通常 "validation"）
    """
    if args.source == "json":
        # 本地 SQuAD v2.0 官方 JSON
        with open(args.input, "r", encoding="utf-8") as f:
            obj = json.load(f)
        data = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        raw: List[Dict[str, Any]] = []
        for art in data:
            for para in art.get("paragraphs", []):
                ctx = para.get("context", "")
                for qa in para.get("qas", []):
                    is_imp = bool(qa.get("is_impossible", False))
                    answers = (
                        ["unanswerable"]
                        if is_imp
                        else [a.get("text", "") for a in qa.get("answers", [])] or [""]
                    )
                    raw.append(
                        {
                            "id": str(qa.get("id")),
                            "question": qa.get("question", ""),
                            "context": ctx,
                            "answers": answers,
                            "is_unanswerable": is_imp,
                            "raw_rec": qa,
                        }
                    )
    elif args.source == "hf":
        # HF 直接加载
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("squad_v2", split=args.split)
        raw = []
        for ex in ds:
            is_imp = bool(ex.get("is_impossible", False))
            if is_imp:
                ans = ex["answers"]["text"] or ["unanswerable"]
            else:
                ans = ex["answers"]["text"] or [""]
            raw.append(
                {
                    "id": str(ex["id"]),
                    "question": ex["question"],
                    "context": ex["context"],
                    "answers": ans,
                    "is_unanswerable": is_imp,
                    "raw_rec": ex,
                }
            )
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    return raw


__all__ = ["load"]


