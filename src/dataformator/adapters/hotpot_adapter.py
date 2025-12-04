# -*- coding: utf-8 -*-
"""
HotpotQA 适配器（仅负责数据加载和标准化，不包含 prompt 逻辑）。
"""

import json
from typing import Any, Dict, List


def _join_context(ctx_list):
    # 原始字段：List[[title, [sent1, sent2, ...]], ...]
    blocks = []
    for item in ctx_list or []:
        if not item or len(item) < 2:
            continue
        title, sents = item[0], item[1]
        text = " ".join(s for s in (sents or []) if s)
        blocks.append(f"[{title}] {text}")
    return "\n".join(blocks)


def load(args) -> List[Dict[str, Any]]:
    """
    加载 HotpotQA 数据集，生成标准格式：
        {id, question, context, answers, is_unanswerable, raw_rec}
    """
    cfg = (getattr(args, "hotpot_config", "distractor") or "distractor").lower()
    if args.source == "json":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw: List[Dict[str, Any]] = []
        for i, ex in enumerate(data):
            raw.append(
                {
                    "id": str(ex.get("_id", f"hotpot_{i}")),
                    "question": ex.get("question", ""),
                    "context": _join_context(ex.get("context", [])),
                    "answers": [ex.get("answer", "")],
                    "is_unanswerable": False,
                    "raw_rec": ex,
                }
            )
    elif args.source == "hf":
        from datasets import load_dataset  # type: ignore[import]

        subset = "fullwiki" if cfg == "fullwiki" else "distractor"
        ds = load_dataset("hotpot_qa", subset, split=args.split)
        raw = []
        for i, ex in enumerate(ds):
            raw.append(
                {
                    "id": str(ex.get("_id", f"hotpot_{i}")),
                    "question": ex["question"],
                    "context": _join_context(ex["context"]),
                    "answers": [ex["answer"]],
                    "is_unanswerable": False,
                    "raw_rec": ex,
                }
            )
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    return raw


__all__ = ["load"]


