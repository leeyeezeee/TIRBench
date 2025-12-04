# -*- coding: utf-8 -*-
"""
Natural Questions 适配器（仅负责数据加载和标准化，不包含 prompt 逻辑）。
"""

import json
from typing import Any, Dict, List


def _window(tokens, start, end, left=80, right=80):
    L = max(0, start - left)
    R = min(len(tokens), end + right)
    return " ".join(tokens[L:R])


def _extract_short_answers_from_rec(rec):
    outs = []
    anns = rec.get("annotations", [])
    for an in anns:
        for sa in an.get("short_answers", []):
            s, e = sa.get("start_token", -1), sa.get("end_token", -1)
            if s >= 0 and e > s:
                outs.append((s, e))
        yno = an.get("yes_no_answer", "NONE")
        if yno in ("YES", "NO"):
            outs.append(("__YN__", yno.lower()))
    return outs


def _from_local(path: str) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec.get("example_id", f"nq_{i}"))
            q = rec.get("question_text", rec.get("question", ""))
            doc = rec.get("document_text", "")
            tokens = doc.split() if isinstance(doc, str) else list(doc)
            spans = _extract_short_answers_from_rec(rec)

            gold, ctx, unans = [], None, False
            span = next(((s, e) for (s, e) in spans if s != "__YN__"), None)
            if span:
                s, e = span
                gold = [" ".join(tokens[s:e]).strip()]
                ctx = _window(tokens, s, e, 80, 80)
            if not gold:
                yn = next((v for (s, v) in spans if s == "__YN__"), None)
                if yn:
                    gold = [yn]
                    ctx = " ".join(tokens[:400])
            if not gold:
                gold = ["unanswerable"]
                unans = True
                ctx = " ".join(tokens[:400])

            raw.append(
                {
                    "id": qid,
                    "question": q,
                    "context": ctx or "",
                    "answers": gold,
                    "is_unanswerable": unans,
                    "raw_rec": rec,
                }
            )
    return raw


def _from_hf(split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore[import]

    try:
        ds = load_dataset("natural_questions", split=split)
    except Exception:
        ds = load_dataset("natural_questions_open", split=split)

    raw: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        qid = str(ex.get("id", ex.get("example_id", f"nq_{i}")))
        q = ex.get("question", ex.get("question_text", ""))

        # 若已有 context + answers[text] 结构，直接使用
        if "context" in ex and "answers" in ex and isinstance(ex["answers"], dict):
            golds = ex["answers"].get("text", []) or ["unanswerable"]
            unans = golds == ["unanswerable"]
            raw.append(
                {
                    "id": qid,
                    "question": q,
                    "context": ex["context"],
                    "answers": golds,
                    "is_unanswerable": unans,
                    "raw_rec": ex,
                }
            )
            continue

        # 兼容 document_text + annotations
        doc = ex.get("document_text", "")
        tokens = doc.split() if isinstance(doc, str) else list(doc)

        spans = []
        anns = ex.get("annotations", [])
        for an in anns:
            for sa in an.get("short_answers", []):
                s, e = sa.get("start_token", -1), sa.get("end_token", -1)
                if s >= 0 and e > s:
                    spans.append((s, e))
            yno = an.get("yes_no_answer", "NONE")
            if yno in ("YES", "NO"):
                spans.append(("__YN__", yno.lower()))

        gold, ctx, unans = [], None, False
        span = next(((s, e) for (s, e) in spans if s != "__YN__"), None)
        if span:
            s, e = span
            gold = [" ".join(tokens[s:e]).strip()]
            ctx = _window(tokens, s, e, 80, 80)
        if not gold:
            yn = next((v for (s, v) in spans if s == "__YN__"), None)
            if yn:
                gold = [yn]
                ctx = " ".join(tokens[:400])
        if not gold:
            gold = ["unanswerable"]
            unans = True
            ctx = " ".join(tokens[:400])

        raw.append(
            {
                "id": qid,
                "question": q,
                "context": ctx or "",
                "answers": gold,
                "is_unanswerable": unans,
                "raw_rec": ex,
            }
        )
    return raw


def load(args) -> List[Dict[str, Any]]:
    """统一入口：根据 source 从本地 JSONL 或 HF 加载 NQ。"""
    if args.source == "json":
        return _from_local(args.input)
    if args.source == "hf":
        return _from_hf(args.split)
    raise ValueError(f"Unsupported source: {args.source}")


__all__ = ["load"]


