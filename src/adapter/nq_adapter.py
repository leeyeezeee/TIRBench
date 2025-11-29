# -*- coding: utf-8 -*-
"""
Natural Questions 适配器
- 本地 JSONL（官方 dev）或 HF（natural_questions / natural_questions_open）
策略：
  1) 优先短答案 token span，窗口截取 context
  2) 其次 yes/no
  3) 都没有 → 不可回答 "unanswerable"
"""
import json

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

def _from_local(path):
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): 
                continue
            rec = json.loads(line)
            qid = str(rec.get("example_id", f"nq_{i}"))
            q = rec.get("question_text", rec.get("question",""))
            doc = rec.get("document_text","")
            tokens = doc.split() if isinstance(doc, str) else list(doc)
            spans = _extract_short_answers_from_rec(rec)

            gold, ctx, unans = [], None, False
            span = next(((s,e) for (s,e) in spans if s != "__YN__"), None)
            if span:
                s, e = span
                gold = [" ".join(tokens[s:e]).strip()]
                ctx = _window(tokens, s, e, 80, 80)
            if not gold:
                yn = next((v for (s,v) in spans if s == "__YN__"), None)
                if yn:
                    gold = [yn]; ctx = " ".join(tokens[:400])
            if not gold:
                gold = ["unanswerable"]; unans = True; ctx = " ".join(tokens[:400])

            raw.append({"id": qid, "question": q, "context": ctx or "",
                        "answers": gold, "is_unanswerable": unans, "raw_rec": rec })
    return raw

def _from_hf(split):
    from datasets import load_dataset
    try:
        ds = load_dataset("natural_questions", split=split)
        mode = "nq"
    except Exception:
        ds = load_dataset("natural_questions_open", split=split)
        mode = "nq_open"

    raw = []
    for i, ex in enumerate(ds):
        qid = str(ex.get("id", ex.get("example_id", f"nq_{i}")))
        q = ex.get("question", ex.get("question_text",""))

        # 若已有 context + answers[text] 结构，直接使用
        if "context" in ex and "answers" in ex and isinstance(ex["answers"], dict):
            golds = ex["answers"].get("text", []) or ["unanswerable"]
            unans = (golds == ["unanswerable"])
            raw.append({"id": qid, "question": q, "context": ex["context"],
                        "answers": golds, "is_unanswerable": unans})
            continue

        # 兼容 document_text + annotations
        doc = ex.get("document_text","")
        tokens = doc.split() if isinstance(doc, str) else list(doc)

        # 复用短答案/YN 提取
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
        span = next(((s,e) for (s,e) in spans if s != "__YN__"), None)
        if span:
            s, e = span
            gold = [" ".join(tokens[s:e]).strip()]
            ctx = _window(tokens, s, e, 80, 80)
        if not gold:
            yn = next((v for (s,v) in spans if s == "__YN__"), None)
            if yn:
                gold = [yn]; ctx = " ".join(tokens[:400])
        if not gold:
            gold = ["unanswerable"]; unans = True; ctx = " ".join(tokens[:400])

        raw.append({"id": qid, "question": q, "context": ctx or "",
                    "answers": gold, "is_unanswerable": unans, "raw_rec": ex })
    return raw

def load(args):
    if args.source == "json":
        return _from_local(args.input)
    elif args.source == "hf":
        return _from_hf(args.split)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

def build_prompt(ex, tokenizer=None):
    content = (
        "Read the context and answer with an exact span from the context. "
        "If it cannot be answered from the context, reply exactly with: unanswerable.\n\n"
        f"Context:\n{ex.context}\n\nQuestion: {ex.question}\nAnswer:"
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [
                {"role":"system","content":"You are a precise open-domain extractive QA assistant."},
                {"role":"user","content":content},
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return content
