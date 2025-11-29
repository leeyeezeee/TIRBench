# -*- coding: utf-8 -*-
"""
HotpotQA 适配器
- 本地 JSON（dev distractor/fullwiki 任意）或 HF 子集
- 注：HF 需要指定配置名；这里默认用 "distractor"。若你在 .sh 改为 fullwiki，可传 --hotpot_config fullwiki
"""
import json

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

def load(args):
    cfg = (getattr(args, "hotpot_config", "distractor") or "distractor").lower()
    if args.source == "json":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = []
        for i, ex in enumerate(data):
            raw.append({
                "id": str(ex.get("_id", f"hotpot_{i}")),
                "question": ex.get("question",""),
                "context": _join_context(ex.get("context", [])),
                "answers": [ex.get("answer","")],
                "is_unanswerable": False,
                "raw_rec": ex
            })
    elif args.source == "hf":
        from datasets import load_dataset
        subset = "fullwiki" if cfg == "fullwiki" else "distractor"
        ds = load_dataset("hotpot_qa", subset, split=args.split)
        raw = []
        for i, ex in enumerate(ds):
            raw.append({
                "id": str(ex.get("_id", f"hotpot_{i}")),
                "question": ex["question"],
                "context": _join_context(ex["context"]),
                "answers": [ex["answer"]],
                "is_unanswerable": False,
                "raw_rec": ex
            })
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    return raw

def build_prompt(ex, tokenizer=None):
    content = (
        "Answer the multi-hop question using ONLY the given context. "
        "Return an exact phrase from the context. "
        "If it cannot be answered from the context, reply exactly with: unanswerable.\n\n"
        f"Context:\n{ex.context}\n\nQuestion: {ex.question}\nAnswer:"
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [
                {"role":"system","content":"You are a careful multi-hop reading comprehension assistant."},
                {"role":"user","content":content},
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return content
