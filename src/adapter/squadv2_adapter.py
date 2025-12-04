# -*- coding: utf-8 -*-
"""
SQuAD v2.0 适配器
返回字段：{id, question, context, answers(list[str]), is_unanswerable(bool)}
"""
import json

def load(args):
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
        raw = []
        for art in data:
            for para in art.get("paragraphs", []):
                ctx = para.get("context", "")
                for qa in para.get("qas", []):
                    is_imp = bool(qa.get("is_impossible", False))
                    answers = ["unanswerable"] if is_imp else [a.get("text","") for a in qa.get("answers", [])] or [""]
                    raw.append({
                        "id": str(qa.get("id")),
                        "question": qa.get("question",""),
                        "context": ctx,
                        "answers": answers,
                        "is_unanswerable": is_imp,
                        "raw_rec": qa
                    })
    elif args.source == "hf":
        # HF 直接加载
        from datasets import load_dataset
        ds = load_dataset("squad_v2", split=args.split)
        raw = []
        for ex in ds:
            is_imp = bool(ex.get("is_impossible", False))
            ans = (ex["answers"]["text"] or ["unanswerable"]) if is_imp else (ex["answers"]["text"] or [""])
            raw.append({
                "id": str(ex["id"]),
                "question": ex["question"],
                "context": ex["context"],
                "answers": ans,
                "is_unanswerable": is_imp,
                "raw_rec": ex
            })
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    # 与 data_cleaning.py 期望一致的列表[dict]
    return raw

def build_prompt(ex, tokenizer=None):
    # 与数学风格一致：优先 chat_template
    content = (
        "Read the context and answer with an exact span from the context.\n"
        "If the question cannot be answered using only the given context, reply exactly with: unanswerable.\n\n"
        f"Context:\n{ex.context}\n\nQuestion: {ex.question}\nAnswer:"
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [
                {"role":"system","content":"You are a precise extractive QA assistant."},
                {"role":"user","content":content},
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return content
