#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter QA datasets (SQuAD v2.0 / HotpotQA / Natural Questions) by removing
items a small model can already answer (zero-shot).

- Auto-download via 🤗 Datasets (or read local SQuAD-like JSON).
- Normalize to official SQuAD nested structure: {version, data:[{title, paragraphs:[{context, qas:[...]}]}]}
- Generate with any HF CausalLM (e.g., Qwen 0.xB).
- Score with SQuAD EM/F1. "Correct" = EM==1 or F1>=threshold (default 0.8).
- Optional handling for unanswerable (SQuAD v2.0 / some NQ cases).
- Save filtered JSON + a CSV log.

Examples:
SQuAD v2.0 (validation):
python filter_qa_three_datasets.py \
  --dataset squad_v2 --source hf --split validation \
  --model Qwen/Qwen2-0.5B-Instruct --use_chat_template \
  --include_unanswerable_hint --handle_unanswerable \
  --device cuda --temperature 0.0 \
  --output squadv2.valid.filtered.json \
  --save_csv squadv2.decisions.csv

HotpotQA (distractor dev):
python filter_qa_three_datasets.py \
  --dataset hotpot_qa --hotpot_config distractor --source hf --split validation \
  --model Qwen/Qwen2-0.5B-Instruct --use_chat_template \
  --device cuda --temperature 0.0 \
  --output hotpot.dev.filtered.json --save_csv hotpot.decisions.csv

Natural Questions (long docs):
python filter_qa_three_datasets.py \
  --dataset nq --source hf --split validation \
  --model Qwen/Qwen2-0.5B-Instruct --use_chat_template \
  --include_unanswerable_hint --handle_unanswerable \
  --device cuda --temperature 0.0 \
  --output nq.valid.filtered.json --save_csv nq.decisions.csv
"""

import argparse, json, re, string
# 导入数学数据集适配器
from adapter.gsm8k_adapter import to_squadlike as gsm8k_to_squadlike
from adapter.aqua_adapter import to_squadlike as aqua_to_squadlike
from adapter.math_adapter import to_squadlike as math_to_squadlike
from adapter.squadv2_adapter import to_squadlike as squadv2_to_squadlike
from adapter.hotpot_adapter import to_squadlike as hotpot_to_squadlike
from adapter.nq_adapter import to_squadlike as nq_to_squadlike
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------
# SQuAD metrics
# ------------------------------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def _normalize_answer(s: str) -> str:
    def lower(text): return text.lower()
    def remove_articles(text): return _ARTICLES_RE.sub(" ", text)
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def white_space_fix(text): return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def metric_match(pred: str, gold_list: List[str], f1_threshold: float = 0.8) -> Tuple[bool, float, float]:
    if not gold_list:
        return False, 0.0, 0.0
    ems = [exact_match_score(pred, g) for g in gold_list]
    f1s = [f1_score(pred, g) for g in gold_list]
    best_em = max(ems) if ems else 0.0
    best_f1 = max(f1s) if f1s else 0.0
    ok = (best_em == 1.0) or (best_f1 >= f1_threshold)
    return ok, best_em, best_f1

# ------------------------------
# Unanswerable heuristic
# ------------------------------
DEFAULT_NO_ANSWER_PATTERNS = [
    "unanswerable",
    "cannot be determined",
    "cannot be answered",
    "not provided in the context",
    "not mentioned in the passage",
    "insufficient information",
    "no answer",
    "unknown",
]

def looks_like_no_answer(text: str, patterns: List[str]) -> bool:
    t = text.strip().lower()
    for p in patterns:
        if p in t:
            return True
    return False

# ------------------------------
# SQuAD-like nested iter + remove
# ------------------------------
def squad_iter_nested(js: Dict[str, Any]) -> Iterable[Tuple[Tuple, Dict[str, Any]]]:
    """Yield ((ai, pi, qi), sample_dict) from official SQuAD-like nested structure."""
    for ai, article in enumerate(js["data"]):
        for pi, para in enumerate(article.get("paragraphs", [])):
            context = para.get("context", "")
            for qi, qa in enumerate(para.get("qas", [])):
                qid = qa.get("id", f"a{ai}_p{pi}_q{qi}")
                question = qa.get("question", "")
                is_imp = bool(qa.get("is_impossible", False))
                answers = qa.get("answers", [])
                golds = [a["text"] for a in answers if isinstance(a, dict) and "text" in a]
                yield (ai, pi, qi), {
                    "id": qid,
                    "question": question,
                    "context": context,
                    "is_impossible": is_imp,
                    "answers": golds,
                }

def nested_remove(js: Dict[str, Any], remove_paths: set) -> Dict[str, Any]:
    """Return deep-copied json with selected qas removed."""
    out = deepcopy(js)
    new_data = []
    for ai, article in enumerate(out["data"]):
        new_paragraphs = []
        for pi, para in enumerate(article.get("paragraphs", [])):
            qas = para.get("qas", [])
            kept_qas = []
            for qi, qa in enumerate(qas):
                if (ai, pi, qi) not in remove_paths:
                    kept_qas.append(qa)
            if kept_qas:
                para["qas"] = kept_qas
                new_paragraphs.append(para)
        if new_paragraphs:
            article["paragraphs"] = new_paragraphs
            new_data.append(article)
    out["data"] = new_data
    return out




# ------------------------------
# Prompting & generation
# ------------------------------
SYS_MSG = "You are a helpful assistant."
USER_TMPL = (
    "Answer the question based on the given context. "
    "Respond with a short, direct answer.\n\n"
    "{unans_hint}"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

def build_prompt(tokenizer, question: str, context: str, use_chat: bool, include_unans_hint: bool) -> str:
    hint = "If the answer cannot be found in the context, reply with 'unanswerable'.\n\n" if include_unans_hint else ""
    user_text = USER_TMPL.format(unans_hint=hint, context=context, question=question)
    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "system", "content": SYS_MSG},
                    {"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYS_MSG}\n\n{user_text}"

def truncate_context(tokenizer, question: str, context: str, max_input_tokens: int) -> str:
    reserve = max(128, min(512, max_input_tokens // 6))  # room for question & overhead
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    budget = max(1, max_input_tokens - len(q_tokens) - reserve)
    ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(ctx_tokens) <= budget:
        return context
    tail = tokenizer.decode(ctx_tokens[-budget:], skip_special_tokens=True)
    return tail

@torch.inference_mode()
def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature if temperature > 0 else None,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_id,
    )
    out_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()

# ------------------------------
# IO helpers
# ------------------------------
def read_squad_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Filter QA datasets by removing items a small model already answers.")
    # dataset selection & source
    ap.add_argument("--dataset", choices=["squad_v2", "hotpot_qa", "nq", "gsm8k", "aqua", "math"], default="squad_v2",
                    help="Choose which dataset to use.")
    ap.add_argument("--source", choices=["hf", "json"], default="hf",
                    help="Use 'hf' for HuggingFace download or 'json' for a local SQuAD-like JSON.")
    ap.add_argument("--split", default="validation",
                    help="HF split name (e.g., 'train'/'validation'). If not present, will fallback automatically.")
    ap.add_argument("--input", default=None, help="Path to local SQuAD-like JSON when --source=json.")
    ap.add_argument("--output", required=True, help="Path to write filtered JSON.")
    ap.add_argument("--save_csv", default=None, help="Optional CSV to log per-item decisions.")

    # dataset-specific options
    ap.add_argument("--hotpot_config", choices=["distractor", "fullwiki"], default="distractor",
                    help="HotpotQA configuration to use when --dataset=hotpot_qa.")

    # model & decoding
    ap.add_argument("--model", required=True, help="HF model name or local path (e.g., Qwen/Qwen2-0.5B-Instruct).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Use tokenizer.apply_chat_template if available (for chat/instruct models).")
    ap.add_argument("--include_unanswerable_hint", action="store_true",
                    help="Tell the model to output 'unanswerable' when no answer appears in context.")
    ap.add_argument("--max_input_tokens", type=int, default=2048, help="Token cap for the prompt (context will be truncated).")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)

    # scoring
    ap.add_argument("--f1_threshold", type=float, default=0.8, help="Correct if EM==1 or F1>=threshold.")
    ap.add_argument("--handle_unanswerable", action="store_true",
                    help="Treat 'unanswerable' predictions as correct when gold answers empty.")
    ap.add_argument("--no_answer_patterns", nargs="*", default=DEFAULT_NO_ANSWER_PATTERNS,
                    help="Phrases indicating 'no answer' (lowercase).")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # Load model
    print(f"[Load] model={args.model} device={args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.device.startswith("cuda") else None,
    )
    if not args.device.startswith("cuda"):
        model = model.to(args.device)

    # Load data -> normalize to SQuAD-like nested JSON

    if args.source == "hf":
        print(f"[Data] loading via datasets: {args.dataset} / split={args.split}")
        from datasets import load_dataset

        if args.dataset == "squad_v2":
            ds = load_dataset("squad_v2")
            split_name = args.split if args.split in ds else ("validation" if "validation" in ds else "train")
            nested = squadv2_to_squadlike(ds[split_name])

        elif args.dataset == "hotpot_qa":
            ds = load_dataset("hotpot_qa", args.hotpot_config)
            split_name = args.split if args.split in ds else ("validation" if "validation" in ds else "train")
            nested = hotpot_to_squadlike(ds[split_name])

        elif args.dataset == "nq":
            ds = load_dataset("natural_questions")
            if args.split in ds:
                split_name = args.split
            elif "validation" in ds:
                split_name = "validation"
            elif "dev" in ds:
                split_name = "dev"
            elif "test" in ds:
                split_name = "test"
            else:
                split_name = "train"
            nested = nq_to_squadlike(ds[split_name])

        elif args.dataset == "gsm8k":
            ds = load_dataset("gsm8k")
            split_name = args.split if args.split in ds else ("test" if "test" in ds else "train")
            nested = gsm8k_to_squadlike(ds[split_name])

        elif args.dataset == "aqua":
            ds = load_dataset("aqua_rat")
            split_name = args.split if args.split in ds else ("validation" if "validation" in ds else "train")
            nested = aqua_to_squadlike(ds[split_name])

        elif args.dataset == "math":
            ds = load_dataset("math")
            split_name = args.split if args.split in ds else ("test" if "test" in ds else "train")
            nested = math_to_squadlike(ds[split_name])

        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    else:
        if not args.input:
            raise ValueError("--input is required when --source=json")
        print(f"[Data] reading local SQuAD-like JSON: {args.input}")
        nested = read_squad_json(args.input)

    # Iterate & evaluate
    items = list(squad_iter_nested(nested))
    print(f"[Eval] total samples: {len(items)}")
    remove_paths = set()
    logs: List[Tuple[str, float, float, int, str]] = []

    for (ai, pi, qi), ex in tqdm(items, desc="Evaluating"):
        qid = ex["id"]
        question = ex["question"] or ""
        context = ex["context"] or ""
        answers = ex["answers"] or []
        is_imp = bool(ex.get("is_impossible", False))

        # truncate context
        if args.max_input_tokens:
            context = truncate_context(tokenizer, question, context, args.max_input_tokens)

        prompt = build_prompt(
            tokenizer,
            question=question,
            context=context,
            use_chat=args.use_chat_template,
            include_unans_hint=args.include_unanswerable_hint
        )

        pred = generate_answer(
            model, tokenizer, prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

        # scoring
        if (not is_imp) and len(answers) > 0:
            ok, em, f1 = metric_match(pred, answers, f1_threshold=args.f1_threshold)
        else:
            # gold is unanswerable
            if args.handle_unanswerable and looks_like_no_answer(pred, [p.lower() for p in args.no_answer_patterns]):
                ok, em, f1 = True, 1.0, 1.0
            else:
                ok, em, f1 = False, 0.0, 0.0

        logs.append((qid, em, f1, int(ok), pred))
        if ok:
            remove_paths.add((ai, pi, qi))

    # Remove & save
    filtered = nested_remove(nested, remove_paths)
    write_json(filtered, args.output)
    removed = len(remove_paths)
    kept = len(items) - removed
    print(f"[Done] removed {removed} / {len(items)}; kept {kept}; wrote: {args.output}")

    # CSV log
    if args.save_csv:
        try:
            import csv
            with open(args.save_csv, "w", newline="", encoding="utf-8") as cf:
                w = csv.writer(cf)
                w.writerow(["id", "EM", "F1", "Correct", "Prediction"])
                for qid, em, f1, ok, pred in logs:
                    w.writerow([qid, f"{em:.3f}", f"{f1:.3f}", ok, pred])
            print(f"[Log] decisions saved to: {args.save_csv}")
        except Exception as e:
            print(f"[Warn] CSV save failed: {e}")

if __name__ == "__main__":
    main()
