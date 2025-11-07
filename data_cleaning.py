#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-backend (vLLM / SGLang / Transformers) data cleaning:
- Filter out "easy" samples that a small model can already solve.
- Default backend: vLLM (Ascend: use vLLM-Ascend).
- Dataset adapters live in adapter/*.py (must implement load(args)->List[Example]).
- Supports extractive QA (SQuADv2/Hotpot/NQ) and math QA (AQuA/GSM8K/MATH).

Outputs:
  - --output <*.json>: filtered examples (single JSON array; same schema).
  - --save_csv <*.csv>: decision log (optional).
  - --export_squad_like: extra SQuAD-like JSON (extractive only).

Usage (Ascend 910B, vLLM-Ascend, 4 cards):
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
  python data_cleaning.py \
      --backend vllm --tp 4 \
      --dataset squadv2 --source json --input datasets/SQuAD/dev-v2.0.json \
      --model /data/models/Qwen2.5-0.5B-Instruct \
      --tokenizer_path /data/models/Qwen2.5-0.5B-Instruct \
      --use_chat_template --include_unanswerable_hint --handle_unanswerable \
      --temperature 0.0 --batch_size 8 \
      --output squadv2.filtered.jsonl --save_csv squadv2.decisions.csv
"""

from __future__ import annotations
import argparse, json, os, re, importlib, math
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Iterable, Dict, Any, Optional, Callable
from collections import Counter
from time import time
from pathlib import Path
from tqdm import tqdm

# =============== Example schema ===============
@dataclass
class Example:
    id: str
    question: str
    context: str
    answers: List[str]
    is_unanswerable: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

# =============== CLI ===============
def build_args():
    ap = argparse.ArgumentParser("Filter 'easy' samples with a small model.")

    # dataset
    ap.add_argument("--dataset", required=True,
                    choices=["squadv2", "hotpot", "nq", "aqua", "gsm8k", "math"])
    ap.add_argument("--source", choices=["hf", "json"], default="hf")
    ap.add_argument("--input", type=str, default=None, help="Local file path when --source=json")
    ap.add_argument("--split", type=str, default="validation")

    # backends
    ap.add_argument("--backend", choices=["vllm", "sglang", "hf"], default="vllm")
    ap.add_argument("--model", type=str, required=True, help="HF repo or local dir (or SGLang model name)")
    ap.add_argument("--tokenizer_path", type=str, default=None)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_input_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    # vLLM
    ap.add_argument("--tp", type=int, default=1, help="Tensor parallel size (multi-card) for vLLM")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)

    # SGLang client (OpenAI-compatible API)
    ap.add_argument("--sglang_api_base", type=str, default=os.getenv("SGLANG_API_BASE", "http://127.0.0.1:30000"))
    ap.add_argument("--sglang_api_key", type=str, default=os.getenv("SGLANG_API_KEY", None))
    ap.add_argument("--sglang_model", type=str, default=None, help="If server uses alias different from --model")
    ap.add_argument("--concurrency", type=int, default=8, help="SGLang client concurrency")

    # HF
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cpu / npu:0")
    ap.add_argument("--device_map", type=str, default=None, help='Transformers: e.g., "auto"')

    # filtering & outputs
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--f1_threshold", type=float, default=0.8)
    ap.add_argument("--export_squad_like", action="store_true")
    ap.add_argument("--output", type=str, required=True, help="Filtered dataset JSON path")
    ap.add_argument("--save_csv", type=str, default=None)

    # dataset hints
    ap.add_argument("--include_unanswerable_hint", action="store_true")
    ap.add_argument("--handle_unanswerable", action="store_true")
    ap.add_argument("--hotpot_config", type=str, default="distractor")

    return ap.parse_args()

# =============== Adapters ===============
ADAPTER_REG = {
    "squadv2": "adapter.squadv2_adapter",
    "hotpot":  "adapter.hotpot_adapter",
    "nq":      "adapter.nq_adapter",
    "aqua":    "adapter.aqua_adapter",
    "gsm8k":   "adapter.gsm8k_adapter",
    "math":    "adapter.math_adapter",
}

def load_by_adapter(args) -> Tuple[List[Example], Any]:
    mod = importlib.import_module(ADAPTER_REG[args.dataset])
    if not hasattr(mod, "load"):
        raise AttributeError(f"{ADAPTER_REG[args.dataset]} must implement load(args)->List[Example]")
    raw = mod.load(args)
    exs: List[Example] = []
    for x in raw:
        if isinstance(x, Example):
            exs.append(x)
        elif isinstance(x, dict):
            exs.append(Example(
                id=str(x.get("id")),
                question=x.get("question",""),
                context=x.get("context",""),
                answers=list(x.get("answers", [])),
                is_unanswerable=bool(x.get("is_unanswerable", False)),
                meta={k:v for k,v in x.items() if k not in ["id","question","context","answers","is_unanswerable"]}
            ))
        else:
            raise TypeError("Adapter.load must return Example or dict.")
    return exs, mod

# =============== Extractive metrics (EM/F1) ===============
_ART = re.compile(r"\b(a|an|the)\b", re.UNICODE)
def _norm(s: str) -> str:
    s = s.lower()
    s = _ART.sub(" ", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    return " ".join(s.split())

def f1(pred: str, gold: str) -> float:
    pt, gt = _norm(pred).split(), _norm(gold).split()
    if not pt or not gt: return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0: return 0.0
    P, R = n/len(pt), n/len(gt)
    return 2*P*R/(P+R)

def match_extractive(pred: str, golds: List[str], thr: float=0.8) -> Tuple[bool,float,float]:
    if not golds: return False, 0.0, float(_norm(pred)=="")
    f1s = [f1(pred,g) for g in golds]
    ems = [float(_norm(pred)==_norm(g)) for g in golds]
    return (max(f1s)>=thr or max(ems)>=1.0, max(f1s), max(ems))

# =============== Math metrics ===============
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FRAC  = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
_PCT   = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%$")

def _strip_tex(s: str) -> str:
    s = s.strip()
    m = _BOXED.search(s)
    if m: s = m.group(1)
    s = s.replace("$","")
    s = _FRAC.sub(lambda m: str(float(m.group(1))/float(m.group(2)) if m.group(2)!='0' else m.group(1)), s)
    return " ".join(s.replace(",", " ").split())

def _num(s: str) -> Optional[float]:
    s = _strip_tex(s)
    if "####" in s: s = s.split("####")[-1].strip()
    m = _PCT.match(s)
    if m:
        try: return float(m.group(1)) / 100.0
        except: pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums: return None
    try: return float(nums[-1])
    except: return None

def match_math(pred: str, golds: List[str]) -> Tuple[bool,float,float]:
    pn = _num(pred)
    for g in golds:
        gn = _num(g)
        if pn is not None and gn is not None:
            if gn == 0:
                if abs(pn) <= 1e-9: return True, 0.0, 1.0
            else:
                rel = abs(pn-gn)/(abs(gn)+1e-12)
                if rel <= 1e-6: return True, rel, float(pn==gn)
    # fallback textual
    pt = _norm(_strip_tex(pred))
    gts = [_norm(_strip_tex(g)) for g in golds]
    hit = pt in gts
    return hit, (0.0 if hit else 1.0), float(hit)

# =============== Prompt builders ===============
def default_prompt(ex: Example, tokenizer=None, use_chat_template=False,
                   include_unanswerable_hint=False, handle_unanswerable=False,
                   task_type: str="extractive") -> str:
    if task_type == "extractive":
        hint = "\nIf the question cannot be answered from the context, reply with: 'unanswerable'." \
               if include_unanswerable_hint else ""
        content = f"Context:\n{ex.context}\n\nQuestion: {ex.question}\nAnswer:{hint}\n"
        if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            msgs=[{"role":"system","content":"You are a helpful RC assistant."},
                  {"role":"user","content":content}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return content

    # math
    content = (
        "Solve the following problem. Respond with the final numeric answer only. "
        "If it is a fraction/percentage, give the numeric form.\n"
        f"Problem: {ex.question}\nAnswer:"
    )
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        msgs=[{"role":"system","content":"You are a careful math solver."},
              {"role":"user","content":content}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return content

# =============== Backends ===============
def build_generator(args):
    """
    Return: tokenizer_like, generate_batch(prompts)->List[str], max_model_len
    """
    max_model_len = args.max_input_tokens + args.max_new_tokens

    # ---------- vLLM (default) ----------
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model,
            tokenizer=args.tokenizer_path or args.model,
            tensor_parallel_size=args.tp,                 # multi-card here
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        tok = llm.get_tokenizer()
        sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_new_tokens)

        def _gen_batch(prompts: List[str]) -> List[str]:
            outs = llm.generate(prompts, sp)
            return [o.outputs[0].text for o in outs]

        return tok, _gen_batch, max_model_len

    # ---------- SGLang (OpenAI-compatible client) ----------
    if args.backend == "sglang":
        import requests, concurrent.futures
        # 为了 chat template，我们尽量加载本地 tokenizer；失败也不影响调用
        tok = None
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                args.tokenizer_path or args.model,
                trust_remote_code=True, local_files_only=args.local_files_only
            )
        except Exception:
            tok = None

        api_base = args.sglang_api_base.rstrip("/")
        model_id = args.sglang_model or args.model
        headers = {"Content-Type":"application/json"}
        if args.sglang_api_key:
            headers["Authorization"] = f"Bearer {args.sglang_api_key}"

        def _one(prompt: str) -> str:
            # 用 chat.completions 端点
            payload = {
                "model": model_id,
                "messages": [{"role":"user","content": prompt}],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_new_tokens
            }
            r = requests.post(f"{api_base}/v1/chat/completions",
                              headers=headers, data=json.dumps(payload), timeout=120)
            r.raise_for_status()
            js = r.json()
            return js["choices"][0]["message"]["content"]

        def _gen_batch(prompts: List[str]) -> List[str]:
            # 简单并发客户端（服务器端负责多卡/并行）
            outs = [None]*len(prompts)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,args.concurrency)) as ex:
                futs = {ex.submit(_one, p): idx for idx,p in enumerate(prompts)}
                for fut in concurrent.futures.as_completed(futs):
                    idx = futs[fut]
                    try:
                        outs[idx] = fut.result()
                    except Exception as e:
                        outs[idx] = f"[ERROR:{e}]"
            return outs

        return tok, _gen_batch, max_model_len

    # ---------- Transformers ----------
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=args.local_files_only
    )
    model_kwargs = dict(trust_remote_code=True, local_files_only=args.local_files_only)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map  # e.g., "auto" (multi-card if supported)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # device placement
    if not args.device_map:
        if args.device.startswith("cuda"):
            model = model.to("cuda")
        elif args.device.startswith("npu"):
            import torch_npu  # noqa
            model = model.to(args.device)
        else:
            model = model.to("cpu")

    def _gen_batch(prompts: List[str]) -> List[str]:
        outs = []
        for p in prompts:
            inputs = tok(p, return_tensors="pt", truncation=True,
                         max_length=args.max_input_tokens)
            # move to device
            if not args.device_map:
                inputs = inputs.to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=args.temperature,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
            )
            text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(text)
        return outs

    return tok, _gen_batch, max_model_len

# =============== IO helpers ===============
def write_json(path: str, examples: Iterable[Example]):
    data = [asdict(ex) for ex in examples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(path: str, rows: List[Dict[str, Any]]):
    import csv
    keys = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def export_squad_like(path: str, examples: List[Example]):
    data = []
    paras = []
    for ex in examples:
        paras.append({
            "context": ex.context,
            "qas": [{
                "id": ex.id,
                "question": ex.question,
                "answers": [{"text": a, "answer_start": -1} for a in ex.answers],
                "is_impossible": ex.is_unanswerable
            }]
        })
    data.append({"title":"filtered","paragraphs":paras})
    obj = {"version":"filtered-1.0","data":data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =============== Main ===============
def main():
    args = build_args()

    # load data
    print(f"[Load] dataset={args.dataset} source={args.source} split={args.split} input={args.input}")
    examples, adapter_mod = load_by_adapter(args)
    print(f"[Load] {len(examples)} examples")

    task_type = "extractive" if args.dataset in ("squadv2","hotpot","nq") else "math"

    # backend
    print(f"[Init] backend={args.backend} model={args.model}")
    tokenizer, generate_batch, _ = build_generator(args)

    # adapter-specific prompt if provided
    build_prompt_fn: Optional[Callable] = getattr(adapter_mod, "build_prompt", None)

    def make_prompt(ex: Example) -> str:
        if build_prompt_fn is not None:
            return build_prompt_fn(ex, tokenizer)
        return default_prompt(
            ex, tokenizer=tokenizer, use_chat_template=args.use_chat_template,
            include_unanswerable_hint=args.include_unanswerable_hint,
            handle_unanswerable=args.handle_unanswerable,
            task_type=task_type
        )

    bs = max(1, args.batch_size)
    kept: List[Example] = []
    logs: List[Dict[str, Any]] = []

    t0 = time()
    for i in tqdm(range(0, len(examples), bs), ncols=100, desc="Filtering"):
        batch = examples[i:i+bs]
        prompts = [make_prompt(ex) for ex in batch]
        preds = generate_batch(prompts)

        for ex, pred in zip(batch, preds):
            pred = (pred or "").strip()
            if task_type == "extractive":
                hit, s1, s2 = match_extractive(pred, ex.answers, args.f1_threshold)
            else:
                hit, s1, s2 = match_math(pred, ex.answers)

            is_easy = bool(hit)  # hit = 小模型能答 → 过滤
            if not is_easy:
                kept.append(ex)
            logs.append({
                "id": ex.id, "dataset": args.dataset, "is_easy": int(is_easy),
                "score1": s1, "score2": s2, "pred": pred,
                "gold": ex.answers[0] if ex.answers else "",
                "question": (ex.question[:120]+"...") if len(ex.question)>120 else ex.question
            })
    t1 = time()
    print(f"[Done] kept {len(kept)} / {len(examples)} in {t1-t0:.1f}s")

    # statistics
    total = len(logs)
    hits = sum(r["is_easy"] for r in logs)
    accuracy = hits / total if total else 0.0
    print(f"[Stats] accuracy={accuracy:.4f} ({hits}/{total})")

    # summary export for math datasets
    if args.dataset in ("aqua", "gsm8k", "math"):
        summary_dir = Path("results") / "math_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{args.dataset}.json"
        summary_payload = {
            "dataset": args.dataset,
            "total_examples": total,
            "easy_examples": hits,
            "filtered_examples": len(kept),
            "accuracy": accuracy,
            "parameters": {
                "model": args.model,
                "backend": args.backend,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        print(f"[Write] summary json -> {summary_path}")

    # outputs
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_json(args.output, kept)
    print(f"[Write] filtered json -> {args.output}")
    if args.save_csv and logs:
        write_csv(args.save_csv, logs)
        print(f"[Write] decisions csv -> {args.save_csv}")
    if args.export_squad_like and task_type == "extractive":
        p = Path(args.output).with_suffix(".squad.json")
        export_squad_like(str(p), kept)
        print(f"[Write] squad-like json -> {p}")

if __name__ == "__main__":
    main()
