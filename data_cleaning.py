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
_TEXT  = re.compile(r"\\text\{([^}]*)\}")

def _strip_tex(s: str) -> str:
    """
    清理 LaTeX 格式，尽可能提取数值或简化表达式
    """
    s = s.strip()
    
    # 提取 \boxed{} 中的内容
    m = _BOXED.search(s)
    if m: 
        s = m.group(1)
    
    # 移除美元符号
    s = s.replace("$", "")
    
    # 尝试处理 \frac{a}{b}，只处理 a,b 都是数字的情况
    def safe_frac_sub(match):
        try:
            numerator = match.group(1).strip()
            denominator = match.group(2).strip()
            # 只处理纯数字的分数
            if denominator != '0':
                num_val = float(numerator)
                den_val = float(denominator)
                return str(num_val / den_val)
            else:
                return match.group(1)
        except (ValueError, ZeroDivisionError):
            # 如果不是数字，保留原始格式（去掉 \frac 但保留内容）
            return f"({match.group(1)})/({match.group(2)})"
    
    s = _FRAC.sub(safe_frac_sub, s)
    
    # 处理其他常见的 LaTeX 命令
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)  # \text{euros} -> euros
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)  # \mathrm{...} -> ...
    s = re.sub(r'\\sqrt\[(\d+)\]\{([^}]*)\}', r'root\1(\2)', s)  # \sqrt[3]{x} -> root3(x)
    s = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)  # \sqrt{x} -> sqrt(x)
    
    # 移除多余的空格和逗号
    s = s.replace(",", " ")
    s = " ".join(s.split())
    
    return s

def _normalize_latex_answer(s: str) -> str:
    """
    标准化 LaTeX 答案用于比较
    处理各种格式：区间、方程、表达式等
    """
    s = s.strip()
    
    # 提取 \boxed{} 内容
    m = _BOXED.search(s)
    if m:
        s = m.group(1)
    
    # 标准化常见符号
    s = s.replace("$", "")
    s = s.replace(" ", "")
    s = s.replace("\\,", "")
    s = s.replace("\\:", "")
    s = s.replace("\\;", "")
    
    # 标准化分数：\frac{a}{b} -> a/b
    s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', s)
    
    # 标准化根号
    s = re.sub(r'\\sqrt\[(\d+)\]\{([^}]*)\}', r'root\1(\2)', s)
    s = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)
    
    # 标准化文本
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    
    # 标准化括号
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\{", "{").replace("\\}", "}")
    
    # 移除其他常见 LaTeX 命令
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    
    return s.lower().strip()

def _num(s: str) -> Optional[float]:
    """
    从字符串中提取数值
    注意：对于 GSM8K 等简单数字，不应该过度处理
    """
    if not s:
        return None
    
    s = s.strip()
    
    # 优先处理 #### 格式（GSM8K 标准格式）
    if "####" in s:
        s = s.split("####")[-1].strip()
    
    # 只对包含 LaTeX 的字符串调用 _strip_tex
    if "\\" in s or "$" in s:
        try:
            s = _strip_tex(s).strip()
        except:
            pass
    
    # 移除常见的单位词和符号
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r'\b(clips|dollars|cents|people|items|units|days|hours|years|months)\b', '', s, flags=re.IGNORECASE)
    s = s.strip()
    
    # 处理百分比
    m = _PCT.match(s)
    if m:
        try: 
            return float(m.group(1)) / 100.0
        except: 
            pass
    
    # 提取所有数字（支持负数和小数）
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums: 
        return None
    
    # 返回最后一个数字（通常是最终答案）
    try: 
        return float(nums[-1])
    except: 
        return None

def _extract_answer_from_response(s: str, dataset: str) -> str:
    """
    从模型响应中提取答案，根据数据集类型使用不同策略
    """
    if not s:
        return ""
    
    s = s.strip()
    
    if dataset == "gsm8k":
        # GSM8K: 严格提取 #### 后的内容
        if "####" in s:
            ans = s.split("####")[-1].strip()
            # 移除可能的单位词
            ans = re.sub(r'\b(clips|dollars|cents|people|items|units|days|hours)\b', '', ans, flags=re.IGNORECASE)
            ans = ans.strip()
            # 提取数字部分
            nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
            if nums:
                return nums[0]  # 返回第一个数字
            return ans
        else:
            # 如果没有 ####，尝试从最后一行提取
            lines = s.strip().split('\n')
            for line in reversed(lines):
                nums = re.findall(r"-?\d+(?:\.\d+)?", line)
                if nums:
                    return nums[-1]
            return s
    
    elif dataset == "math":
        # MATH: 优先提取 \boxed{} 中的内容
        m = _BOXED.search(s)
        if m:
            return m.group(1).strip()
        # 如果没有 \boxed{}，返回原文
        return s
    
    else:
        # 其他数据集，返回原文
        return s

def match_math(pred: str, golds: List[str], dataset: str = None) -> Tuple[bool, float, float]:
    """
    匹配数学答案
    - GSM8K: 纯数字比较
    - MATH: LaTeX + 文本比较
    """
    if not pred or not golds:
        return False, 1.0, 0.0
    
    pred = pred.strip()
    
    # 根据数据集提取答案
    if dataset:
        pred_extracted = _extract_answer_from_response(pred, dataset)
    else:
        pred_extracted = pred
    
    # ============ GSM8K 专用逻辑：纯数字比较 ============
    if dataset == "gsm8k":
        pred_num = _num(pred_extracted)
        if pred_num is None:
            # 如果提取失败，尝试从原始预测中提取
            pred_num = _num(pred)
        
        for g in golds:
            gold_num = _num(g)
            if pred_num is not None and gold_num is not None:
                # 数值比较（允许小误差）
                if abs(pred_num - gold_num) < 1e-6:
                    return True, 0.0, 1.0
        
        # GSM8K 失败就是失败，不需要其他策略
        return False, 1.0, 0.0
    
    # ============ MATH 专用逻辑：多策略比较 ============
    if dataset == "math":
        # 策略 1: 数值比较（如果答案是纯数字）
        pred_num = _num(pred_extracted)
        for g in golds:
            gold_num = _num(g)
            if pred_num is not None and gold_num is not None:
                if gold_num == 0:
                    if abs(pred_num) <= 1e-9:
                        return True, 0.0, 1.0
                else:
                    rel = abs(pred_num - gold_num) / (abs(gold_num) + 1e-12)
                    if rel <= 1e-4:
                        return True, rel, float(abs(pred_num - gold_num) < 1e-9)
        
        # 策略 2: LaTeX 标准化比较
        try:
            pred_latex = _normalize_latex_answer(pred_extracted)
            for g in golds:
                gold_latex = _normalize_latex_answer(g)
                if pred_latex and gold_latex and pred_latex == gold_latex:
                    return True, 0.0, 1.0
        except:
            pass
        
        # 策略 3: 直接文本比较（坐标、区间等）
        pred_clean = pred_extracted.strip().replace(" ", "").lower()
        for g in golds:
            gold_clean = g.strip().replace(" ", "").lower()
            if pred_clean == gold_clean:
                return True, 0.0, 1.0
        
        # 策略 4: 标准化文本比较
        try:
            pt = _norm(_strip_tex(pred_extracted))
            for g in golds:
                gt = _norm(_strip_tex(g))
                if pt and gt and pt == gt:
                    return True, 0.0, 1.0
        except:
            pass
        
        return False, 1.0, 0.0
    
    # ============ 其他数据集：默认数值比较 ============
    pred_num = _num(pred)
    for g in golds:
        gold_num = _num(g)
        if pred_num is not None and gold_num is not None:
            if abs(pred_num - gold_num) < 1e-6:
                return True, 0.0, 1.0
    
    return False, 1.0, 0.0

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
    # 用于保存答案对比（仅 gsm8k 和 math）
    answer_comparisons: List[Dict[str, Any]] = []

    t0 = time()
    for i in tqdm(range(0, len(examples), bs), ncols=100, desc="Filtering"):
        batch = examples[i:i+bs]
        prompts = [make_prompt(ex) for ex in batch]
        preds = generate_batch(prompts)

        for ex, pred in zip(batch, preds):
            pred = (pred or "").strip()
            
            # AQuA 使用多选题匹配
            if args.dataset == "aqua":
                # 从 context 中提取 options
                options = []
                if ex.context.startswith("Options:"):
                    options = [line.strip() for line in ex.context.split('\n')[1:] if line.strip()]
                hit, s1, s2 = match_multiple_choice(pred, ex.answers, options)
            elif task_type == "extractive":
                hit, s1, s2 = match_extractive(pred, ex.answers, args.f1_threshold)
            else:
                hit, s1, s2 = match_math(pred, ex.answers)

            # 对于 gsm8k 和 math，保存答案对比
            if args.dataset in ("gsm8k", "math"):
                pred_extracted = _extract_answer_from_response(pred, args.dataset)
                gold_extracted = ex.answers[0] if ex.answers else ""
                answer_comparisons.append({
                    "id": ex.id,
                    "question": ex.question,
                    "model_raw_output": pred,
                    "model_extracted_answer": pred_extracted,
                    "gold_raw_answer": gold_extracted,
                    "gold_extracted_number": str(_num(gold_extracted)) if _num(gold_extracted) is not None else gold_extracted,
                    "model_extracted_number": str(_num(pred_extracted)) if _num(pred_extracted) is not None else pred_extracted,
                    "match": bool(hit),
                    "match_score": float(s2)
                })

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
    
    # 输出答案对比 JSON（仅 gsm8k 和 math）
    if args.dataset in ("gsm8k", "math") and answer_comparisons:
        comparison_dir = Path("results") / "answer_comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = comparison_dir / f"{args.dataset}_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(answer_comparisons, f, ensure_ascii=False, indent=2)
        print(f"[Write] answer comparison json -> {comparison_path}")
        
def match_multiple_choice(pred: str, golds: List[str], options: List[str] = None) -> Tuple[bool, float, float]:
    """
    匹配多选题答案（如 AQuA）
    支持：
    - 直接匹配字母：A, B, C, D, E
    - 从文本中提取：The answer is E
    - 通过数字反推：如果输出 23，且选项 E)23，则匹配 E
    """
    pred = pred.strip()
    
    # 1. 直接提取字母（A-E）
    # 匹配单独的字母或 "answer is X" 格式
    letter_match = re.search(r'\b([A-E])\b', pred.upper())
    if letter_match:
        pred_letter = letter_match.group(1)
        for g in golds:
            if pred_letter.upper() == g.upper():
                return True, 0.0, 1.0
    
    # 2. 如果提供了 options，尝试通过数字反推
    if options:
        pred_num = _num(pred)
        if pred_num is not None:
            # 遍历选项，看哪个选项包含这个数字
            for opt in options:
                # 选项格式：A)21, B)21.5, C)22, D)22.5, E)23
                opt_match = re.match(r'([A-E])\)(.*)', opt.strip())
                if opt_match:
                    opt_letter = opt_match.group(1)
                    opt_value_str = opt_match.group(2).strip()
                    opt_num = _num(opt_value_str)
                    if opt_num is not None and abs(pred_num - opt_num) < 1e-6:
                        # 找到了对应的选项，检查是否是正确答案
                        for g in golds:
                            if opt_letter.upper() == g.upper():
                                return True, 0.0, 1.0
    
    # 3. Fallback 文本匹配
    pt = _norm(pred)
    for g in golds:
        gt = _norm(g)
        if pt == gt or gt in pt or pt in gt:
            return True, 0.0, 1.0
    
    return False, 1.0, 0.0

if __name__ == "__main__":
    main()
