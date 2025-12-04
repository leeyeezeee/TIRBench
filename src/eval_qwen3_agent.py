#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval Qwen3-4B-thinking + Qwen-Agent(with tools) on cleaned datasets:

- Natural language extractive QA: SQuADv2 / HotpotQA / NQ
- Math QA: AQuA / GSM8K / MATH
"""
import argparse
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

# ===== Adapter registry (reuse existing TIRBench adapters) =====
ADAPTER_REG = {
    "squadv2": "adapter.squadv2_adapter",
    "hotpot":  "adapter.hotpot_adapter",
    "nq":      "adapter.nq_adapter",
    "aqua":    "adapter.aqua_adapter",
    "gsm8k":   "adapter.gsm8k_adapter",
    "math":    "adapter.math_adapter",
}

@dataclass
class Example:
    id: str
    question: str
    context: str
    answers: List[str]
    is_unanswerable: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

def load_by_adapter(args) -> Tuple[List[Example], Any]:
    import importlib
    mod = importlib.import_module(ADAPTER_REG[args.dataset])
    raw = mod.load(args)
    exs: List[Example] = []
    for x in raw:
        if isinstance(x, Example):
            exs.append(x)
        elif isinstance(x, dict):
            exs.append(Example(
                id=str(x.get("id")),
                question=x.get("question", ""),
                context=x.get("context", ""),
                answers=list(x.get("answers", [])),
                is_unanswerable=bool(x.get("is_unanswerable", False)),
                meta={k: v for k, v in x.items()
                      if k not in ["id", "question", "context", "answers", "is_unanswerable"]}
            ))
        else:
            raise TypeError("Adapter.load must return Example or dict.")
    return exs, mod

# ===== Extractive QA metrics =====
_ART = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ART.sub(" ", s)
    s = " ".join(s.split())
    return s

def f1(pred: str, gold: str) -> float:
    pt = _norm(pred).split()
    gt = _norm(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = {}
    for t in pt:
        if t in gt:
            common[t] = common.get(t, 0) + 1
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pt)
    rec = num_same / len(gt)
    return 2 * prec * rec / (prec + rec)

def match_extractive(pred: str, golds: List[str], thr: float) -> Tuple[bool, float, float]:
    if not golds:
        return False, 0.0, 0.0
    f1s = [f1(pred, g) for g in golds]
    ems = [float(_norm(pred) == _norm(g)) for g in golds]
    best_f1 = max(f1s)
    best_em = max(ems)
    hit = (best_f1 >= thr) or (best_em >= 1.0)
    return hit, best_f1, best_em

_UNANS = {
    "", "unanswerable", "no answer", "noanswer",
    "cannot answer", "unknown", "cannot be determined",
    "not answerable", "impossible to answer",
}

def _is_unanswerable_pred(pred: str) -> bool:
    if not pred:
        return True
    t = _norm(pred)
    if t in _UNANS:
        return True
    toks = set(t.split())
    return any(u in toks for u in _UNANS if u)

def _is_yes_no_answers(golds: List[str]) -> bool:
    s = {_norm(a) for a in (golds or [])}
    return len(s) > 0 and s.issubset({"yes", "no"})

# strip <think> blocks & extract short answer
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:：]\s*(.+)",
    re.IGNORECASE,
)

def extract_qa_answer(s: str) -> str:
    """Remove <think> blocks and try to extract the final short answer."""
    if not s:
        return ""
    text = _THINK_BLOCK_RE.sub("", s).strip()
    m = _ANSWER_LINE_RE.search(text)
    if m:
        ans = m.group(1).strip()
    else:
        ans = text.strip().splitlines()[-1].strip()
    ans = ans.strip(' ."\'')
    low = ans.lower()
    for u in _UNANS:
        if not u:
            continue
        if u in low and len(low.split()) <= 5:
            return "unanswerable"
    return ans

def _judge_squad(pred: str, ex: Example, args) -> Tuple[bool, float, float]:
    if ex.is_unanswerable or not ex.answers or _norm(ex.answers[0]) in _UNANS:
        hit = _is_unanswerable_pred(pred)
        return hit, 1.0 if hit else 0.0, 1.0 if hit else 0.0
    thr = args.squad_f1_threshold or args.f1_threshold
    return match_extractive(pred, ex.answers, thr)

def _judge_hotpot(pred: str, ex: Example, args) -> Tuple[bool, float, float]:
    thr = args.hotpot_f1_threshold or args.f1_threshold
    return match_extractive(pred, ex.answers, thr)

def _judge_nq(pred: str, ex: Example, args) -> Tuple[bool, float, float]:
    if ex.is_unanswerable or not ex.answers or _norm(ex.answers[0]) in _UNANS:
        hit = _is_unanswerable_pred(pred)
        return hit, 1.0 if hit else 0.0, 1.0 if hit else 0.0
    if _is_yes_no_answers(ex.answers):
        hit = (_norm(pred) in {"yes", "no"} and _norm(pred) in {_norm(a) for a in ex.answers})
        return hit, 1.0 if hit else 0.0, 1.0 if hit else 0.0
    thr = args.nq_f1_threshold or args.f1_threshold
    return match_extractive(pred, ex.answers, thr)

# ===== Math metrics (simple for AQuA / GSM8K / MATH) =====
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _last_number(s: str) -> str:
    """Extract the last number in a string, or ''."""
    if not s:
        return ""
    nums = _NUM_RE.findall(s)
    return nums[-1] if nums else ""

def _norm_math_expr(s: str) -> str:
    """Normalize a math expression or short answer for string comparison."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("$", "")
    s = s.replace(" ", "")
    s = s.strip(".")
    s = s.replace(",", "")
    return s

def judge_math(dataset: str, pred: str, ex: Example) -> Tuple[bool, float, float]:
    """
    Lightweight judging for AQuA / GSM8K / MATH.
    """
    golds = ex.answers or []
    if not golds:
        return False, 0.0, 0.0

    if dataset == "aqua":
        # multiple choice: compare first letter
        p = pred.strip().upper()
        if p:
            p = p[0]
        gold_set = {(g.strip().upper()[0] if g.strip() else "") for g in golds}
        hit = p in gold_set
        score = float(hit)
        return hit, score, score

    if dataset == "gsm8k":
        pnum = _last_number(pred)
        hit = False
        for g in golds:
            gnum = _last_number(g)
            if pnum and gnum:
                try:
                    if float(pnum) == float(gnum):
                        hit = True
                        break
                except ValueError:
                    continue
        score = float(hit)
        return hit, score, score

    if dataset == "math":
        pnorm = _norm_math_expr(pred)
        gold_norms = {_norm_math_expr(g) for g in golds}
        hit = bool(pnorm and pnorm in gold_norms)
        score = float(hit)
        return hit, score, score

    return False, 0.0, 0.0

def judge_easy(dataset: str, pred: str, ex: Example, args) -> Tuple[bool, float, float]:
    """Return (hit, f1_like, em_like). For math, f1/em are just accuracy."""
    # Math datasets
    if dataset in ("aqua", "gsm8k", "math"):
        return judge_math(dataset, pred, ex)

    # Extractive QA
    if args.decision == "em":
        em = float(_norm(pred) in {_norm(a) for a in (ex.answers or [])})
        return (em >= 1.0), 0.0, em
    if args.decision == "f1":
        return match_extractive(pred, ex.answers, args.f1_threshold)

    if args.decision in ("auto", "squad_v2") and dataset == "squadv2":
        return _judge_squad(pred, ex, args)
    if args.decision in ("auto", "hotpot") and dataset == "hotpot":
        return _judge_hotpot(pred, ex, args)
    if args.decision in ("auto", "nq") and dataset == "nq":
        return _judge_nq(pred, ex, args)

    return match_extractive(pred, ex.answers, args.f1_threshold)

# ===== Qwen-Agent: tools + assistant =====
@register_tool("calculator")
class Calculator(BaseTool):
    """A simple calculator tool as an example."""
    description = "Evaluate a basic math expression like '2+3*4'."
    parameters = [{
        "name": "expression",
        "type": "string",
        "description": "A Python-style arithmetic expression, e.g. '2+3*4'",
        "required": True,
    }]

    def call(self, params: str, **kwargs) -> str:
        import json as _json
        try:
            data = _json.loads(params)
            expr = data.get("expression", "")
            result = eval(expr, {"__builtins__": {}})
            return _json.dumps({"result": result})
        except Exception as e:
            return _json.dumps({"error": str(e)})

def build_agent(dataset: str) -> Assistant:
    """
    Configure Qwen-Agent to talk to local vLLM OpenAI server hosting Qwen3-4B-thinking.

    NOTE: model 路径只在启动 vLLM server 时指定，这里只配：
      - model 名称（客户端请求里的 model 字段）
      - model_server URL
    """
    llm_cfg = {
        "model": "Qwen_Qwen3-4B",              # 和你请求里的 model 一致就行
        "model_server": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "generate_cfg": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
        },
    }
    if dataset in ("aqua", "gsm8k", "math"):
        system = (
            "You are a math problem solving assistant. "
            "Think step by step inside <think>...</think>, "
            "then give a single final numeric or symbolic answer. "
            "Format the last line as 'Final answer: ...'."
        )
    else:
        system = (
            "You are an extractive QA assistant. "
            "You must answer using ONLY the provided context when possible. "
            "If the question cannot be answered from the context, reply exactly with 'unanswerable'. "
            "You can use tools (like a calculator) if needed. "
            "Think step by step inside <think>...</think>, "
            "then give the final short answer on the last line as 'Final answer: ...'."
        )
    tools = ["calculator"]
    assistant = Assistant(
        llm=llm_cfg,
        system_message=system,
        function_list=tools,
    )
    return assistant

def call_agent(assistant: Assistant, prompt: str) -> str:
    """Run a single turn and return the assistant's final text reply."""
    messages = [{"role": "user", "content": prompt}]
    last_chunk = None
    for last_chunk in assistant.run(messages=messages):
        pass
    if not last_chunk:
        return ""
    assistant_msgs = [m for m in last_chunk if m.get("role") == "assistant"]
    if not assistant_msgs:
        return ""
    return assistant_msgs[-1].get("content", "")

def main():
    ap = argparse.ArgumentParser("Eval Qwen3-4B-thinking + tools on cleaned datasets.")
    ap.add_argument("--dataset", required=True,
                    choices=["squadv2", "hotpot", "nq", "aqua", "gsm8k", "math"])
    ap.add_argument("--source", choices=["json", "hf"], default="json")
    ap.add_argument("--input", type=str, required=True,
                    help="Path to cleaned dataset in original schema (json/jsonl).")
    ap.add_argument("--split", type=str, default="validation",
                    help="HF split name (only used when --source hf).")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="If >0, only evaluate on the first N examples.")
    ap.add_argument("--decision", type=str, default="auto",
                    choices=["auto", "em", "f1", "squad_v2", "hotpot", "nq"],
                    help="Only affects extractive QA datasets.")
    ap.add_argument("--f1_threshold", type=float, default=0.8)
    ap.add_argument("--squad_f1_threshold", type=float, default=0.8)
    ap.add_argument("--hotpot_f1_threshold", type=float, default=0.8)
    ap.add_argument("--nq_f1_threshold", type=float, default=0.8)
    args = ap.parse_args()

    print(f"[Load] dataset={args.dataset} source={args.source} input={args.input}")
    examples, adapter_mod = load_by_adapter(args)
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"[Eval] {len(examples)} examples")

    assistant = build_agent(args.dataset)

    total = 0
    total_hit = 0
    sum_f1 = 0.0
    sum_em = 0.0

    t0 = time.time()
    for ex in tqdm(examples, desc="Evaluating"):
        # 尽量复用你 adapter 里已经写好的 build_prompt
        if hasattr(adapter_mod, "build_prompt"):
            prompt = adapter_mod.build_prompt(ex, tokenizer=None)
        else:
            if args.dataset in ("aqua", "gsm8k", "math"):
                prompt = (
                    f"Question:\\n{ex.question}\\n\\n"
                    "Solve the problem step by step. "
                    "Think inside <think>...</think>, then output\\n"
                    "Final answer: <your answer here>"
                )
            else:
                prompt = (
                    f"Context:\\n{ex.context}\\n\\n"
                    f"Question: {ex.question}\\n"
                    "Think inside <think>...</think>, then output\\n"
                    "Final answer: <short answer here>"
                )

        raw_resp = call_agent(assistant, prompt)
        short_ans = extract_qa_answer(raw_resp)

        hit, s1, s2 = judge_easy(args.dataset, short_ans, ex, args)

        total += 1
        if hit:
            total_hit += 1
        sum_f1 += s1
        sum_em += s2

    t1 = time.time()
    acc = total_hit / total if total else 0.0
    avg_f1 = sum_f1 / total if total else 0.0
    avg_em = sum_em / total if total else 0.0
    print(f"[Done] dataset={args.dataset} n={total} "
          f"acc={acc:.4f} avg_f1={avg_f1:.4f} avg_em={avg_em:.4f} "
          f"time={t1-t0:.1f}s")

if __name__ == "__main__":
    main()
