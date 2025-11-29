#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agentic Reasoning controller evaluation for TIRBench-style QA datasets.

- Controller LLM 逐步选择动作:
    SEARCH  : 在当前样本上下文中“查阅”信息
    CODE    : 运行一小段 Python 代码进行计算
    MINDMAP : 用 LLM 整理当前证据为思维导图式摘要
    ANSWER  : 给出最终回答

- 依赖:
    - openai>=1.0.0
    - 现有的 adapter: adapter.squadv2_adapter, adapter.hotpot_adapter, adapter.nq_adapter, ...

用法示例:

    # 先在另一终端起 vLLM:
    # python -m vllm.entrypoints.openai.api_server \
    #   --model /root/autodl-tmp/models/Qwen_Qwen3-4B \
    #   --tokenizer /root/autodl-tmp/models/Qwen_Qwen3-4B \
    #   --tensor-parallel-size 4 \
    #   --max-model-len 2304 \
    #   --port 8000 \
    #   --served-model-name Qwen_Qwen3-4B

    export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
    export OPENAI_API_KEY="EMPTY"

    python eval_agentic_reasoning.py \
        --dataset squadv2 \
        --source json \
        --input /root/autodl-tmp/TIRBench/data/squadv2.json \
        --max-examples 500 \
        --model Qwen_Qwen3-4B \
        --save-json results/agentic_squadv2.json

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------ 路径 & Adapter 映射 ------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent  # 根据你放的位置必要时改一下
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

ADAPTER_REG: Dict[str, str] = {
    # 自然语言 QA
    "squadv2": "adapter.squadv2_adapter",
    "hotpotqa": "adapter.hotpot_adapter",
    "nq": "adapter.nq_adapter",
    # 如果你有数学类 adapter 也可以加上:
    # "aqua": "adapter.aqua_adapter",
    # "gsm8k": "adapter.gsm8k_adapter",
    # "math": "adapter.math_adapter",
}

# ------------------------ OpenAI 客户端 ------------------------

from openai import OpenAI


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


# ------------------------ 数据结构 & 评测工具 ------------------------


@dataclass
class QAExample:
    id: str
    question: str
    context: str
    answers: List[str]
    is_unanswerable: bool = False


def normalize_example(obj: Any) -> QAExample:
    """将 adapter 返回的 dict / 对象统一成 QAExample."""
    if isinstance(obj, dict):
        return QAExample(
            id=str(obj.get("id", "")),
            question=obj.get("question", ""),
            context=obj.get("context", ""),
            answers=list(obj.get("answers", [])),
            is_unanswerable=bool(obj.get("is_unanswerable", False)),
        )
    # 简单兜底: 假设是带属性的对象
    return QAExample(
        id=str(getattr(obj, "id", "")),
        question=getattr(obj, "question", ""),
        context=getattr(obj, "context", ""),
        answers=list(getattr(obj, "answers", [])),
        is_unanswerable=bool(getattr(obj, "is_unanswerable", False)),
    )


import re
import string


def _normalize_answer(s: str) -> str:
    """SQuAD 风格的答案归一化."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = min(common.get(t, 0) + 1, gt_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em_f1(prediction: str, answers: List[str]) -> Tuple[float, float]:
    if not answers:
        return 0.0, 0.0
    em = 0.0
    f1 = 0.0
    for a in answers:
        em = max(em, float(_normalize_answer(prediction) == _normalize_answer(a)))
        f1 = max(f1, _f1_score(prediction, a))
    return em, f1


# ------------------------ Agentic 工具实现 ------------------------


@dataclass
class ReasoningState:
    question: str
    context: str
    step: int = 0
    evidence: List[str] = None
    search_history: List[str] = None
    code_history: List[str] = None
    mindmap: Optional[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.search_history is None:
            self.search_history = []
        if self.code_history is None:
            self.code_history = []


class SearchAgent:
    """简化版 SearchAgent: 对于 SQuAD/Hotpot/NQ，只返回当前样本 context.

    你以后可以替换成 BM25 / Faiss / 真正的 web search.
    """

    def search(self, ex: QAExample, query: str) -> str:
        # 这里简单地返回 context，顺便记录 query
        return ex.context or ""


class CodeAgent:
    """极简 Python 代码解释器，仅用于数值计算.

    安全性有限，只建议在你自己受控的环境里使用。
    """

    SAFE_BUILTINS = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
    }

    def run(self, code: str) -> str:
        local_env: Dict[str, Any] = {}
        try:
            exec(
                code,
                {"__builtins__": self.SAFE_BUILTINS},
                local_env,
            )
            if "result" in local_env:
                return repr(local_env["result"])
            # 没有约定返回值，就把局部变量都打出来
            return repr(local_env)
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {e}"


class MindMapAgent:
    """利用 LLM 把当前证据整理成思维导图式摘要."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def update(self, state: ReasoningState, note: str = "") -> str:
        content = (
            "You are a mind-map assistant.\n"
            "Given the question, current context snippet and accumulated evidence,\n"
            "summarize them into a mind-map style outline with bullet points and short phrases.\n\n"
            f"Question:\n{state.question}\n\n"
            f"Context (truncated):\n{state.context[:800]}\n\n"
            "Current evidence:\n"
            + "\n".join(f"- {e}" for e in state.evidence[-8:])
        )
        if note:
            content += f"\n\nUser note for update:\n{note}\n"
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You write compact mind-map style notes."},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


# ------------------------ 控制器（Agentic Reasoning Loop） ------------------------

CONTROLLER_SYSTEM_PROMPT = """You are an agentic reasoning controller that solves question answering tasks.

You have four high-level actions:

1. "SEARCH": Re-read or retrieve relevant parts of the context.
   - Use this when you are not sure which parts of the passage are relevant.
   - Provide a "search_query" field describing what you want to look for.

2. "CODE": Execute a small Python snippet to compute something.
   - Use this for arithmetic or algorithmic sub-problems.
   - Provide a "code" field. The environment will run it and give you the result.

3. "MINDMAP": Organize what you know so far into a concise mind-map style summary.
   - Use this when the problem is complex and you need to structure evidence.
   - Provide an optional "note" describing what to focus on.

4. "ANSWER": When you are ready to give the final answer to the user.
   - Provide "final_answer" – a short span that answers the question.

At each step, you MUST output a single JSON object with the following keys:
- "action": one of ["SEARCH", "CODE", "MINDMAP", "ANSWER"]
- Optional fields depending on the action:
  - for SEARCH:    "search_query": string
  - for CODE:      "code": string
  - for MINDMAP:   "note": string (optional)
  - for ANSWER:    "final_answer": string

Do NOT include any other text outside the JSON.
"""


def extract_json(text: str) -> Dict[str, Any]:
    """从模型输出中尽量提取 JSON."""
    text = text.strip()
    # 去掉 ```json ``` 包裹
    if text.startswith("```"):
        # ```json\n{...}\n```
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[: -3].strip()
    # 找到最外层的大括号
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    try:
        return json.loads(text)
    except Exception:
        # 失败就返回一个简单的 ANSWER 动作
        return {"action": "ANSWER", "final_answer": text}


class AgenticController:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_steps: int = 6,
        temperature: float = 0.3,
    ):
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature

        self.search_agent = SearchAgent()
        self.code_agent = CodeAgent()
        self.mindmap_agent = MindMapAgent(client, model)

    def build_state_prompt(self, state: ReasoningState) -> str:
        """把当前状态串成一段文本喂给 controller LLM."""
        parts = [
            f"Question:\n{state.question}",
            "\nContext snippet (may be incomplete):",
            state.context[:800] or "(empty)",
        ]
        if state.evidence:
            parts.append("\nRecent evidence:")
            for e in state.evidence[-6:]:
                parts.append(f"- {e}")
        if state.mindmap:
            parts.append("\nCurrent mind map (if any):")
            parts.append(state.mindmap[:800])
        parts.append(
            "\nDecide the next best action (SEARCH, CODE, MINDMAP, or ANSWER) "
            "and respond with a JSON object as specified."
        )
        return "\n".join(parts)

    def run(self, ex: QAExample) -> Tuple[str, List[Dict[str, Any]]]:
        """运行一个样本，返回 (prediction, trace)."""
        state = ReasoningState(question=ex.question, context=ex.context)
        trace: List[Dict[str, Any]] = []

        final_answer: Optional[str] = None

        for step in range(self.max_steps):
            state.step = step
            user_content = self.build_state_prompt(state)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
            )
            msg = resp.choices[0].message
            action_obj = extract_json(msg.content or "")

            action = (action_obj.get("action") or "").upper()
            step_record: Dict[str, Any] = {
                "step": step,
                "raw_output": msg.model_dump(),
                "parsed_action": action_obj,
            }

            if action == "SEARCH":
                query = action_obj.get("search_query") or ex.question
                out = self.search_agent.search(ex, query)
                state.search_history.append(query)
                state.evidence.append(f"[SEARCH] {out[:400]}")
                step_record["tool"] = "SEARCH"
                step_record["query"] = query
                step_record["tool_output"] = out[:400]

            elif action == "CODE":
                code = action_obj.get("code") or ""
                out = self.code_agent.run(code)
                state.code_history.append(code)
                state.evidence.append(f"[CODE] code={code!r} result={out!r}")
                step_record["tool"] = "CODE"
                step_record["code"] = code
                step_record["tool_output"] = out

            elif action == "MINDMAP":
                note = action_obj.get("note") or ""
                mm = self.mindmap_agent.update(state, note)
                state.mindmap = mm
                state.evidence.append(f"[MINDMAP] {mm[:200]}")
                step_record["tool"] = "MINDMAP"
                step_record["note"] = note
                step_record["tool_output"] = mm

            elif action == "ANSWER":
                final_answer = action_obj.get("final_answer") or ""
                step_record["tool"] = "ANSWER"
                step_record["final_answer"] = final_answer
                trace.append(step_record)
                break

            else:
                # 未知动作，当作直接回答
                final_answer = action_obj.get("final_answer") or ""
                step_record["tool"] = "ANSWER?"
                step_record["final_answer"] = final_answer
                trace.append(step_record)
                break

            trace.append(step_record)

        # 如果 max_steps 内还没 ANSWER，再强制问一次
        if final_answer is None:
            fallback_prompt = (
                f"Question: {ex.question}\n\nContext:\n{ex.context}\n\n"
                "Based on the context, answer the question with a short span. "
                "If it cannot be answered, reply exactly with: unanswerable."
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise extractive QA assistant.",
                    },
                    {"role": "user", "content": fallback_prompt},
                ],
                temperature=0.0,
            )
            final_answer = (resp.choices[0].message.content or "").strip()
            trace.append(
                {
                    "step": self.max_steps,
                    "fallback": True,
                    "final_answer": final_answer,
                }
            )

        return final_answer.strip(), trace


# ------------------------ 数据加载 & 主流程 ------------------------


def load_by_adapter(args) -> Tuple[List[QAExample], Any]:
    import importlib

    if args.dataset not in ADAPTER_REG:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    mod_path = ADAPTER_REG[args.dataset]
    mod = importlib.import_module(mod_path)

    if not hasattr(mod, "load"):
        raise ValueError(f"Adapter {mod_path} has no `load(args)` function.")

    print(
        f"[Load] dataset={args.dataset} source={args.source} "
        f"input={getattr(args, 'input', None)}"
    )
    raw = mod.load(args)  # list[dict] 或对象
    examples = [normalize_example(x) for x in raw]

    if args.max_examples is not None and args.max_examples > 0:
        examples = examples[: args.max_examples]

    return examples, mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Reasoning Controller Evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(ADAPTER_REG.keys()),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="json",
        choices=["json", "hf"],
        help="Data source type, must be supported by adapter.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Local dataset path when source=json.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="HF split name when source=hf.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Max number of examples to evaluate (for quick debug).",
    )

    # LLM & client config
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL (vLLM server).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key (vLLM can ignore it).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "Qwen_Qwen3-4B"),
        help="Model name exposed by the OpenAI server.",
    )
    parser.add_argument(
        "--controller-steps",
        type=int,
        default=6,
        help="Max reasoning steps of the controller.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Controller sampling temperature.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save per-example results (JSON).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    client = make_client(args.base_url, args.api_key)
    controller = AgenticController(
        client=client,
        model=args.model,
        max_steps=args.controller_steps,
        temperature=args.temperature,
    )

    examples, _ = load_by_adapter(args)
    n = len(examples)
    print(f"[Eval] {n} examples, model={args.model}")

    all_results: List[Dict[str, Any]] = []
    n_em = 0.0
    n_f1 = 0.0

    t0 = time.time()
    for idx, ex in enumerate(examples, 1):
        pred, trace = controller.run(ex)
        em, f1 = compute_em_f1(pred, ex.answers)
        n_em += em
        n_f1 += f1

        all_results.append(
            {
                "id": ex.id,
                "question": ex.question,
                "context": ex.context,
                "gold_answers": ex.answers,
                "prediction": pred,
                "em": em,
                "f1": f1,
                "trace": trace,
            }
        )

        if idx % 10 == 0 or idx == n:
            print(
                f"Evaluating: {idx}/{n}  "
                f"running_em={n_em/idx:.4f}  running_f1={n_f1/idx:.4f}",
                flush=True,
            )

    t1 = time.time()
    avg_em = n_em / n if n > 0 else 0.0
    avg_f1 = n_f1 / n if n > 0 else 0.0
    acc = avg_em  # 对抽取式 QA，acc=EM

    print(
        f"[Done] dataset={args.dataset} n={n} "
        f"acc={acc:.4f} avg_f1={avg_f1:.4f} avg_em={avg_em:.4f} time={t1 - t0:.1f}s"
    )

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[Save] results -> {out_path}")


if __name__ == "__main__":
    main()
