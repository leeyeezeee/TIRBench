#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment execution logic for data cleaning.
This module contains all the core logic for running data cleaning experiments.
"""

from __future__ import annotations
import json
import os
import re
import random
import sys
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
from time import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np

# Import configuration loaders
sys.path.insert(0, str(Path(__file__).parent))
from dataformator.dataset_loader import DatasetLoader
try:
    from modelloader.llm_agent import LLMAgent
except ImportError:
    LLMAgent = None


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

_UNANS = {
    "", "unanswerable", "no answer", "noanswer",
    "cannot answer", "unknown", "cannot be determined",
    "not answerable", "impossible to answer",
}

def _is_unanswerable_pred(pred: str) -> bool:
    """判断模型输出是否等价于"不可回答" """
    t = _norm(pred)
    if t in _UNANS:
        return True
    toks = set(t.split())
    return any(u in toks for u in _UNANS if u)

def _is_yes_no_answers(golds):
    s = {_norm(a) for a in (golds or [])}
    return len(s) > 0 and s.issubset({"yes", "no"})

# ===== 自然语言 QA 抽短答案（只给 squadv2/hotpot/nq 用） =====
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:：]\s*(.+)",
    re.IGNORECASE,
)

def extract_qa_answer(s: str) -> str:
    """从模型自由生成的文本里抽出一个短答案，用于 SQuAD / Hotpot / NQ 的 F1/EM 打分。"""
    if not s:
        return ""
    # 去掉 <think>...</think>
    text = _THINK_BLOCK_RE.sub("", s).strip()

    # 1) 优先用最后一个 "Answer:" / "Final answer:" 行
    last = None
    for m in _ANSWER_LINE_RE.finditer(text):
        last = m
    if last is not None:
        ans = last.group(1).strip()
    else:
        # 2) 兜底：取最后一行非空文本
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        ans = lines[-1]

    # 去掉收尾标点
    ans = ans.strip().strip(" .\"'")

    # 3) 把各种"不知道/不可回答"的说法统一成 "unanswerable"
    low = ans.lower()
    for u in _UNANS:
        if not u:
            continue
        if u in low and len(low.split()) <= 5:
            return "unanswerable"

    return ans

def _judge_squad(pred, ex, config):
    # SQuAD v2：有 NoAns；命中 NoAns 或 HasAns 用 F1/EM
    is_unanswerable = ex.get("is_unanswerable", False)
    answers = ex.get("answers", [])
    if is_unanswerable or not answers or _norm(answers[0]) in _UNANS:
        hit = _is_unanswerable_pred(pred)
        return hit, 1.0 if hit else 0.0, 1.0 if hit else 0.0
    thr = getattr(config, 'squad_f1_threshold', None) or getattr(config, 'f1_threshold', 0.8)
    return match_extractive(pred, answers, thr)

def _judge_hotpot(pred, ex, config):
    # Hotpot：只有答案，不考虑 NoAns
    answers = ex.get("answers", [])
    thr = getattr(config, 'hotpot_f1_threshold', None) or getattr(config, 'f1_threshold', 0.8)
    return match_extractive(pred, answers, thr)

def _judge_nq(pred, ex, config):
    # NQ：Yes/No → EM；短答案 → F1；也可能出现 NoAns
    is_unanswerable = ex.get("is_unanswerable", False)
    answers = ex.get("answers", [])
    if is_unanswerable or not answers or _norm(answers[0]) in _UNANS:
        hit = _is_unanswerable_pred(pred)
        return hit, 1.0 if hit else 0.0, 1.0 if hit else 0.0
    if _is_yes_no_answers(answers):
        # 统一成 yes/no 之后做 EM
        hit = (_norm(pred) in {"yes", "no"} and _norm(pred) in {_norm(a) for a in answers})
        return hit, 1.0 if hit else 0.0, 1.0 if hit else 0.0
    thr = getattr(config, 'nq_f1_threshold', None) or getattr(config, 'f1_threshold', 0.8)
    return match_extractive(pred, answers, thr)

def judge_easy(dataset: str, pred: str, ex, config):
    """按数据集返回 (hit, f1, em)"""
    answers = ex.get("answers", [])
    decision = getattr(config, 'decision', 'auto')
    if decision == "em":
        em = float(_norm(pred) in {_norm(a) for a in (answers or [])})
        return (em >= 1.0), 0.0, em
    if decision == "f1":
        return match_extractive(pred, answers, getattr(config, 'f1_threshold', 0.8))

    # auto：按数据集走合适规则
    if decision in ("auto","squad_v2") and dataset == "squadv2":
        return _judge_squad(pred, ex, config)
    if decision in ("auto","hotpot") and dataset == "hotpot":
        return _judge_hotpot(pred, ex, config)
    if decision in ("auto","nq") and dataset == "nq":
        return _judge_nq(pred, ex, config)

    return match_extractive(pred, answers, getattr(config, 'f1_threshold', 0.8))

# =============== Math metrics ===============
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FRAC  = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
_PCT   = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%$")
_TEXT  = re.compile(r"\\text\{([^}]*)\}")

def _strip_tex(s: str) -> str:
    """清理 LaTeX 格式，尽可能提取数值或简化表达式"""
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
            if denominator != '0':
                num_val = float(numerator)
                den_val = float(denominator)
                return str(num_val / den_val)
            else:
                return match.group(1)
        except (ValueError, ZeroDivisionError):
            return f"({match.group(1)})/({match.group(2)})"
    
    s = _FRAC.sub(safe_frac_sub, s)
    
    # 处理其他常见的 LaTeX 命令
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\sqrt\[(\d+)\]\{([^}]*)\}', r'root\1(\2)', s)
    s = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', s)
    
    # 移除多余的空格和逗号
    s = s.replace(",", " ")
    s = " ".join(s.split())
    
    return s

def _normalize_latex_answer(s: str) -> str:
    """标准化 LaTeX 答案用于比较"""
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
    """从字符串中提取数值"""
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
    """从模型响应中提取答案，根据数据集类型使用不同策略"""
    if not s:
        return ""
    
    s = s.strip()
    
    if dataset == "gsm8k":
        # GSM8K: 严格提取 #### 后的内容
        if "####" in s:
            ans = s.split("####")[-1].strip()
            ans = re.sub(r'\b(clips|dollars|cents|people|items|units|days|hours)\b', '', ans, flags=re.IGNORECASE)
            ans = ans.strip()
            nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
            if nums:
                return nums[0]
            return ans
        else:
            lines = s.strip().split('\n')
            for line in reversed(lines):
                nums = re.findall(r"-?\d+(?:\.\d+)?", line)
                if nums:
                    return nums[-1]
            return s
    
    elif dataset in ["math","omini"]:
        # MATH: 优先提取 \boxed{} 中的内容
        m = _BOXED.search(s)
        if m:
            return m.group(1).strip()
        return s
    
    else:
        return s

def match_math(pred: str, golds: List[str], dataset: str = None) -> Tuple[bool, float, float]:
    """匹配数学答案"""
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
            pred_num = _num(pred)
        
        for g in golds:
            gold_num = _num(g)
            if pred_num is not None and gold_num is not None:
                if abs(pred_num - gold_num) < 1e-6:
                    return True, 0.0, 1.0
        
        return False, 1.0, 0.0
    
    # ============ MATH 专用逻辑：多策略比较 ============
    if dataset in ["math","omini"]:
        # 策略 1: 数值比较
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
        
        # 策略 3: 直接文本比较
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

def match_multiple_choice(pred: str, golds: List[str], options: List[str] = None) -> Tuple[bool, float, float]:
    """匹配多选题答案（如 AQuA）"""
    pred = pred.strip()
    
    # 1. 直接提取字母（A-E）
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
            for opt in options:
                opt_match = re.match(r'([A-E])\)(.*)', opt.strip())
                if opt_match:
                    opt_letter = opt_match.group(1)
                    opt_value_str = opt_match.group(2).strip()
                    opt_num = _num(opt_value_str)
                    if opt_num is not None and abs(pred_num - opt_num) < 1e-6:
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

# =============== LLM Agent Helper ===============
def build_agent_from_args(config, use_tools: bool = False) -> Tuple[LLMAgent, Any]:
    """
    Build LLMAgent from config object.
    
    Args:
        config: Configuration object (from Sacred _config, converted to object with attribute access)
        use_tools: Whether to enable tools (if False, tools are disabled even if configured)
    
    Returns:
        Tuple of (LLMAgent instance, tokenizer)
    """
    if LLMAgent is None:
        raise ImportError("LLMAgent is not available. Please ensure modelloader.llm_agent is importable.")
    
    # Build agent directly from config object
    agent = LLMAgent(config)
    
    return agent, agent.tokenizer

# =============== IO helpers ===============
def _to_native(obj):
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj

def export_squadv2_original(input_path: str, output_path: str, examples: List[Dict[str, Any]]):
    """对 SQuAD v2.0 本地 JSON：读入原始文件，按 examples 中保留的 id 过滤掉 easy 样本"""
    with open(input_path, "r", encoding="utf-8") as f:
        orig = json.load(f)

    keep_ids = {str(ex.get("id")) for ex in examples}

    new_data = []
    for art in orig.get("data", []):
        new_paras = []
        for para in art.get("paragraphs", []):
            qas = para.get("qas", [])
            filtered_qas = [qa for qa in qas if str(qa.get("id")) in keep_ids]
            if not filtered_qas:
                continue

            para_new = dict(para)
            para_new["qas"] = filtered_qas
            new_paras.append(para_new)

        if not new_paras:
            continue

        art_new = dict(art)
        art_new["paragraphs"] = new_paras
        new_data.append(art_new)

    obj = dict(orig)
    obj["data"] = new_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_json(path, examples):
    data = []
    for ex in examples:
        meta = ex.get("meta", {})
        raw = meta.get("raw_rec")
        if raw is not None:
            data.append(_to_native(raw))
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

def export_squad_like(path: str, examples: List[Dict[str, Any]]):
    data = []
    paras = []
    for ex in examples:
        paras.append({
            "context": ex.get("context", ""),
            "qas": [{
                "id": ex.get("id"),
                "question": ex.get("question", ""),
                "answers": [{"text": a, "answer_start": -1} for a in ex.get("answers", [])],
                "is_impossible": ex.get("is_unanswerable", False)
            }]
        })
    data.append({"title":"filtered","paragraphs":paras})
    obj = {"version":"filtered-1.0","data":data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =============== Main Experiment Logic ===============
def run_experiment(_run, _config, _log):
    """
    Main experiment logic.
    This function contains all the core execution logic for data cleaning experiments.
    """
    # Use _config directly (already merged and converted to object in main.py)
    config = _config
    
    # Handle input_path vs input
    input_path = getattr(config, "input_path", None)
    input_val = getattr(config, "input", None)
    if input_path and not input_val:
        config.input = input_path
    
    # Merge dataset config if present
    dataset_config = getattr(config, "dataset_config", None)
    if dataset_config:
        default_source = getattr(dataset_config, "default_source", None)
        default_input = getattr(dataset_config, "default_input", None)
        if default_source and not getattr(config, "source", None):
            config.source = default_source
        if default_input and not getattr(config, "input", None):
            config.input = os.path.expandvars(default_input)
    
    # Validate required parameters
    if not getattr(config, "dataset", None):
        raise ValueError("dataset is required")
    if not getattr(config, "model", None):
        raise ValueError("model is required")
    if not getattr(config, "output", None):
        raise ValueError("output is required")
    
    # Log configuration to Sacred
    _run.log_scalar("config.dataset", getattr(config, "dataset", None))
    _run.log_scalar("config.backend", getattr(config, "backend", None))
    _run.log_scalar("config.model", getattr(config, "model", None))

    # 全局运行时间（加载+推理+写结果）
    t_start = time()
    start_iso = datetime.now().isoformat(timespec="seconds")

    # 给这次 run 建一个目录
    run_tag = getattr(config, "run_tag", None)
    dataset = getattr(config, "dataset", None)
    run_id = run_tag or f"{dataset}__{datetime.now():%y%m%d_%H%M%S}"
    run_dir = Path(getattr(config, "results_dir", "results")) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # load data using DatasetLoader
    source = getattr(config, "source", None)
    split = getattr(config, "split", None)
    input_val = getattr(config, "input", None)
    print(f"[Load] dataset={dataset} source={source} split={split} input={input_val}")
    dataset_loader = DatasetLoader(dataset)
    examples = dataset_loader.load(config)
    print(f"[Load] {len(examples)} examples")

    task_type = "extractive" if dataset in ("squadv2","hotpot","nq") else "math"

    max_eval_examples = getattr(config, "max_eval_examples", 0)
    if max_eval_examples and max_eval_examples > 0 and len(examples) > max_eval_examples:
        seed = getattr(config, "seed", 42)
        random.seed(seed)
        examples = random.sample(examples, max_eval_examples)
        print(f"[Debug] Only evaluate {len(examples)} examples (random subset, seed={seed})")

    # backend - always use LLMAgent (with or without tools)
    backend = getattr(config, "backend", None)
    model = getattr(config, "model", None)
    print(f"[Init] backend={backend} model={model}")
    
    if LLMAgent is None:
        raise ImportError("LLMAgent is required but not available. Please ensure modelloader.llm_agent is importable.")
    
    # Determine if tools should be enabled
    use_tools = getattr(config, "use_agent_tools", False)
    
    # Build agent from config
    agent, tokenizer = build_agent_from_args(config, use_tools=use_tools)
    
    def make_prompt(ex: Dict[str, Any]) -> str:
        # Use DatasetLoader's build_prompt
        prompt = dataset_loader.build_prompt(ex, tokenizer)
        if prompt:
            return prompt
        # Fallback: build simple prompt
        question = ex.get("question", "")
        context = ex.get("context", "")
        if task_type == "extractive":
            hint = "\nIf the question cannot be answered from the context, reply with: 'unanswerable'." \
                   if getattr(config, "include_unanswerable_hint", False) else ""
            content = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:{hint}\n"
            if getattr(config, "use_chat_template", False) and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                msgs = [{"role":"system","content":"You are a helpful RC assistant."},
                        {"role":"user","content":content}]
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return content
        else:  # math
            content = (
                "Solve the following problem. Respond with the final numeric answer only. "
                "If it is a fraction/percentage, give the numeric form.\n"
                f"Problem: {question}\nAnswer:"
            )
            if getattr(config, "use_chat_template", False) and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                msgs = [{"role":"system","content":"You are a careful math solver."},
                        {"role":"user","content":content}]
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return content
    
    def generate_batch(prompts: List[str]) -> List[str]:
        """Generate responses using LLMAgent"""
        if use_tools:
            # Use chat mode with tools
            return [agent.chat(p, use_tools=True) for p in prompts]
        else:
            # Use simple generation without tools
            return [agent.generate(p) for p in prompts]

    bs = max(1, getattr(config, "batch_size", 8))
    kept: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []
    answer_comparisons: List[Dict[str, Any]] = []

    t0 = time()
    for i in tqdm(range(0, len(examples), bs), ncols=100, desc="Filtering"):
        batch = examples[i:i+bs]
        prompts = [make_prompt(ex) for ex in batch]
        preds = generate_batch(prompts)

        for j, (ex, pred) in enumerate(zip(batch, preds)):
            # 原始完整输出
            raw_pred = (pred or "").strip()
            used_prompt = prompts[j] if getattr(config, "store_prompt", False) else None

            # 默认用于打分的文本
            pred_for_score = raw_pred

            # 对自然语言抽取式三套（squadv2 / hotpot / nq），先把短答案抽出来再打分
            dataset_name = dataset
            if task_type == "extractive" and dataset_name in ("squadv2", "hotpot", "nq"):
                short = extract_qa_answer(raw_pred)
                if short:
                    pred_for_score = short
            
            # AQuA 使用多选题匹配
            answers = ex.get("answers", [])
            if dataset_name == "aqua":
                # 从 context 中提取 options
                options = []
                context = ex.get("context", "")
                if context.startswith("Options:"):
                    options = [line.strip() for line in context.split('\n')[1:] if line.strip()]
                hit, s1, s2 = match_multiple_choice(pred, answers, options)
            elif task_type == "extractive" and dataset_name in ("squadv2", "hotpot", "nq"):
                # 自然语言三套用新的判定（judge_easy）
                hit, s1, s2 = judge_easy(dataset_name, pred_for_score, ex, config)

            elif task_type == "extractive":
                # 其他将来可能的抽取式数据集，仍走旧的 F1/EM 兜底
                hit, s1, s2 = match_extractive(pred, answers, getattr(config, "f1_threshold", 0.8))
            else:
                hit, s1, s2 = match_math(pred, answers, dataset_name)

            # 对于 gsm8k 和 math，保存答案对比
            if dataset_name in ("gsm8k", "math","omini"):
                pred_extracted = _extract_answer_from_response(pred, dataset_name)
                gold_extracted = answers[0] if answers else ""
                answer_comparisons.append({
                    "id": ex.get("id"),
                    "question": ex.get("question", ""),
                    "model_raw_output": pred,
                    "model_extracted_answer": pred_extracted,
                    "gold_raw_answer": gold_extracted,
                    "gold_extracted_number": str(_num(gold_extracted)) if _num(gold_extracted) is not None else gold_extracted,
                    "model_extracted_number": str(_num(pred_extracted)) if _num(pred_extracted) is not None else pred_extracted,
                    "match": bool(hit),
                    "match_score": float(s2)
                })

            is_easy = bool(hit)  # hit = 小模型能答 → 过滤
            if getattr(config, "eval_only", False) or not is_easy:
                if getattr(config, "store_think", False):
                    if "meta" not in ex:
                        ex["meta"] = {}
                    ex["meta"]["model_output"] = raw_pred
                    ex["meta"]["model_answer"] = pred_for_score

                kept.append(ex)
            question = ex.get("question", "")
            log_item = {
                "id": ex.get("id"),
                "dataset": dataset_name,
                "is_easy": int(is_easy),
                "score1": float(s1),
                "score2": float(s2),
                "pred": pred_for_score,
                "answers": answers,
                "is_unanswerable": bool(ex.get("is_unanswerable", False)),
                "question": (question[:120] + "...") if len(question) > 120 else question,
            }
            if getattr(config, "store_think", False):
                log_item["pred_raw"] = raw_pred
            if getattr(config, "store_prompt", False):
                log_item["prompt"] = used_prompt
            context = ex.get("context", "")
            store_context = getattr(config, "store_context", 0)
            if store_context and context:
                log_item["context_head"] = context[:store_context]

            logs.append(log_item)
    t1 = time()
    print(f"[Done] kept {len(kept)} / {len(examples)} in {t1-t0:.1f}s")

    eval_time_sec = t1 - t0
    per_example_time = eval_time_sec / max(1, len(examples))
    end_iso = datetime.now().isoformat(timespec="seconds")
    wall_time_sec = time() - t_start

    # 1) 先做抽样（只对 squadv2/hotpot/nq）
    sample_size = getattr(config, "sample_size", None)
    if dataset_name in ("squadv2", "hotpot", "nq") and sample_size and len(kept) > sample_size:
        f1_by_id = {r["id"]: (float(r["score1"]) if r.get("score1") is not None else float("inf"))
                    for r in logs if not r["is_easy"]}

        kept_sorted = sorted(kept, key=lambda ex: f1_by_id.get(ex.get("id"), float("inf")))
        kept = kept_sorted[:sample_size]
        print(f"[Sample] selected {len(kept)} hardest examples by lowest F1.")
    
    # 2) 再计算统计量（total / hits / accuracy）
    total = len(logs)
    hits = sum(int(r["is_easy"]) for r in logs)
    accuracy = hits / total if total else 0.0
    print(f"[Stats] accuracy={accuracy:.4f} ({hits}/{total})")

    if not getattr(config, "no_logs", False):
        log_dir = Path("results") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_tag = getattr(config, "run_tag", None)
        tag = f"_{run_tag}" if run_tag else ""
        log_path = log_dir / f"{dataset_name}{tag}_{ts}.jsonl"

        with open(log_path, "w", encoding="utf-8") as f:
            for r in logs:
                f.write(json.dumps({k: v for k, v in r.items() if v is not None},
                                ensure_ascii=False) + "\n")

        print(f"[Write] run logs -> {log_path}")
    else:
        print("[Write] skip run logs because --no_logs is set")

    # 3) 自然语言三套的 summary
    if dataset_name in ("squadv2", "hotpot", "nq"):
        def _avg(vals):
            return float(sum(vals)/len(vals)) if vals else 0.0

        f1_all_vals = [float(r["score1"]) for r in logs if r.get("score1") is not None]
        avg_f1_all  = _avg(f1_all_vals)

        kept_ids = {ex.get("id") for ex in kept}
        f1_kept_vals = [float(r["score1"]) for r in logs
                        if (r.get("score1") is not None and r["id"] in kept_ids)]
        avg_f1_kept = _avg(f1_kept_vals)

        test_time_before_sec = eval_time_sec
        test_time_after_sec_est = per_example_time * len(kept)

        summary_dir = Path("results") / "natural_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "dataset": dataset_name,
            "total_examples_before": len(examples),
            "total_examples_after": len(kept),
            "test_time_before_sec": test_time_before_sec,
            "test_time_after_sec_est": test_time_after_sec_est,
            "avg_f1_before": avg_f1_all,
            "avg_f1_after": avg_f1_kept,
            "accuracy_easy_ratio": accuracy,
            "run_start": start_iso,
            "run_end": end_iso,
            "run_wall_time_sec": wall_time_sec,
            "per_example_time_sec": per_example_time,
            "parameters":{
                "model": getattr(config, "model", None),
                "backend": getattr(config, "backend", None),
                "max_input_tokens": getattr(config, "max_input_tokens", 2048),
                "max_new_tokens": getattr(config, "max_new_tokens", 64),
                "batch_size": getattr(config, "batch_size", 8),
                "temperature": getattr(config, "temperature", 0.0),
                "top_p": getattr(config, "top_p", 1.0),
                "f1_threshold": getattr(config, "f1_threshold", 0.8),
                "sample_size": sample_size,
                "seed": getattr(config, "seed", 42),
            },
        }
        with open(summary_dir / f"{dataset_name}.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        print(f"[Write] natural summary -> {summary_dir / f'{dataset_name}.json'}")

    # statistics
    total = len(logs)
    hits = sum(r["is_easy"] for r in logs)
    accuracy = hits / total if total else 0.0
    print(f"[Stats] accuracy={accuracy:.4f} ({hits}/{total})")

    # summary export for math datasets
    if dataset_name in ("aqua", "gsm8k", "math","omini"):
        summary_dir = Path("results") / "math_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{dataset_name}.json"

        test_time_before_sec = eval_time_sec
        accuracy_after = 0.0
        test_time_after_sec_est = per_example_time * len(kept)

        summary_payload = {
            "dataset": dataset_name,
            "total_examples_before": len(logs),
            "total_examples_after": len(kept),
            "test_time_before_sec": test_time_before_sec,
            "test_time_after_sec_est": test_time_after_sec_est,
            "accuracy_before": accuracy,
            "accuracy_after": accuracy_after,
            "run_start": start_iso,
            "run_end": end_iso,
            "run_wall_time_sec": wall_time_sec,
            "per_example_time_sec": per_example_time,
            "parameters":{
                "model": getattr(config, "model", None),
                "backend": getattr(config, "backend", None),
                "max_input_tokens": getattr(config, "max_input_tokens", 2048),
                "max_new_tokens": getattr(config, "max_new_tokens", 64),
                "batch_size": getattr(config, "batch_size", 8),
                "temperature": getattr(config, "temperature", 0.0),
                "top_p": getattr(config, "top_p", 1.0),
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        print(f"[Write] summary json -> {summary_path}")

    # outputs
    output_path = getattr(config, "output", None)
    if not getattr(config, "eval_only", False):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        source_val = getattr(config, "source", None)
        input_val = getattr(config, "input", None)
        if dataset_name == "squadv2" and source_val == "json" and input_val:
            export_squadv2_original(input_val, output_path, kept)
        else:
            write_json(output_path, kept)

        print(f"[Write] filtered json -> {output_path}")

        # 2) 额外：带 think + answer 的 cleaned 数据集
        if getattr(config, "store_think", False):
            rich_records = []
            for ex in kept:
                meta = ex.get("meta", {})
                base = meta.get("raw_rec")

                if base is None:
                    base = {
                        "id": ex.get("id"),
                        "question": ex.get("question", ""),
                        "context": ex.get("context", ""),
                        "answers": ex.get("answers", []),
                        "is_unanswerable": bool(ex.get("is_unanswerable", False)),
                    }
                else:
                    base = _to_native(base)

                if "model_output" in meta:
                    base["model_output"] = meta["model_output"]
                if "model_answer" in meta:
                    base["model_answer"] = meta["model_answer"]

                rich_records.append(base)

            results_dir = getattr(config, "results_dir", "results")
            rich_dir = Path(results_dir) / "filtered_with_model"
            rich_dir.mkdir(parents=True, exist_ok=True)
            rich_path = rich_dir / (Path(output_path).stem + ".with_model.json")
            with open(rich_path, "w", encoding="utf-8") as f:
                json.dump(rich_records, f, ensure_ascii=False, indent=2)
            print(f"[Write] filtered json with model output -> {rich_path}")

    save_csv = getattr(config, "save_csv", None)
    if save_csv and logs:
        write_csv(save_csv, logs)
        print(f"[Write] decisions csv -> {save_csv}")

    if getattr(config, "export_squad_like", False) and task_type == "extractive":
        p = Path(output_path).with_suffix(".squad.json")
        export_squad_like(str(p), kept)
        print(f"[Write] squad-like json -> {p}")
    
    # 输出答案对比 JSON（仅 gsm8k 和 math）
    if dataset_name in ("gsm8k", "math","omini") and answer_comparisons:
        comparison_dir = Path("results") / "answer_comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = comparison_dir / f"{dataset_name}_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(answer_comparisons, f, ensure_ascii=False, indent=2)
        print(f"[Write] answer comparison json -> {comparison_path}")
    
    # Return summary for Sacred logging
    return {
        "kept": len(kept),
        "total": len(examples),
        "accuracy": accuracy,
        "wall_time": wall_time_sec,
    }

