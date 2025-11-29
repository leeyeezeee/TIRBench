"""
MATH 数据集转 SQuAD-like 结构（简化版）
直接使用 answer 字段
"""
import json
import pandas as pd
from typing import List, Dict, Any


def load(args):
    """
    加载 MATH 数据集，直接生成 data_cleaning.py 需要的格式
    
    Args:
        args: 命令行参数，包含 source, input, split 等
    
    Returns:
        List[dict]: 包含 id, question, context, answers 字段的字典列表
    """
    # 根据 source 类型加载数据
    if args.source == "json":
        # 从本地 parquet 文件加载
        from pathlib import Path
        
        suffix = Path(args.input).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            with open(args.input, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(args.input)
            raw_data = df.to_dict("records")
        else:
            raise ValueError("Unsupported input format")
        
    elif args.source == "hf":
        # 从 HuggingFace 加载
        from datasets import load_dataset
        dataset = load_dataset("hendrycks/competition_math", split=args.split)
        raw_data = list(dataset)
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    
    # 直接转换为所需格式（一次循环）
    examples = []
    for i, ex in enumerate(raw_data):
        qid = ex.get("id", f"math_{i}")
        question = ex.get("problem", "")
        # 直接使用 answer 字段
        answer = ex.get("answer", "").strip()
        
        examples.append({
            "id": qid,
            "question": question,
            "context": "",
            "answers": [answer],
            "is_unanswerable": False,
            "raw_rec":ex
        })
    
    return examples


def build_prompt(ex, tokenizer=None):
    """
    MATH 数据集专用 prompt，处理竞赛级数学题
    
    MATH 包含竞赛级数学问题，答案可能包含 LaTeX 格式（如 \\boxed{}）
    """
    content = (
        f"Problem: {ex.question}\n\n"
        "Solve this competition mathematics problem.\n"
        "Provide your final answer using \\boxed{{}} notation.\n"
        "Examples:\n"
        "- For a number: \\boxed{{42}}\n"
        "- For a coordinate: \\boxed{{(3, -1)}}\n"
        "- For an expression: \\boxed{{\\frac{{-33}}{{2}}}}\n"
        "- For an interval: \\boxed{{[0, 3)}}\n\n"
        "Your solution:"
    )
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [
                {"role": "system", "content": "You are an expert mathematics solver. Always provide your final answer in \\boxed{{}} format."},
                {"role": "user", "content": content}
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except:
            pass
    
    return content


