"""
Omini 数据集转 SQuAD-like 结构
包含 question, thinking, answer 三个字段
"""
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


def load(args):
    """
    加载 Omini 数据集，直接生成 data_cleaning.py 需要的格式
    
    Args:
        args: 命令行参数，包含 source, input, split 等
    
    Returns:
        List[dict]: 包含 id, question, context, answers 字段的字典列表
    """
    # 根据 source 类型加载数据，自动检测文件格式
    if args.source == "json":
        # 自动检测文件格式
        suffix = Path(args.input).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            with open(args.input, "r", encoding="utf-8") as f:
                if suffix == ".jsonl":
                    raw_data = [json.loads(line) for line in f]
                else:
                    raw_data = json.load(f)
                    # 如果是单个字典，转换为列表
                    if isinstance(raw_data, dict):
                        raw_data = [raw_data]
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(args.input)
            raw_data = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported input format: {suffix}. Expected .json, .jsonl, .parquet, or .pq")
    elif args.source == "hf":
        # 从 HuggingFace 加载（如果将来有）
        from datasets import load_dataset
        dataset = load_dataset("omini", split=args.split)
        raw_data = list(dataset)
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    
    # 直接转换为所需格式（一次循环）
    examples = []
    for i, ex in enumerate(raw_data):
        qid = ex.get("id", f"omini_{i}")
        question = ex.get("question", "")
        thinking = ex.get("thinking", "")
        answer = ex.get("answer", "").strip()
        
        # 将 thinking 作为 context，question 作为 question
        # 如果 thinking 为空，则 context 也为空
        context = thinking if thinking else ""
        
        examples.append({
            "id": qid,
            "question": question,
            "context": context,
            "answers": [answer],
            "is_unanswerable": False,
            "raw_rec":ex
        })
    
    return examples


def build_prompt(ex, tokenizer=None):
    """
    Omini 数据集专用 prompt，利用 thinking 字段提供思考过程提示
    
    Omini 包含数学或逻辑问题，带有思考过程描述（thinking）和答案
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
                {"role": "system", "content": "You are an expert problem solver. Analyze the problem carefully and provide a clear solution."},
                {"role": "user", "content": content}
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except:
            pass
    
    return content

