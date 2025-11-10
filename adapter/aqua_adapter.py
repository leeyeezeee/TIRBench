"""
AQuA 数据集转 SQuAD-like 结构
直接使用选项标签作为答案
"""
import json
import pandas as pd
from typing import List, Dict, Any


def load(args):
    """
    加载 AQuA 数据集，直接生成 data_cleaning.py 需要的格式
    
    Args:
        args: 命令行参数，包含 source, input, split 等
    
    Returns:
        List[dict]: 包含 id, question, context, answers 字段的字典列表
    """
    # 根据 source 类型加载数据
    if args.source == "json":
        # 从本地 parquet 文件加载
        df = pd.read_parquet(args.input)
        raw_data = df.to_dict('records')
    elif args.source == "hf":
        # 从 HuggingFace 加载
        from datasets import load_dataset
        dataset = load_dataset("aqua_rat", "raw", split=args.split)
        raw_data = list(dataset)
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    
    # 直接转换为所需格式（一次循环）
    examples = []
    for i, ex in enumerate(raw_data):
        qid = ex.get("id", f"aqua_{i}")
        question = ex["question"]
        # 确保 options 是列表（从 numpy array 转换）
        options = list(ex.get("options", []))
        correct_label = ex.get("correct", "")
        
        # 构建 context（包含所有选项）
        context = "Options:\n" + "\n".join(options)
        
        examples.append({
            "id": qid,
            "question": question,
            "context": context,
            "answers": [correct_label],
            "is_unanswerable": False
        })
    
    return examples


def build_prompt(ex, tokenizer=None):
    """
    AQuA 专用 prompt，强调选择选项标签
    
    AQuA 是多选题格式，需要模型从 A-E 中选择正确选项
    """
    # AQuA 的 context 包含选项列表
    content = (
        f"{ex.question}\n\n"
        f"{ex.context}\n\n"
        "Please select the correct option letter (A, B, C, D, or E) as your answer.\n"
        "Respond with only the letter of the correct answer."
    )
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [
                {"role": "system", "content": "You are a precise math problem solver. Always respond with only the option letter."},
                {"role": "user", "content": content}
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except:
            pass
    
    return content


