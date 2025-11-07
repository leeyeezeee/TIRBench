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
        df = pd.read_parquet(args.input)
        raw_data = df.to_dict('records')
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
            "is_unanswerable": False
        })
    
    return examples


