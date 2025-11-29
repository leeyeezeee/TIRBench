"""
GSM8k 数据集转 SQuAD-like 结构
"""
import json
import re
import pandas as pd
from typing import List, Dict, Any


def load(args):
    """
    加载 GSM8K 数据集，直接生成 data_cleaning.py 需要的格式
    
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
        dataset = load_dataset("gsm8k", "main", split=args.split)
        raw_data = list(dataset)
    else:
        raise ValueError(f"Unsupported source: {args.source}")
    
    # 直接转换为所需格式（一次循环）
    examples = []
    for i, ex in enumerate(raw_data):
        qid = ex.get("id", f"gsm8k_{i}")
        question = ex["question"]
        
        # 提取最终答案（不包含推理过程）
        raw_answer = ex["answer"].strip()
        final_answer = extract_final_answer(raw_answer)
        
        examples.append({
            "id": qid,
            "question": question,
            "context": "",
            "answers": [final_answer],
            "is_unanswerable": False,
            "raw_rec":ex
        })
    
    return examples


def extract_final_answer(answer_text):
    """
    从 GSM8K 的 answer 字段中提取最终答案
    格式: "推理过程... ### 最终答案"
    """
    # 使用 ### 分隔符提取最终答案
    if "####" in answer_text:
        final_answer = answer_text.split("####")[-1].strip()
    elif "###" in answer_text:
        final_answer = answer_text.split("###")[-1].strip()
    else:
        # 如果没有分隔符，尝试提取最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        final_answer = numbers[-1] if numbers else answer_text
    
    return final_answer


def build_prompt(ex, tokenizer=None):
    """
    GSM8K 专用 prompt，适合小学数学应用题
    
    GSM8K 包含小学阶段的多步骤应用题，需要清晰的数值答案
    """
    content = (
        f"Problem: {ex.question}\n\n"
        "Solve this math problem step by step, then provide your final answer.\n"
        "Format your final answer as: #### [number]\n"
        "For example:\n"
        "Step 1: Calculate 48/2 = 24\n"
        "Step 2: Calculate 48 + 24 = 72\n"
        "#### 72\n\n"
        "Your solution:"
    )
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            msgs = [
                {"role": "system", "content": "You are a helpful math tutor. Always end your answer with '#### [number]' where [number] is the final numeric answer."},
                {"role": "user", "content": content}
            ]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except:
            pass
    
    return content
