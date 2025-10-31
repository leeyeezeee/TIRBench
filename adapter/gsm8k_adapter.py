"""
GSM8k 数据集转 SQuAD-like 结构
"""
import json

def to_squadlike(gsm8k_data):
    data = []
    for i, ex in enumerate(gsm8k_data):
        qid = ex.get("id", f"gsm8k_{i}")
        question = ex["question"]
        answer = ex["answer"].strip()
        qa = {
            "id": qid,
            "question": question,
            "is_impossible": False,
            "answers": [{"text": answer, "answer_start": 0}],
        }
        data.append({"title": "GSM8k", "paragraphs": [{"context": "", "qas": [qa]}]})
    return {"version": "gsm8k->squad", "data": data}

# 用法示例：
# with open('gsm8k.json', 'r', encoding='utf-8') as f:
#     gsm8k_data = json.load(f)
# squad_like = gsm8k_to_squadlike(gsm8k_data)
