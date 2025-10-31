"""
AQuA 数据集转 SQuAD-like 结构
"""
import json

def to_squadlike(aqua_data):
    data = []
    for i, ex in enumerate(aqua_data):
        qid = ex.get("id", f"aqua_{i}")
        question = ex["question"]
        options = ex.get("options", [])
        correct = ex.get("correct", "")
        context = "Options:\n" + "\n".join(options)
        qa = {
            "id": qid,
            "question": question,
            "is_impossible": False,
            "answers": [{"text": correct, "answer_start": 0}],
        }
        data.append({"title": "AQuA", "paragraphs": [{"context": context, "qas": [qa]}]})
    return {"version": "aqua->squad", "data": data}

# 用法示例：
# with open('aqua.json', 'r', encoding='utf-8') as f:
#     aqua_data = json.load(f)
# squad_like = aqua_to_squadlike(aqua_data)
