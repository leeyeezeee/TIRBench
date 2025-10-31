"""
MATH 数据集转 SQuAD-like 结构
"""
import json

def to_squadlike(math_data):
    data = []
    for i, ex in enumerate(math_data):
        qid = ex.get("id", f"math_{i}")
        question = ex["problem"]
        # 取最终答案（假设 solution 以 "####" 分隔，取前面部分）
        answer = ex["solution"].strip().split("\n####")[0].strip()
        qa = {
            "id": qid,
            "question": question,
            "is_impossible": False,
            "answers": [{"text": answer, "answer_start": 0}],
        }
        data.append({"title": "MATH", "paragraphs": [{"context": "", "qas": [qa]}]})
    return {"version": "math->squad", "data": data}

# 用法示例：
# with open('math.json', 'r', encoding='utf-8') as f:
#     math_data = json.load(f)
# squad_like = math_to_squadlike(math_data)
