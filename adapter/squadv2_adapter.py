"""
SQuAD v2.0 数据集转 SQuAD-like 结构
"""
from collections import defaultdict

def to_squadlike(hf_split):
    group = defaultdict(lambda: defaultdict(list))  # title -> context -> [qa...]
    for ex in hf_split:
        title = ex.get("title") or "No Title"
        ctx = ex["context"]
        qa = {
            "id": ex["id"],
            "question": ex["question"],
            "is_impossible": bool(ex.get("is_impossible", False)),
            "answers": [
                {"text": t, "answer_start": s}
                for t, s in zip(ex["answers"]["text"], ex["answers"]["answer_start"])
            ],
        }
        if "plausible_answers" in ex and ex["plausible_answers"]:
            qa["plausible_answers"] = [
                {"text": t, "answer_start": s}
                for t, s in zip(
                    ex["plausible_answers"]["text"],
                    ex["plausible_answers"].get("answer_start", [0]*len(ex["plausible_answers"]["text"]))
                )
            ]
        group[title][ctx].append(qa)
    data = []
    for title, ctx_map in group.items():
        paragraphs = [{"context": ctx, "qas": qas} for ctx, qas in ctx_map.items()]
        data.append({"title": title, "paragraphs": paragraphs})
    return {"version": "v2.0", "data": data}
