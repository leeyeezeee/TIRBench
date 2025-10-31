"""
HotpotQA 数据集转 SQuAD-like 结构
"""
def to_squadlike(hf_split):
    data = []
    for ex in hf_split:
        qid = ex["id"]
        q = ex["question"]
        parts = []
        for title, sents in ex["context"]:
            parts.append(f"{title} : " + " ".join(sents))
        ctx = "\n\n".join(parts)
        ans_text = str(ex["answer"])
        lower_ctx = ctx.lower()
        lower_ans = ans_text.lower()
        start = lower_ctx.find(lower_ans)
        if start < 0:
            start = 0  # span start best-effort
        qa = {
            "id": qid,
            "question": q,
            "is_impossible": False,
            "answers": [{"text": ans_text, "answer_start": int(start)}],
        }
        data.append({"title": "HotpotQA", "paragraphs": [{"context": ctx, "qas": [qa]}]})
    return {"version": "hotpot_qa->squad", "data": data}
