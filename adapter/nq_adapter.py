"""
Natural Questions 数据集转 SQuAD-like 结构
"""
def to_squadlike(hf_split):
    data = []
    for i, ex in enumerate(hf_split):
        q = ex.get("question_text") or ex.get("question") or ""
        ctx = ex.get("document_text") or ex.get("document") or ""
        golds = []
        anns = ex.get("annotations", [])
        for ann in anns:
            sa = ann.get("short_answers") or {}
            texts = sa.get("text") or []
            for t in texts:
                if t:
                    golds.append(str(t))
        is_imp = (len(golds) == 0)
        answers = []
        if not is_imp:
            ctx_low = ctx.lower()
            for g in golds:
                pos = ctx_low.find(g.lower())
                answers.append({"text": g, "answer_start": max(0, pos)})
        qa = {
            "id": ex.get("id") or ex.get("example_id") or f"nq_{i}",
            "question": q,
            "is_impossible": bool(is_imp),
            "answers": answers,
        }
        data.append({"title": "NaturalQuestions", "paragraphs": [{"context": ctx, "qas": [qa]}]})
    return {"version": "nq->squad", "data": data}
