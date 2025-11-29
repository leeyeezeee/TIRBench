"""
LLM JSONL Evaluator

Features
- Supports three backends: vllm (default), transformers, sglang.
- Default model set to a Qwen3-4B style identifier (user can override).
- Reads a JSONL dataset; each line must contain: paper_id, question, answer (ground_truth), rationale, reference_figure, all_figures.
- Produces an output JSONL (and optional CSV) with: query, prediction_answer, ground_truth, think_content, raw_output, is_correct.
- "think_content" is SAFELY sanitized by default (chain-of-thought is redacted to comply with safety practices).
  Use --think-mode summarized to get a short justification (model asked to give a brief reasoning). 
  Use --think-mode raw_if_present to capture only the model-exposed <think>...</think> or similar segments if they appear; they will be truncated.
- Works with reasoning-style models that emit <think> or special tags, by providing --reasoning-tag-start and --reasoning-tag-end.
- vLLM usage assumes local weights or a HF model path. sglang usage assumes sglang runtime installed.
- Transformers path is CPU/GPU depending on your environment and args.
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

# --------------------------
# Prompt templates (plain text; no extra chat wrappers)
# --------------------------

SYSTEM_PROMPT_COMPACT = (
    "Put your thought content inside <think>...</think> and the answer inside <answer>...</answer>.\n"
)

USER_PROMPT_WITH_BRIEF_JUST = """\
Question:
{question}

Instruction:
1) Provide the final answer in one sentence.
2) Then give a brief explanation in one short sentence (no step-by-step).
Return format:
Answer: <your answer>
Why: <very brief explanation>
"""

USER_PROMPT_FINAL_ONLY = """\
Question:{question}
Begin!\n
"""

# --------------------------
# Utilities
# --------------------------

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def maybe_extract_between(text: str, start_tag: Optional[str], end_tag: Optional[str]) -> Optional[str]:
    if not text or not start_tag or not end_tag:
        return None
    try:
        start = text.find(start_tag)
        end = text.find(end_tag, start + len(start_tag)) if start != -1 else -1
        if start != -1 and end != -1:
            return text[start + len(start_tag):end].strip()
    except Exception:
        pass
    return None

def sanitize_think_content(raw_text: str, mode: str, max_chars: int = 500) -> str:
    if not raw_text:
        return ""
    stripped = re.sub(r"<\|?think\|?>.*?</\|?think\|?>", "[[hidden]]", raw_text, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r"<think>.*?</think>", "[[hidden]]", stripped, flags=re.DOTALL | re.IGNORECASE)

    if mode == "redact":
        return "[[hidden]]"

    if mode == "summarized":
        m = re.search(r"(?im)^why:\s*(.+)$", stripped)
        if m:
            return m.group(1).strip()[:max_chars]
        m2 = re.search(r"(?im)^answer:\s*(.+)$", stripped)
        if m2:
            first = m2.group(1).strip()
            return f"Answer summary: {first[:max_chars]}"
        return stripped[:max_chars]

    if mode == "raw_if_present":
        return stripped[:max_chars]

    return "[[hidden]]"

def parse_answer_and_why(text: str) -> Dict[str, str]:
    ans, why = "", ""
    if not text:
        return {"answer": ans, "why": why}
    m = re.search(r"(?im)^answer:\s*(.+)$", text)
    if m:
        ans = m.group(1).strip()
    m = re.search(r"(?im)^why:\s*(.+)$", text)
    if m:
        why = m.group(1).strip()
    if not ans:
        ans = text.strip()
    return {"answer": ans, "why": why}

def normalize(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def compute_accuracy_from_rows(rows: List[Dict[str, Any]]) -> float:
    total, correct = 0, 0
    for r in rows:
        total += 1
        if "is_correct" in r:
            correct += 1 if r.get("is_correct") is True or r.get("is_correct") == "TRUE" else 0
        else:
            pa = normalize(r.get("prediction_answer"))
            gt = normalize(r.get("ground_truth"))
            correct += 1 if pa == gt else 0
    return (correct / total) if total else 0.0

# --------------------------
# Backends
# --------------------------

def generate_vllm(prompts: List[str], model: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]] = None) -> List[str]:
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError(f"vLLM import failed: {e}")
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
    llm = LLM(model=model, trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]

def generate_transformers(prompts: List[str], model: str, max_tokens: int, temperature: float, top_p: float, device: Optional[str]) -> List[str]:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError(f"Transformers import failed: {e}")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    )
    if device:
        m.to(device)
    out_texts = []
    gen_kwargs = dict(do_sample=True, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, eos_token_id=tok.eos_token_id)
    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(**inputs, **gen_kwargs)
        text = tok.decode(out[0], skip_special_tokens=True)
        if text.startswith(p):
            text = text[len(p):]
        out_texts.append(text.strip())
    return out_texts

def generate_sglang(prompts: List[str], model: str, max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]] = None) -> List[str]:
    try:
        import sglang as sgl
    except Exception as e:
        raise RuntimeError(f"sglang import failed: {e}")
    @sgl.function
    def run(s, prompt):
        s += prompt
        s += sgl.gen(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
    out_texts = []
    for p in prompts:
        out = run.run(prompt=p, model=model)
        out_texts.append(out.text().strip())
    return out_texts

# --------------------------
# Evaluation
# --------------------------

@dataclass
class EvalRow:
    query: str
    prediction_answer: str
    ground_truth: str
    think_content: str
    raw_output: str
    is_correct: bool

def build_prompt(question: str, brief_justify: bool) -> str:
    # Plain text prompt: system line + user template, no chat tags
    if brief_justify:
        return f"{SYSTEM_PROMPT_COMPACT}\n\n{USER_PROMPT_WITH_BRIEF_JUST.format(question=question)}"
    else:
        return f"{SYSTEM_PROMPT_COMPACT}\n\n{USER_PROMPT_FINAL_ONLY.format(question=question)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/google_spiqa.jsonl", help="Path to input JSONL dataset")
    parser.add_argument("--out", type=str, default="results/google_spiqa.jsonl", help="Output JSONL path")
    parser.add_argument("--out-csv", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers", "sglang"], default="vllm")
    parser.add_argument("--model", type=str, default="models/Qwen_Qwen3-4B", help="Model path/name. You can pass a Qwen3 4B think model if available.")
    parser.add_argument("--device", type=str, default="cuda:0", help="transformers device, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stop", type=str, nargs="*", default=None, help="Optional stop strings")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for vLLM")
    parser.add_argument("--think-mode", type=str, choices=["redact", "summarized", "raw_if_present"], default="raw_if_present",
                        help="How to fill think_content in outputs")
    parser.add_argument("--reasoning-tag-start", type=str, default="<think>", help="e.g., <think>")
    parser.add_argument("--reasoning-tag-end", type=str, default="</think>", help="e.g., </think>")
    parser.add_argument("--brief-justify", action="store_true", help="Ask model for a brief, non-step explanation (recommended for think_mode=summarized)")
    parser.add_argument("--max-samples", type=int, default=10, help="Only evaluate the first N samples; 0 means all")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)

    # If results already exist: skip inference, compute & print accuracy
    if out_path.exists():
        existing_rows = list(read_jsonl(out_path))
        if not existing_rows:
            print(f"Existing file '{out_path}' is empty. Nothing to evaluate.", file=sys.stderr)
            sys.exit(1)
        acc = compute_accuracy_from_rows(existing_rows)
        total = len(existing_rows)
        correct = int(round(acc * total))
        print(f"[Skip inference] Loaded {total} rows from {out_path}")
        print(f"Accuracy: {correct}/{total} = {acc*100:.2f}%")
        if args.out_csv:
            import csv
            with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query", "prediction_answer", "ground_truth", "think_content", "raw_output", "is_correct"])
                for r in existing_rows:
                    is_corr = r.get("is_correct")
                    if is_corr is None:
                        is_corr = normalize(r.get("prediction_answer")) == normalize(r.get("ground_truth"))
                    w.writerow([
                        r.get("query", ""),
                        r.get("prediction_answer", ""),
                        r.get("ground_truth", ""),
                        r.get("think_content", ""),
                        r.get("raw_output", ""),
                        "TRUE" if is_corr else "FALSE"
                    ])
            print(f"CSV also written to {args.out_csv}")
        return

    records = list(read_jsonl(data_path))
    if not records:
        print("No data found in the JSONL.", file=sys.stderr)
        sys.exit(1)

    if args.max_samples and args.max_samples > 0:
        records = records[:args.max_samples]

    # Build prompts
    prompts, questions, gts = [], [], []
    for r in records:
        q = r.get("question") or ""
        gt = r.get("answer") or ""
        questions.append(q)
        gts.append(gt)
        prompts.append(build_prompt(q, brief_justify=args.brief_justify))

    # Inference
    if args.backend == "vllm":
        texts = generate_vllm(prompts, args.model, args.max_new_tokens, args.temperature, args.top_p, stop=args.stop)
    elif args.backend == "transformers":
        texts = generate_transformers(prompts, args.model, args.max_new_tokens, args.temperature, args.top_p, device=args.device or None)
    else:
        texts = generate_sglang(prompts, args.model, args.max_new_tokens, args.temperature, args.top_p, stop=args.stop)

    # Parse & package results
    out_rows: List[EvalRow] = []
    for q, gt, raw in zip(questions, gts, texts):
        parsed = parse_answer_and_why(raw)
        ans = parsed["answer"]
        think_segment = ""
        if args.reasoning_tag_start and args.reasoning_tag_end:
            think_segment = maybe_extract_between(raw, args.reasoning_tag_start, args.reasoning_tag_end) or ""
        think_content = sanitize_think_content(
            think_segment if think_segment else raw,
            mode=args.think_mode
        )
        is_correct = normalize(ans) == normalize(gt)

        out_rows.append(EvalRow(
            query=q,
            prediction_answer=ans,
            ground_truth=gt,
            think_content=think_content,
            raw_output=raw,             # <--- full text here
            is_correct=bool(is_correct),
        ))

    # Write JSONL
    write_jsonl(out_path, (asdict(r) for r in out_rows))

    # Optional CSV
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query", "prediction_answer", "ground_truth", "think_content", "raw_output", "is_correct"])
            for r in out_rows:
                w.writerow([r.query, r.prediction_answer, r.ground_truth, r.think_content, r.raw_output, "TRUE" if r.is_correct else "FALSE"])

    # Accuracy to terminal
    total = len(out_rows)
    correct = sum(1 for r in out_rows if r.is_correct)
    acc = (correct / total) if total else 0.0
    print(f"Wrote {total} rows to {out_path}")
    if args.out_csv:
        print(f"CSV also written to {args.out_csv}")
    print(f"Accuracy: {correct}/{total} = {acc*100:.2f}%")

if __name__ == "__main__":
    main()
