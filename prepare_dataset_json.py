#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script for materializing benchmark datasets into JSON files with
custom sampling rules:

- AQuA: keep the entire test set and top it up with random training
        samples until the combined set reaches a target size (default 500).
- MATH: export the full test split to JSON.
- GSM8K: start from a filtered JSON file; if it contains fewer than the
         target size (default 500) examples, sample extra items from the
         test split (without replacement) until the target is met.
- Omini: start from a filtered JSON file; if it contains fewer than the
         target size (default 500) examples, sample extra items from the
         original dataset by id (avoiding duplicates) until the target is met.

All Parquet inputs are converted to pure Python types (lists / dicts /
numbers / strings) so the resulting JSON files remain serializer-friendly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

try:  # Optional, only needed when Parquet columns are numpy scalars.
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _to_native(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain Python types."""
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_to_native(v) for v in value.tolist()]
    if hasattr(value, "tolist"):  # pandas Series / Index
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _load_parquet_records(path: str | Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(path)
    records = df.to_dict(orient="records")
    return [_to_native(rec) for rec in records]


def _load_json_records(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON file {path} must contain a list of objects.")
    return data


def _write_json(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Dataset-specific routines
# ---------------------------------------------------------------------------
def prepare_aqua(
    train_path: str,
    test_path: str,
    output_path: str,
    target_size: int = 500,
    seed: int = 42,
) -> None:
    train = _load_parquet_records(train_path)
    test = _load_parquet_records(test_path)
    if len(test) > target_size:
        raise ValueError(
            f"AQuA test split already has {len(test)} examples, "
            f"which exceeds the requested target_size={target_size}."
        )
    needed = max(0, target_size - len(test))
    rng = random.Random(seed)
    if needed > len(train):
        raise ValueError(
            f"Not enough AQuA training samples ({len(train)}) to reach "
            f"{target_size} total examples."
        )
    sampled = rng.sample(train, needed) if needed else []
    combined = test + sampled
    _write_json(output_path, combined)


def prepare_math(test_path: str, output_path: str) -> None:
    test = _load_parquet_records(test_path)
    _write_json(output_path, test)


def _gsm8k_key(record: Dict[str, Any]) -> Any:
    """
    Produce a deduplication key for GSM8K samples.
    Prefer explicit 'id' if present; otherwise fall back to (question, answer).
    """
    if "id" in record and record["id"] not in (None, ""):
        return record["id"]
    question = (record.get("question") or "").strip()
    answer = (record.get("answer") or "").strip()
    return (question, answer)


def prepare_gsm8k(
    filtered_json: str,
    test_path: str,
    output_path: str,
    target_size: int = 500,
    seed: int = 42,
) -> None:
    filtered = _load_json_records(filtered_json)
    if len(filtered) >= target_size:
        _write_json(output_path, filtered[:target_size])
        return

    test_records = _load_parquet_records(test_path)
    existing_keys = {_gsm8k_key(rec) for rec in filtered}
    candidates = [rec for rec in test_records if _gsm8k_key(rec) not in existing_keys]

    needed = target_size - len(filtered)
    if needed > len(candidates):
        raise ValueError(
            f"Only {len(candidates)} unused GSM8K test examples available, "
            f"but need {needed} more to reach {target_size}."
        )

    rng = random.Random(seed)
    extras = rng.sample(candidates, needed)
    combined = filtered + extras
    _write_json(output_path, combined)


def prepare_omini(
    filtered_json: str,
    original_path: str,
    output_path: str,
    target_size: int = 500,
    seed: int = 42,
) -> None:
    """
    Prepare Omini dataset: read filtered JSON, if less than target_size,
    sample additional records from original dataset by id (avoiding duplicates).
    
    Requirements:
    - Completely preserve all filtered data (no truncation unless exceeding target_size)
    - Use 'id' field for deduplication
    - Merge filtered data with sampled data from original dataset
    - Final dataset should have exactly target_size records (or all filtered if >= target_size)
    """
    # Load filtered dataset (completely preserved)
    filtered = _load_json_records(filtered_json)
    print(f"[Omini] Loaded {len(filtered)} filtered examples")
    
    # If filtered data already meets or exceeds target_size, keep first target_size records
    if len(filtered) >= target_size:
        print(f"[Omini] Filtered data ({len(filtered)}) >= target_size ({target_size}), keeping first {target_size} records")
        _write_json(output_path, filtered[:target_size])
        return

    # Load original dataset (can be JSON or Parquet)
    original_path_obj = Path(original_path)
    if original_path_obj.suffix.lower() in {".json", ".jsonl"}:
        original_records = _load_json_records(original_path)
    elif original_path_obj.suffix.lower() in {".parquet", ".pq"}:
        original_records = _load_parquet_records(original_path)
    else:
        raise ValueError(
            f"Unsupported original file format: {original_path_obj.suffix}. "
            "Expected .json, .jsonl, .parquet, or .pq"
        )
    print(f"[Omini] Loaded {len(original_records)} original examples")

    # Extract existing IDs from filtered data for deduplication
    # Use str() to handle different id types (int, str, etc.)
    existing_ids = set()
    for rec in filtered:
        rec_id = rec.get("id")
        if rec_id is not None:
            existing_ids.add(str(rec_id))
    print(f"[Omini] Found {len(existing_ids)} unique IDs in filtered data")

    # Filter candidates: records from original dataset not in filtered set (by id)
    candidates = []
    for rec in original_records:
        rec_id = rec.get("id")
        if rec_id is not None and str(rec_id) not in existing_ids:
            candidates.append(rec)
    
    print(f"[Omini] Found {len(candidates)} candidate examples from original dataset (not in filtered)")

    # Calculate how many more records we need
    needed = target_size - len(filtered)
    
    if needed <= 0:
        # This shouldn't happen due to earlier check, but handle it anyway
        _write_json(output_path, filtered)
        return
    
    if needed > len(candidates):
        raise ValueError(
            f"Not enough unused Omini examples to reach target size. "
            f"Filtered data has {len(filtered)} examples, need {needed} more to reach {target_size}, "
            f"but only {len(candidates)} unused examples available in original dataset."
        )

    # Randomly sample needed records from candidates
    rng = random.Random(seed)
    extras = rng.sample(candidates, needed)
    print(f"[Omini] Sampled {len(extras)} additional examples from original dataset")

    # Merge: filtered data first (completely preserved), then sampled extras
    combined = filtered + extras
    print(f"[Omini] Combined dataset has {len(combined)} examples (target: {target_size})")
    
    # Verify no duplicates by id
    combined_ids = {str(rec.get("id", "")) for rec in combined if rec.get("id") is not None}
    if len(combined_ids) != len(combined):
        print(f"[WARNING] Found duplicate IDs in combined dataset! "
              f"Unique IDs: {len(combined_ids)}, Total records: {len(combined)}")
    
    _write_json(output_path, combined)
    print(f"[Omini] Saved combined dataset to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset JSON files with custom sampling rules."
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    aqua = sub.add_parser("aqua", help="Export AQuA test + sampled train to JSON.")
    aqua.add_argument("--train", required=True, help="Path to AQuA train parquet.")
    aqua.add_argument("--test", required=True, help="Path to AQuA test parquet.")
    aqua.add_argument("--output", required=True, help="Destination JSON file.")
    aqua.add_argument("--target_size", type=int, default=500)
    aqua.add_argument("--seed", type=int, default=42)

    math = sub.add_parser("math", help="Export MATH test parquet to JSON.")
    math.add_argument("--test", required=True, help="Path to MATH test parquet.")
    math.add_argument("--output", required=True, help="Destination JSON file.")

    gsm = sub.add_parser(
        "gsm8k",
        help="Ensure filtered GSM8K data reaches target size using test split.",
    )
    gsm.add_argument("--filtered", required=True, help="Filtered JSON input file.")
    gsm.add_argument("--test", required=True, help="Path to GSM8K test parquet.")
    gsm.add_argument("--output", required=True, help="Destination JSON file.")
    gsm.add_argument("--target_size", type=int, default=500)
    gsm.add_argument("--seed", type=int, default=42)

    omini = sub.add_parser(
        "omini",
        help="Ensure filtered Omini data reaches target size using original dataset.",
    )
    omini.add_argument("--filtered", required=True, help="Filtered JSON input file.")
    omini.add_argument("--original", required=True, help="Path to original Omini dataset (JSON/JSONL/Parquet).")
    omini.add_argument("--output", required=True, help="Destination JSON file.")
    omini.add_argument("--target_size", type=int, default=500)
    omini.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = build_args()
    if args.dataset == "aqua":
        prepare_aqua(
            train_path=args.train,
            test_path=args.test,
            output_path=args.output,
            target_size=args.target_size,
            seed=args.seed,
        )
    elif args.dataset == "math":
        prepare_math(test_path=args.test, output_path=args.output)
    elif args.dataset == "gsm8k":
        prepare_gsm8k(
            filtered_json=args.filtered,
            test_path=args.test,
            output_path=args.output,
            target_size=args.target_size,
            seed=args.seed,
        )
    elif args.dataset == "omini":
        prepare_omini(
            filtered_json=args.filtered,
            original_path=args.original,
            output_path=args.output,
            target_size=args.target_size,
            seed=args.seed,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()

