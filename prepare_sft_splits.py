#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from glob import glob
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from synth_utils import IDK_DEFAULT, NO_CONTEXT_DEFAULT, ParquetShardWriter

GROUP_CORRECT = "correct"
GROUP_REFUSAL = "refusal"
GROUP_NEGATIVE = "negative"
GROUPS = (GROUP_CORRECT, GROUP_REFUSAL, GROUP_NEGATIVE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build class-aware train/validation splits from parquet shards. "
            "Outputs grouped train splits and balanced validation groups."
        )
    )
    ap.add_argument(
        "--input_glob",
        action="append",
        required=True,
        help=(
            "Glob for source parquet files. Can be passed multiple times. "
            "Example: --input_glob '/home/jovyan/.../synth_ctx_idk.rank*/part-*.parquet'"
        ),
    )
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--val_per_group", type=int, default=5_000, help="Target validation rows per group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--idk_text", type=str, default=IDK_DEFAULT)
    ap.add_argument("--no_context_text", type=str, default=NO_CONTEXT_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=2048, help="Parquet read batch size")
    ap.add_argument("--flush_rows", type=int, default=10_000, help="Flush parquet writers every N rows")
    ap.add_argument("--shard_size", type=int, default=200_000, help="Rows per output parquet shard")
    ap.add_argument("--parquet_compression", type=str, default="zstd")
    ap.add_argument("--keep_raw_answers", action="store_true", help="Keep source answer text instead of canonicalizing")
    ap.add_argument(
        "--allow_nonempty_output_dir",
        action="store_true",
        help="Allow writing into a non-empty output_dir",
    )
    return ap.parse_args()


def parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def canonicalize_answer(answer: str, idk_text: str, no_context_text: str) -> str:
    a = answer.strip()
    n = normalize_text(a)

    idk_norm = normalize_text(idk_text)
    no_ctx_norm = normalize_text(no_context_text)

    # Common refusal variants in synthetic generations.
    idk_variants = {
        "i don't know",
        "i dont know",
        "i cannot find the answer in the context",
        "i can't find the answer in the context",
        "cannot find the answer in the context",
        "can't find the answer in the context",
        "not found in the context",
    }
    no_ctx_variants = {
        "i dont see any relevant context.",
        "i dont see any relevant context",
        "i don't see any relevant context.",
        "i don't see any relevant context",
        "no relevant context",
    }

    if n == idk_norm or n in idk_variants:
        return idk_text
    if n == no_ctx_norm or n in no_ctx_variants:
        return no_context_text
    return a


def classify_example(context: str, canonical_answer: str, negative_flag: bool, idk_text: str, no_context_text: str) -> str:
    # Missing-context behavior should be evaluated as refusal, not negative.
    if context.strip() == "" or canonical_answer == no_context_text:
        return GROUP_REFUSAL
    if negative_flag:
        return GROUP_NEGATIVE
    if canonical_answer == idk_text:
        return GROUP_REFUSAL
    return GROUP_CORRECT


def transform_row(row: Dict[str, Any], idk_text: str, no_context_text: str, keep_raw_answers: bool) -> Optional[Dict[str, Any]]:
    query = str(row.get("query") or "").strip()
    context = str(row.get("context") or "").strip()
    raw_answer = str(row.get("answer") or "").strip()
    if not query or not raw_answer:
        return None

    negative = parse_bool(row.get("negative", False))
    canonical_answer = canonicalize_answer(raw_answer, idk_text=idk_text, no_context_text=no_context_text)
    answer = raw_answer if keep_raw_answers else canonical_answer

    group = classify_example(
        context=context,
        canonical_answer=canonical_answer,
        negative_flag=negative,
        idk_text=idk_text,
        no_context_text=no_context_text,
    )

    return {
        "query": query,
        "context": context,
        "answer": answer,
        "negative": negative,
        "example_type": group,
    }


def iter_parquet_rows(paths: Iterable[str], batch_size: int) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    cols = ["query", "context", "answer", "negative"]
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                yield row


def resolve_input_paths(input_globs: List[str]) -> List[str]:
    paths: List[str] = []
    for pat in input_globs:
        paths.extend(glob(pat))
    paths = sorted(set(paths))
    return [p for p in paths if p.endswith(".parquet")]


def ensure_output_dir(path: str, allow_nonempty: bool) -> None:
    os.makedirs(path, exist_ok=True)
    if allow_nonempty:
        return
    if os.listdir(path):
        raise ValueError(
            f"Output directory is not empty: {path}. "
            "Use --allow_nonempty_output_dir if you want to append."
        )


def flush_buffers(
    writers: Dict[str, ParquetShardWriter],
    buffers: Dict[str, List[Dict[str, Any]]],
    flush_rows: int,
    force: bool = False,
) -> Dict[str, int]:
    written = {k: 0 for k in writers}
    for key, writer in writers.items():
        if not buffers[key]:
            continue
        if force or len(buffers[key]) >= flush_rows:
            writer.write_rows(buffers[key])
            written[key] += len(buffers[key])
            buffers[key] = []
    return written


def main() -> None:
    args = parse_args()
    random_gen = random.Random(args.seed)

    input_paths = resolve_input_paths(args.input_glob)
    if not input_paths:
        raise ValueError("No parquet files matched --input_glob")

    ensure_output_dir(args.output_dir, allow_nonempty=args.allow_nonempty_output_dir)

    print(f"Found {len(input_paths)} parquet files")
    print("Pass 1/2: counting groups")

    counts = {g: 0 for g in GROUPS}
    total_seen = 0
    total_kept = 0
    dropped = 0

    for row in iter_parquet_rows(input_paths, batch_size=args.batch_size):
        total_seen += 1
        out = transform_row(
            row,
            idk_text=args.idk_text,
            no_context_text=args.no_context_text,
            keep_raw_answers=args.keep_raw_answers,
        )
        if out is None:
            dropped += 1
            continue
        counts[out["example_type"]] += 1
        total_kept += 1

    val_targets = {g: min(args.val_per_group, counts[g]) for g in GROUPS}
    val_positions: Dict[str, Set[int]] = {}
    for g in GROUPS:
        if val_targets[g] == 0:
            val_positions[g] = set()
            continue
        val_positions[g] = set(random_gen.sample(range(counts[g]), k=val_targets[g]))

    print("Pass 2/2: writing grouped train/validation splits")

    train_paths = {g: os.path.join(args.output_dir, f"train_{g}") for g in GROUPS}
    val_paths = {g: os.path.join(args.output_dir, f"val_{g}") for g in GROUPS}
    os.makedirs(os.path.join(args.output_dir, "train_all"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val_all"), exist_ok=True)
    for p in train_paths.values():
        os.makedirs(p, exist_ok=True)
    for p in val_paths.values():
        os.makedirs(p, exist_ok=True)

    train_writers = {
        **{f"train_{g}": ParquetShardWriter(out_dir=train_paths[g], shard_size=args.shard_size, compression=args.parquet_compression) for g in GROUPS},
        "train_all": ParquetShardWriter(
            out_dir=os.path.join(args.output_dir, "train_all"),
            shard_size=args.shard_size,
            compression=args.parquet_compression,
        ),
    }
    val_writers = {
        **{f"val_{g}": ParquetShardWriter(out_dir=val_paths[g], shard_size=args.shard_size, compression=args.parquet_compression) for g in GROUPS},
        "val_all": ParquetShardWriter(
            out_dir=os.path.join(args.output_dir, "val_all"),
            shard_size=args.shard_size,
            compression=args.parquet_compression,
        ),
    }

    train_buffers = {k: [] for k in train_writers}
    val_buffers = {k: [] for k in val_writers}
    train_written = {k: 0 for k in train_writers}
    val_written = {k: 0 for k in val_writers}

    seen_per_group = {g: 0 for g in GROUPS}

    for row in iter_parquet_rows(input_paths, batch_size=args.batch_size):
        out = transform_row(
            row,
            idk_text=args.idk_text,
            no_context_text=args.no_context_text,
            keep_raw_answers=args.keep_raw_answers,
        )
        if out is None:
            continue

        g = out["example_type"]
        position = seen_per_group[g]
        seen_per_group[g] += 1

        if position in val_positions[g]:
            val_buffers[f"val_{g}"].append(out)
            val_buffers["val_all"].append(out)
            written = flush_buffers(val_writers, val_buffers, flush_rows=args.flush_rows, force=False)
            for key, n in written.items():
                val_written[key] += n
        else:
            train_buffers[f"train_{g}"].append(out)
            train_buffers["train_all"].append(out)
            written = flush_buffers(train_writers, train_buffers, flush_rows=args.flush_rows, force=False)
            for key, n in written.items():
                train_written[key] += n

    written = flush_buffers(train_writers, train_buffers, flush_rows=args.flush_rows, force=True)
    for key, n in written.items():
        train_written[key] += n
    written = flush_buffers(val_writers, val_buffers, flush_rows=args.flush_rows, force=True)
    for key, n in written.items():
        val_written[key] += n

    for writer in train_writers.values():
        writer.close()
    for writer in val_writers.values():
        writer.close()

    train_counts = {g: train_written[f"train_{g}"] for g in GROUPS}
    val_counts = {g: val_written[f"val_{g}"] for g in GROUPS}

    stats = {
        "input_files": len(input_paths),
        "total_rows_seen": total_seen,
        "total_rows_kept": total_kept,
        "dropped_rows": dropped,
        "source_counts_by_group": counts,
        "requested_val_per_group": args.val_per_group,
        "validation_counts_by_group": val_counts,
        "train_counts_by_group": train_counts,
        "train_all_rows": train_written["train_all"],
        "val_all_rows": val_written["val_all"],
        "idk_text": args.idk_text,
        "no_context_text": args.no_context_text,
        "input_globs": args.input_glob,
    }

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(f"Done. Wrote splits to: {args.output_dir}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
