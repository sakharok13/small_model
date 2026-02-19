#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a global missing-task plan for HotpotQA reasoning trace generation.

Workflow:
1) Reconstruct the full intended task set across all original ranks.
2) Scan completed outputs across all rank directories.
3) Compute missing tasks globally.
4) Redistribute missing tasks evenly to worker task files.
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Sequence, Set, Tuple

from datasets import Dataset, load_dataset


ALL_VERSIONS: Tuple[str, str, str] = ("v1", "v2", "v3")


def normalize_out_dir(path: str) -> str:
    p = (path or "").strip().rstrip("/")
    if not p:
        return p
    if p.endswith(".parquet"):
        p = os.path.dirname(p)
    p = re.sub(r"\.rank\d+$", "", p)
    return p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--plan_dir",
        type=str,
        default="",
        help="Directory to write rebalance plan files. Default: <out_dir>.rebalance_plan",
    )
    ap.add_argument("--dataset_name", type=str, default="hotpot_qa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--split", type=str, default="train")

    ap.add_argument("--num_shards", type=int, default=8, help="Original rank count used for generation")
    ap.add_argument("--num_workers", type=int, default=8, help="How many worker task files to emit")
    ap.add_argument("--max_samples", type=int, default=0, help="Same meaning as generator: 0 means full split")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--context_mode", type=str, choices=["all", "gold_titles", "supporting"], default="all")
    ap.add_argument("--max_context_chars", type=int, default=6000)
    ap.add_argument("--prompt_version", type=str, choices=["v1", "v2", "v3", "mix", "all"], default="mix")

    args = ap.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max_samples must be >= 0")
    if not args.plan_dir:
        args.plan_dir = f"{args.out_dir}.rebalance_plan"
    args.out_dir = normalize_out_dir(args.out_dir)
    args.plan_dir = args.plan_dir.strip().rstrip("/")
    return args


def local_target(global_max: int, world: int, rank: int) -> int:
    base = global_max // world
    rem = global_max % world
    return base + (1 if rank < rem else 0)


def make_task_key(query: Any, context: Any, prompt_version: Any, reference_answer: Any) -> str:
    h = hashlib.blake2b(digest_size=16)
    for part in (query, context, prompt_version, reference_answer):
        text = str("" if part is None else part)
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def parse_context_field(raw: Any) -> List[Tuple[str, List[str]]]:
    out: List[Tuple[str, List[str]]] = []
    if isinstance(raw, dict):
        titles = raw.get("title") or []
        sents = raw.get("sentences") or []
        for title, sent_list in zip(titles, sents):
            if isinstance(sent_list, str):
                sent_list = [sent_list]
            if not isinstance(sent_list, list):
                sent_list = []
            out.append((str(title), [str(x) for x in sent_list]))
        return out

    if isinstance(raw, list):
        for item in raw:
            title = ""
            sent_list: List[str] = []
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                if isinstance(item[1], str):
                    sent_list = [item[1]]
                elif isinstance(item[1], list):
                    sent_list = [str(x) for x in item[1]]
            elif isinstance(item, dict):
                title = str(item.get("title") or "")
                s = item.get("sentences") or item.get("sentence") or []
                if isinstance(s, str):
                    sent_list = [s]
                elif isinstance(s, list):
                    sent_list = [str(x) for x in s]
            if title or sent_list:
                out.append((title, sent_list))
    return out


def parse_supporting_facts(raw: Any) -> Tuple[Set[Tuple[str, int]], Set[str]]:
    pairs: Set[Tuple[str, int]] = set()
    titles: Set[str] = set()

    if isinstance(raw, dict):
        raw_titles = raw.get("title") or []
        raw_ids = raw.get("sent_id") or []
        for title, sid in zip(raw_titles, raw_ids):
            try:
                sid_int = int(sid)
            except Exception:
                continue
            t = str(title)
            pairs.add((t, sid_int))
            titles.add(t)
        return pairs, titles

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = str(item[0])
            try:
                sid_int = int(item[1])
            except Exception:
                continue
            pairs.add((title, sid_int))
            titles.add(title)
    return pairs, titles


def format_doc(title: str, sentences: Sequence[str]) -> str:
    text = " ".join(x.strip() for x in sentences if str(x).strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if title.strip():
        return f"{title.strip()}\n{text}"
    return text


def build_hotpot_context(example: Dict[str, Any], mode: str, max_context_chars: int) -> str:
    docs = parse_context_field(example.get("context"))
    sf_pairs, sf_titles = parse_supporting_facts(example.get("supporting_facts"))

    selected: List[Tuple[str, List[str]]] = []
    if mode == "all":
        selected = docs
    elif mode == "gold_titles":
        selected = [(title, sents) for title, sents in docs if title in sf_titles]
        if not selected:
            selected = docs
    else:
        by_title: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for title, sents in docs:
            for i, sent in enumerate(sents):
                if (title, i) in sf_pairs:
                    by_title[title].append((i, sent))
        for title, _ in docs:
            if title not in by_title:
                continue
            ranked = sorted(by_title[title], key=lambda t: t[0])
            selected.append((title, [x[1] for x in ranked]))
        if not selected:
            selected = docs

    blocks: List[str] = []
    for title, sents in selected:
        block = format_doc(title=title, sentences=sents)
        if block:
            blocks.append(block)
    ctx = "\n\n".join(blocks).strip()
    if max_context_chars > 0 and len(ctx) > max_context_chars:
        return ctx[:max_context_chars].rstrip()
    return ctx


def load_hotpot_examples(args: argparse.Namespace, world: int, rank: int) -> List[Dict[str, str]]:
    ds: Dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if world > 1:
        ds = ds.shard(num_shards=world, index=rank)

    out: List[Dict[str, str]] = []
    for row in ds:
        query = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        context = build_hotpot_context(row, mode=args.context_mode, max_context_chars=args.max_context_chars)
        if not query or not context:
            continue
        out.append(
            {
                "query": query,
                "context": context,
                "reference_answer": answer,
            }
        )
    return out


def pick_prompt_versions(configured: str, rng: random.Random) -> List[str]:
    if configured == "all":
        return list(ALL_VERSIONS)
    if configured == "mix":
        return [rng.choice(list(ALL_VERSIONS))]
    return [configured]


def scan_completed_keys_all_ranks(out_dir: str) -> Tuple[int, Set[str], Dict[str, int], List[str]]:
    rank_dirs = sorted(p for p in glob(f"{out_dir}.rank*") if os.path.isdir(p))
    if os.path.isdir(out_dir):
        rank_dirs.append(out_dir)

    seen_dirs: List[str] = []
    completed_keys: Set[str] = set()
    rows_on_disk = 0
    counts_by_version: Dict[str, int] = {"v1": 0, "v2": 0, "v3": 0}

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to scan existing parquet shards.") from exc

    for d in rank_dirs:
        files = sorted(glob(os.path.join(d, "part-*.parquet")))
        if not files:
            continue
        seen_dirs.append(d)
        for path in files:
            pf = pq.ParquetFile(path)
            rows_on_disk += pf.metadata.num_rows
            names = set(pf.schema_arrow.names)
            if "query" not in names or "context" not in names:
                continue

            read_cols = ["query", "context"]
            if "prompt_version" in names:
                read_cols.append("prompt_version")
            if "reference_answer" in names:
                read_cols.append("reference_answer")
            table = pq.read_table(path, columns=read_cols)
            data = table.to_pydict()

            queries = data.get("query") or []
            contexts = data.get("context") or []
            prompt_versions = data.get("prompt_version") or [""] * len(queries)
            reference_answers = data.get("reference_answer") or [""] * len(queries)
            n = min(len(queries), len(contexts))

            for i in range(n):
                pv = prompt_versions[i] if i < len(prompt_versions) else ""
                ra = reference_answers[i] if i < len(reference_answers) else ""
                key = make_task_key(queries[i], contexts[i], pv, ra)
                if key in completed_keys:
                    continue
                completed_keys.add(key)
                if pv in counts_by_version:
                    counts_by_version[pv] += 1

    return rows_on_disk, completed_keys, counts_by_version, seen_dirs


def write_jsonl(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    all_tasks: List[Dict[str, str]] = []
    all_keys: Set[str] = set()
    target_by_version: Dict[str, int] = {"v1": 0, "v2": 0, "v3": 0}

    for rank in range(args.num_shards):
        rng = random.Random(args.seed + rank)
        examples = load_hotpot_examples(args=args, world=args.num_shards, rank=rank)

        if args.max_samples == 0:
            local_max = len(examples)
        else:
            local_max = min(local_target(args.max_samples, world=args.num_shards, rank=rank), len(examples))

        if local_max <= 0:
            continue

        rng.shuffle(examples)
        examples = examples[:local_max]

        for ex in examples:
            versions = pick_prompt_versions(args.prompt_version, rng=rng)
            for version in versions:
                key = make_task_key(
                    query=ex["query"],
                    context=ex["context"],
                    prompt_version=version,
                    reference_answer=ex["reference_answer"],
                )
                if key in all_keys:
                    continue
                all_keys.add(key)
                task = {
                    "query": ex["query"],
                    "context": ex["context"],
                    "reference_answer": ex["reference_answer"],
                    "prompt_version": version,
                    "task_key": key,
                }
                all_tasks.append(task)
                target_by_version[version] = target_by_version.get(version, 0) + 1

    rows_on_disk, completed_keys, completed_by_version, scanned_dirs = scan_completed_keys_all_ranks(args.out_dir)

    missing = [task for task in all_tasks if task["task_key"] not in completed_keys]
    shuffle_rng = random.Random(args.seed + 1009)
    shuffle_rng.shuffle(missing)

    os.makedirs(args.plan_dir, exist_ok=True)
    for old in glob(os.path.join(args.plan_dir, "tasks_rank*.jsonl")):
        try:
            os.remove(old)
        except OSError:
            pass

    tasks_per_worker: List[List[Dict[str, str]]] = [[] for _ in range(args.num_workers)]
    for i, task in enumerate(missing):
        tasks_per_worker[i % args.num_workers].append(task)

    for worker_rank, rows in enumerate(tasks_per_worker):
        out_path = os.path.join(args.plan_dir, f"tasks_rank{worker_rank}.jsonl")
        write_jsonl(out_path, rows)

    summary = {
        "out_dir": args.out_dir,
        "plan_dir": args.plan_dir,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "num_shards": args.num_shards,
        "num_workers": args.num_workers,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "prompt_version": args.prompt_version,
        "target_total": len(all_tasks),
        "target_by_prompt_version": target_by_version,
        "existing_rows_on_disk": rows_on_disk,
        "existing_unique_completed": len(completed_keys),
        "existing_by_prompt_version": completed_by_version,
        "missing_total": len(missing),
        "missing_by_worker": [len(x) for x in tasks_per_worker],
        "scanned_output_dirs": scanned_dirs,
    }

    summary_path = os.path.join(args.plan_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote rebalance plan to {args.plan_dir}")


if __name__ == "__main__":
    main()
