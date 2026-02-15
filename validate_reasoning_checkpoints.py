#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from glob import glob
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from transformers import AutoTokenizer
except Exception:  # noqa: BLE001
    AutoTokenizer = None


DEFAULT_SPECIAL_TOKENS = {
    "context_start_token": "[context_start]",
    "context_end_token": "[context_end]",
    "query_start_token": "[query_start]",
    "query_end_token": "[query_end]",
    "thinking_start_token": "[thinking_start]",
    "thinking_end_token": "[thinking_end]",
    "answer_start_token": "[answer_start]",
    "answer_end_token": "[answer_end]",
}

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def parse_csv_strings(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def parse_bool_flag(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def checkpoint_step(path: str) -> int:
    base = os.path.basename(path.rstrip("/"))
    m = CHECKPOINT_RE.match(base)
    if not m:
        return -1
    return int(m.group(1))


def collect_eval_targets(run_dir: str, mode: str) -> List[Tuple[str, str]]:
    ckpts = sorted(
        [p for p in glob(os.path.join(run_dir, "checkpoint-*")) if os.path.isdir(p)],
        key=checkpoint_step,
    )
    if mode == "final":
        return [("final", run_dir)]
    if mode == "latest":
        if ckpts:
            latest = ckpts[-1]
            return [(os.path.basename(latest), latest)]
        return [("final", run_dir)]

    # all
    out: List[Tuple[str, str]] = [(os.path.basename(p), p) for p in ckpts]
    out.append(("final", run_dir))
    return out


def infer_group_from_summary(summary: Dict[str, Any]) -> Optional[str]:
    prompt_versions = summary.get("prompt_versions_filter")
    if isinstance(prompt_versions, list):
        cleaned = [str(x).strip() for x in prompt_versions if str(x).strip()]
        if len(cleaned) == 1:
            return cleaned[0]

    args_block = summary.get("args")
    if isinstance(args_block, dict):
        pv = args_block.get("prompt_versions", "")
        tokens = [x.strip() for x in str(pv).split(",") if x.strip()]
        if len(tokens) == 1:
            return tokens[0]
    return None


def validate_run_summary(
    run_dir: str,
    group: str,
    expected_train_rows: int,
    expected_special_tokens: Dict[str, str],
) -> Dict[str, Any]:
    summary_path = os.path.join(run_dir, "sft_reasoning_summary.json")
    result: Dict[str, Any] = {
        "run_dir": run_dir,
        "group": group,
        "summary_path": summary_path,
        "ok": True,
        "errors": [],
        "warnings": [],
    }

    if not os.path.exists(summary_path):
        result["ok"] = False
        result["errors"].append(f"Missing {summary_path}")
        return result

    try:
        summary = read_json(summary_path)
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["errors"].append(f"Failed to parse {summary_path}: {exc}")
        return result

    result["summary"] = summary

    train_rows = summary.get("train_rows")
    if isinstance(train_rows, int):
        result["train_rows"] = train_rows
        if expected_train_rows > 0 and train_rows != expected_train_rows:
            result["ok"] = False
            result["errors"].append(
                f"train_rows={train_rows}, expected={expected_train_rows}"
            )
    else:
        result["ok"] = False
        result["errors"].append("Missing integer train_rows in sft_reasoning_summary.json")

    expected_group = [group]
    got_group = summary.get("prompt_versions_filter")
    if got_group != expected_group:
        result["ok"] = False
        result["errors"].append(
            f"prompt_versions_filter={got_group}, expected={expected_group}"
        )

    pv_stats = summary.get("train_prompt_version_filter")
    if isinstance(pv_stats, dict):
        allowed = pv_stats.get("allowed")
        if allowed != expected_group:
            result["ok"] = False
            result["errors"].append(
                f"train_prompt_version_filter.allowed={allowed}, expected={expected_group}"
            )
    else:
        result["ok"] = False
        result["errors"].append("Missing train_prompt_version_filter block")

    args_block = summary.get("args", {})
    if not isinstance(args_block, dict):
        args_block = {}
        result["warnings"].append("Missing args block in summary; cannot fully verify special tokens")

    for key, expected in expected_special_tokens.items():
        got = args_block.get(key)
        if got != expected:
            result["ok"] = False
            result["errors"].append(f"{key}={got!r}, expected={expected!r}")

    if not parse_bool_flag(args_block.get("require_thinking", True)):
        result["ok"] = False
        result["errors"].append("require_thinking is False, expected True")

    if not parse_bool_flag(args_block.get("filter_correct_only", True)):
        result["ok"] = False
        result["errors"].append("filter_correct_only is False, expected True")

    return result


def find_run_candidates(
    run_dirs: Sequence[str],
    runs_root: Optional[str],
    groups: Sequence[str],
) -> List[Dict[str, Any]]:
    groups_set = set(groups)
    candidates: List[str] = []

    for rd in run_dirs:
        if rd:
            candidates.append(os.path.abspath(rd))

    if runs_root:
        root = os.path.abspath(runs_root)
        candidates.extend(
            os.path.dirname(p)
            for p in glob(os.path.join(root, "**", "sft_reasoning_summary.json"), recursive=True)
        )

    uniq_dirs = sorted(set(candidates))
    out: List[Dict[str, Any]] = []
    for run_dir in uniq_dirs:
        summary_path = os.path.join(run_dir, "sft_reasoning_summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            summary = read_json(summary_path)
        except Exception:  # noqa: BLE001
            continue
        group = infer_group_from_summary(summary=summary)
        if not group or group not in groups_set:
            continue
        out.append(
            {
                "run_dir": run_dir,
                "run_name": os.path.basename(run_dir.rstrip("/")),
                "group": group,
                "summary": summary,
                "summary_path": summary_path,
            }
        )
    return out


def detect_gpus(gpus_arg: str) -> List[str]:
    if gpus_arg:
        out = [x.strip() for x in gpus_arg.split(",") if x.strip()]
        if out:
            return out

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        out = [x.strip() for x in visible.split(",") if x.strip()]
        if out:
            return out

    try:
        import torch

        n = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        n = 0
    if n > 0:
        return [str(i) for i in range(n)]
    return ["0"]


def load_tokenizer_check(
    model_path: str,
    trust_remote_code: bool,
    local_files_only: bool,
    expected_special_tokens: Dict[str, str],
    cache: Dict[str, Dict[str, Any]],
    cache_lock: threading.Lock,
) -> Dict[str, Any]:
    with cache_lock:
        if model_path in cache:
            return cache[model_path]

    result: Dict[str, Any] = {
        "ok": True,
        "missing_tokens": [],
        "token_ids": {},
        "errors": [],
    }
    if AutoTokenizer is None:
        result["ok"] = False
        result["errors"].append("transformers is not installed")
        return result
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        unk_id = tokenizer.unk_token_id
        for name, token in expected_special_tokens.items():
            tid = tokenizer.convert_tokens_to_ids(token)
            result["token_ids"][name] = tid
            if not isinstance(tid, int) or tid < 0 or (unk_id is not None and tid == unk_id):
                result["ok"] = False
                result["missing_tokens"].append(token)
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["errors"].append(str(exc))

    with cache_lock:
        cache[model_path] = result
    return result


def analyze_predictions_format(
    predictions_path: str,
    thinking_start_token: str,
    thinking_end_token: str,
    answer_start_token: str,
    answer_end_token: str,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "ok": True,
        "errors": [],
        "num_rows": 0,
        "required_keys_missing_rows": 0,
        "raw_has_thinking_start": 0,
        "raw_has_thinking_end": 0,
        "raw_has_answer_start": 0,
        "raw_has_answer_end": 0,
        "parsed_nonempty_answer": 0,
        "parsed_nonempty_thinking": 0,
    }
    required_keys = {
        "prediction_answer",
        "prediction_thinking",
        "raw_generation",
        "em",
        "f1",
    }

    if not os.path.exists(predictions_path):
        stats["ok"] = False
        stats["errors"].append(f"Missing predictions file: {predictions_path}")
        return stats

    try:
        with open(predictions_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                stats["num_rows"] += 1

                if any(k not in row for k in required_keys):
                    stats["required_keys_missing_rows"] += 1

                raw = str(row.get("raw_generation", ""))
                thinking = str(row.get("prediction_thinking", "")).strip()
                answer = str(row.get("prediction_answer", "")).strip()

                if thinking_start_token and thinking_start_token in raw:
                    stats["raw_has_thinking_start"] += 1
                if thinking_end_token and thinking_end_token in raw:
                    stats["raw_has_thinking_end"] += 1
                if answer_start_token and answer_start_token in raw:
                    stats["raw_has_answer_start"] += 1
                if answer_end_token and answer_end_token in raw:
                    stats["raw_has_answer_end"] += 1
                if thinking:
                    stats["parsed_nonempty_thinking"] += 1
                if answer:
                    stats["parsed_nonempty_answer"] += 1
    except Exception as exc:  # noqa: BLE001
        stats["ok"] = False
        stats["errors"].append(f"Failed to read {predictions_path}: {exc}")
        return stats

    n = max(1, int(stats["num_rows"]))
    stats["rates"] = {
        "thinking_start_rate": stats["raw_has_thinking_start"] / n,
        "thinking_end_rate": stats["raw_has_thinking_end"] / n,
        "answer_start_rate": stats["raw_has_answer_start"] / n,
        "answer_end_rate": stats["raw_has_answer_end"] / n,
        "nonempty_thinking_rate": stats["parsed_nonempty_thinking"] / n,
        "nonempty_answer_rate": stats["parsed_nonempty_answer"] / n,
        "missing_required_keys_rate": stats["required_keys_missing_rows"] / n,
    }
    return stats


def evaluate_format_thresholds(
    format_stats: Dict[str, Any],
    min_thinking_start_rate: float,
    min_thinking_end_rate: float,
    min_answer_start_rate: float,
    min_answer_end_rate: float,
    min_nonempty_thinking_rate: float,
    min_nonempty_answer_rate: float,
    max_missing_required_keys_rate: float,
) -> List[str]:
    if not format_stats.get("ok", False):
        return list(format_stats.get("errors", []))

    rates = format_stats.get("rates", {})
    failures: List[str] = []

    def check(name: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> None:
        raw = rates.get(name)
        try:
            value = float(raw)
        except Exception:  # noqa: BLE001
            failures.append(f"{name} is missing")
            return
        if min_value is not None and value < min_value:
            failures.append(f"{name}={value:.4f} < {min_value:.4f}")
        if max_value is not None and value > max_value:
            failures.append(f"{name}={value:.4f} > {max_value:.4f}")

    check("thinking_start_rate", min_value=min_thinking_start_rate)
    check("thinking_end_rate", min_value=min_thinking_end_rate)
    check("answer_start_rate", min_value=min_answer_start_rate)
    check("answer_end_rate", min_value=min_answer_end_rate)
    check("nonempty_thinking_rate", min_value=min_nonempty_thinking_rate)
    check("nonempty_answer_rate", min_value=min_nonempty_answer_rate)
    check("missing_required_keys_rate", max_value=max_missing_required_keys_rate)
    return failures


def build_eval_command(
    args: argparse.Namespace,
    model_path: str,
    output_dir: str,
) -> List[str]:
    cmd = [
        args.python_bin,
        args.eval_script,
        "--model_path",
        model_path,
        "--split",
        args.eval_split,
        "--max_samples",
        str(args.eval_max_samples),
        "--seed",
        str(args.eval_seed),
        "--batch_size",
        str(args.eval_batch_size),
        "--max_prompt_tokens",
        str(args.eval_max_prompt_tokens),
        "--max_new_tokens",
        str(args.eval_max_new_tokens),
        "--temperature",
        str(args.eval_temperature),
        "--top_p",
        str(args.eval_top_p),
        "--context_mode",
        args.eval_context_mode,
        "--max_context_chars",
        str(args.eval_max_context_chars),
        "--device_map",
        args.eval_device_map,
        "--dtype",
        args.eval_dtype,
        "--output_dir",
        output_dir,
        "--context_start_token",
        args.context_start_token,
        "--context_end_token",
        args.context_end_token,
        "--query_start_token",
        args.query_start_token,
        "--query_end_token",
        args.query_end_token,
        "--thinking_start_token",
        args.thinking_start_token,
        "--thinking_end_token",
        args.thinking_end_token,
        "--answer_start_token",
        args.answer_start_token,
        "--answer_end_token",
        args.answer_end_token,
    ]
    if args.base_model:
        cmd.extend(["--base_model", args.base_model])
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.local_files_only:
        cmd.append("--local_files_only")
    return cmd


def choose_output_dir(
    run_dir: str,
    run_name: str,
    label: str,
    output_root: Optional[str],
    eval_subdir: str,
) -> str:
    if output_root:
        return os.path.join(output_root, run_name, label)
    return os.path.join(run_dir, eval_subdir, label)


def choose_log_path(
    run_dir: str,
    run_name: str,
    label: str,
    output_root: Optional[str],
    eval_subdir: str,
) -> str:
    if output_root:
        return os.path.join(output_root, run_name, "logs", f"{label}.log")
    return os.path.join(run_dir, eval_subdir, "logs", f"{label}.log")


def resolve_summary_dir(args: argparse.Namespace, run_records: Sequence[Dict[str, Any]]) -> str:
    if args.summary_dir:
        return os.path.abspath(args.summary_dir)
    if args.output_root:
        return os.path.abspath(args.output_root)
    if args.runs_root:
        return os.path.abspath(args.runs_root)
    if run_records:
        return os.path.abspath(os.path.dirname(run_records[0]["run_dir"]))
    return os.getcwd()


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=(
            "Validate reasoning-SFT checkpoints for v1/v2/v3 runs: "
            "metadata checks (6k/group + special tokens) + HotpotQA 6k eval in parallel on multiple GPUs."
        )
    )
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    ap.add_argument(
        "--eval_script",
        type=str,
        default=os.path.join(here, "benchmark_hotpotqa_reasoning.py"),
        help="Path to benchmark_hotpotqa_reasoning.py",
    )
    ap.add_argument("--runs_root", type=str, default=None, help="Root directory with SFT run folders")
    ap.add_argument("--run_dir", action="append", default=[], help="Explicit run directory (repeatable)")
    ap.add_argument("--groups", type=str, default="v1,v2,v3")
    ap.add_argument("--expected_train_rows", type=int, default=6000)
    ap.add_argument("--allow_missing_groups", action="store_true")
    ap.add_argument("--eval_checkpoints", type=str, default="all", choices=["all", "latest", "final"])

    ap.add_argument("--output_root", type=str, default=None, help="Optional root for eval outputs")
    ap.add_argument(
        "--eval_subdir",
        type=str,
        default="hotpot_eval_6k",
        help="Subdir in each run dir when --output_root is not set",
    )
    ap.add_argument("--summary_dir", type=str, default=None, help="Where to write aggregate JSON/CSV summaries")
    ap.add_argument("--skip_existing", action="store_true")

    ap.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids (default: auto-detect)")
    ap.add_argument("--jobs_per_gpu", type=int, default=1, help="Concurrent jobs per GPU (default: 1)")

    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--base_model", type=str, default=None)

    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--eval_max_samples", type=int, default=6000)
    ap.add_argument("--eval_seed", type=int, default=42)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--eval_max_new_tokens", type=int, default=256)
    ap.add_argument("--eval_temperature", type=float, default=0.0)
    ap.add_argument("--eval_top_p", type=float, default=1.0)
    ap.add_argument("--eval_context_mode", type=str, default="all", choices=["all", "gold_titles", "supporting"])
    ap.add_argument("--eval_max_context_chars", type=int, default=6000)
    ap.add_argument("--eval_device_map", type=str, default="auto")
    ap.add_argument("--eval_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])

    ap.add_argument("--context_start_token", type=str, default=DEFAULT_SPECIAL_TOKENS["context_start_token"])
    ap.add_argument("--context_end_token", type=str, default=DEFAULT_SPECIAL_TOKENS["context_end_token"])
    ap.add_argument("--query_start_token", type=str, default=DEFAULT_SPECIAL_TOKENS["query_start_token"])
    ap.add_argument("--query_end_token", type=str, default=DEFAULT_SPECIAL_TOKENS["query_end_token"])
    ap.add_argument("--thinking_start_token", type=str, default=DEFAULT_SPECIAL_TOKENS["thinking_start_token"])
    ap.add_argument("--thinking_end_token", type=str, default=DEFAULT_SPECIAL_TOKENS["thinking_end_token"])
    ap.add_argument("--answer_start_token", type=str, default=DEFAULT_SPECIAL_TOKENS["answer_start_token"])
    ap.add_argument("--answer_end_token", type=str, default=DEFAULT_SPECIAL_TOKENS["answer_end_token"])

    ap.add_argument("--min_thinking_start_rate", type=float, default=0.90)
    ap.add_argument("--min_thinking_end_rate", type=float, default=0.85)
    ap.add_argument("--min_answer_start_rate", type=float, default=0.95)
    ap.add_argument("--min_answer_end_rate", type=float, default=0.95)
    ap.add_argument("--min_nonempty_thinking_rate", type=float, default=0.90)
    ap.add_argument("--min_nonempty_answer_rate", type=float, default=0.98)
    ap.add_argument("--max_missing_required_keys_rate", type=float, default=0.0)

    ap.add_argument("--no_fail_on_metadata", action="store_true")
    ap.add_argument("--no_fail_on_missing_tokens", action="store_true")
    ap.add_argument("--no_fail_on_format", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    groups = parse_csv_strings(args.groups)
    if not groups:
        raise ValueError("--groups is empty")

    special_tokens = {
        "context_start_token": args.context_start_token,
        "context_end_token": args.context_end_token,
        "query_start_token": args.query_start_token,
        "query_end_token": args.query_end_token,
        "thinking_start_token": args.thinking_start_token,
        "thinking_end_token": args.thinking_end_token,
        "answer_start_token": args.answer_start_token,
        "answer_end_token": args.answer_end_token,
    }

    run_records = find_run_candidates(
        run_dirs=args.run_dir,
        runs_root=args.runs_root,
        groups=groups,
    )
    if not run_records:
        raise RuntimeError("No reasoning SFT run dirs found. Pass --runs_root and/or --run_dir.")

    seen_groups = {r["group"] for r in run_records}
    missing_groups = [g for g in groups if g not in seen_groups]
    if missing_groups and not args.allow_missing_groups:
        raise RuntimeError(
            f"Missing groups: {missing_groups}. Use --allow_missing_groups to continue."
        )

    run_checks: Dict[str, Dict[str, Any]] = {}
    for rec in run_records:
        run_dir = rec["run_dir"]
        group = rec["group"]
        run_checks[run_dir] = validate_run_summary(
            run_dir=run_dir,
            group=group,
            expected_train_rows=args.expected_train_rows,
            expected_special_tokens=special_tokens,
        )

    jobs: List[Dict[str, Any]] = []
    for rec in run_records:
        run_dir = rec["run_dir"]
        run_name = rec["run_name"]
        group = rec["group"]
        targets = collect_eval_targets(run_dir=run_dir, mode=args.eval_checkpoints)
        for label, model_path in targets:
            output_dir = choose_output_dir(
                run_dir=run_dir,
                run_name=run_name,
                label=label,
                output_root=args.output_root,
                eval_subdir=args.eval_subdir,
            )
            log_path = choose_log_path(
                run_dir=run_dir,
                run_name=run_name,
                label=label,
                output_root=args.output_root,
                eval_subdir=args.eval_subdir,
            )
            jobs.append(
                {
                    "run_dir": run_dir,
                    "run_name": run_name,
                    "group": group,
                    "label": label,
                    "model_path": model_path,
                    "output_dir": output_dir,
                    "log_path": log_path,
                }
            )

    jobs = sorted(jobs, key=lambda x: (x["group"], x["run_name"], x["label"]))
    if not jobs:
        raise RuntimeError("No evaluation jobs created (no checkpoints/final dirs found).")

    summary_dir = resolve_summary_dir(args=args, run_records=run_records)
    os.makedirs(summary_dir, exist_ok=True)
    results_path = os.path.join(summary_dir, "validation_results.json")
    ranked_path = os.path.join(summary_dir, "validation_ranked.json")
    best_by_group_path = os.path.join(summary_dir, "best_by_group.json")
    csv_path = os.path.join(summary_dir, "validation_results.csv")

    print(f"Discovered run dirs: {len(run_records)}")
    print(f"Planned eval jobs: {len(jobs)}")
    print(f"Summary dir: {summary_dir}")
    print(f"Eval script: {args.eval_script}")
    print(f"Eval max samples: {args.eval_max_samples}")

    if args.dry_run:
        dry_rows: List[Dict[str, Any]] = []
        for job in jobs:
            row = dict(job)
            row["command"] = build_eval_command(args=args, model_path=job["model_path"], output_dir=job["output_dir"])
            row["run_summary_check"] = run_checks.get(job["run_dir"], {})
            dry_rows.append(row)
        write_json(results_path, dry_rows)
        print(f"Dry-run saved planned jobs: {results_path}")
        return

    gpus = detect_gpus(args.gpus)
    workers_per_gpu = max(1, int(args.jobs_per_gpu))
    print(f"GPU workers: gpus={gpus}, jobs_per_gpu={workers_per_gpu}")

    q: Queue = Queue()
    for idx, job in enumerate(jobs):
        item = dict(job)
        item["job_idx"] = idx
        q.put(item)

    tokenizer_cache: Dict[str, Dict[str, Any]] = {}
    tokenizer_cache_lock = threading.Lock()

    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()
    stop_event = threading.Event()

    def persist_results() -> None:
        with results_lock:
            snapshot = list(results)
        write_json(results_path, snapshot)

    def worker_loop(gpu_id: str, slot: int) -> None:
        while not stop_event.is_set():
            try:
                job = q.get_nowait()
            except Empty:
                return

            try:
                run_dir = job["run_dir"]
                run_check = run_checks.get(run_dir, {})
                output_dir = job["output_dir"]
                metrics_path = os.path.join(output_dir, "metrics.json")
                preds_path = os.path.join(output_dir, "predictions.jsonl")
                os.makedirs(os.path.dirname(job["log_path"]) or ".", exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                row: Dict[str, Any] = {
                    "job_idx": job["job_idx"],
                    "gpu_id": gpu_id,
                    "worker_slot": slot,
                    "group": job["group"],
                    "run_name": job["run_name"],
                    "run_dir": job["run_dir"],
                    "checkpoint_label": job["label"],
                    "model_path": job["model_path"],
                    "output_dir": output_dir,
                    "log_path": job["log_path"],
                    "run_summary_check": run_check,
                    "status": "unknown",
                }

                validation_failures: List[str] = []

                if args.skip_existing and os.path.exists(metrics_path):
                    row["status"] = "skipped_existing"
                    row["metrics"] = read_json(metrics_path)
                else:
                    cmd = build_eval_command(
                        args=args,
                        model_path=job["model_path"],
                        output_dir=output_dir,
                    )
                    row["command"] = cmd
                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    env["TOKENIZERS_PARALLELISM"] = "false"

                    t0 = time.time()
                    with open(job["log_path"], "w", encoding="utf-8") as log_fh:
                        log_fh.write("COMMAND:\n")
                        log_fh.write(" ".join(cmd) + "\n\n")
                        log_fh.flush()
                        proc = subprocess.run(
                            cmd,
                            env=env,
                            stdout=log_fh,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    dt = time.time() - t0
                    row["eval_return_code"] = proc.returncode
                    row["elapsed_sec"] = round(dt, 2)
                    if proc.returncode != 0:
                        row["status"] = "eval_failed"
                        with results_lock:
                            results.append(row)
                        persist_results()
                        continue

                    if os.path.exists(metrics_path):
                        row["metrics"] = read_json(metrics_path)
                    else:
                        row["status"] = "eval_failed_missing_metrics"
                        with results_lock:
                            results.append(row)
                        persist_results()
                        continue

                tok_check = load_tokenizer_check(
                    model_path=job["model_path"],
                    trust_remote_code=args.trust_remote_code,
                    local_files_only=args.local_files_only,
                    expected_special_tokens=special_tokens,
                    cache=tokenizer_cache,
                    cache_lock=tokenizer_cache_lock,
                )
                row["tokenizer_check"] = tok_check
                if not tok_check.get("ok", False):
                    validation_failures.append(
                        "Missing special tokens in tokenizer: "
                        + ",".join(tok_check.get("missing_tokens", []))
                    )

                fmt_stats = analyze_predictions_format(
                    predictions_path=preds_path,
                    thinking_start_token=args.thinking_start_token,
                    thinking_end_token=args.thinking_end_token,
                    answer_start_token=args.answer_start_token,
                    answer_end_token=args.answer_end_token,
                )
                row["format_check"] = fmt_stats
                fmt_failures = evaluate_format_thresholds(
                    format_stats=fmt_stats,
                    min_thinking_start_rate=args.min_thinking_start_rate,
                    min_thinking_end_rate=args.min_thinking_end_rate,
                    min_answer_start_rate=args.min_answer_start_rate,
                    min_answer_end_rate=args.min_answer_end_rate,
                    min_nonempty_thinking_rate=args.min_nonempty_thinking_rate,
                    min_nonempty_answer_rate=args.min_nonempty_answer_rate,
                    max_missing_required_keys_rate=args.max_missing_required_keys_rate,
                )
                if fmt_failures:
                    validation_failures.extend(fmt_failures)

                metrics = row.get("metrics", {})
                if isinstance(metrics, dict):
                    try:
                        row["exact_match"] = float(metrics.get("exact_match"))
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        row["f1"] = float(metrics.get("f1"))
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        row["num_samples"] = int(metrics.get("num_samples"))
                    except Exception:  # noqa: BLE001
                        pass
                    if args.eval_max_samples > 0 and row.get("num_samples", 0) != args.eval_max_samples:
                        validation_failures.append(
                            f"num_samples={row.get('num_samples')} expected={args.eval_max_samples}"
                        )

                run_summary_ok = bool(run_check.get("ok", False))
                row["validation_failures"] = validation_failures
                row["validation_ok"] = not validation_failures

                hard_fail = False
                if not run_summary_ok and not args.no_fail_on_metadata:
                    hard_fail = True
                if (not tok_check.get("ok", False)) and (not args.no_fail_on_missing_tokens):
                    hard_fail = True
                if fmt_failures and not args.no_fail_on_format:
                    hard_fail = True
                if any("num_samples=" in x for x in validation_failures):
                    hard_fail = True

                if not run_summary_ok:
                    row["run_summary_errors"] = run_check.get("errors", [])
                    row["run_summary_warnings"] = run_check.get("warnings", [])

                if hard_fail:
                    row["status"] = "validation_failed"
                else:
                    row["status"] = "ok"

                with results_lock:
                    results.append(row)
                persist_results()
            except Exception as exc:  # noqa: BLE001
                failure_row = {
                    "job_idx": job.get("job_idx"),
                    "gpu_id": gpu_id,
                    "worker_slot": slot,
                    "group": job.get("group"),
                    "run_name": job.get("run_name"),
                    "run_dir": job.get("run_dir"),
                    "checkpoint_label": job.get("label"),
                    "model_path": job.get("model_path"),
                    "output_dir": job.get("output_dir"),
                    "log_path": job.get("log_path"),
                    "status": "worker_exception",
                    "error": str(exc),
                }
                with results_lock:
                    results.append(failure_row)
                persist_results()
            finally:
                q.task_done()

    threads: List[threading.Thread] = []
    for gpu in gpus:
        for slot in range(workers_per_gpu):
            th = threading.Thread(target=worker_loop, args=(gpu, slot), daemon=True)
            th.start()
            threads.append(th)

    try:
        for th in threads:
            th.join()
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupted, waiting for worker threads to stop...")
        for th in threads:
            th.join(timeout=1.0)
        raise

    with results_lock:
        final_results = sorted(results, key=lambda r: int(r.get("job_idx", 0)))
    write_json(results_path, final_results)

    ranked = sorted(
        [
            r
            for r in final_results
            if isinstance(r.get("exact_match"), (float, int))
        ],
        key=lambda r: float(r["exact_match"]),
        reverse=True,
    )
    write_json(ranked_path, ranked)

    best_by_group: Dict[str, Dict[str, Any]] = {}
    for row in ranked:
        if row.get("status") != "ok":
            continue
        group = str(row.get("group"))
        if group not in best_by_group:
            best_by_group[group] = {
                "run_name": row.get("run_name"),
                "run_dir": row.get("run_dir"),
                "checkpoint_label": row.get("checkpoint_label"),
                "model_path": row.get("model_path"),
                "output_dir": row.get("output_dir"),
                "exact_match": row.get("exact_match"),
                "f1": row.get("f1"),
                "num_samples": row.get("num_samples"),
            }
    write_json(best_by_group_path, best_by_group)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "status",
                "group",
                "run_name",
                "checkpoint_label",
                "exact_match",
                "f1",
                "num_samples",
                "model_path",
                "output_dir",
            ]
        )
        for row in final_results:
            writer.writerow(
                [
                    row.get("status", ""),
                    row.get("group", ""),
                    row.get("run_name", ""),
                    row.get("checkpoint_label", ""),
                    row.get("exact_match", ""),
                    row.get("f1", ""),
                    row.get("num_samples", ""),
                    row.get("model_path", ""),
                    row.get("output_dir", ""),
                ]
            )

    ok_count = sum(1 for r in final_results if r.get("status") == "ok")
    fail_count = len(final_results) - ok_count
    print(f"Done. Jobs total={len(final_results)} ok={ok_count} failed={fail_count}")
    print(f"Saved: {results_path}")
    print(f"Saved: {ranked_path}")
    print(f"Saved: {best_by_group_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
