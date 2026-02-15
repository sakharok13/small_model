#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import json
import os
import re
import subprocess
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, List, Sequence, Tuple


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s or "").split(",") if x.strip()]


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s or "").split(",") if x.strip()]


def append_repeat_arg(cmd: List[str], flag: str, values: Sequence[str]) -> None:
    for v in values:
        cmd.extend([flag, v])


def format_num(v: Any) -> str:
    s = str(v)
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return re.sub(r"[^a-zA-Z0-9_]", "", s)


def checkpoint_step(path: str) -> int:
    base = os.path.basename(path.rstrip("/"))
    if base.startswith("checkpoint-"):
        try:
            return int(base.split("-", 1)[1])
        except Exception:
            return -1
    return -1


def collect_eval_targets(run_dir: str, mode: str) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    if mode == "all":
        ckpts = sorted(
            [p for p in glob(os.path.join(run_dir, "checkpoint-*")) if os.path.isdir(p)],
            key=lambda p: checkpoint_step(p),
        )
        for p in ckpts:
            targets.append((os.path.basename(p), p))
    targets.append(("final", run_dir))
    return targets


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class RunConfig:
    group: str
    learning_rate: float
    batch_size: int
    grad_accum: int
    epochs: float
    warmup_ratio: float
    weight_decay: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train reasoning SFT separately for prompt_version groups (v1/v2/v3), "
            "evaluate each run on HotpotQA 1k samples, and rank best checkpoints by EM."
        )
    )
    ap.add_argument("--python_bin", type=str, default="python3")
    ap.add_argument("--train_script", type=str, default="finetune_reasoning_traces_sft.py")
    ap.add_argument("--eval_script", type=str, default="benchmark_hotpotqa_reasoning.py")
    ap.add_argument("--base_output_dir", type=str, required=True)

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--train_files", action="append", default=[], help="Parquet files/dirs/globs. Repeatable.")
    ap.add_argument("--eval_files", action="append", default=[], help="Optional parquet eval files for SFT run.")
    ap.add_argument("--groups", type=str, default="v1,v2,v3")

    ap.add_argument("--learning_rates", type=str, default="1e-5,2e-5,5e-5")
    ap.add_argument("--batch_sizes", type=str, default="1,2")
    ap.add_argument("--grad_accums", type=str, default="8,16")
    ap.add_argument("--epochs", type=str, default="1.0")
    ap.add_argument("--warmup_ratios", type=str, default="0.03")
    ap.add_argument("--weight_decays", type=str, default="0.0")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=3072)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_eval_samples", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--evaluation_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--eval_ratio", type=float, default=0.0)
    ap.add_argument("--report_to", type=str, default="none")
    ap.add_argument("--no_filter_correct_only", action="store_true")
    ap.add_argument("--no_require_thinking", action="store_true")
    ap.add_argument("--no_filter_long", action="store_true")

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    ap.add_argument("--eval_checkpoints", type=str, default="all", choices=["all", "final"])
    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--eval_max_samples", type=int, default=1000)
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

    ap.add_argument("--max_runs", type=int, default=0, help="0 means no cap")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_completed", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.base_output_dir, exist_ok=True)

    groups = parse_csv_strings(args.groups)
    if not groups:
        raise ValueError("--groups is empty")
    if not args.train_files:
        raise ValueError("Provide --train_files at least once")

    learning_rates = parse_csv_floats(args.learning_rates)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    grad_accums = parse_csv_ints(args.grad_accums)
    epochs = parse_csv_floats(args.epochs)
    warmup_ratios = parse_csv_floats(args.warmup_ratios)
    weight_decays = parse_csv_floats(args.weight_decays)

    grid: List[RunConfig] = []
    for g, lr, bs, ga, ep, wr, wd in itertools.product(
        groups, learning_rates, batch_sizes, grad_accums, epochs, warmup_ratios, weight_decays
    ):
        grid.append(
            RunConfig(
                group=g,
                learning_rate=lr,
                batch_size=bs,
                grad_accum=ga,
                epochs=ep,
                warmup_ratio=wr,
                weight_decay=wd,
            )
        )

    if args.max_runs > 0:
        grid = grid[: args.max_runs]
    print(f"Planned runs: {len(grid)}")

    results: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(grid):
        run_name = (
            f"{cfg.group}_run{idx:03d}_lr{format_num(cfg.learning_rate)}_bs{cfg.batch_size}"
            f"_ga{cfg.grad_accum}_ep{format_num(cfg.epochs)}_wr{format_num(cfg.warmup_ratio)}"
            f"_wd{format_num(cfg.weight_decay)}"
        )
        run_dir = os.path.join(args.base_output_dir, run_name)
        done_marker = os.path.join(run_dir, "best_checkpoint.json")

        if args.skip_completed and os.path.exists(done_marker):
            print(f"[{idx + 1}/{len(grid)}] skip completed: {run_name}")
            row = {
                "run_name": run_name,
                "run_dir": run_dir,
                "config": cfg.__dict__,
                "status": "skipped_completed",
            }
            try:
                row["best_checkpoint"] = read_json(done_marker)
            except Exception:
                pass
            results.append(row)
            continue

        os.makedirs(run_dir, exist_ok=True)

        train_cmd: List[str] = [
            args.python_bin,
            args.train_script,
            "--output_dir",
            run_dir,
            "--model_name",
            args.model_name,
            "--prompt_versions",
            cfg.group,
            "--max_seq_len",
            str(args.max_seq_len),
            "--max_train_samples",
            str(args.max_train_samples),
            "--max_eval_samples",
            str(args.max_eval_samples),
            "--seed",
            str(args.seed),
            "--learning_rate",
            str(cfg.learning_rate),
            "--per_device_train_batch_size",
            str(cfg.batch_size),
            "--gradient_accumulation_steps",
            str(cfg.grad_accum),
            "--num_train_epochs",
            str(cfg.epochs),
            "--warmup_ratio",
            str(cfg.warmup_ratio),
            "--weight_decay",
            str(cfg.weight_decay),
            "--save_steps",
            str(args.save_steps),
            "--save_total_limit",
            str(args.save_total_limit),
            "--logging_steps",
            str(args.logging_steps),
            "--evaluation_strategy",
            args.evaluation_strategy,
            "--eval_steps",
            str(args.eval_steps),
            "--eval_ratio",
            str(args.eval_ratio),
            "--report_to",
            args.report_to,
            "--run_name",
            run_name,
        ]
        if args.trust_remote_code:
            train_cmd.append("--trust_remote_code")
        if args.local_files_only:
            train_cmd.append("--local_files_only")
        if args.bf16:
            train_cmd.append("--bf16")
        if args.fp16:
            train_cmd.append("--fp16")
        if args.gradient_checkpointing:
            train_cmd.append("--gradient_checkpointing")
        if args.no_filter_correct_only:
            train_cmd.append("--no_filter_correct_only")
        if args.no_require_thinking:
            train_cmd.append("--no_require_thinking")
        if args.no_filter_long:
            train_cmd.append("--no_filter_long")
        append_repeat_arg(train_cmd, "--train_files", args.train_files)
        append_repeat_arg(train_cmd, "--eval_files", args.eval_files)

        print(f"[{idx + 1}/{len(grid)}] train: {run_name}")
        print("Train command:", " ".join(train_cmd))

        row: Dict[str, Any] = {
            "run_name": run_name,
            "run_dir": run_dir,
            "config": cfg.__dict__,
            "train_command": train_cmd,
        }

        if args.dry_run:
            row["status"] = "dry_run"
            results.append(row)
            continue

        train_proc = subprocess.run(train_cmd, check=False)
        row["train_return_code"] = train_proc.returncode
        if train_proc.returncode != 0:
            row["status"] = "train_failed"
            results.append(row)
            with open(os.path.join(args.base_output_dir, "sweep_results.json"), "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            continue

        eval_targets = collect_eval_targets(run_dir, mode=args.eval_checkpoints)
        eval_rows: List[Dict[str, Any]] = []
        best_ckpt: Dict[str, Any] = {}
        best_em = -1.0

        for label, model_path in eval_targets:
            eval_out = os.path.join(run_dir, "hotpot_eval", label)
            os.makedirs(eval_out, exist_ok=True)

            eval_cmd: List[str] = [
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
                eval_out,
            ]
            if args.trust_remote_code:
                eval_cmd.append("--trust_remote_code")
            if args.local_files_only:
                eval_cmd.append("--local_files_only")

            print(f"Eval checkpoint: {label}")
            print("Eval command:", " ".join(eval_cmd))
            eval_proc = subprocess.run(eval_cmd, check=False)

            eval_row: Dict[str, Any] = {
                "label": label,
                "model_path": model_path,
                "output_dir": eval_out,
                "return_code": eval_proc.returncode,
            }
            metrics_path = os.path.join(eval_out, "metrics.json")
            if eval_proc.returncode == 0 and os.path.exists(metrics_path):
                metrics = read_json(metrics_path)
                em = float(metrics.get("exact_match", 0.0))
                f1 = float(metrics.get("f1", 0.0))
                eval_row["exact_match"] = em
                eval_row["f1"] = f1
                eval_row["num_samples"] = int(metrics.get("num_samples", 0))
                if em > best_em:
                    best_em = em
                    best_ckpt = {
                        "label": label,
                        "model_path": model_path,
                        "output_dir": eval_out,
                        "exact_match": em,
                        "f1": f1,
                        "num_samples": int(metrics.get("num_samples", 0)),
                    }
            eval_rows.append(eval_row)

        row["eval_results"] = eval_rows
        if best_ckpt:
            row["best_checkpoint"] = best_ckpt
            row["status"] = "ok"
            with open(done_marker, "w", encoding="utf-8") as fh:
                json.dump(best_ckpt, fh, indent=2, ensure_ascii=False)
        else:
            row["status"] = "eval_failed"
        results.append(row)

        with open(os.path.join(args.base_output_dir, "sweep_results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)

    ranked = [
        r
        for r in results
        if r.get("status") == "ok"
        and isinstance(r.get("best_checkpoint", {}).get("exact_match"), (int, float))
    ]
    ranked = sorted(ranked, key=lambda r: float(r["best_checkpoint"]["exact_match"]), reverse=True)

    best_by_group: Dict[str, Dict[str, Any]] = {}
    for row in ranked:
        group = str(row.get("config", {}).get("group"))
        if group not in best_by_group:
            best_by_group[group] = {
                "run_name": row["run_name"],
                "run_dir": row["run_dir"],
                "best_checkpoint": row["best_checkpoint"],
                "config": row["config"],
            }

    ranked_out = os.path.join(args.base_output_dir, "sweep_ranked.json")
    with open(ranked_out, "w", encoding="utf-8") as fh:
        json.dump(ranked, fh, indent=2, ensure_ascii=False)
    group_out = os.path.join(args.base_output_dir, "best_by_group.json")
    with open(group_out, "w", encoding="utf-8") as fh:
        json.dump(best_by_group, fh, indent=2, ensure_ascii=False)

    print(f"Saved results: {os.path.join(args.base_output_dir, 'sweep_results.json')}")
    print(f"Saved ranking: {ranked_out}")
    print(f"Saved per-group best: {group_out}")
    if ranked:
        print("Best overall:")
        print(json.dumps(ranked[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
