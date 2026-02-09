#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_ratio_sets(s: str) -> List[str]:
    parts = [x.strip() for x in s.split(";") if x.strip()]
    return parts or ["correct:0.70,refusal:0.15,negative:0.15"]


def append_repeat_arg(cmd: List[str], flag: str, values: Sequence[str]) -> None:
    for v in values:
        cmd.extend([flag, v])


@dataclass
class RunConfig:
    learning_rate: float
    batch_size: int
    grad_accum: int
    epochs: float
    mix_strategy: str
    mix_ratios: str
    target_train_samples: int


def extract_losses(metrics: Dict[str, Any], group: str) -> Dict[str, float]:
    if group not in metrics:
        return {}
    out: Dict[str, float] = {}
    block = metrics[group]
    for k, v in block.items():
        if isinstance(v, (int, float)) and k.endswith("_loss"):
            out[k] = float(v)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run SFT hyperparameter and mixture sweep with finetune_qwen3_base.py "
            "and aggregate eval losses by group."
        )
    )
    ap.add_argument("--python_bin", type=str, default="python3")
    ap.add_argument("--train_script", type=str, default="finetune_qwen3_base.py")
    ap.add_argument("--base_output_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)

    # Dataset args (repeatable; passed through to train script).
    ap.add_argument("--train_files", action="append", default=[])
    ap.add_argument("--eval_files", action="append", default=[])
    ap.add_argument("--train_correct_files", action="append", default=[])
    ap.add_argument("--train_refusal_files", action="append", default=[])
    ap.add_argument("--train_negative_files", action="append", default=[])
    ap.add_argument("--eval_correct_files", action="append", default=[])
    ap.add_argument("--eval_refusal_files", action="append", default=[])
    ap.add_argument("--eval_negative_files", action="append", default=[])

    ap.add_argument("--learning_rates", type=str, default="1e-5,2e-5,5e-5")
    ap.add_argument("--batch_sizes", type=str, default="2,4")
    ap.add_argument("--grad_accums", type=str, default="4,8")
    ap.add_argument("--epochs", type=str, default="1.0,2.0")
    ap.add_argument("--mix_strategies", type=str, default="natural,balanced,custom")
    ap.add_argument(
        "--mix_ratio_sets",
        type=str,
        default="correct:0.70,refusal:0.15,negative:0.15;correct:0.50,refusal:0.25,negative:0.25",
        help="Semicolon-separated ratio strings for custom strategy",
    )
    ap.add_argument("--target_train_sizes", type=str, default="0", help="Comma-separated target train sample counts")
    ap.add_argument("--allow_oversample", action="store_true")

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--num_workers_limit", type=int, default=0, help="Reserved; currently unused")

    ap.add_argument("--max_runs", type=int, default=0, help="Stop after N runs (0 = no limit)")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.base_output_dir, exist_ok=True)

    learning_rates = parse_csv_floats(args.learning_rates)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    grad_accums = parse_csv_ints(args.grad_accums)
    epochs = parse_csv_floats(args.epochs)
    mix_strategies = parse_csv_strings(args.mix_strategies)
    mix_ratio_sets = parse_ratio_sets(args.mix_ratio_sets)
    target_train_sizes = parse_csv_ints(args.target_train_sizes)

    run_grid: List[RunConfig] = []
    for lr, bs, ga, ep, mix, tts in itertools.product(
        learning_rates, batch_sizes, grad_accums, epochs, mix_strategies, target_train_sizes
    ):
        if mix == "custom":
            for ratios in mix_ratio_sets:
                run_grid.append(
                    RunConfig(
                        learning_rate=lr,
                        batch_size=bs,
                        grad_accum=ga,
                        epochs=ep,
                        mix_strategy=mix,
                        mix_ratios=ratios,
                        target_train_samples=tts,
                    )
                )
        else:
            run_grid.append(
                RunConfig(
                    learning_rate=lr,
                    batch_size=bs,
                    grad_accum=ga,
                    epochs=ep,
                    mix_strategy=mix,
                    mix_ratios="",
                    target_train_samples=tts,
                )
            )

    if args.max_runs > 0:
        run_grid = run_grid[: args.max_runs]

    print(f"Planned runs: {len(run_grid)}")
    results: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(run_grid):
        run_name = (
            f"run_{idx:03d}_lr{cfg.learning_rate}_bs{cfg.batch_size}_ga{cfg.grad_accum}"
            f"_ep{cfg.epochs}_mix{cfg.mix_strategy}_tts{cfg.target_train_samples}"
        )
        if cfg.mix_strategy == "custom":
            safe_ratio = cfg.mix_ratios.replace(",", "_").replace(":", "-")
            run_name = f"{run_name}_{safe_ratio}"
        run_dir = os.path.join(args.base_output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        cmd: List[str] = [
            args.python_bin,
            args.train_script,
            "--output_dir",
            run_dir,
            "--model_name",
            args.model_name,
            "--max_seq_len",
            str(args.max_seq_len),
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
            "--mix_strategy",
            cfg.mix_strategy,
        ]

        if cfg.target_train_samples > 0:
            cmd.extend(["--target_train_samples", str(cfg.target_train_samples)])
        if cfg.mix_strategy == "custom" and cfg.mix_ratios:
            cmd.extend(["--mix_ratios", cfg.mix_ratios])
        if args.allow_oversample:
            cmd.append("--allow_oversample")
        if args.bf16:
            cmd.append("--bf16")
        if args.fp16:
            cmd.append("--fp16")
        if args.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if args.use_lora:
            cmd.append("--use_lora")

        append_repeat_arg(cmd, "--train_files", args.train_files)
        append_repeat_arg(cmd, "--eval_files", args.eval_files)
        append_repeat_arg(cmd, "--train_correct_files", args.train_correct_files)
        append_repeat_arg(cmd, "--train_refusal_files", args.train_refusal_files)
        append_repeat_arg(cmd, "--train_negative_files", args.train_negative_files)
        append_repeat_arg(cmd, "--eval_correct_files", args.eval_correct_files)
        append_repeat_arg(cmd, "--eval_refusal_files", args.eval_refusal_files)
        append_repeat_arg(cmd, "--eval_negative_files", args.eval_negative_files)

        print(f"[{idx + 1}/{len(run_grid)}] {run_name}")
        print("Command:", " ".join(cmd))

        result_row: Dict[str, Any] = {
            "run_name": run_name,
            "run_dir": run_dir,
            "config": cfg.__dict__,
        }

        if args.dry_run:
            result_row["status"] = "dry_run"
            results.append(result_row)
            continue

        completed = subprocess.run(cmd, check=False)
        result_row["return_code"] = completed.returncode
        result_row["status"] = "ok" if completed.returncode == 0 else "failed"

        metrics_path = os.path.join(run_dir, "all_eval_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)
            result_row["metrics"] = metrics

            losses: Dict[str, float] = {}
            for g in ("all", "correct", "refusal", "negative"):
                losses.update(extract_losses(metrics, g))
            result_row["losses"] = losses

            group_losses = []
            for g in ("correct", "refusal", "negative"):
                key = f"eval_{g}_loss"
                if key in losses:
                    group_losses.append(losses[key])
            if group_losses:
                result_row["mean_group_loss"] = sum(group_losses) / len(group_losses)
        else:
            result_row["metrics"] = {}

        results.append(result_row)

        results_path = os.path.join(args.base_output_dir, "sweep_results.json")
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)

    # Build ranking by mean group loss when available.
    ranked = [
        r for r in results if r.get("status") == "ok" and isinstance(r.get("mean_group_loss"), (int, float))
    ]
    ranked = sorted(ranked, key=lambda r: r["mean_group_loss"])
    rank_path = os.path.join(args.base_output_dir, "sweep_ranked.json")
    with open(rank_path, "w", encoding="utf-8") as fh:
        json.dump(ranked, fh, indent=2, ensure_ascii=False)

    print(f"Saved sweep results: {os.path.join(args.base_output_dir, 'sweep_results.json')}")
    print(f"Saved ranked runs: {rank_path}")
    if ranked:
        best = ranked[0]
        print("Best run:")
        print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
