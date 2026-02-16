#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from typing import List


DEFAULT_DATASETS = [
    "hotpotqa",
    "musique_ans",
    "musique_unans",
    "confiqa_qa",
    "confiqa_mr",
    "confiqa_mc",
    "finqa",
    "convfinqa",
    "tatdqa",
]


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    default_scc_eval_root = os.path.abspath(os.path.join(here, "..", "scc-eval"))
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate answer-only Qwen3 SFT checkpoint through standard scc-eval run.py "
            "(exactly same path as other instruct models)."
        )
    )
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    ap.add_argument("--model", type=str, required=True, help="Path to finetuned checkpoint")
    ap.add_argument("--scc_eval_root", type=str, default=default_scc_eval_root)
    ap.add_argument("--output_dir", type=str, default="", help="Defaults to <scc_eval_root>/results")
    ap.add_argument("--data_dir", type=str, default="/home/jovyan/datasets")
    ap.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)

    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--n_passages", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=4096)

    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scc_eval_root = os.path.abspath(args.scc_eval_root)
    run_py = os.path.join(scc_eval_root, "run.py")
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"Cannot find run.py at {run_py}")

    output_dir = args.output_dir.strip() or os.path.join(scc_eval_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    cmd: List[str] = [
        args.python_bin,
        run_py,
        "--model",
        args.model,
        "--model-type",
        "instruct",
        "--datasets",
        *args.datasets,
        "--output-dir",
        output_dir,
        "--data-dir",
        args.data_dir,
        "--seed",
        str(args.seed),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
    ]

    if args.n_samples is not None:
        cmd.extend(["--n-samples", str(args.n_samples)])
    if args.n_passages is not None:
        cmd.extend(["--n-passages", str(args.n_passages)])
    if args.shuffle:
        cmd.append("--shuffle")
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.top_p is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])

    print("Running scc-eval command:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, check=True, cwd=scc_eval_root)


if __name__ == "__main__":
    main()
