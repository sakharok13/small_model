#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>\s*(.*)$", re.IGNORECASE | re.DOTALL)
THINK_STOP_RE = re.compile(
    r"(</think>|<\|answer_start\|>|<answer_start>|\[answer_start\])",
    re.IGNORECASE,
)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")


def parse_args(default_model_name: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="outputs/reasoning_sft")
    ap.add_argument("--model_name", type=str, default=default_model_name)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--train_files", action="append", default=[], help="Parquet/JSONL file/dir/glob. Repeatable.")
    ap.add_argument("--eval_files", action="append", default=[], help="Optional parquet/JSONL file/dir/glob for eval.")
    ap.add_argument(
        "--prompt_versions",
        type=str,
        default="",
        help="Optional comma-separated prompt_version values to keep (e.g. v1 or v1,v2,v3).",
    )
    ap.add_argument("--eval_ratio", type=float, default=0.01, help="Used only when --eval_files is empty.")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_eval_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--thinking_source",
        type=str,
        choices=["raw_output_think", "column"],
        default="raw_output_think",
        help="Use <think>...</think> from raw_output (recommended) or existing thinking column.",
    )
    ap.add_argument(
        "--fallback_to_thinking_column",
        action="store_true",
        help="When --thinking_source=raw_output_think and <think> is missing, fallback to thinking column.",
    )

    ap.add_argument(
        "--filter_correct_only",
        action="store_true",
        default=True,
        help="Keep only rows where reference_answer == answer (after strip).",
    )
    ap.add_argument("--no_filter_correct_only", action="store_true")
    ap.add_argument("--require_thinking", action="store_true", default=True)
    ap.add_argument("--no_require_thinking", action="store_true")

    ap.add_argument("--max_seq_len", type=int, default=3072)
    ap.add_argument("--min_target_tokens", type=int, default=8)
    ap.add_argument("--filter_long", action="store_true", default=True)
    ap.add_argument("--no_filter_long", action="store_true")

    ap.add_argument("--context_start_token", type=str, default="[context_start]")
    ap.add_argument("--context_end_token", type=str, default="[context_end]")
    ap.add_argument("--query_start_token", type=str, default="[query_start]")
    ap.add_argument("--query_end_token", type=str, default="[query_end]")
    ap.add_argument("--thinking_start_token", type=str, default="[thinking_start]")
    ap.add_argument("--thinking_end_token", type=str, default="[thinking_end]")
    ap.add_argument("--answer_start_token", type=str, default="[answer_start]")
    ap.add_argument("--answer_end_token", type=str, default="[answer_end]")
    ap.add_argument(
        "--register_special_tokens",
        action="store_true",
        help="Add bracket tokens as new special tokens (new embeddings). "
             "Off by default: bracket strings tokenize into existing subwords.",
    )

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--report_to", type=str, default="none")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--logging_dir", type=str, default="")

    ap.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--ddp_backend", type=str, default="nccl")
    ap.add_argument("--ddp_find_unused_parameters", action="store_true")

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    return ap.parse_args()


def expand_paths(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        if not pat:
            continue
        if os.path.isdir(pat):
            out.extend(glob(os.path.join(pat, "*.parquet")))
            out.extend(glob(os.path.join(pat, "*.jsonl")))
            out.extend(glob(os.path.join(pat, "*.json")))
            continue
        matches = glob(pat)
        if matches:
            for m in matches:
                if os.path.isdir(m):
                    out.extend(glob(os.path.join(m, "*.parquet")))
                    out.extend(glob(os.path.join(m, "*.jsonl")))
                    out.extend(glob(os.path.join(m, "*.json")))
                elif m.endswith(".parquet"):
                    out.append(m)
                elif m.endswith(".jsonl") or m.endswith(".json"):
                    out.append(m)
            continue
        if os.path.isfile(pat) and (pat.endswith(".parquet") or pat.endswith(".jsonl") or pat.endswith(".json")):
            out.append(pat)
    return sorted(set(out))


def load_dataset_files(paths: Sequence[str], name: str) -> Dataset:
    if not paths:
        raise ValueError(f"No files provided for {name}")
    exts = {os.path.splitext(p)[1].lower() for p in paths}
    if exts <= {".parquet"}:
        ds = load_dataset("parquet", data_files=list(paths), split="train")
    elif exts <= {".json", ".jsonl"}:
        ds = load_dataset("json", data_files=list(paths), split="train")
    else:
        raise ValueError(f"{name} has mixed/unsupported file extensions: {sorted(exts)}")
    required = {"query", "context", "answer"}
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return ds


def str_clean(x: Any) -> str:
    return str(x if x is not None else "").strip()


def clean_trace_text(text: str) -> str:
    text = str(text or "")
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.replace("<s>", " ").replace("</s>", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_think_from_raw_output(raw_output: Any) -> str:
    raw = str(raw_output or "")
    m = THINK_RE.search(raw)
    if m:
        return clean_trace_text(m.group(1))
    m_open = THINK_OPEN_RE.search(raw)
    if not m_open:
        return ""
    tail = m_open.group(1)
    stop = THINK_STOP_RE.search(tail)
    if stop:
        tail = tail[:stop.start()]
    return clean_trace_text(tail)


def prepare_reasoning_fields(
    ds: Dataset,
    args: argparse.Namespace,
    split_name: str,
) -> Tuple[Dataset, Dict[str, Any]]:
    has_raw_output = "raw_output" in ds.column_names
    has_thinking_col = "thinking" in ds.column_names
    if args.thinking_source == "raw_output_think" and (not has_raw_output):
        raise ValueError(
            f"{split_name} is missing 'raw_output' but --thinking_source=raw_output_think was requested."
        )
    if args.fallback_to_thinking_column and (not has_thinking_col):
        raise ValueError(
            f"{split_name} is missing 'thinking' but --fallback_to_thinking_column was requested."
        )

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        answer = str_clean(ex.get("answer"))
        raw_think = extract_think_from_raw_output(ex.get("raw_output"))
        thinking_col = clean_trace_text(str_clean(ex.get("thinking")))
        thinking = ""
        if args.thinking_source == "raw_output_think":
            thinking = raw_think
            if (not thinking) and args.fallback_to_thinking_column:
                thinking = thinking_col
        else:
            thinking = thinking_col
        return {
            "thinking": thinking,
            "answer": answer,
            "_has_raw_think": int(bool(raw_think)),
            "_has_final_thinking": int(bool(thinking)),
        }

    ds = ds.map(_map, desc=f"Prepare reasoning fields ({split_name})")
    has_raw_think_count = int(sum(ds["_has_raw_think"])) if len(ds) > 0 else 0
    has_final_think_count = int(sum(ds["_has_final_thinking"])) if len(ds) > 0 else 0
    stats = {
        "split_name": split_name,
        "thinking_source": args.thinking_source,
        "fallback_to_thinking_column": bool(args.fallback_to_thinking_column),
        "rows": len(ds),
        "rows_with_raw_think": has_raw_think_count,
        "rows_with_final_thinking": has_final_think_count,
    }
    ds = ds.remove_columns([c for c in ["_has_raw_think", "_has_final_thinking"] if c in ds.column_names])
    return ds, stats


def filter_quality(ds: Dataset, require_thinking: bool, filter_correct_only: bool) -> Tuple[Dataset, Dict[str, int]]:
    before = len(ds)

    def keep_non_empty(ex: Dict[str, Any]) -> bool:
        if not str_clean(ex.get("query")):
            return False
        if not str_clean(ex.get("context")):
            return False
        if not str_clean(ex.get("answer")):
            return False
        if require_thinking and not str_clean(ex.get("thinking")):
            return False
        return True

    ds = ds.filter(keep_non_empty, desc="Filter non-empty fields")
    after_non_empty = len(ds)

    if filter_correct_only:
        if "reference_answer" not in ds.column_names:
            raise ValueError("reference_answer column is required when --filter_correct_only is enabled.")
        ds = ds.filter(
            lambda ex: str_clean(ex.get("reference_answer")) == str_clean(ex.get("answer")),
            desc="Filter reference_answer == answer",
        )
    after_correct = len(ds)

    return ds, {
        "before": before,
        "after_non_empty": after_non_empty,
        "after_correct": after_correct,
    }


def maybe_cap(ds: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0 or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def parse_prompt_versions(spec: str) -> Set[str]:
    return {x.strip() for x in str(spec or "").split(",") if x.strip()}


def filter_prompt_versions(ds: Dataset, allowed: Set[str], split_name: str) -> Tuple[Dataset, Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "split_name": split_name,
        "allowed": sorted(allowed),
        "before": len(ds),
        "after": len(ds),
    }
    if not allowed:
        stats["enabled"] = False
        return ds, stats
    stats["enabled"] = True
    if "prompt_version" not in ds.column_names:
        raise ValueError(
            f"{split_name} is missing required column 'prompt_version' but --prompt_versions was set."
        )

    ds = ds.filter(
        lambda ex: str_clean(ex.get("prompt_version")) in allowed,
        desc=f"Filter prompt_version in {sorted(allowed)} ({split_name})",
    )
    stats["after"] = len(ds)
    return ds, stats


def build_prefix(ex: Dict[str, Any], args: argparse.Namespace) -> str:
    return (
        f"{args.context_start_token}{str_clean(ex['context'])}{args.context_end_token}"
        f"{args.query_start_token}{str_clean(ex['query'])}{args.query_end_token}"
    )


def build_target(ex: Dict[str, Any], args: argparse.Namespace) -> str:
    return (
        f"{args.thinking_start_token}{str_clean(ex['thinking'])}{args.thinking_end_token}"
        f"{args.answer_start_token}{str_clean(ex['answer'])}{args.answer_end_token}"
    )


def build_full_text(ex: Dict[str, Any], args: argparse.Namespace) -> str:
    return f"{build_prefix(ex, args)}{build_target(ex, args)}"


class ThinkingStartCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, args: argparse.Namespace) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prefixes = [build_prefix(ex, self.args) for ex in batch]
        texts = [build_full_text(ex, self.args) for ex in batch]

        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prefix_enc = self.tokenizer(
            prefixes,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        prefix_lens = [len(ids) for ids in prefix_enc["input_ids"]]

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        for i, pref_len in enumerate(prefix_lens):
            if pref_len >= labels.shape[1]:
                labels[i, :] = -100
            else:
                labels[i, :pref_len] = -100
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def filter_long_samples(ds: Dataset, tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dataset:
    def keep(ex: Dict[str, Any]) -> bool:
        prefix_ids = tokenizer(
            build_prefix(ex, args),
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_seq_len,
        )["input_ids"]
        return len(prefix_ids) < (args.max_seq_len - max(1, args.min_target_tokens))

    return ds.filter(keep, desc="Filter long prompts")


def register_special_tokens(tokenizer: AutoTokenizer, args: argparse.Namespace) -> int:
    special = [
        args.context_start_token,
        args.context_end_token,
        args.query_start_token,
        args.query_end_token,
        args.thinking_start_token,
        args.thinking_end_token,
        args.answer_start_token,
        args.answer_end_token,
    ]
    dedup = list(dict.fromkeys(special))
    return tokenizer.add_special_tokens({"additional_special_tokens": dedup})


def build_model_kwargs(args: argparse.Namespace, dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if dtype is None:
        return kwargs
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in sig.parameters:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    return kwargs


def build_training_args(args: argparse.Namespace, eval_enabled: bool) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    eval_strategy_value = args.evaluation_strategy if eval_enabled else "no"

    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "eval_steps": args.eval_steps,
    }
    if is_distributed:
        kwargs["ddp_backend"] = args.ddp_backend
        kwargs["ddp_find_unused_parameters"] = args.ddp_find_unused_parameters
    if args.run_name:
        kwargs["run_name"] = args.run_name
    if args.logging_dir:
        kwargs["logging_dir"] = args.logging_dir
    if "evaluation_strategy" in allowed:
        kwargs["evaluation_strategy"] = eval_strategy_value
    if "eval_strategy" in allowed:
        kwargs["eval_strategy"] = eval_strategy_value

    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**kwargs)


def main(default_model_name: str = "Qwen/Qwen3-0.6B") -> None:
    args = parse_args(default_model_name=default_model_name)
    if args.no_filter_long:
        args.filter_long = False
    if args.no_filter_correct_only:
        args.filter_correct_only = False
    if args.no_require_thinking:
        args.require_thinking = False

    train_paths = expand_paths(args.train_files)
    if not train_paths:
        raise ValueError("No train parquet files found. Pass --train_files file/dir/glob.")
    allowed_prompt_versions = parse_prompt_versions(args.prompt_versions)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.register_special_tokens:
        added = register_special_tokens(tokenizer, args)
        print(f"Added special tokens: {added}")

    train_ds = load_dataset_files(train_paths, name="train")
    train_ds, train_pv_stats = filter_prompt_versions(
        train_ds,
        allowed=allowed_prompt_versions,
        split_name="train",
    )
    train_ds, train_reasoning_stats = prepare_reasoning_fields(
        train_ds,
        args=args,
        split_name="train",
    )
    train_ds, train_stats = filter_quality(
        train_ds,
        require_thinking=args.require_thinking,
        filter_correct_only=args.filter_correct_only,
    )

    eval_ds: Optional[Dataset] = None
    eval_stats: Dict[str, int] = {}
    eval_paths = expand_paths(args.eval_files)
    if eval_paths:
        eval_ds = load_dataset_files(eval_paths, name="eval")
        eval_ds, eval_pv_stats = filter_prompt_versions(
            eval_ds,
            allowed=allowed_prompt_versions,
            split_name="eval",
        )
        eval_ds, eval_reasoning_stats = prepare_reasoning_fields(
            eval_ds,
            args=args,
            split_name="eval",
        )
        eval_ds, eval_stats = filter_quality(
            eval_ds,
            require_thinking=args.require_thinking,
            filter_correct_only=args.filter_correct_only,
        )
        eval_stats["prompt_version_filter"] = eval_pv_stats
        eval_stats["reasoning_prepare"] = eval_reasoning_stats
    elif args.eval_ratio > 0.0 and len(train_ds) > 1:
        shuffled = train_ds.shuffle(seed=args.seed)
        eval_n = max(1, int(len(shuffled) * args.eval_ratio))
        eval_n = min(eval_n, len(shuffled) - 1)
        eval_ds = shuffled.select(range(eval_n))
        train_ds = shuffled.select(range(eval_n, len(shuffled)))
        eval_stats = {"from_train_split": eval_n}

    if args.filter_long:
        before_train = len(train_ds)
        train_ds = filter_long_samples(train_ds, tokenizer, args)
        print(f"Filtered train samples by prompt length: {before_train} -> {len(train_ds)}")
        if eval_ds is not None:
            before_eval = len(eval_ds)
            eval_ds = filter_long_samples(eval_ds, tokenizer, args)
            print(f"Filtered eval samples by prompt length: {before_eval} -> {len(eval_ds)}")

    train_ds = maybe_cap(train_ds, args.max_train_samples, seed=args.seed)
    if eval_ds is not None:
        eval_ds = maybe_cap(eval_ds, args.max_eval_samples, seed=args.seed)

    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty after filtering.")
    if eval_ds is not None and len(eval_ds) == 0:
        print("Eval dataset became empty; disabling eval.")
        eval_ds = None

    dtype: Optional[torch.dtype] = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **build_model_kwargs(args=args, dtype=dtype),
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if args.register_special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    collator = ThinkingStartCollator(tokenizer=tokenizer, max_length=args.max_seq_len, args=args)
    train_args = build_training_args(args=args, eval_enabled=eval_ds is not None)

    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model_name": args.model_name,
        "train_files_count": len(train_paths),
        "eval_files_count": len(eval_paths),
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "prompt_versions_filter": sorted(allowed_prompt_versions),
        "train_prompt_version_filter": train_pv_stats,
        "train_reasoning_prepare": train_reasoning_stats,
        "train_filter_stats": train_stats,
        "eval_filter_stats": eval_stats,
        "loss_masking": (
            "labels are computed only for tokens after prefix; "
            f"supervised segment starts at {args.thinking_start_token}"
        ),
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "sft_reasoning_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
