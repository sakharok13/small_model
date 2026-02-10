#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
import random
from glob import glob
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

GROUPS = ("correct", "refusal", "negative")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="outputs/qwen3_small_idk")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer from local HF cache only")
    ap.add_argument("--max_seq_len", type=int, default=2048)

    # Backward compatibility path (JSONL).
    ap.add_argument("--data_jsonl", type=str, default="", help="Legacy: JSONL dataset path")

    # Unified parquet inputs (can be file, directory, or glob). Repeatable.
    ap.add_argument("--train_files", action="append", default=[], help="Train parquet file/dir/glob")
    ap.add_argument("--eval_files", action="append", default=[], help="Eval parquet file/dir/glob")

    # Grouped inputs for class-aware mixing/evaluation.
    ap.add_argument("--train_correct_files", action="append", default=[], help="Train correct parquet file/dir/glob")
    ap.add_argument("--train_refusal_files", action="append", default=[], help="Train refusal parquet file/dir/glob")
    ap.add_argument("--train_negative_files", action="append", default=[], help="Train negative parquet file/dir/glob")
    ap.add_argument("--eval_correct_files", action="append", default=[], help="Eval correct parquet file/dir/glob")
    ap.add_argument("--eval_refusal_files", action="append", default=[], help="Eval refusal parquet file/dir/glob")
    ap.add_argument("--eval_negative_files", action="append", default=[], help="Eval negative parquet file/dir/glob")

    # Mixing options (used when grouped train files are provided).
    ap.add_argument("--mix_strategy", type=str, choices=["natural", "balanced", "custom"], default="natural")
    ap.add_argument(
        "--mix_ratios",
        type=str,
        default="correct:0.70,refusal:0.15,negative:0.15",
        help="Only for custom mix_strategy; format: correct:0.7,refusal:0.15,negative:0.15",
    )
    ap.add_argument(
        "--target_train_samples",
        type=int,
        default=0,
        help="Target mixed-train size. 0 means automatic size from selected strategy.",
    )
    ap.add_argument("--max_train_samples", type=int, default=0, help="Optional hard cap after mixing")
    ap.add_argument("--allow_oversample", action="store_true", help="Allow sampling with replacement for group mixing")

    # Optional HotpotQA augmentation.
    ap.add_argument("--add_hotpot_train", action="store_true", help="Append HotpotQA train examples to the training set")
    ap.add_argument("--hotpot_dataset_name", type=str, default="hotpot_qa")
    ap.add_argument("--hotpot_dataset_config", type=str, default="distractor")
    ap.add_argument("--hotpot_train_split", type=str, default="train")
    ap.add_argument(
        "--hotpot_context_mode",
        type=str,
        choices=["all", "gold_titles", "supporting"],
        default="all",
        help=(
            "all: all context docs/sentences; "
            "gold_titles: all sentences from supporting-fact titles; "
            "supporting: only supporting-fact sentences."
        ),
    )
    ap.add_argument(
        "--hotpot_max_context_chars",
        type=int,
        default=6000,
        help="Hard cap on HotpotQA context chars (0 disables cap).",
    )

    # Special token setup.
    ap.add_argument("--context_start_token", type=str, default="[context_start]")
    ap.add_argument("--context_end_token", type=str, default="[context_end]")
    ap.add_argument("--query_start_token", type=str, default="[query_start]")
    ap.add_argument("--query_end_token", type=str, default="[query_end]")
    ap.add_argument("--answer_start_token", type=str, default="[answer_start]")
    ap.add_argument("--answer_end_token", type=str, default="[answer_end]")
    ap.add_argument("--no_register_special_tokens", action="store_true")

    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--logging_dir", type=str, default="")
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_to", type=str, default="none")
    ap.add_argument("--ddp_backend", type=str, default="nccl")
    ap.add_argument("--ddp_find_unused_parameters", action="store_true")
    ap.add_argument("--evaluation_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    ap.add_argument("--eval_steps", type=int, default=500)

    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 if supported")
    ap.add_argument("--fp16", action="store_true", help="Use float16")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA",
    )

    ap.add_argument("--filter_long", action="store_true", default=True, help="Drop samples with prompt_len >= max_seq_len")
    ap.add_argument("--no_filter_long", action="store_true", help="Disable long-sample filtering")

    return ap.parse_args()


def expand_paths(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        if not pat:
            continue
        if os.path.isdir(pat):
            out.extend(glob(os.path.join(pat, "*.parquet")))
            continue
        matches = glob(pat)
        if matches:
            out.extend(matches)
            continue
        if os.path.isfile(pat):
            out.append(pat)
    return sorted(set(out))


def parse_mix_ratios(spec: str) -> Dict[str, float]:
    ratios = {k: 0.0 for k in GROUPS}
    if not spec.strip():
        raise ValueError("mix_ratios cannot be empty for mix_strategy=custom")
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid mix ratio term: {chunk}")
        key, val = chunk.split(":", 1)
        key = key.strip()
        if key not in ratios:
            raise ValueError(f"Unknown class in mix ratios: {key}")
        ratios[key] = float(val)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("mix_ratios must sum to > 0")
    return {k: v / total for k, v in ratios.items()}


def build_prompt(query: str, context: str, args: argparse.Namespace) -> str:
    return (
        f"{args.context_start_token}{context}{args.context_end_token}"
        f"{args.query_start_token}{query}{args.query_end_token}"
        f"{args.answer_start_token}"
    )


def build_full_text(query: str, context: str, answer: str, args: argparse.Namespace) -> str:
    return f"{build_prompt(query=query, context=context, args=args)}{answer}{args.answer_end_token}"


def ensure_columns(ds: Dataset, name: str) -> None:
    required = {"query", "context", "answer"}
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_parquet_dataset(paths: Sequence[str], name: str) -> Dataset:
    if not paths:
        raise ValueError(f"No files provided for {name}")
    ds = load_dataset("parquet", data_files=list(paths), split="train")
    ensure_columns(ds, name=name)
    return ds


def with_example_type(ds: Dataset, example_type: str) -> Dataset:
    if "example_type" in ds.column_names:
        return ds
    values = [example_type] * len(ds)
    return ds.add_column("example_type", values)


def sample_dataset(ds: Dataset, n: int, rng: random.Random, oversample: bool) -> Dataset:
    if n <= 0 or len(ds) == 0:
        return ds.select([])
    if oversample:
        idx = [rng.randrange(len(ds)) for _ in range(n)]
    else:
        n = min(n, len(ds))
        idx = rng.sample(range(len(ds)), n)
    return ds.select(idx)


def allocate_by_ratios(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {k: total * ratios.get(k, 0.0) for k in GROUPS}
    base = {k: int(raw[k]) for k in GROUPS}
    remainder = total - sum(base.values())
    if remainder > 0:
        order = sorted(GROUPS, key=lambda k: (raw[k] - base[k]), reverse=True)
        for i in range(remainder):
            base[order[i % len(order)]] += 1
    return base


def build_mixed_train_dataset(group_ds: Dict[str, Dataset], args: argparse.Namespace) -> Tuple[Dataset, Dict[str, Any]]:
    rng = random.Random(args.seed)
    available = {g: len(group_ds[g]) for g in GROUPS}

    if args.mix_strategy == "natural":
        parts = [group_ds[g] for g in GROUPS if len(group_ds[g]) > 0]
        if not parts:
            raise ValueError("No grouped train samples available")
        train_ds = concatenate_datasets(parts)
        if args.target_train_samples > 0:
            take_n = min(args.target_train_samples, len(train_ds))
            train_ds = sample_dataset(train_ds, n=take_n, rng=rng, oversample=False)
        return train_ds.shuffle(seed=args.seed), {
            "strategy": "natural",
            "available": available,
            "selected_per_group": {g: len(group_ds[g]) for g in GROUPS},
            "total_selected": len(train_ds),
        }

    if args.mix_strategy == "balanced":
        if args.target_train_samples > 0:
            each = args.target_train_samples // len(GROUPS)
            remainder = args.target_train_samples - each * len(GROUPS)
            requested = {g: each for g in GROUPS}
            for i, g in enumerate(GROUPS):
                if i < remainder:
                    requested[g] += 1
        else:
            if args.allow_oversample:
                base = max(available.values()) if available else 0
            else:
                base = min(available.values()) if available else 0
            requested = {g: base for g in GROUPS}
    else:
        ratios = parse_mix_ratios(args.mix_ratios)
        total = args.target_train_samples if args.target_train_samples > 0 else sum(available.values())
        requested = allocate_by_ratios(total=total, ratios=ratios)

    sampled: List[Dataset] = []
    selected = {g: 0 for g in GROUPS}
    for g in GROUPS:
        part = sample_dataset(
            group_ds[g],
            n=requested[g],
            rng=rng,
            oversample=args.allow_oversample,
        )
        sampled.append(part)
        selected[g] = len(part)

    train_ds = concatenate_datasets(sampled)
    if len(train_ds) == 0:
        raise ValueError("Mixed train dataset is empty. Check class availability and mixing settings.")
    train_ds = train_ds.shuffle(seed=args.seed)
    return train_ds, {
        "strategy": args.mix_strategy,
        "available": available,
        "requested_per_group": requested,
        "selected_per_group": selected,
        "total_selected": len(train_ds),
        "allow_oversample": args.allow_oversample,
        "mix_ratios": args.mix_ratios,
    }


class AnswerOnlyCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, args: argparse.Namespace) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [build_prompt(ex["query"], ex["context"], self.args) for ex in batch]
        texts = [build_full_text(ex["query"], ex["context"], ex["answer"], self.args) for ex in batch]

        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        prompt_enc = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

        labels = input_ids.clone()
        for i, prompt_len in enumerate(prompt_lens):
            if prompt_len >= labels.size(1):
                labels[i, :] = -100
            else:
                labels[i, :prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def filter_long_samples(ds: Dataset, tokenizer: AutoTokenizer, args: argparse.Namespace) -> Dataset:
    def keep_ex(ex: Dict[str, Any]) -> bool:
        prompt = build_prompt(query=ex["query"], context=ex["context"], args=args)
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_seq_len,
        )["input_ids"]
        return len(prompt_ids) < args.max_seq_len - 1

    return ds.filter(keep_ex)


def maybe_cap_dataset(ds: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def load_grouped_datasets(
    correct_patterns: Sequence[str],
    refusal_patterns: Sequence[str],
    negative_patterns: Sequence[str],
    split_name: str,
) -> Optional[Dict[str, Dataset]]:
    pattern_map = {
        "correct": expand_paths(correct_patterns),
        "refusal": expand_paths(refusal_patterns),
        "negative": expand_paths(negative_patterns),
    }
    if not any(pattern_map[g] for g in GROUPS):
        return None
    if not all(pattern_map[g] for g in GROUPS):
        missing = [g for g in GROUPS if not pattern_map[g]]
        raise ValueError(f"{split_name}: grouped inputs require all classes; missing {missing}")

    out: Dict[str, Dataset] = {}
    for g in GROUPS:
        ds = load_parquet_dataset(pattern_map[g], name=f"{split_name}_{g}")
        out[g] = with_example_type(ds, g)
    return out


def register_special_tokens(tokenizer: AutoTokenizer, args: argparse.Namespace) -> int:
    tokens = [
        args.context_start_token,
        args.context_end_token,
        args.query_start_token,
        args.query_end_token,
        args.answer_start_token,
        args.answer_end_token,
    ]
    uniq: List[str] = []
    for tok in tokens:
        if tok not in uniq:
            uniq.append(tok)
    return tokenizer.add_special_tokens({"additional_special_tokens": uniq})


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
    text = " ".join(x.strip() for x in sentences if str(x).strip()).strip()
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

    blocks = []
    for title, sents in selected:
        block = format_doc(title=title, sentences=sents)
        if block:
            blocks.append(block)
    ctx = "\n\n".join(blocks).strip()
    if max_context_chars > 0 and len(ctx) > max_context_chars:
        return ctx[:max_context_chars].rstrip()
    return ctx


def load_hotpot_train_dataset(args: argparse.Namespace) -> Dataset:
    ds = load_dataset(
        args.hotpot_dataset_name,
        args.hotpot_dataset_config,
        split=args.hotpot_train_split,
    )

    def to_qa(example: Dict[str, Any]) -> Dict[str, Any]:
        query = str(example.get("question") or "").strip()
        answer = str(example.get("answer") or "").strip()
        context = build_hotpot_context(
            example,
            mode=args.hotpot_context_mode,
            max_context_chars=args.hotpot_max_context_chars,
        )
        return {
            "query": query,
            "context": context,
            "answer": answer,
            "example_type": "correct",
            "source": "hotpotqa_train",
        }

    remove_cols = ds.column_names
    ds = ds.map(to_qa, remove_columns=remove_cols, desc="Converting HotpotQA -> query/context/answer")
    ds = ds.filter(lambda ex: bool(ex["query"]) and bool(ex["answer"]))
    return ds


def build_model_load_kwargs(args: argparse.Namespace, dtype: Optional[torch.dtype]) -> Dict[str, Any]:
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


def build_training_arguments(args: argparse.Namespace, eval_enabled: bool) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    eval_strategy_value = args.evaluation_strategy if eval_enabled else "no"
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
        "ddp_backend": args.ddp_backend,
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        "eval_steps": args.eval_steps,
    }

    if args.run_name:
        kwargs["run_name"] = args.run_name
    if args.logging_dir:
        kwargs["logging_dir"] = args.logging_dir

    # transformers version compatibility:
    # old-style: evaluation_strategy; newer-style: eval_strategy
    if "evaluation_strategy" in allowed:
        kwargs["evaluation_strategy"] = eval_strategy_value
    if "eval_strategy" in allowed:
        kwargs["eval_strategy"] = eval_strategy_value

    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**filtered)


def main() -> None:
    args = parse_args()
    if args.no_filter_long:
        args.filter_long = False
    if args.add_hotpot_train and args.filter_long:
        # Keep all Hotpot training examples as requested.
        print("HotpotQA train is enabled; disabling --filter_long so all Hotpot train examples are seen.")
        args.filter_long = False

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_added_tokens = 0
    if not args.no_register_special_tokens:
        num_added_tokens = register_special_tokens(tokenizer, args)
        print(f"Added special tokens: {num_added_tokens}")

    # Train split loading.
    train_grouped = load_grouped_datasets(
        correct_patterns=args.train_correct_files,
        refusal_patterns=args.train_refusal_files,
        negative_patterns=args.train_negative_files,
        split_name="train",
    )
    mix_stats: Dict[str, Any] = {}
    train_ds: Optional[Dataset] = None
    if train_grouped is not None:
        train_ds, mix_stats = build_mixed_train_dataset(train_grouped, args)
    elif args.data_jsonl:
        train_ds = load_dataset("json", data_files=args.data_jsonl, split="train")
        ensure_columns(train_ds, name="train_jsonl")
    else:
        train_paths = expand_paths(args.train_files)
        if train_paths:
            train_ds = load_parquet_dataset(train_paths, name="train")

    if args.add_hotpot_train:
        hotpot_ds = load_hotpot_train_dataset(args)
        print(f"Loaded HotpotQA train rows: {len(hotpot_ds)}")
        if train_ds is None:
            train_ds = hotpot_ds
            mix_stats["hotpot_only"] = True
        else:
            train_ds = concatenate_datasets([train_ds, hotpot_ds]).shuffle(seed=args.seed)
            mix_stats["hotpot_rows_appended"] = len(hotpot_ds)

    if train_ds is None:
        raise ValueError(
            "Provide train data using grouped files or --train_files/--data_jsonl, "
            "or enable --add_hotpot_train."
        )

    train_ds = maybe_cap_dataset(train_ds, max_samples=args.max_train_samples, seed=args.seed)
    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty after loading/mixing/filtering.")

    # Eval split loading.
    eval_grouped = load_grouped_datasets(
        correct_patterns=args.eval_correct_files,
        refusal_patterns=args.eval_refusal_files,
        negative_patterns=args.eval_negative_files,
        split_name="eval",
    )
    eval_ds: Optional[Dataset] = None
    if eval_grouped is not None:
        parts = [eval_grouped[g] for g in GROUPS if len(eval_grouped[g]) > 0]
        if parts:
            eval_ds = concatenate_datasets(parts).shuffle(seed=args.seed)
    else:
        eval_paths = expand_paths(args.eval_files)
        if eval_paths:
            eval_ds = load_parquet_dataset(eval_paths, name="eval")

    if args.filter_long:
        train_before = len(train_ds)
        train_ds = filter_long_samples(train_ds, tokenizer, args)
        print(f"Filtered train samples by prompt length: {train_before} -> {len(train_ds)}")
        if eval_ds is not None:
            eval_before = len(eval_ds)
            eval_ds = filter_long_samples(eval_ds, tokenizer, args)
            print(f"Filtered eval samples by prompt length: {eval_before} -> {len(eval_ds)}")
        if eval_grouped is not None:
            for g in GROUPS:
                before = len(eval_grouped[g])
                eval_grouped[g] = filter_long_samples(eval_grouped[g], tokenizer, args)
                print(f"Filtered eval_{g} samples by prompt length: {before} -> {len(eval_grouped[g])}")
    if eval_ds is not None and len(eval_ds) == 0:
        print("Eval dataset became empty after filtering; disabling periodic eval.")
        eval_ds = None

    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model_load_kwargs = build_model_load_kwargs(args=args, dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_load_kwargs)
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.enable_input_require_grads()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    collator = AnswerOnlyCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        args=args,
    )

    train_args = build_training_arguments(args=args, eval_enabled=(eval_ds is not None))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print(f"Final train rows: {len(train_ds)}")
    if eval_ds is not None:
        print(f"Final eval rows: {len(eval_ds)}")
    if mix_stats:
        print("Mix stats:")
        print(json.dumps(mix_stats, indent=2))

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    all_eval_metrics: Dict[str, Any] = {}
    if eval_ds is not None:
        overall = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix="eval_all")
        all_eval_metrics["all"] = overall
        print(json.dumps(overall, indent=2))

    if eval_grouped is not None:
        for g in GROUPS:
            if len(eval_grouped[g]) == 0:
                continue
            metrics = trainer.evaluate(eval_dataset=eval_grouped[g], metric_key_prefix=f"eval_{g}")
            all_eval_metrics[g] = metrics
            print(json.dumps(metrics, indent=2))

    run_meta = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "num_added_special_tokens": num_added_tokens,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "mix_stats": mix_stats,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "all_eval_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(all_eval_metrics, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
