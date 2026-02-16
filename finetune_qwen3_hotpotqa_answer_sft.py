#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


INSTRUCT_SYSTEM_INSTRUCTION = (
    "You are an expert in retrieval-based question answering. "
    "Answer the question using only the information provided in the context below. "
    "If the context does not contain enough information to answer, "
    'respond with "unknown".'
)

ANSWER_FORMAT_HINT_STANDARD = (
    'a short phrase (a few words), or "unknown" if unanswerable from the context'
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fine-tune Qwen3 on full HotpotQA train split for answer-only output "
            "(no reasoning), with standard instruct chat template and enable_thinking=False."
        )
    )
    ap.add_argument("--output_dir", type=str, default="outputs/qwen3_hotpot_answer_sft")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--dataset_name", type=str, default="hotpot_qa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="", help="Optional eval split, e.g. validation")
    ap.add_argument("--n_passages", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_samples", type=int, default=0, help="0 means full train split")
    ap.add_argument("--max_eval_samples", type=int, default=0)

    ap.add_argument("--max_seq_len", type=int, default=3072)
    ap.add_argument("--filter_long", action="store_true")
    ap.add_argument("--min_target_tokens", type=int, default=4)

    ap.add_argument("--answer_prefix", type=str, default="Answer: ")
    ap.add_argument("--answer_format_hint", type=str, default=ANSWER_FORMAT_HINT_STANDARD)

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
    ap.add_argument("--evaluation_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--report_to", type=str, default="none")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--logging_dir", type=str, default="")
    ap.add_argument("--ddp_backend", type=str, default="nccl")
    ap.add_argument("--ddp_find_unused_parameters", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    return ap.parse_args()


def apply_chat_template_no_thinking(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in sig.parameters:
            kwargs["enable_thinking"] = False
    except Exception:
        pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def format_passage(title: str, sentences: List[str]) -> str:
    return f"{title} | {' '.join(sentences)}"


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


def parse_supporting_titles(raw: Any) -> set[str]:
    titles: set[str] = set()
    if isinstance(raw, dict):
        for title in raw.get("title") or []:
            titles.add(str(title))
        return titles
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and item:
                titles.add(str(item[0]))
    return titles


def build_hotpot_passages(
    sample: Dict[str, Any],
    n_passages: int,
    seed: int,
) -> List[str]:
    docs = parse_context_field(sample.get("context"))
    gold_titles = parse_supporting_titles(sample.get("supporting_facts"))

    gold_passages: List[str] = []
    distractor_passages: List[str] = []
    for title, sentences in docs:
        passage = format_passage(title, sentences)
        if title in gold_titles:
            gold_passages.append(passage)
        else:
            distractor_passages.append(passage)

    rng = random.Random(seed)
    if n_passages <= 0:
        out = gold_passages + distractor_passages
        rng.shuffle(out)
        return out

    if len(gold_passages) > n_passages:
        gold_passages = rng.sample(gold_passages, n_passages)
    if len(gold_passages) >= n_passages:
        rng.shuffle(gold_passages)
        return gold_passages

    n_distractors = n_passages - len(gold_passages)
    if len(distractor_passages) > n_distractors:
        distractor_passages = rng.sample(distractor_passages, n_distractors)

    passages = gold_passages + distractor_passages
    rng.shuffle(passages)
    return passages


def clean_text(x: Any) -> str:
    return str(x if x is not None else "").strip()


def build_instruct_user_content(
    question: str,
    passages: List[str],
    answer_format_hint: str,
) -> str:
    context = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return (
        f"{INSTRUCT_SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Your answer should be {answer_format_hint}.\n"
        f'End your response with "Answer:" followed by your final answer.'
    )


def maybe_cap_dataset(ds: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


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
        "ddp_backend": args.ddp_backend,
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
    }
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


class ChatAnswerCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        answer_format_hint: str,
        answer_prefix: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_format_hint = answer_format_hint
        self.answer_prefix = answer_prefix

    def build_prompt_text(self, question: str, passages: List[str]) -> str:
        user_content = build_instruct_user_content(
            question=question,
            passages=passages,
            answer_format_hint=self.answer_format_hint,
        )
        return apply_chat_template_no_thinking(
            tokenizer=self.tokenizer,
            messages=[{"role": "user", "content": user_content}],
            add_generation_prompt=True,
        )

    def build_train_text(self, question: str, passages: List[str], answer: str) -> str:
        user_content = build_instruct_user_content(
            question=question,
            passages=passages,
            answer_format_hint=self.answer_format_hint,
        )
        assistant = f"{self.answer_prefix}{answer}".strip()
        return apply_chat_template_no_thinking(
            tokenizer=self.tokenizer,
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant},
            ],
            add_generation_prompt=False,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [self.build_prompt_text(ex["question"], ex["passages"]) for ex in batch]
        texts = [self.build_train_text(ex["question"], ex["passages"], ex["answer"]) for ex in batch]

        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
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


def build_sft_dataset(split_ds: Dataset, n_passages: int, seed: int, split_name: str) -> Dataset:
    def convert(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        passages = build_hotpot_passages(
            sample=example,
            n_passages=n_passages,
            seed=seed + idx,
        )
        return {
            "question": clean_text(example.get("question")),
            "answer": clean_text(example.get("answer")),
            "passages": passages,
            "source_split": split_name,
        }

    ds = split_ds.map(
        convert,
        with_indices=True,
        remove_columns=split_ds.column_names,
        desc=f"Build {split_name} rows",
    )
    ds = ds.filter(
        lambda ex: bool(clean_text(ex.get("question")))
        and bool(clean_text(ex.get("answer")))
        and bool(ex.get("passages")),
        desc=f"Filter invalid {split_name} rows",
    )
    return ds


def filter_long_samples(
    ds: Dataset,
    collator: ChatAnswerCollator,
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    min_target_tokens: int,
) -> Dataset:
    def keep(ex: Dict[str, Any]) -> bool:
        prompt = collator.build_prompt_text(ex["question"], ex["passages"])
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"]
        return len(prompt_ids) < (max_seq_len - max(1, min_target_tokens))

    return ds.filter(keep, desc="Filter overlong prompts")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_train = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)
    raw_train = maybe_cap_dataset(raw_train, args.max_train_samples, seed=args.seed)
    train_ds = build_sft_dataset(raw_train, n_passages=args.n_passages, seed=args.seed, split_name=args.train_split)

    eval_ds: Optional[Dataset] = None
    if args.eval_split:
        raw_eval = load_dataset(args.dataset_name, args.dataset_config, split=args.eval_split)
        raw_eval = maybe_cap_dataset(raw_eval, args.max_eval_samples, seed=args.seed)
        eval_ds = build_sft_dataset(
            raw_eval,
            n_passages=args.n_passages,
            seed=args.seed + 1_000_000,
            split_name=args.eval_split,
        )
        if len(eval_ds) == 0:
            print("Eval dataset became empty; disabling eval.")
            eval_ds = None

    collator = ChatAnswerCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        answer_format_hint=args.answer_format_hint,
        answer_prefix=args.answer_prefix,
    )

    if args.filter_long:
        before_train = len(train_ds)
        train_ds = filter_long_samples(
            ds=train_ds,
            collator=collator,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            min_target_tokens=args.min_target_tokens,
        )
        print(f"Filtered train by length: {before_train} -> {len(train_ds)}")
        if eval_ds is not None:
            before_eval = len(eval_ds)
            eval_ds = filter_long_samples(
                ds=eval_ds,
                collator=collator,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                min_target_tokens=args.min_target_tokens,
            )
            print(f"Filtered eval by length: {before_eval} -> {len(eval_ds)}")

    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty after preprocessing.")
    if eval_ds is not None and len(eval_ds) == 0:
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

    trainer = Trainer(
        model=model,
        args=build_training_args(args=args, eval_enabled=eval_ds is not None),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(f"Train rows: {len(train_ds)}")
    if eval_ds is not None:
        print(f"Eval rows: {len(eval_ds)}")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds) if eval_ds is not None else 0,
        "n_passages": args.n_passages,
        "chat_template_mode": "default tokenizer chat template with enable_thinking=False",
        "target_format": "assistant answer only, prefixed by 'Answer:'",
        "answer_prefix": args.answer_prefix,
        "answer_format_hint": args.answer_format_hint,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "sft_hotpot_answer_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
