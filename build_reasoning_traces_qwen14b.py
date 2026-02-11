#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate reasoning-trace training data with Qwen3-14B.

Output parquet fields:
  - query
  - context
  - thinking
  - answer
  - source
  - reference_answer

Sources:
  - PleIAs/SYNTH (streaming)
  - hotpot_qa (train)
"""

import argparse
import inspect
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from synth_utils import ParquetShardWriter, truncate_text, valid_row, with_rank_suffix_dir


SYSTEM_PROMPT_TEMPLATE = (
    # "You are generating supervised reasoning traces for a smaller QA model.\n"
    "Rules:\n"
    "1) Use ONLY the provided CONTEXT.\n"
    "2) Think briefly and concisely.\n"
    "3) Output exactly two XML blocks and nothing else:\n"
    "   <think>very brief reasoning</think>\n"
    "   <answer>final short answer</answer>\n"
    "4) If the answer is not explicitly in context, answer EXACTLY: {idk}\n"
)

IDK_DEFAULT = "I can't find the answer in the context."

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--max_samples", type=int, default=500_000, help="Global target across all ranks")
    ap.add_argument("--hotpot_ratio", type=float, default=0.70, help="Target share of hotpot rows in final output")
    ap.add_argument("--repeat_hotpot", action="store_true", default=True, help="Cycle hotpot rows when hotpot target > hotpot unique rows")
    ap.add_argument("--no_repeat_hotpot", action="store_true", help="Disable hotpot repetition")

    ap.add_argument("--hotpot_dataset_name", type=str, default="hotpot_qa")
    ap.add_argument("--hotpot_dataset_config", type=str, default="distractor")
    ap.add_argument("--hotpot_split", type=str, default="train")
    ap.add_argument("--hotpot_context_mode", type=str, choices=["all", "gold_titles", "supporting"], default="all")
    ap.add_argument("--hotpot_max_context_chars", type=int, default=6000)

    ap.add_argument("--synth_dataset", type=str, default="PleIAs/SYNTH")
    ap.add_argument("--synth_split", type=str, default="train")
    ap.add_argument("--synth_lang", type=str, default="en")
    ap.add_argument("--min_context_chars", type=int, default=300)
    ap.add_argument("--max_context_chars", type=int, default=6000)
    ap.add_argument("--max_query_chars", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--enable_thinking", action="store_true", default=True)
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)

    ap.add_argument("--idk_text", type=str, default=IDK_DEFAULT)
    ap.add_argument("--max_thinking_words", type=int, default=80)
    ap.add_argument("--allow_empty_thinking", action="store_true")

    ap.add_argument("--shard_size", type=int, default=50_000)
    ap.add_argument("--save_every", type=int, default=5_000)
    ap.add_argument("--parquet_compression", type=str, default="zstd")

    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_model_len", type=int, default=8192)
    ap.add_argument("--vllm_dtype", type=str, default="auto")
    ap.add_argument(
        "--distributed_output_mode",
        type=str,
        choices=["per_rank", "single"],
        default="per_rank",
    )

    args = ap.parse_args()
    if args.no_repeat_hotpot:
        args.repeat_hotpot = False
    if args.disable_thinking:
        args.enable_thinking = False
    if not (0.0 <= args.hotpot_ratio <= 1.0):
        raise ValueError("--hotpot_ratio must be in [0, 1]")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be > 0")

    if args.enable_thinking:
        if args.temperature is None:
            args.temperature = 0.6
        if args.top_p is None:
            args.top_p = 0.95
    else:
        if args.temperature is None:
            args.temperature = 0.7
        if args.top_p is None:
            args.top_p = 0.8
    if args.top_k is None:
        args.top_k = 20
    return args


def local_target(global_max: int, world: int, rank: int) -> int:
    base = global_max // world
    rem = global_max % world
    return base + (1 if rank < rem else 0)


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


def load_hotpot_examples(args: argparse.Namespace, world: int, rank: int) -> List[Dict[str, Any]]:
    ds: Dataset = load_dataset(
        args.hotpot_dataset_name,
        args.hotpot_dataset_config,
        split=args.hotpot_split,
    )
    if world > 1:
        ds = ds.shard(num_shards=world, index=rank)

    out: List[Dict[str, Any]] = []
    for row in ds:
        query = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        context = build_hotpot_context(
            row,
            mode=args.hotpot_context_mode,
            max_context_chars=args.hotpot_max_context_chars,
        )
        if not query or not context:
            continue
        out.append(
            {
                "query": query,
                "context": context,
                "source": "hotpot_qa",
                "reference_answer": answer,
            }
        )
    return out


def iter_synth_examples(args: argparse.Namespace, world: int, rank: int) -> Iterator[Dict[str, Any]]:
    ds = load_dataset(args.synth_dataset, split=args.synth_split, streaming=True)
    if world > 1:
        ds = ds.shard(num_shards=world, index=rank)

    for row in ds:
        if not valid_row(
            row=row,
            lang=args.synth_lang,
            min_context_chars=args.min_context_chars,
            max_context_chars=args.max_context_chars,
            max_query_chars=args.max_query_chars,
        ):
            continue
        query = str(row.get("query") or "").strip()
        context = truncate_text(str(row.get("query_seed_text") or ""), args.max_context_chars)
        if not query or not context:
            continue
        yield {
            "query": query,
            "context": context,
            "source": "synth",
            "reference_answer": "",
        }


def build_model_kwargs(args: argparse.Namespace, dtype: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if dtype != "auto":
        if "dtype" in sig.parameters:
            kwargs["dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype
    return kwargs


def build_prompt_text(
    tokenizer,
    query: str,
    context: str,
    system_prompt: str,
    enable_thinking: bool,
) -> str:
    user = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{query}\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@torch.inference_mode()
def generate_batch_hf(
    model,
    tokenizer,
    prompt_texts: List[str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(gen, skip_special_tokens=False)


def generate_batch_vllm(
    llm,
    prompt_texts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompt_texts, sampling_params=sampling)
    texts: List[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
            continue
        texts.append(out.outputs[0].text)
    return texts


def strip_meta_tokens(text: str) -> str:
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.replace("<s>", " ").replace("</s>", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_content(text: str) -> str:
    text = strip_meta_tokens(text)
    text = re.sub(r"</?(think|answer)>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text.strip()
    toks = text.strip().split()
    if len(toks) <= max_words:
        return " ".join(toks)
    return " ".join(toks[:max_words]).strip()


def parse_reasoning_output(raw_text: str, max_thinking_words: int) -> Tuple[str, str]:
    text = raw_text.strip()

    think_match = THINK_RE.search(text)
    answer_match = ANSWER_RE.search(text)

    thinking = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    if not answer:
        if think_match:
            answer = text[think_match.end() :].strip()
        else:
            answer = text.strip()

    if not thinking and answer_match:
        prefix = text[: answer_match.start()].strip()
        prefix = clean_content(prefix)
        if prefix:
            thinking = prefix

    thinking = clean_content(thinking)
    answer = clean_content(answer)
    thinking = truncate_words(thinking, max_thinking_words=max_thinking_words)
    return thinking, answer


def main() -> None:
    args = parse_args()

    from accelerate import Accelerator

    accelerator = Accelerator()
    world = accelerator.num_processes
    rank = accelerator.process_index

    if args.use_vllm and world > 1:
        raise RuntimeError(
            "vLLM already handles multi-GPU with tensor_parallel_size. "
            "Use a single process when --use_vllm is enabled."
        )

    local_max = local_target(args.max_samples, world=world, rank=rank)
    if local_max <= 0:
        print(f"Rank {rank}: nothing to do (local_max=0).")
        return

    random.seed(args.seed + rank)

    out_dir = args.out_dir
    if world > 1 and args.distributed_output_mode == "per_rank":
        out_dir = with_rank_suffix_dir(args.out_dir, rank)
    os.makedirs(out_dir, exist_ok=True)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(idk=args.idk_text)
    target_hotpot = int(round(local_max * args.hotpot_ratio))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = None
    llm = None
    if args.use_vllm:
        from vllm import LLM

        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            dtype=args.vllm_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        dtype: Any = "auto"
        model_kwargs = build_model_kwargs(args=args, dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=None if world > 1 else "auto",
            **model_kwargs,
        )
        if world > 1:
            model.to(accelerator.device)
        model.eval()

    hotpot_examples = load_hotpot_examples(args=args, world=world, rank=rank)
    random.shuffle(hotpot_examples)
    hotpot_unique = len(hotpot_examples)
    hotpot_idx = 0

    synth_iter = iter_synth_examples(args=args, world=world, rank=rank)

    writer = ParquetShardWriter(
        out_dir=out_dir,
        shard_size=args.shard_size,
        compression=args.parquet_compression,
    )

    pending: List[Dict[str, Any]] = []
    out_buffer: List[Dict[str, Any]] = []
    buffer_flush_rows = args.save_every if args.save_every > 0 else args.shard_size

    produced_total = 0
    produced_by_source: Dict[str, int] = {"hotpot_qa": 0, "synth": 0}
    skipped_parse = 0
    skipped_empty = 0
    synth_exhausted = False
    hotpot_enabled = hotpot_unique > 0

    pbar = tqdm(
        total=local_max,
        desc=f"reasoning traces (rank {rank})",
        unit="ex",
        disable=not accelerator.is_local_main_process,
    )

    def get_next_hotpot() -> Optional[Dict[str, Any]]:
        nonlocal hotpot_idx
        if not hotpot_examples:
            return None
        if hotpot_idx >= len(hotpot_examples):
            if not args.repeat_hotpot:
                return None
            random.shuffle(hotpot_examples)
            hotpot_idx = 0
        ex = hotpot_examples[hotpot_idx]
        hotpot_idx += 1
        return ex

    def flush_pending() -> None:
        nonlocal pending, out_buffer, produced_total, skipped_parse, skipped_empty
        if not pending or produced_total >= local_max:
            pending = []
            return

        prompt_texts = [x["prompt_text"] for x in pending]
        if args.use_vllm:
            completions = generate_batch_vllm(
                llm=llm,
                prompt_texts=prompt_texts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        else:
            completions = generate_batch_hf(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

        for ex, raw in zip(pending, completions):
            if produced_total >= local_max:
                break
            thinking, answer = parse_reasoning_output(raw_text=raw, max_thinking_words=args.max_thinking_words)
            if not answer:
                skipped_empty += 1
                continue
            if (not args.allow_empty_thinking) and (not thinking):
                skipped_parse += 1
                continue

            row = {
                "query": ex["query"],
                "context": ex["context"],
                "thinking": thinking,
                "answer": answer,
                "source": ex["source"],
                "reference_answer": ex["reference_answer"],
            }
            out_buffer.append(row)
            produced_total += 1
            produced_by_source[ex["source"]] = produced_by_source.get(ex["source"], 0) + 1

        pending = []
        if len(out_buffer) >= buffer_flush_rows:
            writer.write_rows(out_buffer)
            out_buffer = []

    try:
        while produced_total < local_max:
            need_hotpot = hotpot_enabled and (produced_by_source["hotpot_qa"] < target_hotpot)
            ex: Optional[Dict[str, Any]]

            if need_hotpot:
                ex = get_next_hotpot()
                if ex is None:
                    hotpot_enabled = False
                    continue
            else:
                if synth_exhausted:
                    break
                try:
                    ex = next(synth_iter)
                except StopIteration:
                    synth_exhausted = True
                    continue

            prompt_text = build_prompt_text(
                tokenizer=tokenizer,
                query=ex["query"],
                context=ex["context"],
                system_prompt=system_prompt,
                enable_thinking=args.enable_thinking,
            )
            pending.append(
                {
                    "query": ex["query"],
                    "context": ex["context"],
                    "source": ex["source"],
                    "reference_answer": ex["reference_answer"],
                    "prompt_text": prompt_text,
                }
            )

            if len(pending) >= args.batch_size:
                flush_pending()
                pbar.n = produced_total
                pbar.refresh()
    finally:
        flush_pending()
        if out_buffer:
            writer.write_rows(out_buffer)
            out_buffer = []
        writer.close()
        pbar.n = produced_total
        pbar.refresh()
        pbar.close()

    stats = {
        "global_max_samples": args.max_samples,
        "local_target": local_max,
        "produced_total": produced_total,
        "produced_by_source": produced_by_source,
        "target_hotpot_rows": target_hotpot,
        "hotpot_unique_rows_local": hotpot_unique,
        "repeat_hotpot": args.repeat_hotpot,
        "skipped_parse": skipped_parse,
        "skipped_empty_answer": skipped_empty,
        "synth_exhausted": synth_exhausted,
        "rank": rank,
        "world": world,
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Done. Wrote reasoning traces to {out_dir}")


if __name__ == "__main__":
    main()
