#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate reasoning traces from HotpotQA train only, using Qwen3-14B.

Output parquet fields:
  - query
  - context
  - thinking
  - answer
  - prompt_version
  - reference_answer
"""

import argparse
import inspect
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from synth_utils import ParquetShardWriter, with_rank_suffix_dir


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

IDK_DEFAULT = "I can't find the answer in the context."

PROMPT_V1 = (
    "You are a QA assistant.\n"
    "Rules:\n"
    "1) Use only the provided CONTEXT.\n"
    "2) Think briefly, then answer.\n"
    "3) Output exactly:\n"
    "   <think>brief reasoning</think>\n"
    "   <answer>final answer</answer>\n"
    "4) If answer is not in context, output EXACTLY this in <answer>: {idk}\n"
)

PROMPT_V2_SHORT = (
    "You are a QA assistant.\n"
    "Rules:\n"
    "1) Use only CONTEXT.\n"
    "2) Keep reasoning extremely short (max 20 words).\n"
    "3) Output exactly:\n"
    "   <think>extremely short reasoning</think>\n"
    "   <answer>final answer</answer>\n"
    "4) If answer is not in context, output EXACTLY this in <answer>: {idk}\n"
)

PROMPT_V3_STRUCTURED = (
    "You are a QA assistant.\n"
    "Use only CONTEXT and output exactly:\n"
    "<think>\n"
    "facts: key evidence from context\n"
    "bridge: how evidence connects\n"
    "check: why this answers the question\n"
    "</think>\n"
    "<answer>final answer</answer>\n"
    "If answer is not in context, output EXACTLY this in <answer>: {idk}\n"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--dataset_name", type=str, default="hotpot_qa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 means use full local Hotpot split shard; >0 caps rows globally across ranks.",
    )
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--context_mode", type=str, choices=["all", "gold_titles", "supporting"], default="all")
    ap.add_argument("--max_context_chars", type=int, default=6000)

    ap.add_argument(
        "--prompt_version",
        type=str,
        choices=["v1", "v2", "v3", "mix"],
        default="mix",
        help="v1=basic, v2=very short thinking, v3=structured thinking, mix=random among all three",
    )
    ap.add_argument("--idk_text", type=str, default=IDK_DEFAULT)
    ap.add_argument("--max_thinking_words", type=int, default=80)
    ap.add_argument("--max_thinking_words_v2", type=int, default=20)
    ap.add_argument("--allow_empty_thinking", action="store_true")

    ap.add_argument("--enable_thinking", action="store_true", default=True)
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--min_p", type=float, default=0.0, help="Qwen3 thinking default is 0")

    ap.add_argument("--shard_size", type=int, default=50_000)
    ap.add_argument("--save_every", type=int, default=5_000)
    ap.add_argument("--parquet_compression", type=str, default="zstd")

    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_model_len", type=int, default=8192)
    ap.add_argument("--vllm_dtype", type=str, default="auto")
    vllm_eager_group = ap.add_mutually_exclusive_group()
    vllm_eager_group.add_argument("--vllm_enforce_eager", dest="vllm_enforce_eager", action="store_true")
    vllm_eager_group.add_argument("--vllm_no_enforce_eager", dest="vllm_enforce_eager", action="store_false")
    ap.set_defaults(vllm_enforce_eager=True)
    vllm_sanitize_group = ap.add_mutually_exclusive_group()
    vllm_sanitize_group.add_argument(
        "--vllm_sanitize_dist_env",
        dest="vllm_sanitize_dist_env",
        action="store_true",
        help="Unset torch.distributed env vars before creating each vLLM instance.",
    )
    vllm_sanitize_group.add_argument(
        "--vllm_no_sanitize_dist_env",
        dest="vllm_sanitize_dist_env",
        action="store_false",
    )
    ap.set_defaults(vllm_sanitize_dist_env=True)
    ap.add_argument(
        "--distributed_output_mode",
        type=str,
        choices=["per_rank", "single"],
        default="per_rank",
    )

    args = ap.parse_args()
    if args.max_samples < 0:
        raise ValueError("--max_samples must be >= 0")
    if args.disable_thinking:
        args.enable_thinking = False

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
    ds: Dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if world > 1:
        ds = ds.shard(num_shards=world, index=rank)

    out: List[Dict[str, Any]] = []
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


def system_prompt_for_version(version: str, idk_text: str) -> str:
    if version == "v1":
        return PROMPT_V1.format(idk=idk_text)
    if version == "v2":
        return PROMPT_V2_SHORT.format(idk=idk_text)
    if version == "v3":
        return PROMPT_V3_STRUCTURED.format(idk=idk_text)
    raise ValueError(f"Unknown prompt version: {version}")


def pick_prompt_version(configured: str, rng: random.Random) -> str:
    if configured != "mix":
        return configured
    return rng.choice(["v1", "v2", "v3"])


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
    min_p: float,
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

    gen_kwargs: Dict[str, Any] = {
        **enc,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,  # Never greedy for Qwen3 thinking-mode generation
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    # min_p is available in newer transformers generation APIs.
    if "min_p" in inspect.signature(model.generate).parameters:
        gen_kwargs["min_p"] = min_p

    out = model.generate(**gen_kwargs)
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(gen, skip_special_tokens=False)


def generate_batch_vllm(
    llm,
    prompt_texts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> List[str]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
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
    thinking = truncate_words(thinking, max_words=max_thinking_words)
    return thinking, answer


def thinking_cap_for_version(version: str, args: argparse.Namespace) -> int:
    if version == "v2":
        return args.max_thinking_words_v2
    return args.max_thinking_words


def env_distributed_info() -> Tuple[int, int, int, bool]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_local_main = local_rank == 0
    return world, rank, local_rank, is_local_main


def pin_vllm_to_single_gpu(local_rank: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        gpu = str(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        return gpu

    parts = [x.strip() for x in visible.split(",") if x.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else "0"

    gpu = parts[local_rank % len(parts)]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu


def sanitize_dist_env_for_vllm() -> None:
    # accelerate/torchrun env leaks into vLLM worker subprocesses and can cause hangs.
    keys = [
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
    ]
    for key in keys:
        os.environ.pop(key, None)
    # Let each vLLM process choose its own local init endpoint.
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)


def main() -> None:
    args = parse_args()

    accelerator = None
    if args.use_vllm:
        world, rank, local_rank, is_local_main = env_distributed_info()
        if world > 1:
            if args.vllm_tensor_parallel_size != 1:
                raise ValueError(
                    "When launching multiple processes with --use_vllm, "
                    "--vllm_tensor_parallel_size must be 1."
                )
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
            os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
            os.environ.setdefault("GLOO_USE_IPV6", "0")
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
            bound_gpu = pin_vllm_to_single_gpu(local_rank=local_rank)
            if is_local_main:
                print(
                    f"[vLLM multi-proc] world={world} | this rank={rank} "
                    f"local_rank={local_rank} on GPU {bound_gpu}"
                )
            if args.vllm_sanitize_dist_env:
                sanitize_dist_env_for_vllm()
    else:
        from accelerate import Accelerator

        accelerator = Accelerator()
        world = accelerator.num_processes
        rank = accelerator.process_index
        local_rank = accelerator.local_process_index
        is_local_main = accelerator.is_local_main_process

    rng = random.Random(args.seed + rank)

    out_dir = args.out_dir
    if world > 1 and args.distributed_output_mode == "per_rank":
        out_dir = with_rank_suffix_dir(args.out_dir, rank)
    os.makedirs(out_dir, exist_ok=True)

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
            enforce_eager=args.vllm_enforce_eager,
        )
    else:
        dtype: Any = "auto"
        model_kwargs = build_model_kwargs(args=args, dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=None if world > 1 else "auto",
            **model_kwargs,
        )
        if world > 1 and accelerator is not None:
            model.to(accelerator.device)
        model.eval()

    examples = load_hotpot_examples(args=args, world=world, rank=rank)
    rng.shuffle(examples)

    if not examples:
        raise RuntimeError("No HotpotQA examples after filtering.")

    if args.max_samples == 0:
        local_max = len(examples)
    else:
        local_max = min(local_target(args.max_samples, world=world, rank=rank), len(examples))

    if local_max <= 0:
        print(f"Rank {rank}: nothing to do (local_max=0).")
        return

    examples = examples[:local_max]

    writer = ParquetShardWriter(
        out_dir=out_dir,
        shard_size=args.shard_size,
        compression=args.parquet_compression,
    )

    pending: List[Dict[str, Any]] = []
    out_buffer: List[Dict[str, Any]] = []
    buffer_flush_rows = args.save_every if args.save_every > 0 else args.shard_size
    version_counts: Dict[str, int] = {"v1": 0, "v2": 0, "v3": 0}

    produced = 0
    skipped_parse = 0
    skipped_empty = 0
    pbar = tqdm(
        total=local_max,
        desc=f"hotpot reasoning traces (rank {rank})",
        unit="ex",
        disable=not is_local_main,
    )

    def flush_pending() -> None:
        nonlocal pending, out_buffer, produced, skipped_parse, skipped_empty
        if not pending or produced >= local_max:
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
                min_p=args.min_p,
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
                min_p=args.min_p,
            )

        for ex, raw in zip(pending, completions):
            if produced >= local_max:
                break
            max_words = thinking_cap_for_version(ex["prompt_version"], args)
            thinking, answer = parse_reasoning_output(raw_text=raw, max_thinking_words=max_words)
            if not answer:
                # Fallback so we keep more rows from full train split.
                answer = clean_content(raw)
            if not thinking:
                # Fallback compact thought when tags are missing.
                thinking = truncate_words(clean_content(raw), max_words=max_words)
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
                "prompt_version": ex["prompt_version"],
                "reference_answer": ex["reference_answer"],
            }
            out_buffer.append(row)
            produced += 1
            version_counts[ex["prompt_version"]] = version_counts.get(ex["prompt_version"], 0) + 1

        pending = []
        if len(out_buffer) >= buffer_flush_rows:
            writer.write_rows(out_buffer)
            out_buffer = []

    try:
        for ex in examples:
            if produced >= local_max:
                break

            version = pick_prompt_version(args.prompt_version, rng=rng)
            system_prompt = system_prompt_for_version(version, idk_text=args.idk_text)
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
                    "reference_answer": ex["reference_answer"],
                    "prompt_version": version,
                    "prompt_text": prompt_text,
                }
            )

            if len(pending) >= args.batch_size:
                flush_pending()
                pbar.n = produced
                pbar.refresh()
    finally:
        flush_pending()
        if out_buffer:
            writer.write_rows(out_buffer)
            out_buffer = []
        writer.close()
        pbar.n = produced
        pbar.refresh()
        pbar.close()

    stats = {
        "global_max_samples": args.max_samples,
        "local_target": local_max,
        "produced_total": produced,
        "prompt_version_config": args.prompt_version,
        "produced_by_prompt_version": version_counts,
        "skipped_parse": skipped_parse,
        "skipped_empty_answer": skipped_empty,
        "rank": rank,
        "world": world,
        "args": vars(args),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Done. Wrote Hotpot reasoning traces to {out_dir}")


if __name__ == "__main__":
    main()
