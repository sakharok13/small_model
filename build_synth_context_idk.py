#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a dataset of:
  [CONTEXT][QUERY][answer]
where answer is generated from context using Qwen3-4B-Instruct,
and sometimes CONTEXT is mismatched so answer should be "I don't know".
Some examples intentionally omit context and use a fixed no-context answer.

Output: Parquet shards with fields:
  - query, context, answer
  - negative (bool)

Requirements:
  pip install -U "transformers>=4.51.0" datasets accelerate torch tqdm
Qwen3 recommends transformers>=4.51.0.
"""

import argparse
import os
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


from synth_utils import (
    IDK_DEFAULT,
    NO_CONTEXT_DEFAULT,
    build_messages,
    extract_json_obj,
    load_state,
    ParquetShardWriter,
    restore_random_state,
    save_state,
    truncate_text,
    valid_row,
    with_rank_suffix_dir,
)

IDK = IDK_DEFAULT
NO_CONTEXT_ANSWER = NO_CONTEXT_DEFAULT

SYSTEM_PROMPT_TEMPLATE = (
    "You are a strict question-answering assistant.\n"
    "Rules:\n"
    "0) If CONTEXT is empty, reply with EXACTLY: {no_context}\n"
    "1) Use ONLY the provided CONTEXT.\n"
    "2) If the answer is not explicitly stated in CONTEXT, reply with EXACTLY: {idk}\n"
    "3) Do not use outside knowledge. Do not guess.\n"
    "4) Reply with only the final answer text. No preamble.\n"
)

# Optional JSON mode for higher-quality filtering:
# This makes it easier to sanity-check that the model extracted an explicit quote.
USE_JSON_MODE_DEFAULT = True
SYSTEM_PROMPT_JSON_TEMPLATE = (
    "You are a dataset generator.\n"
    "You must answer the QUESTION using ONLY the provided CONTEXT.\n"
    "If CONTEXT is empty, set answer to EXACTLY: {no_context}\n"
    "If the answer is not explicitly stated in CONTEXT, set answer to EXACTLY: {idk}\n"
    "Return STRICT JSON and nothing else with keys:\n"
    '  {{"answer": "...", "quotes": ["verbatim quote 1", "verbatim quote 2"]}}\n'
    'If answer is "{idk}" or "{no_context}", quotes must be [].\n'
)


 


@torch.inference_mode()
def generate_batch_hf(
    model,
    tokenizer,
    prompt_texts: List[str],
    max_new_tokens: int,
) -> List[str]:
    """Greedy generate a batch of completions using HF Transformers."""
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        # Important: prompts already include special tokens from chat template.
        add_special_tokens=False,
    )
    # Put tensors on the main device (works with device_map="auto" for most setups)
    device = model.device
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Slice off the prompt
    gen = out[:, enc["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [t.strip() for t in texts]


def generate_batch_vllm(
    llm,
    prompt_texts: List[str],
    max_new_tokens: int,
) -> List[str]:
    """Greedy generate a batch of completions using vLLM."""
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompt_texts, sampling_params=sampling)
    texts: List[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
            continue
        texts.append(out.outputs[0].text.strip())
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="PleIAs/SYNTH", help="HF dataset name")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for Parquet shards")
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for model loading")

    ap.add_argument("--max_samples", type=int, default=50_000, help="Max output examples (per-rank when distributed)")
    ap.add_argument("--neg_ratio", type=float, default=0.30, help="Probability to create a negative (mismatched context) example per row")
    ap.add_argument("--neg_pool_size", type=int, default=4096, help="Size of pool for sampling mismatched contexts")
    ap.add_argument("--neg_pool_warmup", type=int, default=256, help="Need this many contexts before sampling negatives")
    ap.add_argument("--missing_context_ratio", type=float, default=0.005, help="Probability to add missing-context examples")
    ap.add_argument("--missing_context_max", type=int, default=1000, help="Maximum missing-context examples to add")
    ap.add_argument("--missing_context_answer", type=str, default=NO_CONTEXT_ANSWER, help="Answer when context is missing")
    ap.add_argument("--shard_size", type=int, default=50_000, help="Examples per Parquet shard")
    ap.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="Write buffered rows to the current shard every N examples (0 to only flush at shard_size/end)",
    )
    ap.add_argument("--parquet_compression", type=str, default="zstd", help="Parquet compression codec (zstd|snappy|gzip|none)")

    ap.add_argument("--min_context_chars", type=int, default=300)
    ap.add_argument("--max_context_chars", type=int, default=6000)
    ap.add_argument("--max_query_chars", type=int, default=512)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_json_mode", action="store_true", default=USE_JSON_MODE_DEFAULT, help="Ask Qwen for JSON with quotes and filter by quotes")
    ap.add_argument("--no_json_mode", action="store_true", help="Disable JSON mode")
    ap.add_argument("--use_vllm", action="store_true", help="Use vLLM for fast multi-GPU inference")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size (e.g. 4 or 8)")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90, help="vLLM GPU memory utilization fraction")
    ap.add_argument("--vllm_max_model_len", type=int, default=8192, help="vLLM max model length")
    ap.add_argument("--vllm_dtype", type=str, default="auto", help="vLLM dtype: auto|float16|bfloat16|float32")
    ap.add_argument("--state_path", type=str, default="", help="Path to resume state (default: out_dir/state.json)")
    ap.add_argument("--resume", action="store_true", help="Resume from state file and append shards to out_dir")
    ap.add_argument(
        "--distributed_output_mode",
        type=str,
        choices=["per_rank", "single"],
        default="per_rank",
        help="In multi-GPU, write per-rank output dirs (recommended) or a single shared dir (unsafe).",
    )

    args = ap.parse_args()
    if args.no_json_mode:
        args.use_json_mode = False
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be > 0")

    from accelerate import Accelerator

    accelerator = Accelerator()
    world = accelerator.num_processes
    rank = accelerator.process_index

    if args.use_vllm and world > 1:
        raise RuntimeError(
            "vLLM already handles multi-GPU via --vllm_tensor_parallel_size. "
            "Run with a single process (no accelerate) when using vLLM."
        )

    random.seed(args.seed)

    out_dir = args.out_dir
    if world > 1 and args.distributed_output_mode == "per_rank":
        out_dir = with_rank_suffix_dir(args.out_dir, rank)
    os.makedirs(out_dir, exist_ok=True)
    if not args.state_path:
        args.state_path = os.path.join(out_dir, "state.json")

    state = load_state(args.state_path) if args.resume else {}
    resume_rows = int(state.get("rows_seen", 0))
    rows_seen = resume_rows if args.resume else 0
    written = int(state.get("written", 0)) if args.resume else 0
    missing_context_written = int(state.get("missing_context_written", 0)) if args.resume else 0
    if args.resume:
        restore_random_state(state, args.seed)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        idk=IDK,
        no_context=args.missing_context_answer,
    )
    system_prompt_json = SYSTEM_PROMPT_JSON_TEMPLATE.format(
        idk=IDK,
        no_context=args.missing_context_answer,
    )

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    tokenizer.padding_side = "left"
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype="auto",
            device_map=None if world > 1 else "auto",
            trust_remote_code=args.trust_remote_code,
        )
        if world > 1:
            model.to(accelerator.device)
        model.eval()

    # Stream SYNTH (it's massive)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    if world > 1:
        ds = ds.shard(num_shards=world, index=rank)
    skip_remaining = 0
    if args.resume and resume_rows > 0:
        try:
            ds = ds.skip(resume_rows)
            skip_remaining = 0
            rows_seen = resume_rows
        except Exception:
            # fallback: manual skipping in loop
            skip_remaining = resume_rows
            rows_seen = 0

    neg_pool: Deque[Tuple[str, str]] = deque(maxlen=args.neg_pool_size)  # (context, url)
    if args.resume and state.get("neg_pool"):
        try:
            for ctx, url in state.get("neg_pool", []):
                neg_pool.append((ctx, url))
        except Exception:
            neg_pool.clear()

    # We'll accumulate inference jobs and flush in batches
    pending: List[Dict[str, Any]] = []
    out_buffer: List[Dict[str, Any]] = []
    produced = written
    buffer_flush_rows = args.save_every if args.save_every > 0 else args.shard_size
    shard_writer = ParquetShardWriter(
        out_dir=out_dir,
        shard_size=args.shard_size,
        compression=args.parquet_compression,
    )

    def flush_pending() -> None:
        nonlocal pending, produced, missing_context_written, out_buffer
        if not pending:
            return

        # Build prompts for only those that need model inference
        prompt_texts = [ex["prompt_text"] for ex in pending if ex["needs_infer"]]
        completions: List[str] = []
        if prompt_texts:
            if args.use_vllm:
                completions = generate_batch_vllm(
                    llm=llm,
                    prompt_texts=prompt_texts,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                completions = generate_batch_hf(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_texts=prompt_texts,
                    max_new_tokens=args.max_new_tokens,
                )

        # Assign outputs back
        comp_i = 0
        for ex in pending:
            answer = ex["answer"]
            quotes: List[str] = ex.get("quotes", [])

            if ex["needs_infer"]:
                raw = completions[comp_i]
                comp_i += 1

                if args.use_json_mode:
                    obj = extract_json_obj(raw)
                    if obj is None:
                        # Drop noisy sample
                        continue
                    answer = (obj.get("answer") or "").strip()
                    quotes = obj.get("quotes") or []
                    if not isinstance(quotes, list):
                        quotes = []

                    if answer not in (IDK, args.missing_context_answer):
                        # Filtering: require at least one quote that appears verbatim in context
                        if len(quotes) == 0:
                            continue
                        if any((not isinstance(q, str)) or (q.strip() == "") for q in quotes):
                            continue
                        ctx = ex["context"]
                        if not all(q in ctx for q in quotes):
                            continue
                else:
                    answer = raw.strip()
                    # Optional cleanup: force canonical IDK if it starts with it
                    if answer.lower().startswith("i don't know"):
                        answer = IDK

                # If we *expect* IDK (negative) but model didn't comply, force it
                if ex["force_idk"]:
                    answer = IDK
                    quotes = []

            # Final canonicalization
            answer = answer.strip()
            if answer == "":
                continue
            if answer.lower() == "i dont know":
                answer = IDK

            out = {
                "negative": ex["is_negative"],
                "query": ex["query"],
                "context": ex["context"],
                "answer": answer,
            }

            out_buffer.append(out)
            produced += 1

        pending = []
        if len(out_buffer) >= buffer_flush_rows:
            shard_writer.write_rows(out_buffer)
            written += len(out_buffer)
            out_buffer = []
            save_state(
                args.state_path,
                rows_seen=rows_seen,
                written=written,
                neg_pool=neg_pool,
                extra={"missing_context_written": missing_context_written},
            )

    pbar = tqdm(
        total=args.max_samples,
        desc=f"writing examples (rank {rank})",
        unit="ex",
        initial=produced,
        disable=not accelerator.is_local_main_process,
    )

    try:
        for row in ds:
            if produced >= args.max_samples:
                break
            rows_seen += 1
            if skip_remaining > 0:
                skip_remaining -= 1
                continue

            if not valid_row(
                row=row,
                lang=args.lang,
                min_context_chars=args.min_context_chars,
                max_context_chars=args.max_context_chars,
                max_query_chars=args.max_query_chars,
            ):
                continue

            query = row["query"].strip()
            context = truncate_text(row["query_seed_text"], args.max_context_chars)
            url = row.get("query_seed_url") or ""
            seed_license = row.get("seed_license") or ""
            synth_id = row.get("synth_id") or ""

            # Always keep a positive candidate (may still become IDK if not answerable)
            # We'll run inference for positives.
            # Optionally add missing-context example
            if (
                args.missing_context_ratio > 0.0
                and missing_context_written < args.missing_context_max
                and random.random() < args.missing_context_ratio
            ):
                pending.append(
                    {
                        "synth_id": synth_id,
                        "query_seed_url": url,
                        "seed_license": seed_license,
                        "query": query,
                        "context": "",
                        "is_negative": True,
                        "is_missing_context": True,
                        "needs_infer": False,
                        "force_idk": False,
                        "prompt_text": "",
                        "answer": args.missing_context_answer,
                        "quotes": [],
                    }
                )
                missing_context_written += 1

            msg_prompt, _ = build_messages(
                query=query,
                context=context,
                answer=None,
                json_mode=args.use_json_mode,
                system_prompt=system_prompt,
                system_prompt_json=system_prompt_json,
                idk_text=IDK,
            )
            prompt_text = tokenizer.apply_chat_template(
                msg_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

            pending.append(
                {
                    "synth_id": synth_id,
                    "query_seed_url": url,
                    "seed_license": seed_license,
                    "query": query,
                    "context": context,
                    "is_negative": False,
                    "needs_infer": True,
                    "force_idk": False,
                    "prompt_text": prompt_text,
                    "answer": "",
                }
            )

            # Sometimes add a negative (mismatched context)
            if len(neg_pool) >= args.neg_pool_warmup and random.random() < args.neg_ratio:
                # sample a context from pool with a different URL when possible
                ctx_neg, url_neg = random.choice(list(neg_pool))
                tries = 0
                while tries < 10 and url_neg == url:
                    ctx_neg, url_neg = random.choice(list(neg_pool))
                    tries += 1

                ctx_neg = truncate_text(ctx_neg, args.max_context_chars)

                # For negatives, we generally *don't need* model inference; we force IDK
                pending.append(
                    {
                        "synth_id": synth_id,
                        "query_seed_url": url,
                        "seed_license": seed_license,
                        "query": query,
                        "context": ctx_neg,
                        "is_negative": True,
                        "needs_infer": False,  # change to True if you want model to generate IDK (slower, noisier)
                        "force_idk": True,
                        "prompt_text": "",
                        "answer": IDK,
                        "quotes": [],
                    }
                )

            # Update negative pool
            neg_pool.append((context, url))

            # Flush batches
            if len(pending) >= args.batch_size:
                flush_pending()
                pbar.n = produced
                pbar.refresh()
    finally:
        # final flush
        flush_pending()
        if out_buffer:
            shard_writer.write_rows(out_buffer)
            written += len(out_buffer)
            out_buffer = []
        shard_writer.close()
        pbar.n = produced
        pbar.refresh()
        pbar.close()
        save_state(
            args.state_path,
            rows_seen=rows_seen,
            written=written,
            neg_pool=neg_pool,
            extra={"missing_context_written": missing_context_written},
        )

    print(f"Done. Wrote {written} examples to {out_dir}")


if __name__ == "__main__":
    main()
