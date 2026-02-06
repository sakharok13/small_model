#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a dataset of:
  [query][CONTEXT][answer]
where answer is generated from context using Qwen3-4B-Instruct,
and sometimes CONTEXT is mismatched so answer should be "I don't know".

Output: JSONL with fields:
  - query, context, answer
  - is_negative (bool)
  - synth_id, query_seed_url, seed_license
  - text (chat formatted)
  - prompt_len (token length to mask so training loss applies only to answer tokens)

Requirements:
  pip install -U "transformers>=4.51.0" datasets accelerate torch tqdm
Qwen3 recommends transformers>=4.51.0.
"""

import argparse
import json
import os
import random
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


IDK = "I can't find the answer in the context"

SYSTEM_PROMPT = (
    "You are a strict question-answering assistant.\n"
    "Rules:\n"
    "1) Use ONLY the provided CONTEXT.\n"
    f"2) If the answer is not explicitly stated in CONTEXT, reply with EXACTLY: {IDK}\n"
    "3) Do not use outside knowledge. Do not guess.\n"
    "4) Reply with only the final answer text. No preamble.\n"
)

# Optional JSON mode for higher-quality filtering:
# This makes it easier to sanity-check that the model extracted an explicit quote.
USE_JSON_MODE_DEFAULT = True
SYSTEM_PROMPT_JSON = (
    "You are a dataset generator.\n"
    "You must answer the QUESTION using ONLY the provided CONTEXT.\n"
    f"If the answer is not explicitly stated in CONTEXT, set answer to EXACTLY: {IDK}\n"
    "Return STRICT JSON and nothing else with keys:\n"
    '  {"answer": "...", "quotes": ["verbatim quote 1", "verbatim quote 2"]}\n'
    f'If answer is "{IDK}", quotes must be [].\n'
)


def valid_row(
    row: Dict[str, Any],
    lang: str,
    min_context_chars: int,
    max_context_chars: int,
    max_query_chars: int,
) -> bool:
    """Filter SYNTH rows to ones suitable for context-grounded QA."""
    if row.get("language") != lang:
        return False
    q = (row.get("query") or "").strip()
    ctx = (row.get("query_seed_text") or "").strip()

    if not q or not ctx:
        return False

    if len(q) > max_query_chars:
        return False

    if len(ctx) < min_context_chars:
        return False

    # We'll truncate later if too long
    if len(ctx) > max_context_chars * 10:
        # Extremely long contexts are often noisy; drop them early
        return False

    return True


def truncate_text(s: str, max_chars: int) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    # Keep head; context usually has the needed answer near the beginning in SYNTH seeds
    return s[:max_chars].rstrip()


def build_user_content(query: str, context: str, json_mode: bool) -> str:
    if json_mode:
        return (
            "QUESTION:\n"
            f"{query}\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "Return JSON."
        )
    return (
        "QUESTION:\n"
        f"{query}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        f"Answer using only CONTEXT. If not in context, reply exactly: {IDK}"
    )


def build_messages(
    query: str,
    context: str,
    answer: Optional[str],
    json_mode: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Returns:
      - messages_prompt: system + user (no assistant answer)   [for inference prompt & prompt_len]
      - messages_full:   system + user + assistant(answer)     [for training text]
    """
    system = SYSTEM_PROMPT_JSON if json_mode else SYSTEM_PROMPT
    messages_prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_content(query, context, json_mode=json_mode)},
    ]
    messages_full = messages_prompt + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    return messages_prompt, messages_full


def extract_json_obj(s: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the first {...} JSON object from model output.
    Qwen may occasionally wrap; we keep it robust.
    """
    # Greedy match; output should be short
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


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
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for model loading")

    ap.add_argument("--max_samples", type=int, default=50_000, help="Max output examples (after filtering)")
    ap.add_argument("--neg_ratio", type=float, default=0.30, help="Probability to create a negative (mismatched context) example per row")
    ap.add_argument("--neg_pool_size", type=int, default=4096, help="Size of pool for sampling mismatched contexts")
    ap.add_argument("--neg_pool_warmup", type=int, default=256, help="Need this many contexts before sampling negatives")

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

    args = ap.parse_args()
    if args.no_json_mode:
        args.use_json_mode = False

    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
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
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()

    # Stream SYNTH (it's massive)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    neg_pool: Deque[Tuple[str, str]] = deque(maxlen=args.neg_pool_size)  # (context, url)
    written = 0

    # We'll accumulate inference jobs and flush in batches
    pending: List[Dict[str, Any]] = []

    def flush_pending(fh) -> None:
        nonlocal pending, written
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

                    if answer != IDK:
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

            # Build chat-formatted training text and prompt_len
            messages_prompt, messages_full = build_messages(
                query=ex["query"],
                context=ex["context"],
                answer=answer,
                json_mode=False,  # IMPORTANT: training data should be plain answer behavior
            )

            # prompt_len includes the assistant header (generation prompt)
            prompt_ids = tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_len = len(prompt_ids)

            full_text = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )

            out = {
                "synth_id": ex["synth_id"],
                "query_seed_url": ex["query_seed_url"],
                "seed_license": ex["seed_license"],
                "is_negative": ex["is_negative"],
                "query": ex["query"],
                "context": ex["context"],
                "answer": answer,
                # helpful debugging
                "quotes": quotes if args.use_json_mode else [],
                # for SFT
                "text": full_text,
                "prompt_len": prompt_len,
            }

            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

        pending = []

    with open(args.out_jsonl, "w", encoding="utf-8") as fh:
        pbar = tqdm(total=args.max_samples, desc="writing examples", unit="ex")

        for row in ds:
            if written >= args.max_samples:
                break

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
            msg_prompt, _ = build_messages(query=query, context=context, answer=None, json_mode=args.use_json_mode)
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
                flush_pending(fh)
                # update pbar by delta written in flush? easiest: set to written
                pbar.n = written
                pbar.refresh()

        # final flush
        flush_pending(fh)
        pbar.n = written
        pbar.refresh()
        pbar.close()

    print(f"Done. Wrote {written} examples to {args.out_jsonl}")


if __name__ == "__main__":
    main()
