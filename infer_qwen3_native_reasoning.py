#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for Qwen3 native structured reasoning output.

Expected model output format:
  <|thinking_start|> ... <|thinking_end|>
  <|answer_start|> ... <|answer_end|>

Supports:
  - Single query/context inference
  - HotpotQA batch inference with JSONL output
"""

import argparse
import inspect
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
THINK_SPAN_RE = re.compile(
    r"<\|thinking_start\|>\s*(.*?)\s*<\|thinking_end\|>",
    re.IGNORECASE | re.DOTALL,
)
ANSWER_SPAN_RE = re.compile(
    r"<\|answer_start\|>\s*(.*?)\s*<\|answer_end\|>",
    re.IGNORECASE | re.DOTALL,
)
THINK_OPEN_RE = re.compile(r"<\|thinking_start\|>\s*(.*)$", re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<\|answer_start\|>\s*(.*)$", re.IGNORECASE | re.DOTALL)

# Backward-compatible fallbacks.
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:\-]\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


SYSTEM_PROMPT = """You are a precise and detail-oriented assistant tasked with answering factual questions based solely on the provided context.

1. Identify whether the question is bridge or comparison.
2. Extract exact entities and attributes from context.
3. Cross-reference evidence step by step.
4. Use only explicitly supported context details.
5. Do not use external knowledge.

Output format requirements (MANDATORY):
<|thinking_start|>concise reasoning based only on context<|thinking_end|>
<|answer_start|>exact final answer<|answer_end|>

Do not output anything after <|answer_end|>.
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Infer with Qwen3 using native thinking/answer token spans."
    )
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_model_len", type=int, default=8192)
    ap.add_argument("--vllm_dtype", type=str, default="auto")
    ap.add_argument("--num_shards", type=int, default=1, help="Dataset sharding across parallel workers.")
    ap.add_argument("--shard_index", type=int, default=0, help="Shard index in [0, num_shards).")
    ap.add_argument(
        "--distributed_output_mode",
        type=str,
        choices=["per_rank", "single"],
        default="per_rank",
        help="When num_shards>1, write per-rank outputs or force single output path.",
    )

    ap.add_argument("--query", type=str, default="", help="Single-query mode: question.")
    ap.add_argument("--context", type=str, default="", help="Single-query mode: context.")

    ap.add_argument("--dataset_name", type=str, default="hotpot_qa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--context_mode", type=str, choices=["all", "gold_titles", "supporting"], default="all")
    ap.add_argument("--max_context_chars", type=int, default=6000)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all samples in the selected shard.")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--enable_thinking", action="store_true", default=True)
    ap.add_argument("--disable_thinking", action="store_true")

    ap.add_argument("--output_jsonl", type=str, default="")
    ap.add_argument("--pretty_print", action="store_true")
    args = ap.parse_args()

    if args.disable_thinking:
        args.enable_thinking = False
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1 and args.use_vllm and args.vllm_tensor_parallel_size != 1:
        raise ValueError(
            "When launching multiple shard workers with --use_vllm, --vllm_tensor_parallel_size must be 1."
        )
    return args


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


def build_prompt_text(
    tokenizer: Any,
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
    model: Any,
    tokenizer: Any,
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
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else None,
        "top_p": top_p if temperature > 0.0 else None,
        "top_k": top_k if temperature > 0.0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if "min_p" in inspect.signature(model.generate).parameters:  # type: ignore[name-defined]
        gen_kwargs["min_p"] = min_p
    if temperature <= 0.0:
        gen_kwargs.pop("temperature")
        gen_kwargs.pop("top_p")
        gen_kwargs.pop("top_k")

    out = model.generate(**gen_kwargs)
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(gen, skip_special_tokens=False)


def generate_batch_vllm(
    llm: Any,
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
    return [out.outputs[0].text if out.outputs else "" for out in outputs]


def strip_meta_tokens(text: str) -> str:
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.replace("<s>", " ").replace("</s>", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_field(text: str) -> str:
    text = strip_meta_tokens(str(text or ""))
    text = text.replace("**", " ")
    return re.sub(r"\s+", " ", text).strip(" \t\n\r\"'`")


def parse_structured_output(raw_text: str) -> Dict[str, Any]:
    raw = str(raw_text or "")
    thinking = ""
    answer = ""
    parse_method = "none"

    think_match = THINK_SPAN_RE.search(raw)
    answer_match = ANSWER_SPAN_RE.search(raw)
    if think_match:
        thinking = think_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
        parse_method = "native_thinking_answer_tokens" if think_match else "native_answer_tokens"

    if not thinking:
        think_open = THINK_OPEN_RE.search(raw)
        if think_open:
            tail = think_open.group(1)
            end_pos = tail.find("<|answer_start|>")
            thinking = tail[:end_pos].strip() if end_pos >= 0 else tail.strip()
            if parse_method == "none":
                parse_method = "native_thinking_start_only"

    if not answer:
        answer_open = ANSWER_OPEN_RE.search(raw)
        if answer_open:
            answer = answer_open.group(1).strip()
            if parse_method == "none":
                parse_method = "native_answer_start_only"

    if not thinking:
        tag = THINK_TAG_RE.search(raw)
        if tag:
            thinking = tag.group(1).strip()
            if parse_method == "none":
                parse_method = "think_tag"

    if not answer:
        tag = ANSWER_TAG_RE.search(raw)
        if tag:
            answer = tag.group(1).strip()
            if parse_method == "none":
                parse_method = "answer_tag"

    if not answer:
        marker = ANSWER_MARKER_RE.search(raw)
        if marker:
            answer = marker.group(1).strip()
            if parse_method == "none":
                parse_method = "answer_marker"

    if not answer:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if lines:
            answer = lines[-1]
            if not thinking and len(lines) > 1:
                thinking = " ".join(lines[:-1])
            if parse_method == "none":
                parse_method = "last_line_fallback"

    thinking = normalize_field(thinking)
    answer = normalize_field(answer)
    parse_ok = bool(answer)

    return {
        "thinking": thinking,
        "answer": answer,
        "parse_ok": parse_ok,
        "parse_method": parse_method,
        "raw_output": raw,
    }


def with_rank_suffix_path(path: str, rank: int) -> str:
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".jsonl"
    return f"{root}.rank{rank}{ext}"


def resolve_output_jsonl_path(args: argparse.Namespace) -> str:
    base = args.output_jsonl.strip()
    if not base:
        os.makedirs("outputs", exist_ok=True)
        base = f"outputs/hotpot_native_reasoning_{args.split}.jsonl"
    if args.num_shards > 1 and args.distributed_output_mode == "per_rank":
        return with_rank_suffix_path(base, args.shard_index)
    return base


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.use_vllm:
        from vllm import LLM

        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            dtype=args.vllm_dtype,
        )
        return None, tokenizer, llm

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer, None


def infer_batch(
    records: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    llm: Any,
    args: argparse.Namespace,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for start in tqdm(
        range(0, len(records), args.batch_size),
        desc=f"infer rank {args.shard_index}",
        unit="batch",
        disable=not show_progress,
    ):
        batch = records[start : start + args.batch_size]
        prompts = [
            build_prompt_text(
                tokenizer=tokenizer,
                query=str(x["query"]),
                context=str(x["context"]),
                system_prompt=SYSTEM_PROMPT,
                enable_thinking=args.enable_thinking,
            )
            for x in batch
        ]

        if args.use_vllm:
            raws = generate_batch_vllm(
                llm=llm,
                prompt_texts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )
        else:
            raws = generate_batch_hf(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompts,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
            )

        for rec, raw in zip(batch, raws):
            parsed = parse_structured_output(raw)
            outputs.append(
                {
                    "query": rec["query"],
                    "context": rec["context"],
                    "reference_answer": rec.get("reference_answer", ""),
                    **parsed,
                }
            )
    return outputs


def main() -> None:
    args = parse_args()
    is_local_main = args.shard_index == 0
    model, tokenizer, llm = load_model_and_tokenizer(args)

    # Single query/context mode.
    if args.query.strip() and args.context.strip():
        rows = [{"query": args.query.strip(), "context": args.context.strip(), "reference_answer": ""}]
        out = infer_batch(
            rows,
            model=model,
            tokenizer=tokenizer,
            llm=llm,
            args=args,
            show_progress=True,
        )[0]
        if args.pretty_print:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(out, ensure_ascii=False))
        if args.output_jsonl:
            with open(args.output_jsonl, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        return

    # Dataset mode.
    ds: Dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.num_shards > 1:
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index)
    rows: List[Dict[str, Any]] = []
    for ex in ds:
        q = str(ex.get("question") or "").strip()
        c = build_hotpot_context(ex, mode=args.context_mode, max_context_chars=args.max_context_chars)
        if not q or not c:
            continue
        rows.append(
            {
                "query": q,
                "context": c,
                "reference_answer": str(ex.get("answer") or "").strip(),
            }
        )
        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break

    if not rows:
        raise RuntimeError("No rows to infer.")

    outputs = infer_batch(
        rows,
        model=model,
        tokenizer=tokenizer,
        llm=llm,
        args=args,
        show_progress=is_local_main,
    )
    parse_ok_count = sum(1 for x in outputs if x.get("parse_ok"))
    summary = {
        "total": len(outputs),
        "parse_ok": parse_ok_count,
        "parse_ok_rate": (parse_ok_count / len(outputs)) if outputs else 0.0,
        "rank": args.shard_index,
        "num_shards": args.num_shards,
        "split": args.split,
    }

    output_jsonl = resolve_output_jsonl_path(args)
    if output_jsonl:
        out_dir = os.path.dirname(output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as fh:
            for row in outputs:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Rank {args.shard_index}: wrote {len(outputs)} rows to {output_jsonl}")
    else:
        print(json.dumps(outputs[0], indent=2, ensure_ascii=False))

    summary_path = output_jsonl + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
