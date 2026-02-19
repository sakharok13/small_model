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
  - raw_output
  - parse_ok
  - parse_method
"""

import argparse
import hashlib
import inspect
import json
import os
import random
import re
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from synth_utils import ParquetShardWriter, with_rank_suffix_dir


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
QWEN3_ANSWER_SPAN_RE = re.compile(r"<\|answer_start\|>\s*(.*?)\s*<\|answer_end\|>", re.IGNORECASE | re.DOTALL)
QWEN3_ANSWER_OPEN_RE = re.compile(r"<\|answer_start\|>\s*(.*)$", re.IGNORECASE | re.DOTALL)
PLAIN_ANSWER_SPAN_RE = re.compile(r"<answer_start>\s*(.*?)\s*<answer_end>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_TAG_RE = re.compile(r"<answer>\s*(.*)$", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_XML_RE = re.compile(r"###\s*<final answer>\s*(.*?)\s*</final answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_LINE_RE = re.compile(r"(?:^|\n)\s*final\s+answer\s*[:\-]\s*(.+)", re.IGNORECASE)
STANDALONE_BOLD_LINE_RE = re.compile(r"(?:^|\n)\s*\*\*(.+?)\*\*\s*(?=\n|$)", re.DOTALL)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)

IDK_DEFAULT = "I can't find the answer in the context."

PROMPT_V4 = """You are a precise and detail-oriented assistant tasked with answering factual questions based solely on the provided context. Follow these steps:

1. **Identify the Task Type**: Determine if the question is a *comparison* (e.g., "Which has more X?") or *bridge* (e.g., "X is part of Y, which is linked to Z?"). For *bridge* tasks, map relationships between entities step-by-step (e.g., show -> original series -> air date). For *comparison* tasks, focus on quantifiable or definitional attributes.

2. **Extract Key Entities**: Highlight the main subjects (e.g., "Katniss Everdeen," "Het Huis Anubis," "The Chimes") and their attributes (e.g., age, release year, album titles) from the context. Use **exact names** (e.g., "The Chimes" vs. "Chimes") to avoid ambiguity.

3. **Cross-Reference Information**:
   - For *bridge* tasks, connect disparate pieces of information (e.g., link a TV show to its source material, a character to their age).
   - For *comparison* tasks, directly compare attributes (e.g., album release years, character ages) using explicit data from the context.

4. **Prioritize Exact Matches**: Use **precisely stated details** from the context (e.g., "September 2006" for "Het Huis Anubis") and avoid assumptions. Verify that all claims are explicitly stated or logically inferred (e.g., Katniss's age remains 16 in *Catching Fire* as no new age is mentioned).

5. **Conclude with a Clear Answer**: For *bridge* tasks, ensure the chain of relationships is complete (e.g., "The Dutch-Belgian series 'Het Huis Anubis' first aired in 2006"). For *comparison* tasks, explicitly state the result (e.g., "The song was from the album *The Chimes*").

**Domain-Specific Knowledge to Include**:
- **TV/Film**: Recognize adaptations (e.g., *House of Anubis* is based on *Het Huis Anubis*), note air dates, and distinguish between original works and remakes.
- **Music**: Identify albums, cover songs, and their original sources (e.g., The Chimes' cover of U2's song is on their album *The Chimes*).
- **Literature**: Use character ages from novels (e.g., Katniss Everdeen is 16 in *The Hunger Games*).

**Example**: If asked, "Which album was Pauline Henry's cover of U2's song from?" extract the album name (*The Chimes*) from the context, ensuring no confusion with the original song's album (*The Joshua Tree*). For a *bridge* task like "The Dutch-Belgian series *House of Anubis* was based on what show, which first aired in what year?" link the series to *Het Huis Anubis* and use its air date (2006).

**Final Answer Format (MANDATORY)**:
- Output your final answer using Qwen3 native boundary tokens exactly once:
<|answer_start|>your exact final answer<|answer_end|>
- Put nothing after `<|answer_end|>`.
- Keep the answer span concise, with no extra explanation inside the span.
- Do not use external knowledge or assumptions."""

ALL_VERSIONS: Tuple[str] = ("v4",)


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
        "--tasks_jsonl",
        type=str,
        default="",
        help=(
            "Optional precomputed task file (JSONL). When set, dataset loading/sharding "
            "is bypassed and each line must include query/context/reference_answer/prompt_version."
        ),
    )
    ap.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Manual dataset sharding for simple multi-process vLLM launch.",
    )
    ap.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Manual shard index in [0, num_shards).",
    )
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
        choices=["v4"],
        default="v4",
        help="Only v4 is supported for Hotpot reasoning trace generation.",
    )
    ap.add_argument("--idk_text", type=str, default=IDK_DEFAULT)
    ap.add_argument("--max_thinking_words", type=int, default=80)
    ap.add_argument("--allow_empty_thinking", action="store_true")
    ap.add_argument(
        "--fallback_to_reference_answer",
        action="store_true",
        default=True,
        help="If parsing fails to extract an answer, use reference_answer instead of dropping the row.",
    )
    ap.add_argument(
        "--no_fallback_to_reference_answer",
        dest="fallback_to_reference_answer",
        action="store_false",
        help="Disable fallback to reference_answer on parse failures.",
    )
    ap.add_argument(
        "--parse_failures_jsonl",
        type=str,
        default="",
        help="Optional path for JSONL parse-failure logs (defaults to <out_dir>/parse_failures.jsonl).",
    )

    ap.add_argument("--enable_thinking", action="store_true", default=True)
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--min_p", type=float, default=0.0, help="Qwen3 thinking default is 0")

    ap.add_argument("--shard_size", type=int, default=50_000)
    ap.add_argument("--save_every", type=int, default=5_000)
    ap.add_argument("--parquet_compression", type=str, default="zstd")
    ap.add_argument(
        "--skip_existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help=(
            "When output part-*.parquet files already exist, skip already-generated "
            "tasks using (query, context, prompt_version, reference_answer) keys."
        ),
    )
    ap.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Disable resume/skip behavior and append everything.",
    )

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
    if args.max_samples < 0:
        raise ValueError("--max_samples must be >= 0")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
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


def make_task_key(query: Any, context: Any, prompt_version: Any, reference_answer: Any) -> str:
    h = hashlib.blake2b(digest_size=16)
    for part in (query, context, prompt_version, reference_answer):
        text = str("" if part is None else part)
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def scan_existing_outputs(out_dir: str) -> Tuple[int, Set[str], Dict[str, int]]:
    files = sorted(glob(os.path.join(out_dir, "part-*.parquet")))
    if not files:
        return 0, set(), {"v4": 0}
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to scan existing parquet shards for resume.") from exc

    total_rows_on_disk = 0
    keys: Set[str] = set()
    counts_by_version: Dict[str, int] = {"v4": 0}

    for path in files:
        pf = pq.ParquetFile(path)
        total_rows_on_disk += pf.metadata.num_rows
        names = set(pf.schema_arrow.names)
        if "query" not in names or "context" not in names:
            continue
        read_cols = ["query", "context"]
        if "prompt_version" in names:
            read_cols.append("prompt_version")
        if "reference_answer" in names:
            read_cols.append("reference_answer")
        table = pq.read_table(path, columns=read_cols)
        data = table.to_pydict()

        queries = data.get("query") or []
        contexts = data.get("context") or []
        n = min(len(queries), len(contexts))
        prompt_versions = data.get("prompt_version")
        if prompt_versions is None:
            prompt_versions = [""] * n
        reference_answers = data.get("reference_answer")
        if reference_answers is None:
            reference_answers = [""] * n

        for i in range(n):
            pv = prompt_versions[i] if i < len(prompt_versions) else ""
            ra = reference_answers[i] if i < len(reference_answers) else ""
            key = make_task_key(
                query=queries[i],
                context=contexts[i],
                prompt_version=pv,
                reference_answer=ra,
            )
            if key in keys:
                continue
            keys.add(key)
            if pv in counts_by_version:
                counts_by_version[pv] += 1

    return total_rows_on_disk, keys, counts_by_version


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


def load_tasks_from_jsonl(path: str) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    if not path:
        return tasks
    if not os.path.exists(path):
        raise FileNotFoundError(f"--tasks_jsonl file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc

            query = str(row.get("query") or "").strip()
            context = str(row.get("context") or "").strip()
            reference_answer = str(row.get("reference_answer") or "").strip()
            prompt_version = "v4"
            if not query or not context:
                continue
            task_key = make_task_key(
                query=query,
                context=context,
                prompt_version=prompt_version,
                reference_answer=reference_answer,
            )

            tasks.append(
                {
                    "query": query,
                    "context": context,
                    "reference_answer": reference_answer,
                    "prompt_version": prompt_version,
                    "task_key": task_key,
                }
            )
    return tasks


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
    _ = idk_text
    if version == "v4":
        return PROMPT_V4
    raise ValueError(f"Unknown prompt version: {version}")


def pick_prompt_versions(configured: str, rng: random.Random) -> List[str]:
    _ = configured
    _ = rng
    return ["v4"]


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
    text = text.replace("**", " ")
    return re.sub(r"\s+", " ", text).strip()


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text.strip()
    toks = text.strip().split()
    if len(toks) <= max_words:
        return " ".join(toks)
    return " ".join(toks[:max_words]).strip()


def linewise_tail(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize_answer_text(answer: str) -> str:
    if not answer:
        return ""
    bold_spans = list(BOLD_RE.finditer(answer))
    if bold_spans:
        answer = bold_spans[-1].group(1)
    answer = clean_content(answer)
    if not answer:
        return ""
    answer = re.sub(r"^#+\s*", "", answer)
    answer = re.sub(r"^\s*(?:answer|final answer)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.replace("**", " ").strip()
    # Keep only the final line when model spills multi-line thoughts into answer.
    answer = linewise_tail(answer) or answer
    answer = answer.strip(" \t\n\r\"'`")
    if re.match(r"^(facts|bridge|check)\s*:", answer, flags=re.IGNORECASE):
        return ""
    return answer


def parse_reasoning_output(raw_text: str, max_thinking_words: int) -> Tuple[str, str, bool, str]:
    raw = str(raw_text).replace("<s>", " ").replace("</s>", " ").strip()
    text = SPECIAL_TOKEN_RE.sub(" ", raw).strip()
    think_match = THINK_RE.search(text)
    answer_match = ANSWER_RE.search(text)

    thinking = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    parse_method = "answer_tag" if answer_match else ""
    answer_start: Optional[int] = answer_match.start() if answer_match else None

    qwen_match = QWEN3_ANSWER_SPAN_RE.search(raw)
    if qwen_match:
        answer = qwen_match.group(1).strip()
        parse_method = "qwen3_answer_tokens"
        answer_start = None
        if not thinking:
            thinking = clean_content(raw[: qwen_match.start()].strip())

    if not answer:
        plain_span = PLAIN_ANSWER_SPAN_RE.search(raw)
        if plain_span:
            answer = plain_span.group(1).strip()
            parse_method = "plain_answer_tokens"
            answer_start = None
            if not thinking:
                thinking = clean_content(raw[: plain_span.start()].strip())

    if not answer:
        qwen_open = QWEN3_ANSWER_OPEN_RE.search(raw)
        if qwen_open:
            answer = qwen_open.group(1).strip()
            parse_method = "qwen3_answer_start_only"
            answer_start = None
            if not thinking:
                thinking = clean_content(raw[: qwen_open.start()].strip())

    if not answer and think_match:
        tail = text[think_match.end() :].strip()
        open_tag = ANSWER_OPEN_TAG_RE.search(tail)
        if open_tag:
            answer = open_tag.group(1).strip()
            parse_method = "answer_open_tag_tail"
            answer_start = think_match.end() + open_tag.start()
        else:
            marker = ANSWER_MARKER_RE.search(tail)
            if marker:
                answer = marker.group(1).strip()
                parse_method = "answer_marker_tail"
                answer_start = think_match.end() + marker.start()
            else:
                answer = linewise_tail(tail)
                if answer:
                    parse_method = "tail_last_line"
                    answer_start = think_match.end()

    if not answer:
        xml_match = FINAL_ANSWER_XML_RE.search(text)
        if xml_match:
            answer = xml_match.group(1).strip()
            parse_method = "final_answer_xml"
            answer_start = xml_match.start()

    if not answer:
        final_line_matches = list(FINAL_ANSWER_LINE_RE.finditer(text))
        if final_line_matches:
            last_match = final_line_matches[-1]
            answer = last_match.group(1).strip()
            parse_method = "final_answer_line"
            answer_start = last_match.start()

    if not thinking and answer_match:
        prefix = text[: answer_match.start()].strip()
        prefix = re.sub(r"<think>", " ", prefix, flags=re.IGNORECASE)
        prefix = clean_content(prefix)
        if prefix:
            thinking = prefix

    if not answer and not answer_match:
        marker = ANSWER_MARKER_RE.search(text)
        if marker:
            answer = marker.group(1).strip()
            parse_method = "answer_marker"
            answer_start = marker.start()
            if not thinking:
                prefix = text[: marker.start()].strip()
                thinking = clean_content(prefix)
        else:
            bold_line_matches = list(STANDALONE_BOLD_LINE_RE.finditer(text))
            if bold_line_matches:
                last_bold = bold_line_matches[-1]
                answer = last_bold.group(1).strip()
                parse_method = "standalone_bold_line"
                answer_start = last_bold.start()
            else:
                bold_matches = list(BOLD_RE.finditer(text))
                if bold_matches:
                    last_bold = bold_matches[-1]
                    answer = last_bold.group(1).strip()
                    parse_method = "last_bold_span"
                    answer_start = last_bold.start()

        if not answer:
            # Last-resort split: use last non-empty line as answer.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                answer = lines[-1]
                parse_method = "last_line_fallback"
                if not thinking and len(lines) > 1:
                    thinking = " ".join(lines[:-1])

    if not thinking and answer_start is not None and answer_start > 0:
        thinking = clean_content(text[:answer_start].strip())

    thinking = clean_content(thinking)
    answer = normalize_answer_text(answer)
    if thinking and answer and thinking.lower() == answer.lower():
        answer = ""
        parse_method = ""
    thinking = truncate_words(thinking, max_words=max_thinking_words)
    return thinking, answer, bool(answer), parse_method or "unknown"


def thinking_cap_for_version(version: str, args: argparse.Namespace) -> int:
    _ = version
    return args.max_thinking_words


def contains_answer_substring(raw_text: str, answer: str) -> bool:
    expected = str(answer or "").strip().lower()
    if not expected:
        return False
    return expected in strip_meta_tokens(str(raw_text or "")).lower()


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stats(out_dir: str, stats: Dict[str, Any]) -> None:
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()

    accelerator = None
    if args.use_vllm:
        world = args.num_shards
        rank = args.shard_index
        is_local_main = rank == 0
        if world > 1 and args.vllm_tensor_parallel_size != 1:
            raise ValueError(
                "When launching multiple processes with --use_vllm, "
                "--vllm_tensor_parallel_size must be 1."
            )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if is_local_main:
            gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            print(f"[vLLM simple multi-proc] world={world} rank={rank} CUDA_VISIBLE_DEVICES={gpu_env}")
    else:
        from accelerate import Accelerator

        accelerator = Accelerator()
        world = accelerator.num_processes
        rank = accelerator.process_index
        is_local_main = accelerator.is_local_main_process

    rng = random.Random(args.seed + rank)

    out_dir = args.out_dir
    if world > 1 and args.distributed_output_mode == "per_rank":
        out_dir = with_rank_suffix_dir(args.out_dir, rank)
    os.makedirs(out_dir, exist_ok=True)

    task_mode = bool(args.tasks_jsonl)
    examples: List[Dict[str, Any]] = []
    assigned_tasks: List[Dict[str, str]] = []
    if task_mode:
        assigned_tasks = load_tasks_from_jsonl(args.tasks_jsonl)
        target_rows = len(assigned_tasks)
        local_max = target_rows
        if is_local_main:
            print(f"Rank {rank}: loaded {target_rows} preassigned tasks from {args.tasks_jsonl}")
    else:
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
        rows_per_example = 1
        target_rows = local_max * rows_per_example

    if target_rows <= 0:
        print(f"Rank {rank}: nothing to do (target_rows=0).")
        return

    existing_rows_on_disk = 0
    completed_keys: Set[str] = set()
    existing_version_counts: Dict[str, int] = {"v4": 0}
    if args.skip_existing:
        existing_rows_on_disk, completed_keys, existing_version_counts = scan_existing_outputs(out_dir)
        if is_local_main and existing_rows_on_disk > 0:
            print(
                f"Rank {rank}: found {existing_rows_on_disk} rows on disk, "
                f"{len(completed_keys)} unique completed tasks. Resuming remaining work."
            )

    produced = min(len(completed_keys), target_rows)
    initial_produced = produced
    if produced >= target_rows:
        stats = {
            "task_mode": task_mode,
            "tasks_jsonl": args.tasks_jsonl if task_mode else "",
            "global_max_samples": args.max_samples,
            "local_examples_target": local_max,
            "target_rows": target_rows,
            "produced_total": produced,
            "produced_new": 0,
            "existing_rows_on_disk_at_start": existing_rows_on_disk,
            "existing_unique_rows_at_start": initial_produced,
            "completed_unique_rows": produced,
            "prompt_version_config": args.prompt_version,
            "produced_by_prompt_version": existing_version_counts,
            "skipped_existing_task": 0,
            "skipped_parse": 0,
            "skipped_empty_answer": 0,
            "fallback_reference_answer_used": 0,
            "parse_failures_logged": 0,
            "parse_failures_jsonl": args.parse_failures_jsonl.strip()
            or os.path.join(out_dir, "parse_failures.jsonl"),
            "rank": rank,
            "world": world,
            "args": vars(args),
        }
        write_stats(out_dir, stats)
        print(f"Done. Wrote Hotpot reasoning traces to {out_dir}")
        return

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

    writer = ParquetShardWriter(
        out_dir=out_dir,
        shard_size=args.shard_size,
        compression=args.parquet_compression,
    )

    pending: List[Dict[str, Any]] = []
    out_buffer: List[Dict[str, Any]] = []
    buffer_flush_rows = args.save_every if args.save_every > 0 else args.shard_size
    version_counts: Dict[str, int] = dict(existing_version_counts)

    skipped_parse = 0
    skipped_empty = 0
    fallback_reference_answer = 0
    skipped_existing = 0
    parse_failures_logged = 0
    parse_failures_jsonl = args.parse_failures_jsonl.strip() or os.path.join(out_dir, "parse_failures.jsonl")
    pbar = tqdm(
        total=target_rows,
        desc=f"hotpot reasoning traces (rank {rank})",
        unit="ex",
        disable=not is_local_main,
    )
    pbar.n = produced
    pbar.refresh()

    def flush_pending() -> None:
        nonlocal pending, out_buffer, produced, skipped_parse, skipped_empty
        nonlocal fallback_reference_answer, skipped_existing, parse_failures_logged
        if not pending or produced >= target_rows:
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
            if produced >= target_rows:
                break
            if ex["task_key"] in completed_keys:
                skipped_existing += 1
                continue
            max_words = thinking_cap_for_version(ex["prompt_version"], args)
            thinking, answer, parse_ok, parse_method = parse_reasoning_output(
                raw_text=raw,
                max_thinking_words=max_words,
            )
            parse_ok_effective = parse_ok
            if not answer:
                if contains_answer_substring(raw_text=raw, answer=ex["reference_answer"]):
                    answer = str(ex["reference_answer"]).strip()
                    parse_method = "reference_substring_match"
                    parse_ok_effective = False
                elif args.fallback_to_reference_answer and ex["reference_answer"]:
                    answer = str(ex["reference_answer"]).strip()
                    fallback_reference_answer += 1
                    parse_method = "fallback_reference_answer"
                    parse_ok_effective = False
                else:
                    skipped_empty += 1
                    parse_failures_logged += 1
                    append_jsonl(
                        parse_failures_jsonl,
                        {
                            "reason": "empty_answer",
                            "query": ex["query"],
                            "context": ex["context"],
                            "reference_answer": ex["reference_answer"],
                            "prompt_version": ex["prompt_version"],
                            "parse_method": parse_method,
                            "raw_output": str(raw),
                            "parsed_thinking": thinking,
                            "parsed_answer": answer,
                        },
                    )
                    continue

            if not thinking and not args.allow_empty_thinking:
                skipped_parse += 1
                parse_failures_logged += 1
                append_jsonl(
                    parse_failures_jsonl,
                    {
                        "reason": "empty_thinking",
                        "query": ex["query"],
                        "context": ex["context"],
                        "reference_answer": ex["reference_answer"],
                        "prompt_version": ex["prompt_version"],
                        "parse_method": parse_method,
                        "raw_output": str(raw),
                        "parsed_thinking": thinking,
                        "parsed_answer": answer,
                    },
                )
                continue

            if not thinking:
                parse_ok_effective = False

            row = {
                "query": ex["query"],
                "context": ex["context"],
                "thinking": thinking,
                "answer": answer,
                "prompt_version": ex["prompt_version"],
                "reference_answer": ex["reference_answer"],
                "raw_output": str(raw),
                "parse_ok": bool(parse_ok_effective),
                "parse_method": parse_method,
            }
            out_buffer.append(row)
            completed_keys.add(ex["task_key"])
            produced += 1
            version_counts[ex["prompt_version"]] = version_counts.get(ex["prompt_version"], 0) + 1

        pending = []
        if len(out_buffer) >= buffer_flush_rows:
            writer.write_rows(out_buffer)
            out_buffer = []

    try:
        if task_mode:
            for task in assigned_tasks:
                if produced >= target_rows:
                    break
                task_key = task["task_key"]
                if task_key in completed_keys:
                    skipped_existing += 1
                    continue

                version = task["prompt_version"]
                system_prompt = system_prompt_for_version(version, idk_text=args.idk_text)
                prompt_text = build_prompt_text(
                    tokenizer=tokenizer,
                    query=task["query"],
                    context=task["context"],
                    system_prompt=system_prompt,
                    enable_thinking=args.enable_thinking,
                )

                pending.append(
                    {
                        "query": task["query"],
                        "context": task["context"],
                        "reference_answer": task["reference_answer"],
                        "prompt_version": version,
                        "prompt_text": prompt_text,
                        "task_key": task_key,
                    }
                )

                if len(pending) >= args.batch_size:
                    flush_pending()
                    pbar.n = produced
                    pbar.refresh()
        else:
            for ex in examples:
                if produced >= target_rows:
                    break

                versions = pick_prompt_versions(args.prompt_version, rng=rng)
                for version in versions:
                    if produced >= target_rows:
                        break
                    task_key = make_task_key(
                        query=ex["query"],
                        context=ex["context"],
                        prompt_version=version,
                        reference_answer=ex["reference_answer"],
                    )
                    if task_key in completed_keys:
                        skipped_existing += 1
                        continue
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
                            "task_key": task_key,
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
        "task_mode": task_mode,
        "tasks_jsonl": args.tasks_jsonl if task_mode else "",
        "global_max_samples": args.max_samples,
        "local_examples_target": local_max,
        "target_rows": target_rows,
        "produced_total": produced,
        "produced_new": max(0, produced - initial_produced),
        "existing_rows_on_disk_at_start": existing_rows_on_disk,
        "existing_unique_rows_at_start": initial_produced,
        "completed_unique_rows": len(completed_keys),
        "prompt_version_config": args.prompt_version,
        "produced_by_prompt_version": version_counts,
        "skipped_existing_task": skipped_existing,
        "skipped_parse": skipped_parse,
        "skipped_empty_answer": skipped_empty,
        "fallback_reference_answer_used": fallback_reference_answer,
        "parse_failures_logged": parse_failures_logged,
        "parse_failures_jsonl": parse_failures_jsonl,
        "rank": rank,
        "world": world,
        "args": vars(args),
    }
    write_stats(out_dir, stats)
    print(f"Done. Wrote Hotpot reasoning traces to {out_dir}")


if __name__ == "__main__":
    main()
