#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
import random
import re
import string
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark a finetuned checkpoint on HotpotQA using the same prompt structure as training."
    )
    ap.add_argument("--model_path", type=str, required=True, help="Path/name of finetuned checkpoint (or adapter path if --base_model is set)")
    ap.add_argument("--base_model", type=str, default=None, help="Base model path/name when model_path is a LoRA adapter")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer only from local cache/files")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])

    ap.add_argument("--dataset_name", type=str, default="hotpot_qa")
    ap.add_argument("--dataset_config", type=str, default="distractor")
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all samples")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--context_mode",
        type=str,
        default="all",
        choices=["all", "gold_titles", "supporting"],
        help=(
            "all: all context documents/sentences from sample; "
            "gold_titles: all sentences from supporting-fact titles; "
            "supporting: only supporting-fact sentences."
        ),
    )
    ap.add_argument("--max_context_chars", type=int, default=6000, help="Hard cap for context chars (0 disables cap)")

    ap.add_argument(
        "--prompt_template",
        type=str,
        default="[context_start]{context}[context_end][query_start]{query}[query_end][answer_start]",
        help="Must match training prompt structure.",
    )
    ap.add_argument("--answer_end", type=str, default="[answer_end]", help="Answer end marker token/string")
    ap.add_argument("--max_prompt_tokens", type=int, default=1800, help="Prompt truncation limit before generation")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--output_dir", type=str, default="outputs/hotpotqa_benchmark")
    ap.add_argument("--predictions_name", type=str, default="predictions.jsonl")
    ap.add_argument("--metrics_name", type=str, default="metrics.json")
    return ap.parse_args()


def get_torch_dtype(dtype: str) -> Any:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return "auto"


def build_model_kwargs(torch_dtype: Any, trust_remote_code: bool, local_files_only: bool) -> Dict[str, Any]:
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if torch_dtype != "auto":
        if "dtype" in sig.parameters:
            kwargs["dtype"] = torch_dtype
        else:
            kwargs["torch_dtype"] = torch_dtype
    return kwargs


def load_model_and_tokenizer(args: argparse.Namespace):
    torch_dtype = get_torch_dtype(args.dtype)
    model_kwargs = build_model_kwargs(
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    if args.base_model:
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=args.device_map,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base, args.model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=args.device_map,
            **model_kwargs,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


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


def compact_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def format_doc(title: str, sentences: Sequence[str]) -> str:
    text = " ".join(x.strip() for x in sentences if str(x).strip())
    text = compact_whitespace(text)
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
    else:  # supporting
        by_title: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for title, sents in docs:
            for i, sent in enumerate(sents):
                if (title, i) in sf_pairs:
                    by_title[title].append((i, sent))
        for title, sents in docs:
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


def build_prompt(query: str, context: str, template: str) -> str:
    return template.format(query=query, context=context)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        table = str.maketrans("", "", string.punctuation)
        return s.translate(table)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def answer_em(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    answer_end: str,
) -> List[str]:
    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )
    device = model.device
    enc = {k: v.to(device) for k, v in enc.items()}

    eos_ids = [tokenizer.eos_token_id]
    answer_end_id = tokenizer.convert_tokens_to_ids(answer_end) if answer_end else None
    if isinstance(answer_end_id, int) and answer_end_id >= 0 and answer_end_id != tokenizer.unk_token_id:
        eos_ids.append(answer_end_id)
    eos_ids = list(dict.fromkeys([x for x in eos_ids if x is not None]))
    eos_arg: Any = eos_ids[0] if len(eos_ids) == 1 else eos_ids

    gen_kwargs: Dict[str, Any] = {
        **enc,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_arg,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    out = model.generate(**gen_kwargs)

    preds: List[str] = []
    attn = enc["attention_mask"]
    for i in range(out.size(0)):
        in_len = int(attn[i].sum().item())
        gen_ids = out[i, in_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if answer_end and answer_end in text:
            text = text.split(answer_end)[0].strip()
        preds.append(text)
    return preds


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def select_dataset_slice(ds: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_dir(args.output_dir)

    model, tokenizer = load_model_and_tokenizer(args)

    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    ds = select_dataset_slice(ds, max_samples=args.max_samples, seed=args.seed)

    preds_rows: List[Dict[str, Any]] = []
    sum_em = 0.0
    sum_f1 = 0.0
    n = 0

    by_type: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "em": 0.0, "f1": 0.0})
    by_level: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "em": 0.0, "f1": 0.0})

    total = len(ds)
    for start in tqdm(range(0, total, args.batch_size), desc="HotpotQA eval", unit="batch"):
        batch = ds[start : min(start + args.batch_size, total)]
        questions = batch.get("question", [])
        golds = batch.get("answer", [])
        ids = batch.get("id", [None] * len(questions))
        types = batch.get("type", [None] * len(questions))
        levels = batch.get("level", [None] * len(questions))

        prompts: List[str] = []
        contexts: List[str] = []
        for i, question in enumerate(questions):
            ex = {k: v[i] for k, v in batch.items()}
            context = build_hotpot_context(
                ex,
                mode=args.context_mode,
                max_context_chars=args.max_context_chars,
            )
            contexts.append(context)
            prompts.append(build_prompt(query=str(question), context=context, template=args.prompt_template))

        predictions = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            answer_end=args.answer_end,
        )

        for i, pred in enumerate(predictions):
            gold = str(golds[i]) if i < len(golds) else ""
            em = answer_em(prediction=pred, gold=gold)
            f1 = answer_f1(prediction=pred, gold=gold)
            sum_em += em
            sum_f1 += f1
            n += 1

            typ = str(types[i]) if i < len(types) and types[i] is not None else "unknown"
            lvl = str(levels[i]) if i < len(levels) and levels[i] is not None else "unknown"
            by_type[typ]["count"] += 1
            by_type[typ]["em"] += em
            by_type[typ]["f1"] += f1
            by_level[lvl]["count"] += 1
            by_level[lvl]["em"] += em
            by_level[lvl]["f1"] += f1

            preds_rows.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "question": questions[i] if i < len(questions) else "",
                    "gold_answer": gold,
                    "prediction": pred,
                    "em": em,
                    "f1": f1,
                    "type": typ,
                    "level": lvl,
                    "context_chars": len(contexts[i]) if i < len(contexts) else 0,
                    "context_mode": args.context_mode,
                }
            )

    def finalize_breakdown(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for key, v in sorted(stats.items(), key=lambda kv: kv[0]):
            cnt = max(1.0, v["count"])
            out[key] = {
                "count": int(v["count"]),
                "exact_match": 100.0 * (v["em"] / cnt),
                "f1": 100.0 * (v["f1"] / cnt),
            }
        return out

    metrics = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "num_samples": n,
        "context_mode": args.context_mode,
        "max_context_chars": args.max_context_chars,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "exact_match": 100.0 * (sum_em / max(1, n)),
        "f1": 100.0 * (sum_f1 / max(1, n)),
        "by_type": finalize_breakdown(by_type),
        "by_level": finalize_breakdown(by_level),
        "model_path": args.model_path,
        "base_model": args.base_model,
    }

    pred_path = os.path.join(args.output_dir, args.predictions_name)
    metrics_path = os.path.join(args.output_dir, args.metrics_name)
    write_jsonl(pred_path, preds_rows)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
