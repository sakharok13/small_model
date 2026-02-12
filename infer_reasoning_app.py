#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from typing import Optional, Tuple

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Path to finetuned checkpoint or adapter")
    ap.add_argument("--base_model", type=str, default=None, help="Base model if model_path is a LoRA adapter")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")

    ap.add_argument("--context_start_token", type=str, default="[context_start]")
    ap.add_argument("--context_end_token", type=str, default="[context_end]")
    ap.add_argument("--query_start_token", type=str, default="[query_start]")
    ap.add_argument("--query_end_token", type=str, default="[query_end]")
    ap.add_argument("--thinking_start_token", type=str, default="[thinking_start]")
    ap.add_argument("--thinking_end_token", type=str, default="[thinking_end]")
    ap.add_argument("--answer_start_token", type=str, default="[answer_start]")
    ap.add_argument("--answer_end_token", type=str, default="[answer_end]")
    return ap.parse_args()


def build_prompt(query: str, context: str, args: argparse.Namespace) -> str:
    return (
        f"{args.context_start_token}{context}{args.context_end_token}"
        f"{args.query_start_token}{query}{args.query_end_token}"
    )


def _clean_text(text: str) -> str:
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.replace("<s>", " ").replace("</s>", " ")
    return re.sub(r"\s+", " ", text).strip()


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        return text[i + len(start) :].strip()
    return text[i + len(start) : j].strip()


def parse_generation(raw: str, args: argparse.Namespace) -> Tuple[str, str]:
    thinking = _between(raw, args.thinking_start_token, args.thinking_end_token)
    answer = _between(raw, args.answer_start_token, args.answer_end_token)

    if not thinking:
        # Fallback when [thinking_end] is missing but [answer_start] exists.
        i_think = raw.find(args.thinking_start_token)
        i_ans = raw.find(args.answer_start_token)
        if i_think >= 0 and i_ans > i_think:
            thinking = raw[i_think + len(args.thinking_start_token) : i_ans].strip()

    if not answer:
        # Fallback when [answer_end] is missing.
        i_ans = raw.find(args.answer_start_token)
        if i_ans >= 0:
            answer = raw[i_ans + len(args.answer_start_token) :].strip()

    return _clean_text(thinking), _clean_text(answer)


def load_model(
    model_path: str,
    base_model: Optional[str],
    dtype: str,
    trust_remote_code: bool,
    device_map: str,
):
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    if base_model:
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    answer_end_token: str,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    eos_ids = [tokenizer.eos_token_id]
    answer_end_id = tokenizer.convert_tokens_to_ids(answer_end_token) if answer_end_token else None
    if isinstance(answer_end_id, int) and answer_end_id >= 0 and answer_end_id != tokenizer.unk_token_id:
        eos_ids.append(answer_end_id)
    eos_ids = [x for x in dict.fromkeys(eos_ids) if x is not None]

    gen_kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0.0 else None,
        top_p=top_p if temperature > 0.0 else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids if len(eos_ids) > 1 else eos_ids[0],
    )
    if temperature <= 0.0:
        gen_kwargs.pop("temperature")
        gen_kwargs.pop("top_p")

    out = model.generate(**gen_kwargs)
    gen = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(
        model_path=args.model_path,
        base_model=args.base_model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )

    def infer(query: str, context: str, max_new_tokens: int, temperature: float, top_p: float):
        query = (query or "").strip()
        context = (context or "").strip()
        if not query:
            return "", "", "Query is empty.", ""
        if not context:
            return "", "", "Context is empty.", ""

        prompt = build_prompt(query=query, context=context, args=args)
        raw = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            answer_end_token=args.answer_end_token,
        )
        thinking, answer = parse_generation(raw=raw, args=args)
        return thinking, answer, raw, prompt

    with gr.Blocks() as demo:
        gr.Markdown("# Reasoning Inference (Query + Context -> Thinking + Answer)")
        gr.Markdown(
            "Prompt format: "
            f"`{args.context_start_token}context{args.context_end_token}` + "
            f"`{args.query_start_token}query{args.query_end_token}`"
        )

        with gr.Row():
            query = gr.Textbox(label="Query", lines=2, placeholder="Enter question")
            context = gr.Textbox(label="Context", lines=12, placeholder="Paste context")

        with gr.Row():
            max_new_tokens = gr.Slider(64, 4096, value=args.max_new_tokens, step=32, label="max_new_tokens")
            temperature = gr.Slider(0.0, 1.2, value=args.temperature, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=args.top_p, step=0.05, label="top_p")

        run_btn = gr.Button("Generate")

        with gr.Row():
            thinking = gr.Textbox(label="Thinking", lines=10)
            answer = gr.Textbox(label="Answer", lines=4)
        raw = gr.Textbox(label="Raw Generated Text", lines=8)
        prompt = gr.Textbox(label="Rendered Prompt", lines=4)

        run_btn.click(
            fn=infer,
            inputs=[query, context, max_new_tokens, temperature, top_p],
            outputs=[thinking, answer, raw, prompt],
        )

    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

