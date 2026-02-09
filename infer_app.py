#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import threading
from typing import Generator, Optional

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def build_prompt(query: str, context: str, template: str) -> str:
    return template.format(query=query, context=context)


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
        # LoRA/PEFT adapter case
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


def generate_stream(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    answer_end: str,
) -> Generator[str, None, None]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    acc = ""
    for text in streamer:
        acc += text
        if answer_end and answer_end in acc:
            acc = acc.split(answer_end)[0]
            yield acc
            break
        yield acc

    thread.join()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Path to finetuned checkpoint or adapter")
    ap.add_argument("--base_model", type=str, default=None, help="Base model name/path if model_path is a LoRA adapter")
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for loading")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)

    ap.add_argument(
        "--prompt_template",
        type=str,
        default="[context_start]{context}[context_end][query_start]{query}[query_end][answer_start]",
        help="Template used for inference prompt.",
    )
    ap.add_argument(
        "--answer_end",
        type=str,
        default="[answer_end]",
        help="String that marks the end of the answer (optional).",
    )
    args = ap.parse_args()

    model, tokenizer = load_model(
        model_path=args.model_path,
        base_model=args.base_model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )

    def infer(query: str, context: str) -> Generator[str, None, None]:
        prompt = build_prompt(query, context, args.prompt_template)
        yield from generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            answer_end=args.answer_end,
        )

    with gr.Blocks() as demo:
        gr.Markdown("# Qwen3 Finetuned QA")
        with gr.Row():
            query = gr.Textbox(label="Query", lines=2, placeholder="Enter the question")
            context = gr.Textbox(label="Context", lines=8, placeholder="Paste the context")
        btn = gr.Button("Generate")
        answer = gr.Textbox(label="Answer", lines=6)

        btn.click(fn=infer, inputs=[query, context], outputs=answer)

    demo.queue().launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()
