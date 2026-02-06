import torch
from typing import Any, Dict, List
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def collate_answer_only(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # Tokenize full text (already chat-formatted)
    enc = tokenizer(
        [ex["text"] for ex in batch],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.clone()

    # Mask everything up to prompt_len for each row
    for i, ex in enumerate(batch):
        pl = int(ex["prompt_len"])
        if pl >= labels.size(1):
            labels[i, :] = -100
        else:
            labels[i, :pl] = -100

    # Also mask padding
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
