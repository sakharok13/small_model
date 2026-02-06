import ast
import json
import os
import random
import re
from glob import glob
from typing import Any, Deque, Dict, List, Optional, Tuple

IDK_DEFAULT = "I can't find the answer in the context"


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


def build_user_content(query: str, context: str, json_mode: bool, idk_text: str) -> str:
    if json_mode:
        return (
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "Return JSON."
        )
    return (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        f"Answer using only CONTEXT. If not in context, reply exactly: {idk_text}"
    )


def build_messages(
    query: str,
    context: str,
    answer: Optional[str],
    json_mode: bool,
    system_prompt: str,
    system_prompt_json: str,
    idk_text: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Returns:
      - messages_prompt: system + user (no assistant answer)   [for inference prompt & prompt_len]
      - messages_full:   system + user + assistant(answer)     [for training text]
    """
    system = system_prompt_json if json_mode else system_prompt
    messages_prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_content(query, context, json_mode=json_mode, idk_text=idk_text)},
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


def load_state(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except Exception:
            return {}


def restore_random_state(state: Dict[str, Any], seed: int) -> None:
    if state.get("random_state"):
        try:
            random.setstate(ast.literal_eval(state["random_state"]))
        except Exception:
            random.seed(seed)
    else:
        random.seed(seed)


def save_state(path: str, rows_seen: int, written: int, neg_pool: Deque[Tuple[str, str]]) -> None:
    if not path:
        return
    state = {
        "rows_seen": rows_seen,
        "written": written,
        "random_state": repr(random.getstate()),
        "neg_pool": list(neg_pool),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
    os.replace(tmp, path)


def with_rank_suffix_dir(path: str, rank: int) -> str:
    return f"{path}.rank{rank}"


def next_shard_idx(out_dir: str) -> int:
    pattern = os.path.join(out_dir, "part-*.parquet")
    existing = glob(pattern)
    if not existing:
        return 0
    max_idx = -1
    for p in existing:
        base = os.path.basename(p)
        try:
            idx = int(base.replace("part-", "").replace(".parquet", ""))
            if idx > max_idx:
                max_idx = idx
        except Exception:
            continue
    return max_idx + 1
