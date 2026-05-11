"""JSONL → HuggingFace `datasets.Dataset` conversion for SFT.

Each JSONL line is one example: `{"messages": [{"role": ..., "content": ...}, ...]}`.
The OpenAI chat-completion shape. Tokenizer's chat template renders
the conversation into a single string the trainer ingests.

Examples whose tokenized form exceeds `max_seq_len` are truncated
from the front (keep the most recent turn) so the assistant's final
reply always lands in the loss window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_dataset(path: Path, *, tokenizer: Any, max_seq_len: int):
    from datasets import Dataset  # imported lazily; trainer.py owns the venv

    examples: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
            messages = obj.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            examples.append({"text": text})

    if not examples:
        raise ValueError(f"no usable examples in {path}")

    ds = Dataset.from_list(examples)

    def _truncate_front(row: dict[str, str]) -> dict[str, str]:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if len(ids) <= max_seq_len:
            return row
        kept = ids[-max_seq_len:]
        row["text"] = tokenizer.decode(kept, skip_special_tokens=False)
        return row

    return ds.map(_truncate_front)
