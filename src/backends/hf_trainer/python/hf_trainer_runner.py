#!/usr/bin/env python3
"""HF Trainer runner — BLUT's bundled Python wrapper.

Invoked as:

    python hf_trainer_runner.py <job_spec.json>

The job JSON conforms to the `HfTrainerJob` struct in
`src/backends/hf_trainer/runner.rs`. This script dispatches to
`transformers.Trainer` (task == "sft" / "distill") or
`trl.DPOTrainer` (task == "dpo") and emits one JSON line per
status event on stdout. Stderr is free-form.

Wire schema (matches `runner::StatusLine`):
  {"kind":"step","step":int,"total":int,"loss":float,"lr":float}
  {"kind":"saved","path":str}
  {"kind":"done","checkpoint_dir":str,"final_loss":float}
  {"kind":"failed","error":str}

Auto-managed venv contract: this script only imports packages
listed in `runner::REQUIRED_PKGS`. New dep = bump
`VENV_VERSION` in `venv.rs` so stale envs get rebuilt.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


def emit(line: Dict[str, Any]) -> None:
    """Write one status JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(line, default=str) + "\n")
    sys.stdout.flush()


def fail(msg: str) -> None:
    emit({"kind": "failed", "error": msg})
    sys.exit(1)


def load_spec(spec_path: Path) -> Dict[str, Any]:
    if not spec_path.exists():
        fail(f"job spec not found: {spec_path}")
    try:
        return json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"parse job spec: {exc}")
        return {}  # unreachable


def run_sft(spec: Dict[str, Any]) -> None:
    """SFT via `transformers.Trainer`."""
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    base_model = spec["base_model"]
    train_path = spec["train_dataset_path"]
    eval_path = spec.get("eval_dataset_path")
    output_dir = spec["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {}
    peft = spec.get("peft")
    if peft and peft.get("method") == "qlora":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if peft and peft.get("method") in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if peft["method"] == "qlora":
            model = prepare_model_for_kbit_training(model)
        lc = LoraConfig(
            r=int(peft["rank"]),
            lora_alpha=int(peft["alpha"]),
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lc)

    def fmt(example: Dict[str, Any]) -> Dict[str, Any]:
        # Accept HF chat-template-style {"messages": [...]} or
        # plain {"text": "..."}.
        if "messages" in example:
            chat = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": chat}
        return {"text": example.get("text", "")}

    def tok(example: Dict[str, Any]) -> Dict[str, Any]:
        out = tokenizer(
            example["text"],
            truncation=True,
            max_length=int(spec["seq_len"]),
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    train_ds = train_ds.map(fmt).map(tok, remove_columns=train_ds.column_names)
    eval_ds = None
    if eval_path:
        eval_ds = load_dataset("json", data_files=str(eval_path), split="train")
        eval_ds = eval_ds.map(fmt).map(tok, remove_columns=eval_ds.column_names)

    ta_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(spec["batch_size"]),
        gradient_accumulation_steps=int(spec["grad_accum"]),
        num_train_epochs=int(spec["epochs"]),
        learning_rate=float(spec["lr"]),
        seed=int(spec["seed"]),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        bf16=True,
    )
    ta_kwargs.update(spec.get("extra") or {})
    if eval_ds is not None:
        ta_kwargs.setdefault("eval_strategy", "epoch")

    targs = TrainingArguments(**ta_kwargs)

    class Emitter(TrainerCallback):
        def __init__(self) -> None:
            self.last_loss: Optional[float] = None
            self.total: int = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.total = int(state.max_steps or 0)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                self.last_loss = float(logs["loss"])
            emit(
                {
                    "kind": "step",
                    "step": int(state.global_step or 0),
                    "total": self.total,
                    "loss": self.last_loss,
                    "lr": float(logs.get("learning_rate", 0.0)) if "learning_rate" in logs else None,
                }
            )

        def on_save(self, args, state, control, **kwargs):
            emit(
                {
                    "kind": "saved",
                    "path": str(Path(args.output_dir) / f"checkpoint-{state.global_step}"),
                }
            )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[Emitter()],
    )
    train_out = trainer.train()
    trainer.save_model(str(output_dir))
    # `training_loss` is None when no train step ran (cache hit /
    # zero-epoch sanity run). Don't choke on that — emit a Done
    # line with final_loss=None and let the recipe decide.
    final_loss = (
        float(train_out.training_loss)
        if train_out.training_loss is not None
        else None
    )
    emit(
        {
            "kind": "done",
            "checkpoint_dir": str(output_dir),
            "final_loss": final_loss,
        }
    )


def run_dpo(spec: Dict[str, Any]) -> None:
    """DPO via `trl.DPOTrainer`."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    dpo = spec.get("dpo") or {}
    base_model = spec["base_model"]
    prefs_path = dpo.get("preferences_path") or spec["train_dataset_path"]
    output_dir = spec["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    train_ds = load_dataset("json", data_files=str(prefs_path), split="train")
    # Expected schema: {"prompt": str, "chosen": str, "rejected": str}.

    cfg = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(spec["batch_size"]),
        gradient_accumulation_steps=int(spec["grad_accum"]),
        num_train_epochs=int(spec["epochs"]),
        learning_rate=float(spec["lr"]),
        seed=int(spec["seed"]),
        beta=float(dpo.get("beta", 0.1)),
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        bf16=True,
    )

    # DPO progress fan-out — same Step/Saved schema as SFT so the
    # Rust runner can consume both paths identically.
    class DpoEmitter(TrainerCallback):
        def __init__(self) -> None:
            self.last_loss: Optional[float] = None
            self.total: int = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.total = int(state.max_steps or 0)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                self.last_loss = float(logs["loss"])
            emit(
                {
                    "kind": "step",
                    "step": int(state.global_step or 0),
                    "total": self.total,
                    "loss": self.last_loss,
                    "lr": float(logs.get("learning_rate", 0.0))
                    if "learning_rate" in logs
                    else None,
                }
            )

        def on_save(self, args, state, control, **kwargs):
            emit(
                {
                    "kind": "saved",
                    "path": str(
                        Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    ),
                }
            )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        callbacks=[DpoEmitter()],
    )
    out = trainer.train()
    trainer.save_model(str(output_dir))
    final_loss = (
        float(out.training_loss) if out.training_loss is not None else None
    )
    emit(
        {
            "kind": "done",
            "checkpoint_dir": str(output_dir),
            "final_loss": final_loss,
        }
    )


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: hf_trainer_runner.py <job_spec.json>")
    spec = load_spec(Path(sys.argv[1]))
    task = spec.get("task", "sft")

    try:
        if task in ("sft", "distill"):
            run_sft(spec)
        elif task == "dpo":
            run_dpo(spec)
        else:
            fail(f"unknown task: {task}")
    except Exception as exc:
        emit({"kind": "failed", "error": f"{type(exc).__name__}: {exc}"})
        sys.stderr.write(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
