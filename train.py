#!/usr/bin/env python3
"""
QLoRA fine-tuning with Unsloth — RTX 3060 12GB / 32GB RAM optimized.
Base model: unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit
Dataset: agent_dataset.jsonl (ShareGPT format)
"""

import json
import os
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ── Training config ──────────────────────────────────────────────────────────

MODEL_ID = "unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit"
OUTPUT_DIR = "qwen3-agent-finetune"
DATASET_DIR = "unsloth_data"  # Directory with train.jsonl, valid.jsonl, test.jsonl

MAX_SEQ_LEN = 4096      # 32GB RAM handles this
LORA_RANK = 16          # start here; bump to 32 for next run
LORA_ALPHA = 32         # 2x rank is standard
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

BATCH_SIZE = 2                  # fits in 12GB VRAM with 4-bit quant
GRAD_ACCUM = 4                  # effective batch = 8
EPOCHS = 1
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_8bit"        # 8-bit optimizer saves ~2GB VRAM
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
SAVE_STEPS = 100
SEED = 42


# ── Load model + tokenizer ───────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,             # auto-detect: bf16 on Ampere (RTX 3060)
        load_in_4bit=True,      # QLoRA: 4-bit base weights
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves ~30% VRAM
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    return model, tokenizer


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_dataset(tokenizer) -> tuple[Dataset, Dataset]:
    print(f"Loading dataset from {DATASET_DIR}/ ...")

    # Load train and valid splits
    records_train = []
    with open(Path(DATASET_DIR) / "train.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records_train.append(json.loads(line))

    records_valid = []
    with open(Path(DATASET_DIR) / "valid.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records_valid.append(json.loads(line))

    print(f"  {len(records_train)} training examples")
    print(f"  {len(records_valid)} validation examples")

    def format_example(example):
        convs = example["conversations"]
        messages = []
        for turn in convs:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds_train = Dataset.from_list(records_train)
    ds_train = ds_train.map(format_example, remove_columns=ds_train.column_names)

    ds_valid = Dataset.from_list(records_valid)
    ds_valid = ds_valid.map(format_example, remove_columns=ds_valid.column_names)

    return ds_train, ds_valid


# ── Trainer ──────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset: Dataset, valid_dataset: Dataset) -> None:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=WARMUP_RATIO,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER,
            optim=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            fp16=False,
            bf16=True,          # RTX 3060 supports bf16
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            eval_steps=SAVE_STEPS,
            save_total_limit=2,
            seed=SEED,
            report_to="none",   # set to "wandb" if you want tracking
            dataloader_num_workers=0,
            remove_unused_columns=True,
            packing=True,       # pack short examples together → faster training
        ),
    )

    print("Starting training ...")
    print(f"  Model: {MODEL_ID}")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"  Batch: {BATCH_SIZE} * {GRAD_ACCUM} grad accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Max seq len: {MAX_SEQ_LEN}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Output: {OUTPUT_DIR}/")

    trainer_stats = trainer.train()
    print(f"\nTraining complete in {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"  Loss: {trainer_stats.metrics['train_loss']:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not Path(DATASET_DIR).exists():
        print(f"ERROR: {DATASET_DIR} not found. Run prepare_pc_data.py first.")
        raise SystemExit(1)

    model, tokenizer = load_model()
    dataset_train, dataset_valid = load_dataset(tokenizer)
    train(model, tokenizer, dataset_train, dataset_valid)


if __name__ == "__main__":
    main()
