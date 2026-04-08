# Hybrid Dataset

A training-ready dataset combining real Claude conversations with lightweight augmentations.

## Overview

- **Total conversations:** 4,747
- **Real conversations:** 3,165
- **Augmented:** 1,582 (50% augmentation ratio)
- **Splits:** 80% train, 10% valid, 10% test

## Statistics

- Total turns: 150,619
- Avg turns/conversation: 31.7
- Total tokens: 13.9M
- Format: Conversational (human/gpt turns)

## Formats

### 1. Unsloth Format (PC Training)
```
hybrid_dataset/unsloth_format/
├── train.jsonl  (3,797 examples)
├── valid.jsonl  (475 examples)
└── test.jsonl   (475 examples)
```

**Use with:** `train.py` (Unsloth on PC with RTX 3060)

### 2. MLX Format (Mac Training)
```
hybrid_dataset/mlx_format/
└── dataset.json (all splits in one file)
```

**Use with:** `mlx_lm.lora --data hybrid_dataset/mlx_format/dataset.json`

### 3. ShareGPT Format (General)
```
hybrid_dataset/sharegpt_format/
├── train.jsonl
├── valid.jsonl
└── test.jsonl
```

**Use with:** Any training pipeline supporting ShareGPT format

## Data Sources

### Real Data
- Extracted from 3,165 Claude Code sessions
- Real agent conversations with tool use
- From `real_dataset_split/`

### Augmented Data
Lightweight augmentations (no GPU required):
- **Reverse:** Reordered conversation flow
- **Shorten:** Condensed conversations
- **Paraphrase:** Varied human prompts

## Usage

### On PC (Recommended)
```bash
# With Unsloth
python train.py \
  --data hybrid_dataset/unsloth_format/train.jsonl \
  --valid_data hybrid_dataset/unsloth_format/valid.jsonl \
  --model Qwen/Qwen3.5-7B-Instruct
```

### On Mac
```bash
# With MLX LM
mlx_lm.lora \
  --model mlx-community/Qwen3-4B-MLX-4bit \
  --data hybrid_dataset/mlx_format/dataset.json \
  --train \
  --iters 500 \
  --batch-size 1
```

### Combine with Synthetic Data
After running `generate_synthetic_data.py` on PC:
```bash
cat hybrid_dataset/hybrid_combined.jsonl \
    synthetic_agent_data.jsonl \
    > final_dataset.jsonl
```

## Performance Notes

- **Mac M4 16GB:** Use MLX format with batch_size=1
- **PC RTX 3060 12GB:** Use Unsloth format with batch_size=2-4
- **Larger GPUs:** Can increase batch size for faster training

## Next Steps

1. ✓ Hybrid dataset created
2. ⏳ Generate synthetic data (on PC with GPU)
3. ⏳ Train final model with combined dataset
4. ⏳ Export to GGUF for Ollama
5. ⏳ Test agent capabilities
