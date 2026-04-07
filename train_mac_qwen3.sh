#!/bin/bash
# Train Qwen3 on Mac using MLX LM CLI with extended context
# Optimized for M4 Mini with 16GB unified memory

set -e

echo "========================================"
echo "Qwen3 Training on Mac (4K Context)"
echo "========================================"

# Use MLX-converted model for faster loading
MODEL="mlx-community/Qwen3-8B-Instruct-4bit"
DATA_DIR="mlx_data/jsonl_chunked"
ADAPTER_PATH="qwen3_mac_adapters"

# Training parameters
BATCH_SIZE=1
ITERS=1000
VAL_BATCHES=10
LEARNING_RATE=1e-5
STEPS_PER_REPORT=5
STEPS_PER_EVAL=100
GRAD_ACCUM=8
MAX_SEQ_LENGTH=4096  # Extended context

# LoRA config
NUM_LAYERS=8
LORA_RANK=8

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_DIR"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM grad accum = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  LoRA rank: $LORA_RANK"
echo ""

mlx_lm.lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --fine-tune-type lora \
    --optimizer adam \
    --num-layers "$NUM_LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --iters "$ITERS" \
    --val-batches "$VAL_BATCHES" \
    --learning-rate "$LEARNING_RATE" \
    --steps-per-report "$STEPS_PER_REPORT" \
    --steps-per-eval "$STEPS_PER_EVAL" \
    --grad-accumulation-steps "$GRAD_ACCUM" \
    --adapter-path "$ADAPTER_PATH" \
    --save-every 250 \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --test \
    --test-batches 10

echo "✓ Training complete! Adapter saved to: $ADAPTER_PATH"
