#!/bin/bash
# Train Qwen3.5-9B on Mac using MLX LM CLI
# Optimized for M4 Mini with 16GB unified memory

set -e

echo "========================================"
echo "Qwen3.5-9B Training on Mac (4K Context)"
echo "========================================"

# Use Qwen3.5 9B model (good balance for 16GB Mac)
MODEL="mlx-community/Qwen3.5-9B-MLX-4bit"
DATA_DIR="mlx_data/jsonl_chunked"
ADAPTER_PATH="qwen35_9b_mac_adapters"

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

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_DIR"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  LoRA layers: $NUM_LAYERS"
echo "  Adapter path: $ADAPTER_PATH"
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
    --test-batches 10 \
    --grad-checkpoint

echo ""
echo "✓ Training complete!"
echo "Adapter saved to: $ADAPTER_PATH"
