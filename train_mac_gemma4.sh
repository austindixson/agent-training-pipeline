#!/bin/bash
# Train Gemma 4 e2b on Mac using MLX LM CLI
# Optimized for M4 Mini with 16GB unified memory

set -e

echo "========================================"
echo "Gemma 4 Training on Mac (4K Context)"
echo "========================================"

# Use Gemma 4 e2b (2B model - fits in 16GB)
MODEL="mlx-community/gemma-4-e2b-it-4bit"
DATA_DIR="mlx_data/jsonl_chunked"
ADAPTER_PATH="gemma4_e2b_mac_adapters"

# Training parameters
BATCH_SIZE=2
ITERS=1000
VAL_BATCHES=10
LEARNING_RATE=1e-5
STEPS_PER_REPORT=5
STEPS_PER_EVAL=100
GRAD_ACCUM=4
MAX_SEQ_LENGTH=4096

# LoRA config
NUM_LAYERS=16

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
