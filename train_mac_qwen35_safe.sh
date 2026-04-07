#!/bin/bash
# Train Qwen 3.5 on Mac with conservative memory settings
# For M4 Mini 16GB - tested safe configuration

set -e

echo "========================================"
echo "Qwen 3.5 Training (Safe Mode - 16GB)"
echo "========================================"

MODEL="mlx-community/Qwen3.5-4B-MLX-4bit"
DATA_DIR="mlx_data/jsonl_chunked"
ADAPTER_PATH="qwen35_4b_mac_adapters"

# Ultra-conservative settings to avoid OOM
BATCH_SIZE=1
ITERS=500
VAL_BATCHES=5
LEARNING_RATE=1e-5
STEPS_PER_REPORT=5
STEPS_PER_EVAL=50
GRAD_ACCUM=8
MAX_SEQ_LENGTH=2048

# Minimal LoRA to save memory
NUM_LAYERS=8

echo "Configuration (Ultra-Safe):"
echo "  Model: $MODEL"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  LoRA layers: $NUM_LAYERS"
echo "  Iterations: $ITERS"
echo "  Adapter path: $ADAPTER_PATH"
echo ""
echo "Expected peak memory: ~7-8 GB"
echo "Estimated time: ~1-1.5 hours"
echo ""

# Remove old adapter if exists
rm -rf "$ADAPTER_PATH"

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
    --save-every 100 \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --test \
    --test-batches 5 \
    --grad-checkpoint

echo ""
echo "✓ Training complete!"
echo "Adapter saved to: $ADAPTER_PATH"
