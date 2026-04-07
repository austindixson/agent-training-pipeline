#!/bin/bash
# Train Gemma 3 4B with ultra-conservative memory settings
# For M4 Mini 16GB - avoid OOM at all costs

set -e

echo "========================================"
echo "Gemma 3 Training (Safe Mode - 16GB)"
echo "========================================"

MODEL="mlx-community/gemma-3-4b-it-4bit"
DATA_DIR="mlx_data/jsonl_chunked"
ADAPTER_PATH="gemma3_4b_mac_adapters"

# Ultra-conservative settings to avoid OOM
BATCH_SIZE=1        # Single batch
ITERS=500           # Fewer iters for faster testing
VAL_BATCHES=5       # Fewer validation batches
LEARNING_RATE=1e-5
STEPS_PER_REPORT=5
STEPS_PER_EVAL=50   # More frequent eval
GRAD_ACCUM=8        # Higher accumulation
MAX_SEQ_LENGTH=2048 # Reduced from 4096

# Minimal LoRA to save memory
NUM_LAYERS=8        # Reduced from 16

echo "Configuration (Ultra-Safe):"
echo "  Model: $MODEL"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Max seq length: $MAX_SEQ_LENGTH (reduced from 4096)"
echo "  LoRA layers: $NUM_LAYERS (reduced from 16)"
echo "  Iterations: $ITERS (reduced)"
echo "  Adapter path: $ADAPTER_PATH"
echo ""
echo "Expected peak memory: ~8-9 GB"
echo "Estimated time: ~45-60 minutes"
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
