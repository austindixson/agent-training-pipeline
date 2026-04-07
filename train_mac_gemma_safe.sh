#!/bin/bash
# Train Gemma on Mac using MLX LM CLI
# Optimized for M4 Mini with 16GB unified memory
# Conservative settings to avoid OOM

set -e  # Exit on error

echo "========================================"
echo "Gemma Training on Mac (Safe Mode)"
echo "========================================"

# Configuration
MODEL="mlx-community/gemma-2-2b-it-4bit"  # 2B model for 16GB Mac
DATA_DIR="mlx_data/jsonl_chunked"
ADAPTER_PATH="gemma_mac_adapters"

# Training parameters (very conservative)
BATCH_SIZE=1  # Single batch to minimize memory
ITERS=2000  # More iters with smaller dataset
VAL_BATCHES=10  # Fewer validation batches
LEARNING_RATE=1e-5
STEPS_PER_REPORT=5
STEPS_PER_EVAL=100
GRAD_ACCUM=8  # Effective batch size = 1 * 8 = 8

# LoRA config
NUM_LAYERS=8  # Fewer layers to save memory
LORA_RANK=4  # Smaller rank

# Memory optimization
MAX_SEQ_LENGTH=1024  # Smaller context window

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_DIR (chunked)"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM grad accum = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Iterations: $ITERS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max seq length: $MAX_SEQ_LENGTH"
echo "  LoRA layers: $NUM_LAYERS"
echo "  LoRA rank: $LORA_RANK"
echo ""

# Check if dataset exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Dataset directory not found: $DATA_DIR"
    exit 1
fi

# Run training
echo "Starting training..."
echo "========================================"

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

echo ""
echo "✓ Training complete!"
echo "Adapter saved to: $ADAPTER_PATH"
