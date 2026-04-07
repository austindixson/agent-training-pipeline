#!/bin/bash
# Train Gemma on Mac using MLX LM CLI
# Optimized for M4 Mini with 16GB unified memory

set -e  # Exit on error

echo "========================================"
echo "Gemma Training on Mac (MLX)"
echo "========================================"

# Configuration
MODEL="mlx-community/gemma-2-2b-it-4bit"  # 2B model for 16GB Mac
DATA_DIR="mlx_data/jsonl"
ADAPTER_PATH="gemma_mac_adapters"
OUTPUT_DIR="gemma_mac_output"

# Training parameters (conservative for 16GB)
BATCH_SIZE=2
ITERS=1000
VAL_BATCHES=25
LEARNING_RATE=1e-5
STEPS_PER_REPORT=10
STEPS_PER_EVAL=200
GRAD_ACCUM=4  # Effective batch size = 2 * 4 = 8

# LoRA config
NUM_LAYERS=16  # Target first 16 layers
LORA_RANK=8

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_DIR"
echo "  Batch size: $BATCH_SIZE × $GRAD_ACCUM grad accum = $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Iterations: $ITERS"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA layers: $NUM_LAYERS"
echo "  LoRA rank: $LORA_RANK"
echo "  Adapter path: $ADAPTER_PATH"
echo ""

# Check if dataset exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Dataset directory not found: $DATA_DIR"
    echo "   Run dataset conversion first"
    exit 1
fi

# Check files exist
for split in train valid test; do
    if [ ! -f "$DATA_DIR/${split}.jsonl" ]; then
        echo "❌ Missing file: $DATA_DIR/${split}.jsonl"
        exit 1
    fi
done

echo "✓ Dataset files found"

# Create output directory
mkdir -p "$ADAPTER_PATH"
mkdir -p "$OUTPUT_DIR"

# Run training
echo ""
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
    --save-every 500 \
    --max-seq-length 2048 \
    --test \
    --test-batches 25

echo ""
echo "========================================"
echo "✓ Training complete!"
echo "========================================"
echo ""
echo "Adapter saved to: $ADAPTER_PATH"
echo ""
echo "To use the fine-tuned model:"
echo "  python3 -c \"from mlx_lm import load; model, tokenizer = load('$MODEL'); from mlx_lm.utils import load_adapters; load_adapters(model, '$ADAPTER_PATH'); print('Model loaded!')\""
echo ""
echo "To export for Ollama, run:"
echo "  python3 export_mlx_to_gguf.py --model $MODEL --adapter $ADAPTER_PATH"
