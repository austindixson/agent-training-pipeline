# PC Training Instructions

## What you'll train on your PC

- **Base model**: Qwen3-8B-Instruct (8B parameters, 4-bit quantized)
- **Dataset**: 2,532 real agent conversations from your Claude Code sessions
- **Method**: QLoRA (only ~0.3% of parameters trainable)
- **Hardware**: RTX 3060 12GB VRAM + 32GB RAM
- **Time**: ~2-4 hours for 1 epoch

## Step 1: Copy files to PC

From Mac, copy these to your PC:
```bash
# Copy the dataset directory
scp -r unsloth_data/ user@pc-ip:/path/to/project/

# Copy training scripts
scp train.py merge_gguf.py Modelfile user@pc-ip:/path/to/project/

# Or use GitHub/Drive/Dropbox
```

## Step 2: Install dependencies on PC

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux

# Install Unsloth (CUDA 12.1)
pip install "unsloth[cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
pip install trl transformers datasets accelerate
```

## Step 3: Run training

```bash
python train.py
```

**What to expect:**
- Loading model: ~1 minute
- Training: 2-4 hours (2532 examples × 1 epoch)
- Peak VRAM: ~9-11GB
- Loss should decrease from ~2.0 to <1.5

**Output:**
- `qwen3-agent-finetune/adapter_config.json`
- `qwen3-agent-finetune/adapters.safetensors` (~6MB)

## Step 4: Export to GGUF

```bash
python merge_gguf.py
```

This creates `qwen3-agent.gguf` (~5GB) with q4_k_m quantization.

## Step 5: Copy GGUF back to Mac

```bash
# On PC:
scp qwen3-agent.gguf user@mac-ip:/Users/ghost/Desktop/localagent/
```

## Step 6: Load into Ollama (on Mac)

```bash
ollama create qwen3-agent -f Modelfile
ollama run qwen3-agent
```

## Step 7: Test

```bash
# Test basic inference
ollama run qwen3-agent "What files are in the current directory?"

# Test tool calling
ollama run qwen3-agent "Run 'ls -la' and tell me how many files there are"
```

## Troubleshooting

**Out of memory:**
- Reduce `BATCH_SIZE` from 2 to 1 in train.py
- Reduce `MAX_SEQ_LEN` from 4096 to 2048
- Reduce `LORA_RANK` from 16 to 8

**Training too slow:**
- Increase `BATCH_SIZE` to 4 (if VRAM allows)
- Increase `GRAD_ACCUM` to 8

**Poor quality:**
- Train for 2-3 epochs (change `EPOCHS = 1` to `EPOCHS = 3`)
- Increase `LORA_RANK` to 32
- Use full dataset instead of 80/10/10 split
