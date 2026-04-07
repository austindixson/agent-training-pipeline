# Cross-Platform Fine-Tuning Pipeline

## Phase 1: Dataset (Mac — already done ✓)
```bash
# Already completed:
python3 extract_claude_logs.py --output real_dataset.jsonl
# Result: 3,165 real conversations
```

## Phase 2: Training (PC with RTX 3060)

### 2.1 Install dependencies on PC
```bash
pip install "unsloth[cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
pip install trl transformers datasets accelerate
```

### 2.2 Copy dataset to PC
```bash
# On Mac:
scp real_dataset.jsonl user@pc-ip:/path/to/project/

# Or use GitHub/sync service
```

### 2.3 Run training on PC
```bash
python train.py
```

**Expected performance on RTX 3060:**
- 3,165 examples × 3-5 epochs = ~2-4 hours
- Peak VRAM: ~9-11GB (fits in 12GB)
- Output: `qwen3-agent-finetune/` directory

## Phase 3: Export (PC)

### 3.1 Merge + GGUF on PC
```bash
python merge_gguf.py
```

This creates `qwen3-agent.gguf` (~5GB with q4_k_m quantization)

### 3.2 Copy GGUF back to Mac
```bash
# From PC:
scp qwen3-agent.gguf user@mac-ip:/Users/ghost/Desktop/localagent/
```

## Phase 4: Deploy (Mac)

### 4.1 Create Ollama model
```bash
ollama create qwen3-agent -f Modelfile
ollama run qwen3-agent
```

### 4.2 Test
```bash
ollama run qwen3-agent "List the files in the current directory"
```

## Why This Works

1. **Same base model everywhere**: Qwen3-8B-Instruct runs on both platforms
2. **LoRA is portable**: Adapter weights are platform-agnostic
3. **GGUF is universal**: Runs via Ollama on any OS/architecture
4. **Training happens once**: On fastest hardware (PC CUDA)
5. **Inference happens everywhere**: Mac, PC, mobile

## File Sizes

| File | Size | Location |
|---|---|---|
| `real_dataset.jsonl` | ~15MB | Mac → PC |
| `qwen3-agent-finetune/` | ~6MB (adapter) | PC |
| `qwen3-agent.gguf` | ~5GB | PC → Mac |
| Final model in Ollama | ~5GB | Mac |

## Bandwidth Requirements

- Dataset upload: <1 minute (15MB)
- GGUF download: 10-30 minutes on typical home connection (5GB)
