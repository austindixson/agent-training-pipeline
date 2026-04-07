# Local Agent Fine-Tuning Pipeline

Train your own local AI agent that uses tools (Bash, Read, Write, etc.) by fine-tuning Qwen models on real conversation data.

## 🎯 What This Does

- **Extracts** real agent conversations from your Claude Code sessions
- **Trains** a fine-tuned model that learns tool-calling patterns
- **Exports** to GGUF format for Ollama deployment
- **Runs** entirely on consumer hardware (Mac M4 or PC with RTX 3060)

## 📁 Project Structure

```
localagent/
├── extract_claude_logs.py    # Extract real conversations from ~/.claude/projects/
├── prepare_pc_data.py        # Convert dataset for PC training
├── train.py                  # PC training script (Unsloth)
├── merge_gguf.py             # Export adapter to GGUF
├── Modelfile                 # Ollama model definition
├── tests/
│   └── test_integration.py  # Real integration tests (no mocks)
├── WINDOWS.md                # Windows PC training instructions
├── PC_README.md              # Linux PC training instructions
└── PC_TRAINING_WORKFLOW.md   # Cross-platform workflow guide
```

## 🚀 Quick Start

### Option 1: Train on Windows PC (Recommended - Fastest)

**Hardware:** Windows PC with RTX 3060 12GB VRAM + 32GB RAM
**Time:** 2-4 hours

📖 **See [WINDOWS.md](WINDOWS.md) for complete step-by-step instructions**

Quick overview:
1. Extract conversations on Mac: `python3 extract_claude_logs.py`
2. Prepare dataset: `python3 prepare_pc_data.py`
3. Copy to Windows PC
4. Install dependencies: `pip install "unsloth[cu121]" --extra-index-url https://download.pytorch.org/whl/cu121`
5. Run training: `python train.py`

### Option 2: Train on Linux PC

**Hardware:** Linux PC with NVIDIA GPU
**Time:** 2-4 hours

📖 **See [PC_README.md](PC_README.md) for instructions**

### Option 3: Train on Mac (Proof of Concept)

**Hardware:** Mac M4 16GB unified memory
**Time:** 6-8 hours
**Note:** Slower and memory-constrained. Use PC for production training.

4. **Export to GGUF** (PC):
   ```bash
   python merge_gguf.py
   ```

5. **Load into Ollama** (Mac):
   ```bash
   ollama create qwen3-agent -f Modelfile
   ollama run qwen3-agent
   ```

### Option 2: Train on Mac (Slower - Proof of Concept)

**Hardware:** Mac M4 16GB unified memory
**Time:** 6-8 hours (smaller model)

See `PC_TRAINING_WORKFLOW.md` for detailed Mac MLX training instructions.

## 📊 Dataset

The pipeline extracts real agent conversations from:
- **Source:** `~/.claude/projects/` (all your Claude Code sessions)
- **Format:** ShareGPT JSONL with tool calls and responses
- **Content:** Real Bash, Read, Write, Edit, Grep, Glob, Agent calls
- **Quality:** Only sessions with ≥2 tool calls included

**Typical output:** 2,000-5,000 conversations depending on usage

## 🧪 Testing

```bash
# Run integration tests (real API calls, no mocks)
pytest tests/test_integration.py -v
```

Tests cover:
- Log extraction validation
- OpenRouter API connectivity
- MLX training smoke test
- Dataset quality checks

## 📦 Outputs

| File | Size | Description |
|------|------|-------------|
| `qwen3-agent-finetune/adapters.safetensors` | ~6MB | LoRA adapter (training output) |
| `qwen3-agent.gguf` | ~5GB | Quantized model (Ollama ready) |

## 🔧 Configuration

**Training parameters (PC):**
- Base model: `unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit`
- LoRA rank: 16 (0.3% parameters trainable)
- Batch size: 2 (effective batch = 8 with gradient accumulation)
- Max sequence length: 4096 tokens
- Optimizer: adamw_8bit
- Learning rate: 2e-4

**Hardware requirements:**
- **Minimum:** RTX 3060 12GB VRAM + 32GB RAM
- **Recommended:** RTX 4070+ 16GB VRAM + 64GB RAM

## 📖 Documentation

- `PC_README.md` — Detailed PC training instructions
- `PC_TRAINING_WORKFLOW.md` — Cross-platform workflow diagram
- `Modelfile` — Ollama model configuration

## 🤝 Contributing

This is a personal fine-tuning pipeline. Feel free to adapt for your own use case.

## 📄 License

MIT

## 🙏 Acknowledgments

- **Unsloth** — Efficient QLoRA training
- **MLX** — Apple Silicon ML framework
- **Qwen** — Base model from Alibaba Cloud
- **Ollama** — Local model inference
