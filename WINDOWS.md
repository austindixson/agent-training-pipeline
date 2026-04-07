# Windows PC Training Instructions

Complete guide for training your local agent on Windows PC with NVIDIA GPU (RTX 3060 or better).

## Prerequisites

**Hardware:**
- NVIDIA GPU with 12GB+ VRAM (RTX 3060, RTX 4070, etc.)
- 32GB+ RAM recommended
- 10GB+ free disk space

**Software:**
- Windows 10/11
- Python 3.9-3.12
- Git
- CUDA Toolkit 12.1 (or newer)

## Step 1: Install Python and Dependencies

### 1.1 Install Python (if not already installed)

1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Verify installation:
   ```cmd
   python --version
   ```

### 1.2 Install CUDA Toolkit

1. Download CUDA Toolkit 12.1 from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Run installer and select "Express Installation"
3. Restart your computer

### 1.3 Clone the Repository

```cmd
git clone https://github.com/austindixson/agent-training-pipeline.git
cd agent-training-pipeline
```

### 1.4 Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### 1.5 Install Dependencies

```cmd
pip install "unsloth[cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
pip install trl transformers datasets accelerate
```

**Expected install time:** 5-10 minutes

## Step 2: Prepare Dataset (On Mac First)

### 2.1 Extract Your Claude Code Conversations

On your Mac, run:
```bash
cd /Users/ghost/Desktop/localagent
python3 extract_claude_logs.py --output real_dataset.jsonl
```

### 2.2 Prepare Dataset Splits

```bash
python3 prepare_pc_data.py
```

This creates `unsloth_data/` with train/valid/test splits.

### 2.3 Copy Dataset to Windows

**Option A: Network Share**
1. On Mac: System Settings → Sharing → File Sharing
2. On Windows: `\\Mac-IP\Users\ghost\Desktop\localagent\unsloth_data`

**Option B: USB Drive**
```cmd
# Copy to USB on Mac, then on Windows:
copy E:\unsloth_data\* C:\Users\YourName\agent-training-pipeline\unsloth_data\
```

**Option C: Cloud Storage**
Upload to Google Drive/Dropbox on Mac, download on Windows.

## Step 3: Run Training

### 3.1 Verify Dataset

```cmd
dir unsloth_data
```

You should see:
- `train.jsonl` (~87MB)
- `valid.jsonl` (~10MB)
- `test.jsonl` (~11MB)

### 3.2 Start Training

```cmd
python train.py
```

**What to expect:**
- **Duration:** 2-4 hours for 1 epoch
- **Peak VRAM:** 9-11GB
- **Loss curve:** Should decrease from ~2.0 to <1.5
- **Console output:**
  ```
  Loading unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit ...
  Trainable parameters: 0.297% (1.466M/494.033M)
  Starting training...
  Iter 100: Train loss 1.845, Val loss 1.923
  Iter 200: Train loss 1.612, Val loss 1.754
  ...
  Training complete in 7200.5s
  LoRA adapter saved to qwen3-agent-finetune/
  ```

### 3.3 Monitor GPU Usage

Open Task Manager → Performance → GPU to monitor:
- GPU usage: Should be 80-100%
- VRAM: Should peak at ~10GB
- Temperature: Keep under 85°C

## Step 4: Export to GGUF

### 4.1 Run Export Script

```cmd
python merge_gguf.py
```

**Output:**
```
Loading base model: unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit
Loading LoRA adapter: qwen3-agent-finetune
Merging adapter into base weights...

GGUF exported: qwen3-agent-q4_k_m.gguf (5.2 GB)
```

### 4.2 Verify GGUF File

```cmd
dir qwen3-agent*.gguf
```

File should be ~5GB.

## Step 5: Copy GGUF Back to Mac

### 5.1 Transfer Options

**Option A: Network Share**
```cmd
copy qwen3-agent-q4_k_m.gguf \\Mac-IP\Users\ghost\Desktop\localagent\
```

**Option B: USB Drive**
```cmd
copy qwen3-agent-q4_k_m.gguf E:\
```

**Option C: Cloud Storage**
Upload to Google Drive/Dropbox (may take 30-60 minutes for 5GB)

## Step 6: Load into Ollama (On Mac)

### 6.1 Copy GGUF to Correct Location

```bash
# On Mac
cp ~/Downloads/qwen3-agent-q4_k_m.gguf ~/Desktop/localagent/
cd ~/Desktop/localagent
```

### 6.2 Update Modelfile (if needed)

The Modelfile should reference your GGUF file:
```dockerfile
FROM ./qwen3-agent-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
```

### 6.3 Create Ollama Model

```bash
ollama create qwen3-agent -f Modelfile
ollama run qwen3-agent
```

## Step 7: Test Your Model

### 7.1 Basic Inference Test

```bash
ollama run qwen3-agent "What files are in the current directory?"
```

### 7.2 Tool Calling Test

```bash
ollama run qwen3-agent "Run 'ls -la' and tell me how many files there are"
```

### 7.3 Multi-Step Planning Test

```bash
ollama run qwen3-agent "I need to: 1) Find all Python files, 2) Read the first one, 3) Tell me what it does"
```

## Troubleshooting

### Out of Memory Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Reduce batch size** in `train.py`:
   ```python
   BATCH_SIZE = 1  # Change from 2 to 1
   ```

2. **Reduce sequence length**:
   ```python
   MAX_SEQ_LEN = 2048  # Change from 4096 to 2048
   ```

3. **Close other applications** (Chrome, games, etc.)

### Training Too Slow

**Solutions:**
1. **Increase batch size** (if VRAM allows):
   ```python
   BATCH_SIZE = 4
   ```

2. **Increase gradient accumulation**:
   ```python
   GRAD_ACCUM = 8  # Change from 4 to 8
   ```

### Poor Quality Results

**Solutions:**
1. **Train for more epochs**:
   ```python
   EPOCHS = 3  # Change from 1 to 3
   ```

2. **Increase LoRA rank**:
   ```python
   LORA_RANK = 32  # Change from 16 to 32
   ```

3. **Use full dataset** instead of 80/10/10 split

### Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'unsloth'
```

**Solution:**
```cmd
# Make sure venv is activated
venv\Scripts\activate

# Reinstall dependencies
pip install --upgrade "unsloth[cu121]" --extra-index-url https://download.pytorch.org/whl/cu121
```

### GPU Not Detected

**Symptoms:**
```
RuntimeError: No CUDA GPUs are available
```

**Solutions:**
1. **Check CUDA installation:**
   ```cmd
   nvidia-smi
   ```
   
2. **Reinstall CUDA Toolkit** (see Step 1.2)

3. **Update NVIDIA drivers:**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

## Performance Benchmarks

**Expected training times (RTX 3060):**
- 1 epoch (2,532 examples): 2-4 hours
- 3 epochs: 6-12 hours
- Full dataset (5,000+ examples): 4-8 hours per epoch

**Peak VRAM usage:**
- Batch size 1: ~8GB
- Batch size 2: ~10GB
- Batch size 4: ~11GB

**GGUF conversion time:** 5-10 minutes

## Next Steps

After your model is trained and deployed:

1. **Benchmark performance** — Compare to base Qwen model
2. **Iterate on quality** — Train more epochs or increase dataset size
3. **Deploy to production** — Use in your agent workflows
4. **Share results** — Document your learnings and improvements

## Additional Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Ollama Documentation](https://ollama.ai/)
- [Qwen Models](https://huggingface.co/Qwen)
- [GGUF Format](https://github.com/ggerganov/llama.cpp)
