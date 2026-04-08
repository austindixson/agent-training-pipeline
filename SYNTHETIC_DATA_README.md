# Synthetic Agent Data Generation

Generate diverse training data from real agent trajectories using fine-tuned models.

## Setup

### Requirements (PC with RTX 3060 12GB+)
```bash
pip install unsloth torch transformers datasets
```

## Usage

### Basic Usage
```python
python generate_synthetic_data.py
```

### Custom Configuration
```python
from generate_synthetic_data import main

main(
    model_path="path/to/your/model",
    data_files=["real_dataset_split/real_dataset_part1.jsonl"],
    output_file="synthetic_agent_data.jsonl",
    target_examples=5000,
    batch_size=4
)
```

## What It Does

For each real agent trajectory, generates 2 synthetic variations:

1. **Tool Failure + Recovery**
   - Realistic tool errors (timeout, API failure, wrong result)
   - Agent recovers with alternative approaches
   - Maintains conversational context

2. **Multi-Step Reasoning + Parallel Tools**
   - Deeper analysis before actions
   - Multiple tools used in parallel
   - Shows reasoning process

## Output

- **Format:** JSONL (same schema as input)
- **Augmentation:** 2x your real dataset
- **Quality:** Filtered for valid JSON only

## Model Recommendations

- **Base:** `0xSero/gemma-4-21b-a4b-it-REAP` (open source)
- **Fine-tuned:** Your own `gemma4-reap-agent-ft-final`
- **Quantization:** 4-bit QLoRA for memory efficiency

## Combining Datasets

```bash
# Merge real + synthetic
cat real_dataset_split/*.jsonl synthetic_agent_data.jsonl > combined_dataset.jsonl

# Or use Python
python -c "
import json

real = []
for file in ['real_dataset_split/real_dataset_part1.jsonl',
             'real_dataset_split/real_dataset_part2.jsonl']:
    with open(file) as f:
        real.extend([json.loads(line) for line in f])

synthetic = []
with open('synthetic_agent_data.jsonl') as f:
    synthetic = [json.loads(line) for line in f]

combined = real + synthetic
with open('combined_dataset.jsonl', 'w') as f:
    for item in combined:
        f.write(json.dumps(item) + '\n')

print(f'Combined: {len(combined)} examples')
"
```

## Performance

- **Speed:** ~50-100 examples/minute on RTX 3060
- **VRAM:** ~8-10GB with batch_size=4
- **Quality:** ~90%+ valid JSON rate

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 2 or 1
- Enable gradient checkpointing in model load

### Low Valid JSON Rate
- Lower `temperature` to 0.7
- Increase `max_new_tokens`
- Improve prompt engineering

### Slow Generation
- Increase `batch_size` (if VRAM allows)
- Use model with lower precision
- Process fewer examples (reduce `target_examples`)
