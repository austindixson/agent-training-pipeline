#!/usr/bin/env python3
"""
Generate synthetic agent training data using fine-tuned models.

This script uses a fine-tuned Gemma 4 REAP model to create diverse
variations of real agent trajectories for data augmentation.

Requirements:
  - unsloth
  - PyTorch with CUDA
  - GPU with 12GB+ VRAM
"""

from unsloth import FastLanguageModel
import json
import re
from pathlib import Path
from datasets import load_dataset
from typing import Optional
import torch


def load_model(model_path: str):
    """Load the fine-tuned model for inference."""
    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        dtype=None,
        load_in_4bit=True
    )
    model = FastLanguageModel.for_inference(model)
    print("✓ Model loaded")
    return model, tokenizer


def load_real_data(data_files: list[str]) -> list:
    """Load real agent trajectories from JSONL files."""
    print(f"Loading data from: {data_files}")

    # Try loading with datasets library first
    try:
        dataset = load_dataset("json", data_files=data_files, split="train")
        real_data = [json.loads(ex) for ex in dataset.to_pandas().to_dict(orient="records")]
    except:
        # Fallback to manual loading
        real_data = []
        for file in data_files:
            with open(file) as f:
                for line in f:
                    real_data.append(json.loads(line))

    print(f"✓ Loaded {len(real_data)} examples")
    return real_data


def extract_json_from_response(response: str) -> Optional[dict]:
    """Extract valid JSON from model response."""
    try:
        # Try direct parse first
        return json.loads(response)
    except:
        pass

    # Try to find JSON array in response
    patterns = [
        r'\[[\s\S]*\]',  # JSON array
        r'\{[\s\S]*\}',  # JSON object
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

    return None


def generate_synthetic_prompt(real_example: dict) -> str:
    """Create prompt for synthetic data generation."""
    return f"""You are an expert agent data synthesizer. Given this real agent trajectory, create 2 new diverse variations:

Variation 1: Tool failure + recovery
- Introduce a realistic tool failure (timeout, error, wrong result)
- Show agent recovering with alternative approach
- Maintain conversational flow

Variation 2: Multi-step reasoning + parallel tool use
- Expand reasoning with intermediate steps
- Use 2-3 tools in parallel where appropriate
- Show deeper analysis before action

Real trajectory:
{json.dumps(real_example, indent=2)}

Output ONLY valid JSON array with 2 objects, using the exact same schema as the input. Make them realistic and high-quality.

Output format:
[
  {{
    "conversations": [
      {{"from": "human", "value": "..."}},
      {{"from": "gpt", "value": "..."}}
    ]
  }},
  {{
    "conversations": [
      {{"from": "human", "value": "..."}},
      {{"from": "gpt", "value": "..."}}
    ]
  }}
]"""


def generate_synthetic_batch(
    model,
    tokenizer,
    examples: list[dict],
    batch_size: int = 4,
    max_new_tokens: int = 4096,
    temperature: float = 0.85,
    top_p: float = 0.9
) -> list[dict]:
    """Generate synthetic examples for a batch of real examples."""
    synthetic_list = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompts = [generate_synthetic_prompt(ex) for ex in batch]

        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to("cuda")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and parse
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)

            # Extract synthetic examples from response
            synthetic = extract_json_from_response(response)

            if synthetic:
                if isinstance(synthetic, list):
                    synthetic_list.extend(synthetic)
                else:
                    synthetic_list.append(synthetic)

        # Clear cache
        torch.cuda.empty_cache()

        print(f"  Generated {len(synthetic_list)} examples so far...")

    return synthetic_list


def save_synthetic_data(synthetic_list: list[dict], output_file: str):
    """Save synthetic examples to JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in synthetic_list:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Saved {len(synthetic_list)} examples to {output_file}")


def main(
    model_path: str = "gemma4-reap-agent-ft-final",
    data_files: list[str] = ["real_dataset_split/real_dataset_part1.jsonl",
                               "real_dataset_split/real_dataset_part2.jsonl"],
    output_file: str = "synthetic_agent_data.jsonl",
    target_examples: int = 10000,
    batch_size: int = 4
):
    """Main generation pipeline."""
    print("="*60)
    print("Synthetic Agent Data Generation")
    print("="*60)

    # Load model
    model, tokenizer = load_model(model_path)

    # Load real data
    real_data = load_real_data(data_files)

    # Shuffle for diversity
    random.shuffle(real_data)

    # Calculate how many real examples we need to process
    examples_to_process = min(target_examples // 2, len(real_data))
    selected_examples = real_data[:examples_to_process]

    print(f"\nGenerating synthetic data...")
    print(f"  Real examples: {len(selected_examples)}")
    print(f"  Target synthetic: ~{target_examples}")
    print(f"  Batch size: {batch_size}")
    print()

    # Generate synthetic data
    synthetic_list = generate_synthetic_batch(
        model,
        tokenizer,
        selected_examples,
        batch_size=batch_size
    )

    # Save results
    save_synthetic_data(synthetic_list, output_file)

    print("\n" + "="*60)
    print(f"✓ Complete! Generated {len(synthetic_list)} synthetic examples")
    print("="*60)


if __name__ == "__main__":
    import random
    random.seed(42)

    # Run generation
    main(
        model_path="gemma4-reap-agent-ft-final",  # Change to your model path
        data_files=[
            "real_dataset_split/real_dataset_part1.jsonl",
            "real_dataset_split/real_dataset_part2.jsonl"
        ],
        output_file="synthetic_agent_data.jsonl",
        target_examples=10000,
        batch_size=4
    )
