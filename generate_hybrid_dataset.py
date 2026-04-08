#!/usr/bin/env python3
"""
Generate hybrid dataset combining real and augmented agent conversations.

Creates a training-ready dataset with:
- Real Claude conversations
- Lightweight augmentations (no GPU required)
- Proper train/valid/test splits
- Multiple format outputs (Unsloth, MLX, ShareGPT)
"""

import json
import random
from pathlib import Path
from typing import Literal
from copy import deepcopy
import re


def load_real_dataset(data_files: list[str]) -> list[dict]:
    """Load real Claude conversations from JSONL files."""
    print(f"Loading real data from: {data_files}")

    real_data = []
    for file_path in data_files:
        path = Path(file_path)
        if not path.exists():
            print(f"  ⚠ File not found: {file_path}")
            continue

        with open(path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "conversations" in data and data["conversations"]:
                        real_data.append(data)
                except json.JSONDecodeError:
                    continue

    print(f"✓ Loaded {len(real_data)} real conversations")
    return real_data


def augment_conversation(conversation: dict, aug_type: Literal["reverse", "shorten", "paraphrase"]) -> dict:
    """Create lightweight augmentation without GPU.

    Args:
        conversation: Original conversation with "conversations" list
        aug_type: Type of augmentation

    Returns:
        Augmented conversation copy
    """
    aug = deepcopy(conversation)

    if aug_type == "reverse":
        # Reverse human/gpt order in later turns (creates new flow)
        convs = aug["conversations"]
        if len(convs) >= 4:
            # Keep first exchange, reverse middle
            aug["conversations"] = (
                convs[:2] +
                list(reversed(convs[2:-1])) +
                [convs[-1]]
            )

    elif aug_type == "shorten":
        # Remove every other message after first (creates concise version)
        convs = aug["conversations"]
        if len(convs) >= 6:
            # Keep pattern: human, gpt, human, gpt (skip middle)
            aug["conversations"] = convs[:2] + convs[-2:]

    elif aug_type == "paraphrase":
        # Simple text paraphrasing (tool calls remain same)
        for msg in aug["conversations"]:
            if msg.get("from") == "human":
                # Add variation prefix
                variations = [
                    "Can you help me with ",
                    "I need to ",
                    "Please ",
                    ""
                ]
                if random.random() > 0.5:
                    msg["value"] = random.choice(variations) + msg["value"]

    return aug


def create_hybrid_dataset(
    real_data: list[dict],
    augmentation_ratio: float = 0.5,
    random_seed: int = 42
) -> list[dict]:
    """Create hybrid dataset with real + augmented data.

    Args:
        real_data: Real conversations
        augmentation_ratio: Ratio of augmented to real data (0.5 = 1:1)
        random_seed: Random seed for reproducibility

    Returns:
        Hybrid dataset
    """
    random.seed(random_seed)

    # Calculate how many augmentations to create
    num_augmented = int(len(real_data) * augmentation_ratio)
    num_to_augment = min(num_augmented, len(real_data))

    print(f"\nCreating hybrid dataset...")
    print(f"  Real: {len(real_data)}")
    print(f"  Augmented: {num_to_augment}")
    print(f"  Total: {len(real_data) + num_to_augment}")

    # Select random subset for augmentation
    selected_indices = random.sample(range(len(real_data)), num_to_augment)

    # Create augmentations
    aug_types = ["reverse", "shorten", "paraphrase"]
    augmented_data = []

    for idx in selected_indices:
        aug_type = random.choice(aug_types)
        try:
            aug = augment_conversation(real_data[idx], aug_type)
            augmented_data.append(aug)
        except Exception:
            continue

    # Combine
    hybrid_data = real_data + augmented_data
    random.shuffle(hybrid_data)

    print(f"✓ Created hybrid dataset: {len(hybrid_data)} total")
    return hybrid_data


def save_unsloth_format(
    data: list[dict],
    output_dir: Path,
    split_ratio: tuple = (0.8, 0.1, 0.1)
):
    """Save in Unsloth/ShareGPT format (JSONL).

    Format expected by Unsloth:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    output_dir = Path(output_dir) / "unsloth_format"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split
    n = len(data)
    train_end = int(n * split_ratio[0])
    valid_end = int(n * (split_ratio[0] + split_ratio[1]))

    splits = {
        "train": data[:train_end],
        "valid": data[train_end:valid_end],
        "test": data[valid_end:]
    }

    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  {split_name}: {len(split_data)} → {output_file}")


def save_mlx_format(
    data: list[dict],
    output_dir: Path,
    split_ratio: tuple = (0.8, 0.1, 0.1)
):
    """Save in MLX format (JSON with train/valid/test lists).

    Format expected by MLX LM:
    {"train": [{"text": "..."}, ...], "valid": [...], "test": [...]}
    """
    output_dir = Path(output_dir) / "mlx_format"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert conversations to text format
    text_data = []
    for conv in data:
        turns = []
        for msg in conv.get("conversations", []):
            role = msg.get("from", "human")
            value = msg.get("value", "")

            if role == "human":
                turns.append(f"User: {value}")
            elif role == "gpt":
                turns.append(f"Assistant: {value}")

        text_data.append({"text": "\n\n".join(turns)})

    # Split
    n = len(text_data)
    train_end = int(n * split_ratio[0])
    valid_end = int(n * (split_ratio[0] + split_ratio[1]))

    dataset = {
        "train": text_data[:train_end],
        "valid": text_data[train_end:valid_end],
        "test": text_data[valid_end:]
    }

    # Save combined JSON
    output_file = output_dir / "dataset.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, ensure_ascii=False)

    print(f"  MLX format: {len(dataset['train'])} train, {len(dataset['valid'])} valid, {len(dataset['test'])} test")
    print(f"  → {output_file}")


def save_sharegpt_format(
    data: list[dict],
    output_dir: Path,
    split_ratio: tuple = (0.8, 0.1, 0.1)
):
    """Save in ShareGPT format (separate JSONL files per split).

    Standard ShareGPT format for training pipelines.
    """
    output_dir = Path(output_dir) / "sharegpt_format"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split
    n = len(data)
    train_end = int(n * split_ratio[0])
    valid_end = int(n * (split_ratio[0] + split_ratio[1]))

    splits = {
        "train": data[:train_end],
        "valid": data[train_end:valid_end],
        "test": data[valid_end:]
    }

    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  {split_name}: {len(split_data)} → {output_file}")


def generate_statistics(data: list[dict]) -> dict:
    """Generate dataset statistics."""
    total_conversations = len(data)
    total_turns = sum(len(d.get("conversations", [])) for d in data)
    total_tokens = sum(
        sum(len(msg.get("value", "").split()) for msg in d.get("conversations", []))
        for d in data
    )

    has_tool_calls = sum(
        1 for d in data
        if any("<tool_use>" in msg.get("value", "") for msg in d.get("conversations", []))
    )

    return {
        "total_conversations": total_conversations,
        "total_turns": total_turns,
        "avg_turns_per_conv": total_turns / total_conversations if total_conversations else 0,
        "total_tokens": total_tokens,
        "conversations_with_tools": has_tool_calls,
        "tool_call_percentage": (has_tool_calls / total_conversations * 100) if total_conversations else 0
    }


def main(
    input_files: list[str] = [
        "real_dataset_split/real_dataset_part1.jsonl",
        "real_dataset_split/real_dataset_part2.jsonl"
    ],
    output_dir: str = "hybrid_dataset",
    augmentation_ratio: float = 0.5,
    formats: list[str] = ["unsloth", "mlx", "sharegpt"],
    split_ratio: tuple = (0.8, 0.1, 0.1)
):
    """Main pipeline."""
    print("="*60)
    print("Hybrid Dataset Generation")
    print("="*60)

    # Load real data
    real_data = load_real_dataset(input_files)

    if not real_data:
        print("❌ No data loaded. Check input files.")
        return

    # Create hybrid dataset
    hybrid_data = create_hybrid_dataset(
        real_data,
        augmentation_ratio=augmentation_ratio
    )

    # Generate statistics
    stats = generate_statistics(hybrid_data)

    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"  Total conversations: {stats['total_conversations']:,}")
    print(f"  Total turns: {stats['total_turns']:,}")
    print(f"  Avg turns/conversation: {stats['avg_turns_per_conv']:.1f}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  With tool calls: {stats['conversations_with_tools']:,} ({stats['tool_call_percentage']:.1f}%)")

    # Save in multiple formats
    print("\n" + "="*60)
    print("Saving Datasets")
    print("="*60)

    output_path = Path(output_dir)

    if "unsloth" in formats:
        print("\nUnsloth format:")
        save_unsloth_format(hybrid_data, output_path, split_ratio)

    if "mlx" in formats:
        print("\nMLX format:")
        save_mlx_format(hybrid_data, output_path, split_ratio)

    if "sharegpt" in formats:
        print("\nShareGPT format:")
        save_sharegpt_format(hybrid_data, output_path, split_ratio)

    # Save combined file for reference
    combined_file = output_path / "hybrid_combined.jsonl"
    with open(combined_file, "w") as f:
        for item in hybrid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Combined: {len(hybrid_data)} → {combined_file}")

    # Save metadata
    metadata = {
        "total_conversations": len(hybrid_data),
        "augmentation_ratio": augmentation_ratio,
        "split_ratio": split_ratio,
        "statistics": stats,
        "formats": list(formats)
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Metadata → {metadata_file}")

    print("\n" + "="*60)
    print("✓ Hybrid dataset created successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"\nUse for training:")
    print(f"  PC (Unsloth): {output_path}/unsloth_format/")
    print(f"  Mac (MLX):    {output_path}/mlx_format/dataset.json")
    print(f"  General:      {output_path}/sharegpt_format/")


if __name__ == "__main__":
    main(
        input_files=[
            "real_dataset_split/real_dataset_part1.jsonl",
            "real_dataset_split/real_dataset_part2.jsonl"
        ],
        output_dir="hybrid_dataset",
        augmentation_ratio=0.5,  # 50% augmented data
        formats=["unsloth", "mlx", "sharegpt"],
        split_ratio=(0.8, 0.1, 0.1)  # 80% train, 10% valid, 10% test
    )
