#!/usr/bin/env python3
"""
Prepare dataset for PC training with Unsloth.
Converts raw ShareGPT format to train/valid/test splits.
"""

import json
from pathlib import Path

def main():
    # Load raw dataset
    records = []
    with open("real_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} conversations")

    # Create splits
    total = len(records)
    train_end = int(total * 0.8)
    valid_end = train_end + int(total * 0.1)

    train = records[:train_end]
    valid = records[train_end:valid_end]
    test = records[valid_end:]

    # Create output directory
    output_dir = Path("unsloth_data")
    output_dir.mkdir(exist_ok=True)

    # Write splits
    for split, data in [("train", train), ("valid", valid), ("test", test)]:
        with open(output_dir / f"{split}.jsonl", "w") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")

    print(f"\nCreated unsloth_data/")
    print(f"  train.jsonl: {len(train)} examples ({len(train)*1.0/total:.1%})")
    print(f"  valid.jsonl: {len(valid)} examples ({len(valid)*1.0/total:.1%})")
    print(f"  test.jsonl: {len(test)} examples ({len(test)*1.0/total:.1%})")
    print(f"\nCopy unsloth_data/ to your PC for training.")

if __name__ == "__main__":
    main()
