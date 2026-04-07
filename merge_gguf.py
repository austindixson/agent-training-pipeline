#!/usr/bin/env python3
"""
Merge LoRA adapter back into base model and export to GGUF (q4_k_m).
Requires Unsloth — run on the same PC used for training.

Usage:
    python merge_gguf.py
    python merge_gguf.py --adapter ./qwen3-agent-finetune --output qwen3-agent.gguf --quant q8_0
"""

import argparse
from pathlib import Path

from unsloth import FastLanguageModel

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit"
DEFAULT_ADAPTER_DIR = "qwen3-agent-finetune"
DEFAULT_GGUF_FILE = "qwen3-agent.gguf"
DEFAULT_QUANT = "q4_k_m"      # good balance: quality vs size (~5GB)
MAX_SEQ_LEN = 4096


# ── Merge + export ────────────────────────────────────────────────────────────

def merge_and_export(
    base_model: str,
    adapter_dir: str,
    gguf_file: str,
    quant: str,
) -> None:
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}\n"
            "Run train.py first."
        )

    print(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Loading LoRA adapter: {adapter_dir}")
    model = model.load_adapter(adapter_dir)

    print(f"Merging adapter into base weights ...")
    # save_pretrained_gguf handles merge + quantize + export in one step
    model.save_pretrained_gguf(
        gguf_file.replace(".gguf", ""),   # Unsloth appends extension
        tokenizer,
        quantization_method=quant,
    )

    # Unsloth saves as <name>-<quant>.gguf — locate the actual file
    candidates = list(Path(".").glob(f"*{quant}*.gguf"))
    if candidates:
        actual = candidates[0]
        print(f"\nGGUF exported: {actual}  ({actual.stat().st_size / 1e9:.2f} GB)")
        print(f"\nNext steps:")
        print(f"  ollama create qwen3-agent -f Modelfile")
        print(f"  ollama run qwen3-agent")
    else:
        print(f"\nExport complete. Look for *.gguf in current directory.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA + export GGUF")
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL, help="Base model ID or path")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR, help="LoRA adapter directory")
    parser.add_argument("--output", default=DEFAULT_GGUF_FILE, help="Output GGUF filename")
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization method",
    )
    args = parser.parse_args()

    merge_and_export(
        base_model=args.base,
        adapter_dir=args.adapter,
        gguf_file=args.output,
        quant=args.quant,
    )


if __name__ == "__main__":
    main()
