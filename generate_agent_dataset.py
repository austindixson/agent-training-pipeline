#!/usr/bin/env python3
"""
Dataset generator for agent fine-tuning.
Primary: OpenRouter API (Grok or any available model)
Fallback: Claude CLI via subprocess (uses existing login, no API key needed)
Output: JSONL in ShareGPT format compatible with Unsloth + MLX
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from openai import OpenAI

# Auto-load .env from script directory (puts OPENROUTER_API_KEY into environ)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ──────────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "x-ai/grok-3-mini"   # change to any OpenRouter model
OUTPUT_FILE = "agent_dataset.jsonl"
CHECKPOINT_EVERY = 10

# Agent training scenario templates
SCENARIOS = [
    # Tool use
    "You need to search the web, summarize results, and answer a user question about {topic}.",
    "Use the file_read tool to inspect a config file, then use file_write to update a setting for {topic}.",
    "Call the bash_exec tool to run a shell command related to {topic}, parse the output, and report findings.",
    "Use multiple tools in sequence: search → read → analyze → write a report about {topic}.",
    # Error recovery
    "A tool call fails with a timeout. Retry with exponential backoff and explain your reasoning for {topic}.",
    "You receive malformed JSON from a tool. Parse what you can, flag the issue, and continue for {topic}.",
    "An API returns a 429 rate limit error. Implement a fallback strategy for {topic}.",
    # Memory / multi-turn
    "Across 3 turns, help the user refine their plan for {topic}. Remember previous constraints they mentioned.",
    "The user contradicts themselves between turn 1 and turn 3 about {topic}. Politely surface the conflict.",
    "Summarize a long conversation about {topic} into a concise action plan.",
    # JSON output
    "Return a structured JSON plan with keys: goal, steps, tools_needed, risks for {topic}.",
    "Parse user intent and output a tool call as JSON: {{name, args}} for {topic}.",
    "Generate a task graph JSON with dependencies for completing {topic}.",
    # Self-correction
    "You made a mistake in step 2 of a {topic} task. Detect it, explain what went wrong, and redo correctly.",
    "Your previous answer was incomplete. A follow-up message points this out. Revise for {topic}.",
    # Planning
    "Break down a complex goal about {topic} into subtasks with owners, deadlines, and success criteria.",
    "Given constraints (budget, time, tools), produce the optimal plan for {topic}.",
]

TOPICS = [
    "setting up a Python development environment",
    "debugging a memory leak in a Node.js service",
    "migrating a PostgreSQL database schema",
    "building a REST API with FastAPI",
    "configuring a CI/CD pipeline with GitHub Actions",
    "analyzing server logs for anomalies",
    "scraping and cleaning a dataset",
    "deploying a Docker container to a cloud provider",
    "writing unit tests for a payment service",
    "refactoring a monolith into microservices",
    "optimizing a slow SQL query",
    "setting up Ollama with a custom model",
    "fine-tuning an LLM on custom data",
    "building a RAG pipeline",
    "monitoring a production API with Prometheus",
]

SYSTEM_PROMPT = """You are an expert AI assistant that is highly capable of using tools, \
planning multi-step tasks, recovering from errors, producing structured JSON output, \
and maintaining context across long conversations. \
Always think step by step, use tools when available, and self-correct when you detect mistakes."""


# ── OpenRouter client ────────────────────────────────────────────────────────

def make_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)


def generate_via_openrouter(client: OpenAI, scenario: str) -> list[dict]:
    """Returns list of {from, value} conversation turns."""
    prompt = (
        f"Generate a realistic multi-turn conversation (3-6 turns) between a user and an AI agent.\n"
        f"Scenario: {scenario}\n\n"
        f"Format as JSON array with objects: {{\"from\": \"human\" or \"gpt\", \"value\": \"...\"}}\n"
        f"The agent should use tools (represented as JSON blocks), think step-by-step, "
        f"and produce high-quality responses. Start with the human turn."
    )
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content.strip()
    # Extract JSON array from response
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found in response: {raw[:200]}")
    return json.loads(raw[start:end])


# ── Claude CLI fallback ──────────────────────────────────────────────────────

def generate_via_claude_cli(scenario: str) -> list[dict]:
    """Falls back to local Claude CLI (no API key required, uses existing login)."""
    prompt = (
        f"Generate a realistic multi-turn conversation (3-6 turns) between a user and an AI agent.\n"
        f"Scenario: {scenario}\n\n"
        f"Format as a JSON array with objects: {{\"from\": \"human\" or \"gpt\", \"value\": \"...\"}}\n"
        f"The agent should use tools (represented as JSON blocks), think step-by-step, "
        f"and produce high-quality responses. Start with the human turn.\n"
        f"Return ONLY the JSON array, no other text."
    )
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,  # some prompts take >60s on slower inference
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr[:200]}")
    raw = result.stdout.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in Claude CLI output: {raw[:200]}")
    return json.loads(raw[start:end])


# ── Generation loop ──────────────────────────────────────────────────────────

def build_scenario(idx: int) -> str:
    scenario_tmpl = SCENARIOS[idx % len(SCENARIOS)]
    topic = TOPICS[idx % len(TOPICS)]
    return scenario_tmpl.format(topic=topic)


def load_existing(output_path: Path) -> int:
    """Return count of already-generated examples."""
    if not output_path.exists():
        return 0
    count = 0
    with output_path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def generate_dataset(num: int, output_file: str = OUTPUT_FILE) -> None:
    output_path = Path(output_file)
    already_done = load_existing(output_path)
    if already_done > 0:
        print(f"Resuming from checkpoint: {already_done} examples already generated.")

    try:
        client = make_openrouter_client()
        openrouter_available = True
    except ValueError:
        print("OPENROUTER_API_KEY not set — will use Claude CLI for all examples.")
        client = None
        openrouter_available = False

    generated = already_done
    failed = 0

    with output_path.open("a") as f:
        for i in range(already_done, num):
            scenario = build_scenario(i)
            print(f"[{i+1}/{num}] {scenario[:80]}...", end=" ", flush=True)

            conversations = None

            # Try OpenRouter first
            if openrouter_available:
                try:
                    conversations = generate_via_openrouter(client, scenario)
                    print("✓ openrouter", flush=True)
                except Exception as e:
                    print(f"✗ openrouter ({e.__class__.__name__}) → trying claude CLI", flush=True)

            # Fallback to Claude CLI
            if conversations is None:
                try:
                    conversations = generate_via_claude_cli(scenario)
                    print("✓ claude-cli", flush=True)
                except Exception as e:
                    print(f"✗ claude-cli ({e.__class__.__name__}) — skipping", flush=True)
                    failed += 1
                    continue

            # Validate structure
            if not isinstance(conversations, list) or len(conversations) < 2:
                print("  ✗ invalid conversation structure — skipping")
                failed += 1
                continue

            record = {"conversations": conversations}
            f.write(json.dumps(record) + "\n")
            generated += 1

            # Checkpoint flush every N examples
            if generated % CHECKPOINT_EVERY == 0:
                f.flush()
                print(f"  [checkpoint] {generated} examples saved")

            # Small delay to avoid rate limits
            time.sleep(0.5)

    print(f"\nDone. Generated: {generated}, Failed: {failed}, Output: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global OPENROUTER_MODEL  # must be first use of the name in this scope
    default_model = OPENROUTER_MODEL

    parser = argparse.ArgumentParser(description="Generate agent fine-tuning dataset")
    parser.add_argument("--num", type=int, default=300, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output JSONL file")
    parser.add_argument("--model", type=str, default=default_model, help="OpenRouter model ID")
    args = parser.parse_args()

    OPENROUTER_MODEL = args.model
    generate_dataset(args.num, args.output)


if __name__ == "__main__":
    main()
