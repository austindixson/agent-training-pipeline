#!/usr/bin/env python3
"""
Extract real Claude Code conversation logs from ~/.claude/projects/ and convert
to ShareGPT JSONL format for fine-tuning.

Targets sessions that contain actual tool-calling agent loops (Bash, Read, Write,
Edit, Grep, Glob, Agent, etc.) — the exact behavior we want the model to learn.

Output format per line:
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

Usage:
    python extract_claude_logs.py
    python extract_claude_logs.py --output real_dataset.jsonl --min-tools 2 --max-turns 40
"""

import argparse
import json
import re
from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DEFAULT_OUTPUT = "real_dataset.jsonl"

# Tools that indicate real agent work (not just text replies)
AGENT_TOOLS = {
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "Agent", "WebFetch", "WebSearch", "TodoWrite",
    "mcp__context7__query-docs", "mcp__lossless-recall__recall_grep",
}


def render_tool_call(tool_use: dict) -> str:
    """Render a tool_use block as readable text for the gpt turn."""
    name = tool_use.get("name", "UnknownTool")
    inp = tool_use.get("input", {})
    lines = [f"<tool_call>", f'{{"name": "{name}", "arguments": {json.dumps(inp)}}}', "</tool_call>"]
    return "\n".join(lines)


def render_tool_result(tool_result: dict) -> str:
    """Render a tool_result block as readable text for the human turn."""
    content = tool_result.get("content", "")
    if isinstance(content, list):
        # List of content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        content = "\n".join(parts)
    # Trim very long outputs (log spam, binary data)
    if len(content) > 3000:
        content = content[:2500] + "\n...[truncated]..."
    return f"<tool_response>\n{content}\n</tool_response>"


def extract_text(content) -> str:
    """Extract plain text from a content value (str or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", "").strip())
            elif btype == "thinking":
                pass  # skip internal thinking
            elif btype == "tool_use":
                parts.append(render_tool_call(block))
            elif btype == "tool_result":
                parts.append(render_tool_result(block))
        return "\n".join(p for p in parts if p)
    return ""


def session_has_tool_calls(messages: list[dict], min_tools: int) -> bool:
    tool_count = 0
    for m in messages:
        content = m.get("message", {}).get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("name") in AGENT_TOOLS:
                        tool_count += 1
    return tool_count >= min_tools


def parse_session(path: Path) -> list[dict]:
    """Parse a JSONL session file into a list of message dicts."""
    messages = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only keep actual conversation messages
        if obj.get("type") not in ("user", "assistant"):
            continue
        msg = obj.get("message", {})
        if not msg:
            continue
        messages.append(obj)
    return messages


def messages_to_sharegpt(messages: list[dict]) -> list[dict]:
    """
    Convert raw session messages to ShareGPT conversation turns.

    Strategy:
    - user role messages with plain text → human turn
    - user role messages with tool_result → append to previous human turn or new human turn
    - assistant role messages → gpt turn (text + tool calls inline)
    - Skip system/empty/thinking-only turns
    """
    turns = []
    pending_tool_results: list[str] = []

    for obj in messages:
        msg = obj.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            # Collect tool results
            if isinstance(content, list):
                results = [render_tool_result(b) for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                plain = " ".join(b.get("text", "").strip() for b in content if isinstance(b, dict) and b.get("type") == "text").strip()
                if results:
                    pending_tool_results.extend(results)
                if plain:
                    # Flush pending results first
                    if pending_tool_results:
                        turns.append({"from": "human", "value": "\n\n".join(pending_tool_results)})
                        pending_tool_results = []
                    turns.append({"from": "human", "value": plain})
            elif isinstance(content, str) and content.strip():
                if pending_tool_results:
                    turns.append({"from": "human", "value": "\n\n".join(pending_tool_results)})
                    pending_tool_results = []
                turns.append({"from": "human", "value": content.strip()})

        elif role == "assistant":
            # Flush any pending tool results as a human turn first
            if pending_tool_results:
                turns.append({"from": "human", "value": "\n\n".join(pending_tool_results)})
                pending_tool_results = []

            text = extract_text(content)
            if text:
                turns.append({"from": "gpt", "value": text})

    # Flush remaining
    if pending_tool_results:
        turns.append({"from": "human", "value": "\n\n".join(pending_tool_results)})

    return turns


def is_quality_conversation(turns: list[dict], min_tools: int) -> bool:
    """Filter for conversations that are genuinely useful training examples."""
    if len(turns) < 3:
        return False

    # Must have at least one gpt turn with a tool call
    tool_call_turns = sum(1 for t in turns if t["from"] == "gpt" and "<tool_call>" in t["value"])
    if tool_call_turns < min_tools:
        return False

    # Must have tool responses (real grounding)
    tool_response_turns = sum(1 for t in turns if "<tool_response>" in t["value"])
    if tool_response_turns < 1:
        return False

    # Must start with human
    if turns[0]["from"] != "human":
        return False

    return True


def deduplicate(conversations: list[list[dict]]) -> list[list[dict]]:
    """Remove near-duplicate conversations by first-human-turn fingerprint."""
    seen: set[str] = set()
    unique = []
    for conv in conversations:
        if not conv:
            continue
        key = conv[0]["value"][:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(conv)
    return unique


def extract(
    projects_dir: Path,
    output_file: str,
    min_tools: int,
    max_turns: int,
    include_subagents: bool,
) -> int:
    all_sessions: list[Path] = []

    for jsonl in projects_dir.rglob("*.jsonl"):
        if not include_subagents and "subagents" in str(jsonl):
            continue
        all_sessions.append(jsonl)

    print(f"Found {len(all_sessions)} session files in {projects_dir}")

    conversations: list[list[dict]] = []
    skipped = 0

    for path in all_sessions:
        try:
            messages = parse_session(path)
        except Exception as e:
            skipped += 1
            continue

        if not session_has_tool_calls(messages, min_tools):
            skipped += 1
            continue

        turns = messages_to_sharegpt(messages)

        # Chunk long sessions into windows of max_turns
        for i in range(0, max(1, len(turns) - 4), max_turns // 2):
            window = turns[i : i + max_turns]
            if is_quality_conversation(window, min_tools):
                conversations.append(window)

    conversations = deduplicate(conversations)

    with open(output_file, "w") as f:
        for conv in conversations:
            f.write(json.dumps({"conversations": conv}) + "\n")

    print(f"Extracted {len(conversations)} conversations ({skipped} sessions skipped)")
    print(f"Output: {output_file}")
    return len(conversations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real Claude logs for fine-tuning")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file")
    parser.add_argument("--projects-dir", default=str(CLAUDE_PROJECTS_DIR), help="Claude projects directory")
    parser.add_argument("--min-tools", type=int, default=2, help="Min tool calls per conversation")
    parser.add_argument("--max-turns", type=int, default=40, help="Max turns per training example (windows long sessions)")
    parser.add_argument("--include-subagents", action="store_true", help="Also extract subagent sessions")
    args = parser.parse_args()

    count = extract(
        projects_dir=Path(args.projects_dir),
        output_file=args.output,
        min_tools=args.min_tools,
        max_turns=args.max_turns,
        include_subagents=args.include_subagents,
    )
    print(f"\nDone. {count} training examples ready.")


if __name__ == "__main__":
    main()
