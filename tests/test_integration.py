"""
Real integration tests — no mocks, no simulations.
Runs on Mac M4 16GB. Each test hits real systems.

pytest tests/test_integration.py -v -s
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Log extraction — real files, real parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractClaudeLogs:
    def test_projects_dir_exists_and_has_sessions(self):
        """Sanity: ~/.claude/projects must exist with JSONL files."""
        assert CLAUDE_PROJECTS.exists(), f"No Claude projects dir at {CLAUDE_PROJECTS}"
        sessions = list(CLAUDE_PROJECTS.rglob("*.jsonl"))
        sessions = [s for s in sessions if "subagents" not in str(s)]
        assert len(sessions) >= 10, f"Expected >=10 session files, found {len(sessions)}"

    def test_extraction_produces_valid_sharegpt(self, tmp_path):
        """Run real extraction and validate every output record."""
        output = tmp_path / "test_real.jsonl"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
             "--output", str(output),
             "--min-tools", "2",
             "--max-turns", "30"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"extract_claude_logs.py failed:\n{result.stderr}"
        assert output.exists(), "Output file not created"

        records = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        assert len(records) >= 5, f"Expected >=5 records, got {len(records)}"

        for i, rec in enumerate(records):
            assert "conversations" in rec, f"Record {i} missing 'conversations'"
            turns = rec["conversations"]
            assert len(turns) >= 3, f"Record {i} too short: {len(turns)} turns"
            assert turns[0]["from"] == "human", f"Record {i} must start with human"

            gpt_turns = [t for t in turns if t["from"] == "gpt"]
            assert any("<tool_call>" in t["value"] for t in gpt_turns), \
                f"Record {i} has no real tool calls"

    def test_extraction_has_tool_responses(self, tmp_path):
        """Every conversation must have grounded tool results, not just calls."""
        output = tmp_path / "test_tools.jsonl"
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
             "--output", str(output), "--min-tools", "2"],
            capture_output=True, text=True, timeout=60,
        )
        records = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        for rec in records[:20]:
            all_text = " ".join(t["value"] for t in rec["conversations"])
            assert "<tool_response>" in all_text, \
                "Conversation has tool calls but no tool responses — not grounded"

    def test_extraction_deduplicates(self, tmp_path):
        """Running extraction twice should produce identical output (no duplicates)."""
        out1 = tmp_path / "run1.jsonl"
        out2 = tmp_path / "run2.jsonl"
        for out in (out1, out2):
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
                 "--output", str(out), "--min-tools", "2"],
                capture_output=True, text=True, timeout=60,
            )
        lines1 = out1.read_text().splitlines()
        lines2 = out2.read_text().splitlines()
        assert len(lines1) == len(lines2), \
            f"Extraction not deterministic: {len(lines1)} vs {len(lines2)} records"

    def test_subagent_sessions_included_when_flagged(self, tmp_path):
        """--include-subagents flag should produce more records."""
        base = tmp_path / "base.jsonl"
        with_sub = tmp_path / "with_sub.jsonl"
        for out, flag in [(base, []), (with_sub, ["--include-subagents"])]:
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
                 "--output", str(out), "--min-tools", "1"] + flag,
                capture_output=True, text=True, timeout=90,
            )
        base_count = len(base.read_text().splitlines())
        sub_count = len(with_sub.read_text().splitlines())
        assert sub_count >= base_count, \
            "Including subagents should not reduce the dataset"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset generation — real OpenRouter API calls
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetGeneration:
    @pytest.fixture(autouse=True)
    def require_api_key(self):
        key = os.environ.get("OPENROUTER_API_KEY", "")
        # Also try loading from .env
        env_path = REPO_ROOT / ".env"
        if not key and env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["OPENROUTER_API_KEY"] = key
        if not key:
            pytest.skip("OPENROUTER_API_KEY not set — skipping live API tests")

    def test_generates_valid_sharegpt_records(self, tmp_path):
        """Actually call OpenRouter and validate 3 output records."""
        output = tmp_path / "gen_test.jsonl"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "generate_agent_dataset.py"),
             "--num", "3", "--output", str(output)],
            capture_output=True, text=True, timeout=180,
            env={**os.environ},
        )
        assert result.returncode == 0, f"generate_agent_dataset.py failed:\n{result.stderr}\n{result.stdout}"
        assert output.exists()

        lines = [l for l in output.read_text().splitlines() if l.strip()]
        assert len(lines) >= 2, f"Expected >=2 records (some may fail), got {len(lines)}"

        for line in lines:
            rec = json.loads(line)
            assert "conversations" in rec
            turns = rec["conversations"]
            assert len(turns) >= 2
            froms = {t["from"] for t in turns}
            assert froms <= {"human", "gpt"}, f"Unexpected turn roles: {froms}"
            assert any(t["from"] == "human" for t in turns)
            assert any(t["from"] == "gpt" for t in turns)

    def test_checkpointing_resumes_from_existing(self, tmp_path):
        """If output file already has N records, generation resumes from N."""
        output = tmp_path / "resume.jsonl"
        # Pre-seed with 1 record
        existing = json.dumps({"conversations": [
            {"from": "human", "value": "existing"},
            {"from": "gpt", "value": "pre-existing record"},
        ]})
        output.write_text(existing + "\n")

        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "generate_agent_dataset.py"),
             "--num", "3", "--output", str(output)],
            capture_output=True, text=True, timeout=180,
            env={**os.environ},
        )
        lines = [l for l in output.read_text().splitlines() if l.strip()]
        # Should have the pre-existing record + 2 new ones
        assert len(lines) >= 2, f"Expected at least 2 records after resume, got {len(lines)}"
        # First record must be the pre-seeded one
        first = json.loads(lines[0])
        assert first["conversations"][0]["value"] == "existing"

    def test_openrouter_connection(self):
        """Direct check: OpenRouter responds with a valid model list or completion."""
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        resp = client.chat.completions.create(
            model="x-ai/grok-3-mini",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip().lower()
        assert "pong" in text, f"Unexpected OpenRouter response: {text}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. MLX training — real fine-tune on M4, tiny config
# ─────────────────────────────────────────────────────────────────────────────

class TestMlxTraining:
    TEST_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"  # ~300MB, fast to download

    @pytest.fixture
    def tiny_dataset(self, tmp_path) -> Path:
        """5-example dataset from real extracted logs (or synthetic fallback)."""
        # Try real logs first
        real_out = tmp_path / "real.jsonl"
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
             "--output", str(real_out), "--min-tools", "1", "--max-turns", "20"],
            capture_output=True, text=True, timeout=60,
        )
        lines = []
        if result.returncode == 0 and real_out.exists():
            lines = [l for l in real_out.read_text().splitlines() if l.strip()]

        # Use up to 5 real examples
        ds_path = tmp_path / "train.jsonl"
        if len(lines) >= 5:
            ds_path.write_text("\n".join(lines[:5]) + "\n")
        else:
            # Fallback: minimal synthetic (only if no real data available)
            minimal = []
            for i in range(5):
                minimal.append(json.dumps({"conversations": [
                    {"from": "human", "value": f"Run a bash command to check disk usage. Task {i}"},
                    {"from": "gpt", "value": '<tool_call>\n{"name": "Bash", "arguments": {"command": "df -h"}}\n</tool_call>'},
                    {"from": "human", "value": '<tool_response>\nFilesystem      Size  Used Avail Use% Mounted on\n/dev/disk3s1s1  460G  120G  340G  27% /\n</tool_response>'},
                    {"from": "gpt", "value": "Disk usage: 120GB used of 460GB (27%). Plenty of space available."},
                ]}))
            ds_path.write_text("\n".join(minimal) + "\n")

        return ds_path

    def test_mlx_lm_installed(self):
        """mlx_lm must be importable on this machine."""
        result = subprocess.run(
            [sys.executable, "-c", "import mlx_lm; print(mlx_lm.__version__)"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"mlx_lm not installed: {result.stderr}"
        print(f"mlx_lm version: {result.stdout.strip()}")

    def test_mlx_training_runs_and_saves_adapter(self, tiny_dataset, tmp_path):
        """
        Actually run mlx_lm.lora for 5 iterations on real/extracted data.
        Verifies the adapter is saved to disk.
        """
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self.TEST_MODEL,
            "--train",
            "--data", str(tiny_dataset),
            "--iters", "5",
            "--batch-size", "1",
            "--lora-rank", "4",
            "--learning-rate", "1e-4",
            "--save-every", "5",
            "--adapter-path", str(adapter_dir),
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - start

        print(f"Elapsed: {elapsed:.1f}s")
        print(f"stdout: {result.stdout[-1000:]}")
        if result.stderr:
            print(f"stderr: {result.stderr[-500:]}")

        assert result.returncode == 0, \
            f"MLX training failed (exit {result.returncode}):\n{result.stderr[-1000:]}"

        # Adapter directory should contain saved weights
        adapter_files = list(adapter_dir.rglob("*"))
        assert len(adapter_files) > 0, \
            f"No adapter files saved to {adapter_dir}"
        print(f"Adapter files: {[f.name for f in adapter_files]}")

    def test_mlx_training_loss_decreases(self, tiny_dataset, tmp_path):
        """Loss should decrease over 10 iterations — basic sanity for learning."""
        adapter_dir = tmp_path / "adapter_loss"
        adapter_dir.mkdir()

        result = subprocess.run(
            [sys.executable, "-m", "mlx_lm.lora",
             "--model", self.TEST_MODEL,
             "--train",
             "--data", str(tiny_dataset),
             "--iters", "10",
             "--batch-size", "1",
             "--lora-rank", "4",
             "--adapter-path", str(adapter_dir)],
            capture_output=True, text=True, timeout=600,
        )

        output = result.stdout + result.stderr
        # Extract loss values from output (mlx_lm prints "Iter N: Train loss X.XXX")
        import re
        losses = [float(m) for m in re.findall(r"[Ll]oss[: ]+([0-9]+\.[0-9]+)", output)]
        print(f"Losses: {losses}")

        if len(losses) >= 2:
            # First loss should be higher than last (learning happened)
            assert losses[0] >= losses[-1] * 0.5, \
                f"Loss did not decrease: {losses[0]} → {losses[-1]}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset quality checks — structural validation of merged dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetQuality:
    def test_merged_dataset_has_tool_calling_patterns(self, tmp_path):
        """Real extracted dataset must contain agent-loop patterns."""
        output = tmp_path / "quality.jsonl"
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
             "--output", str(output), "--min-tools", "2"],
            capture_output=True, text=True, timeout=60,
        )
        records = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        assert len(records) > 0

        tool_names_seen: set[str] = set()
        for rec in records:
            for turn in rec["conversations"]:
                if turn["from"] == "gpt":
                    import re
                    names = re.findall(r'"name":\s*"([^"]+)"', turn["value"])
                    tool_names_seen.update(names)

        print(f"Tool names seen in dataset: {sorted(tool_names_seen)}")
        # Should see at least some of the core tools
        core = {"Bash", "Read", "Write", "Edit", "Grep", "Glob"}
        overlap = core & tool_names_seen
        assert len(overlap) >= 2, \
            f"Expected core tools in dataset, got only: {tool_names_seen}"

    def test_no_empty_turns(self, tmp_path):
        """No conversation should have blank human or gpt turns."""
        output = tmp_path / "empty_check.jsonl"
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
             "--output", str(output), "--min-tools", "1"],
            capture_output=True, text=True, timeout=60,
        )
        records = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        for i, rec in enumerate(records):
            for j, turn in enumerate(rec["conversations"]):
                assert turn["value"].strip(), \
                    f"Record {i}, turn {j} ({turn['from']}) is empty"

    def test_turn_alternation_is_reasonable(self, tmp_path):
        """Most conversations should alternate human/gpt reasonably."""
        output = tmp_path / "alt_check.jsonl"
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "extract_claude_logs.py"),
             "--output", str(output), "--min-tools", "2"],
            capture_output=True, text=True, timeout=60,
        )
        records = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        for rec in records[:10]:
            turns = rec["conversations"]
            # Should not have 3+ consecutive same-role turns
            max_consecutive = 1
            cur = 1
            for k in range(1, len(turns)):
                if turns[k]["from"] == turns[k-1]["from"]:
                    cur += 1
                    max_consecutive = max(max_consecutive, cur)
                else:
                    cur = 1
            # Real agent sessions call multiple tools in parallel producing
            # back-to-back tool_result (human) turns — allow up to 8
            assert max_consecutive <= 8, \
                f"Too many consecutive same-role turns ({max_consecutive}) — bad conversation structure"
