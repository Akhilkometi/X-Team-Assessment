"""Verifies the LLM comparison script's documented fallback, not LLM behavior itself.

No API key is required to run this test -- that's the point. See eval/eval_llm_answer.py.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exits_cleanly_regardless_of_key_state():
    # Whether GEMINI_API_KEY is absent, or present in .env but rejected by the API
    # (auth/quota/network), this must never raise -- only ever print why and return 0.
    env_without_key = {k: v for k, v in __import__("os").environ.items() if k != "GEMINI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-m", "eval.eval_llm_answer"],
        cwd=ROOT,
        env=env_without_key,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "skip" in (result.stdout + result.stderr).lower()
