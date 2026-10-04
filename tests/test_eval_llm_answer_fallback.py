"""Verifies the LLM comparison script's documented fallback, not LLM backend behavior itself.

No credentials are required to run this test -- that's the point. See eval/eval_llm_answer.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import eval_llm_answer  # noqa: E402


def test_exits_cleanly_when_no_backend_available(monkeypatch, capsys):
    # Force both backends to report "unavailable" regardless of what's actually in
    # .env on this machine -- GEMINI_API_KEY/HF_TOKEN may be present and genuinely
    # working here, but this test exercises the skip path itself (the brief's
    # "documented fallback / clear skip" allowance), not today's credential state.
    # Stripping os.environ alone wouldn't be enough: _load_env_var() falls back to
    # reading .env off disk, so a real local .env would leak through regardless.
    monkeypatch.setattr(eval_llm_answer, "_build_gemma_generator", lambda: None)
    monkeypatch.setattr(eval_llm_answer, "_build_gemini_generator", lambda: None)

    eval_llm_answer.main()  # must not raise, regardless of key state

    output = capsys.readouterr().out.lower()
    assert "skip" in output
