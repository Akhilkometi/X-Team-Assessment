from pathlib import Path

import pytest

pytest.importorskip("whisper", reason="Part C (voiceqa) is optional; pip install -r requirements-voiceqa.txt to run these")

from triage.classifier import DATA_PATH, fit_model, load_rows, predict_route  # noqa: E402
from voiceqa.asr import transcribe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VOICE_NOTES = ROOT / "media" / "voice"


@pytest.fixture(scope="module")
def model():
    texts, labels = load_rows(DATA_PATH)
    return fit_model(texts, labels)


def test_asr_transcribes_readable_text_from_voice_01():
    text = transcribe(VOICE_NOTES / "voice-01.mp3").lower()
    assert "withdrawal" in text
    assert "wallet" in text


def test_asr_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        transcribe(VOICE_NOTES / "does-not-exist.mp3")


def test_asr_silent_audio_transcribes_to_empty_text(tmp_path):
    # Mirrors mediaqa's empty-OCR-text short-circuit: silence must not crash the model.
    import subprocess

    silent = tmp_path / "silent.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", "-q:a", "9", str(silent)],
        check=True,
        capture_output=True,
    )
    assert transcribe(silent) == ""


def test_fraud_voice_note_routes_to_fraud_report(model):
    # "a withdrawal... I definitely did not make" -- caught by the keyword safety net,
    # the same did-not-make pattern Part A's error analysis added.
    text = transcribe(VOICE_NOTES / "voice-01.mp3")
    assert predict_route(model, text) == "fraud-report"


def test_staking_question_voice_note_routes_to_general(model):
    text = transcribe(VOICE_NOTES / "voice-02.mp3")
    assert predict_route(model, text) == "general"


def test_fee_complaint_voice_note_routes_to_fraud_report(model):
    # Worth flagging, not just asserting: the caller explicitly says "I am not saying
    # anybody stole from me" -- this is a fee/slippage complaint, not a fraud report.
    # It still routes to fraud-report because the keyword rule's "scam\w*" pattern
    # fires on "I am being scammed with these fees", a hedge that never appears in the
    # written-ticket training corpus. This is the keyword rule's documented
    # false-positive risk materializing on a different modality -- see the README.
    text = transcribe(VOICE_NOTES / "voice-03.mp3")
    assert predict_route(model, text) == "fraud-report"
