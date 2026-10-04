"""Local Whisper ASR -- DEMO ONLY, not the Part C submission.

Part C's chosen modality is screenshots (see mediaqa/ocr.py and the README's Part C
section for why voice was NOT selected: ASR cost, accent/noise error, consent/retention
questions). This module exists solely so the local demo UI can show, live, what that
trade-off actually looks like -- including its real failure mode (see the voice-03
example in the README: "scammed" fires the fraud keyword net on a message that is
actually a fee dispute, not fraud -- an honest demonstration, not a bug to silently
patch around).

No network call -- openai-whisper runs the model on-device.
"""
from __future__ import annotations

from pathlib import Path

WHISPER_MODEL_NAME = "base.en"
_model = None  # loaded lazily, once, on first use


def _get_model():
    global _model
    if _model is None:
        import whisper

        _model = whisper.load_model(WHISPER_MODEL_NAME)
    return _model


def transcribe(audio_path: str | Path) -> str:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    model = _get_model()
    result = model.transcribe(str(path), fp16=False)
    return result["text"].strip()
