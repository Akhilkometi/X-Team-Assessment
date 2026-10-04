"""Local Whisper ASR for in-app voice notes. No network call, no hosted transcription API."""
from __future__ import annotations

from pathlib import Path

import whisper

_WHISPER_MODEL_NAME = "base"
_model_cache: "whisper.Whisper | None" = None


def _get_model() -> "whisper.Whisper":
    """Lazily load and cache the ASR model -- mirrors kbqa.retrieve._get_model's pattern
    so repeated transcribe() calls (e.g. in tests or a batch run) don't reload it each time."""
    global _model_cache
    if _model_cache is None:
        _model_cache = whisper.load_model(_WHISPER_MODEL_NAME)
    return _model_cache


def transcribe(audio_path: str | Path) -> str:
    """Transcribe a local audio file to text, entirely on-device."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")
    model = _get_model()
    result = model.transcribe(str(path), fp16=False)
    return str(result["text"]).strip()
