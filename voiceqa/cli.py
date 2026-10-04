"""Batch entry point: python -m voiceqa.cli --input-dir media/voice --output voice_predictions.csv

Reuses the exact Part A pipeline (triage.classifier) on transcribed text -- same fitted
model, same keyword safety net, same fraud-threshold estimation. ASR is the only new
component; routing logic is not duplicated or reimplemented for audio, the same
pattern mediaqa.cli uses for screenshots.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from triage.classifier import DATA_PATH, estimate_fraud_threshold, fit_model, load_rows, predict_route
from voiceqa.asr import transcribe

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Route support voice notes via local ASR + the Part A classifier")
    parser.add_argument("--input-dir", required=True, help="Directory of voice note audio files")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--fraud-threshold", type=float, default=None, help="Override the estimated threshold")
    args = parser.parse_args()
    if args.fraud_threshold is not None and not 0.0 <= args.fraud_threshold <= 1.0:
        parser.error("--fraud-threshold must be between 0 and 1")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"--input-dir is not a directory: {input_dir}")
    audio_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not audio_files:
        parser.error(f"No audio files found in {input_dir}")

    train_texts, train_labels = load_rows(DATA_PATH)
    model = fit_model(train_texts, train_labels)
    fraud_threshold = args.fraud_threshold
    if fraud_threshold is None:
        fraud_threshold = estimate_fraud_threshold(train_texts, train_labels)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["audio", "transcript", "prediction"])
        writer.writeheader()
        for audio_path in audio_files:
            text = transcribe(audio_path).strip()
            prediction = predict_route(model, text, fraud_threshold=fraud_threshold) if text else ""
            writer.writerow({"audio": audio_path.name, "transcript": text, "prediction": prediction})


if __name__ == "__main__":
    main()
