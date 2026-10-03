"""Batch entry point: python -m mediaqa.cli --input-dir media/screenshots --output predictions.csv

Reuses the exact Part A pipeline (triage.classifier) on OCR'd text -- same model, same
keyword safety net, same fraud-threshold estimation. OCR is the only new component;
routing logic is not duplicated.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mediaqa.ocr import extract_text
from triage.classifier import DATA_PATH, estimate_fraud_threshold, fit_model, load_rows, predict_route

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Route support screenshots via OCR + the Part A classifier")
    parser.add_argument("--input-dir", required=True, help="Directory of screenshot images")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--fraud-threshold", type=float, default=None, help="Override the estimated threshold")
    args = parser.parse_args()
    if args.fraud_threshold is not None and not 0.0 <= args.fraud_threshold <= 1.0:
        parser.error("--fraud-threshold must be between 0 and 1")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"--input-dir is not a directory: {input_dir}")
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"No images found in {input_dir}")

    train_texts, train_labels = load_rows(DATA_PATH)
    model = fit_model(train_texts, train_labels)
    fraud_threshold = args.fraud_threshold
    if fraud_threshold is None:
        fraud_threshold = estimate_fraud_threshold(train_texts, train_labels)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["image", "ocr_text", "prediction"])
        writer.writeheader()
        for image_path in images:
            text = extract_text(image_path).strip()
            prediction = predict_route(model, text, fraud_threshold=fraud_threshold) if text else ""
            writer.writerow({"image": image_path.name, "ocr_text": text.replace("\n", " "), "prediction": prediction})


if __name__ == "__main__":
    main()
