"""CSV prediction entry point: python -m triage.predict --input ... --output ..."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from triage.classifier import DATA_PATH, estimate_fraud_threshold, fit_model, load_rows, predict_route


def main() -> None:
    parser = argparse.ArgumentParser(description="Route support messages to a team")
    parser.add_argument("--input", required=True, help="CSV containing a text column")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--fraud-threshold", type=float, default=None, help="Override the grouped-OOF threshold (default: estimate it from training data)")
    args = parser.parse_args()
    if args.fraud_threshold is not None and not 0.0 <= args.fraud_threshold <= 1.0:
        parser.error("--fraud-threshold must be between 0 and 1")
    train_texts, train_labels = load_rows(DATA_PATH)
    model = fit_model(train_texts, train_labels)
    fraud_threshold = args.fraud_threshold
    if fraud_threshold is None:
        fraud_threshold = estimate_fraud_threshold(train_texts, train_labels)
    with Path(args.input).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if "text" not in (reader.fieldnames or []):
            parser.error("Input CSV must contain a text column")
        rows = list(reader)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["text", "prediction"])
        writer.writeheader()
        for row in rows:
            text = row["text"]
            writer.writerow({"text": text, "prediction": predict_route(model, text, fraud_threshold=fraud_threshold)})


if __name__ == "__main__":
    main()
