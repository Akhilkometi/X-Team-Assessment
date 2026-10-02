"""Backward-compatible entry point for the leakage-aware triage evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from triage.classifier import DATA_PATH, estimate_fraud_threshold, fit_model, load_rows, predict_route  # noqa: E402


def predict(text: str) -> str:
    """Train on the available labeled data and return a safety-aware route."""
    texts, labels = load_rows(DATA_PATH)
    model = fit_model(texts, labels)
    threshold = estimate_fraud_threshold(texts, labels)
    return predict_route(model, text, fraud_threshold=threshold)


def main() -> None:
    from eval.eval_classifier import main as evaluate

    evaluate()


if __name__ == "__main__":
    main()
