"""Leakage-aware text classifier and template grouping utilities."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
ROUTES = ("account-access", "transaction-dispute", "fraud-report", "general")

# Routing safety net: these patterns deliberately favor escalation to the fraud queue.
FRAUD_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"unauthori\w*",
        r"never\s+authori[sz]ed",
        r"stolen",
        r"phish\w*",
        r"scam\w*",
        r"hack\w*",
        r"drain\w*",
        r"(?:did\s+not|didn't|never)\s+(?:make|made|authori[sz]e[ds]?|initiate[ds]?)\b",
    )
)

_GREETING = re.compile(r"^(?:hi|hello(?:\s+team)?|hey|quick question|please help|urgent)[:,.!\s-]*", re.I)
_SIGNOFF = re.compile(r"(?:\s*(?:thanks|thank you|please advise|let me know|appreciate any help|this is time sensitive)[.!\s]*)+$", re.I)
_ASSET = re.compile(r"\b(?:bitcoin|btc|ethereum|eth|solana|sol|usdc|usdt|ada|cardano|xrp|litecoin|ltc|dogecoin|doge|polygon|matic)\b", re.I)
_AMOUNT = re.compile(r"(?:\$\s*)?\b\d[\d,]*(?:\.\d+)?\b")


def load_rows(path: str | Path = DATA_PATH) -> tuple[list[str], list[str]]:
    """Load and validate labeled training rows."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"text", "label"}.issubset(rows[0]):
        raise ValueError("Training CSV must contain text and label columns")
    texts = [row["text"].strip() for row in rows]
    labels = [row["label"].strip() for row in rows]
    if any(not text for text in texts) or any(label not in ROUTES for label in labels):
        raise ValueError("Training rows contain empty text or an unknown route")
    return texts, labels


def template_key(text: str) -> str:
    """Normalize superficial template variations for grouped evaluation."""
    normalized = text.lower().strip()
    normalized = _GREETING.sub("", normalized)
    normalized = _SIGNOFF.sub("", normalized)
    normalized = _ASSET.sub(" asset ", normalized)
    normalized = _AMOUNT.sub(" amount ", normalized)
    normalized = re.sub(r"\b(?:iphone|android|tablet|laptop|desktop browser|work computer|phone)\b", " device ", normalized)
    normalized = re.sub(r"[^a-z]+", " ", normalized)
    return " ".join(normalized.split())


def build_model(*, class_weight: str | None = "balanced") -> Pipeline:
    """Build an end-to-end pipeline so vectorization is fit only on training folds."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, C=10.0, class_weight=class_weight),
            ),
        ]
    )


def predict_route(model: Pipeline, text: str, *, fraud_threshold: float = 0.25) -> str:
    """Predict a route, applying a high-recall fraud escalation policy."""
    if any(pattern.search(text) for pattern in FRAUD_PATTERNS):
        return "fraud-report"
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    fraud_index = classes.index("fraud-report")
    if probabilities[fraud_index] >= fraud_threshold:
        return "fraud-report"
    return classes[int(probabilities.argmax())]


def fit_model(texts: Iterable[str], labels: Iterable[str]) -> Pipeline:
    model = build_model(class_weight="balanced")
    model.fit(list(texts), list(labels))
    return model


def estimate_fraud_threshold(
    texts: list[str], labels: list[str], *, precision_floor: float = 0.60, seed: int = 42
) -> float:
    """Choose a recall-maximizing fraud cutoff from grouped out-of-fold scores."""
    groups = [template_key(text) for text in texts]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = [0.0] * len(texts)
    for train_indices, valid_indices in splitter.split(texts, labels, groups):
        model = build_model(class_weight="balanced")
        model.fit([texts[index] for index in train_indices], [labels[index] for index in train_indices])
        fraud_column = list(model.classes_).index("fraud-report")
        fold_scores = model.predict_proba([texts[index] for index in valid_indices])[:, fraud_column]
        for index, score in zip(valid_indices, fold_scores):
            scores[int(index)] = float(score)
    truth = list(labels)
    best: tuple[float, float, float] | None = None
    for threshold in sorted({0.0, *scores, 1.0}):
        predicted = [score >= threshold for score in scores]
        true_positive = sum(is_fraud and is_predicted for is_fraud, is_predicted in zip((label == "fraud-report" for label in truth), predicted))
        false_positive = sum(label != "fraud-report" and is_predicted for label, is_predicted in zip(truth, predicted))
        false_negative = sum(label == "fraud-report" and not is_predicted for label, is_predicted in zip(truth, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        if precision >= precision_floor:
            candidate = (recall, precision, threshold)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise ValueError("No fraud threshold meets the requested precision floor")
    return best[2]
