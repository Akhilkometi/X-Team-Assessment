"""Exploratory comparison: Logistic Regression vs XGBoost vs CatBoost (native text mode).

Not part of the shipped pipeline. triage/classifier.py and requirements.txt are
untouched by this file's conclusions; it exists to make the model choice an
honest, re-runnable comparison instead of an assertion.

Requires xgboost and catboost, which are intentionally NOT in requirements.txt
(the shipped predictor only needs scikit-learn). Install them to re-run:

    pip install xgboost catboost

Usage:
    python -m eval.model_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triage.classifier import ROUTES, load_rows, template_key  # noqa: E402

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None

SEEDS = (11, 23, 37, 51, 71)
PRECISION_FLOOR = 0.60


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=ROUTES, average="macro", zero_division=0),
        "fraud_precision": precision_score(y_true, y_pred, labels=["fraud-report"], average=None, zero_division=0)[0],
        "fraud_recall": recall_score(y_true, y_pred, labels=["fraud-report"], average=None, zero_division=0)[0],
    }


def choose_threshold(y_true: list[str], fraud_scores: np.ndarray) -> float:
    """Same policy as eval_classifier.py: maximize recall subject to a precision floor."""
    truth = np.asarray(y_true)
    candidates = sorted(set([0.0, *map(float, fraud_scores), 1.0]))
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = fraud_scores >= threshold
        tp = int(np.sum(predicted & (truth == "fraud-report")))
        fp = int(np.sum(predicted & (truth != "fraud-report")))
        fn = int(np.sum(~predicted & (truth == "fraud-report")))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision >= PRECISION_FLOOR:
            feasible.append((recall, precision, threshold))
    return max(feasible)[2] if feasible else 1.0


def apply_threshold(probabilities: np.ndarray, classes: list[str], threshold: float) -> list[str]:
    fraud_index = classes.index("fraud-report")
    result = []
    for row in probabilities:
        if row[fraud_index] >= threshold:
            result.append("fraud-report")
        else:
            result.append(classes[int(row.argmax())])
    return result


def build_lr() -> Pipeline:
    from sklearn.linear_model import LogisticRegression

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced")),
        ]
    )


def build_xgb() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    random_state=0,
                ),
            ),
        ]
    )


def fit_predict_catboost(train_texts, train_labels, test_texts, seed: int):
    """CatBoost's native text mode takes raw strings directly; no TF-IDF step."""
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        text_features=[0],
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
    )
    train_pool = Pool(data=[[t] for t in train_texts], label=train_labels, text_features=[0])
    model.fit(train_pool)
    test_pool = Pool(data=[[t] for t in test_texts], text_features=[0])
    probabilities = model.predict_proba(test_pool)
    classes = [str(c) for c in model.classes_]
    return probabilities, classes


def evaluate_model(name: str, texts: list[str], labels: list[str], groups: list[str], fit_predict_proba) -> None:
    """fit_predict_proba(train_texts, train_labels, test_texts, seed) -> (probabilities, classes)"""
    all_truth: list[str] = []
    all_preds_argmax: list[str] = []
    all_probs: list[np.ndarray] = []
    all_classes: list[str] | None = None
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(texts, labels, groups):
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_texts = [texts[i] for i in test_idx]
            probabilities, classes = fit_predict_proba(train_texts, train_labels, test_texts, seed)
            all_classes = classes
            all_truth.extend(labels[i] for i in test_idx)
            all_probs.append(probabilities)
            all_preds_argmax.extend(classes[int(row.argmax())] for row in probabilities)

    argmax_score = metrics(all_truth, all_preds_argmax)
    print(f"\n=== {name} ===")
    print(f"argmax:            {argmax_score}")

    stacked = np.vstack(all_probs)
    fraud_index = all_classes.index("fraud-report")
    threshold = choose_threshold(all_truth, stacked[:, fraud_index])
    tuned_preds = apply_threshold(stacked, all_classes, threshold)
    tuned_score = metrics(all_truth, tuned_preds)
    print(f"tuned (in-sample threshold, NOT nested — optimistic): {tuned_score}  threshold={threshold:.3f}")


def main() -> None:
    texts, labels = load_rows()
    groups = [template_key(text) for text in texts]
    print(f"rows={len(texts)} unique_templates={len(set(groups))}")

    evaluate_model(
        "Logistic Regression (current, shipped)",
        texts, labels, groups,
        lambda tr_t, tr_l, te_t, seed: (
            (lambda m: (m.predict_proba(te_t), list(m.classes_)))(build_lr().fit(tr_t, tr_l))
        ),
    )

    if XGBClassifier is None:
        print("\n=== XGBoost === SKIPPED (not installed)")
    else:
        route_to_index = {route: i for i, route in enumerate(ROUTES)}

        def fit_predict_xgb(tr_t, tr_l, te_t, seed):
            # XGBoost's sklearn wrapper requires integer-encoded labels, not strings.
            encoded = [route_to_index[label] for label in tr_l]
            model = build_xgb()
            model.fit(tr_t, encoded)
            return model.predict_proba(te_t), list(ROUTES)

        evaluate_model("XGBoost (TF-IDF features)", texts, labels, groups, fit_predict_xgb)

    if CatBoostClassifier is None:
        print("\n=== CatBoost (native text) === SKIPPED (not installed)")
    else:
        evaluate_model(
            "CatBoost (native text mode)",
            texts, labels, groups,
            fit_predict_catboost,
        )

    print(
        "\nNote: the 'tuned' row here picks the fraud threshold on the same pooled "
        "predictions it is scored against (not nested per outer fold like "
        "eval_classifier.py), so it is optimistic for all three models equally — "
        "fine for a same-footing comparison, not a number to quote standalone."
    )


if __name__ == "__main__":
    main()
