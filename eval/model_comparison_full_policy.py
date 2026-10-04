"""Nested grouped-CV threshold + keyword rule, matched across LR / XGBoost / CatBoost.

eval/model_comparison.py deliberately isolates model choice from the deployed policy:
it scores argmax and an in-sample (non-nested, "optimistic") threshold, with no keyword
rule at all. eval/eval_classifier.py measures the actual shipped policy -- keyword
rule, then a threshold chosen per outer fold from nested inner out-of-fold
probabilities, then argmax -- but only for the shipped logistic regression model.

This script runs that EXACT SAME protocol with XGBoost and CatBoost swapped in as the
base model, to answer: does the headline 99.6% recall / 62.1% precision number hold if
the base model changes, or is it dominated by the keyword rule either way?

Requires xgboost and catboost (see eval/model_comparison.py); not in requirements.txt.

Usage:
    python -m eval.model_comparison_full_policy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triage.classifier import ROUTES, load_rows, template_key  # noqa: E402
from eval.eval_classifier import SEEDS, choose_threshold, route_predictions, metrics  # noqa: E402
from eval.model_comparison import build_lr, build_xgb, fit_predict_catboost, XGBClassifier, CatBoostClassifier  # noqa: E402


def fit_predict_lr(train_texts, train_labels, test_texts, seed):
    model = build_lr()
    model.fit(train_texts, train_labels)
    return model.predict_proba(test_texts), list(model.classes_)


def fit_predict_xgb(train_texts, train_labels, test_texts, seed):
    route_to_index = {route: i for i, route in enumerate(ROUTES)}
    encoded = [route_to_index[label] for label in train_labels]
    model = build_xgb()
    model.fit(train_texts, encoded)
    return model.predict_proba(test_texts), list(ROUTES)


def get_oof_generic(texts, labels, groups, splitter, fit_predict_proba, seed):
    """Same shape as eval_classifier.get_oof, but the fit/predict step is pluggable."""
    probabilities: list[np.ndarray | None] = [None] * len(texts)
    classes: list[str] | None = None
    for train_idx, test_idx in splitter.split(texts, labels, groups):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        fold_probabilities, fold_classes = fit_predict_proba(train_texts, train_labels, test_texts, seed)
        classes = fold_classes
        for idx, scores in zip(test_idx, fold_probabilities):
            probabilities[int(idx)] = scores
    if classes is None or any(p is None for p in probabilities):
        raise RuntimeError("Cross-validation did not produce a prediction for every row")
    return np.vstack(probabilities), classes


def run_nested(name: str, texts, labels, groups, fit_predict_proba) -> dict:
    all_truth: list[str] = []
    all_preds: list[str] = []
    thresholds: list[float] = []
    for seed in SEEDS:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in outer.split(texts, labels, groups):
            inner_texts = [texts[i] for i in train_idx]
            inner_labels = [labels[i] for i in train_idx]
            inner_groups = [groups[i] for i in train_idx]
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 100)
            inner_probs, inner_classes = get_oof_generic(
                inner_texts, inner_labels, inner_groups, inner, fit_predict_proba, seed
            )
            threshold = choose_threshold(inner_labels, inner_probs, np.array(inner_classes))
            thresholds.append(threshold)
            test_texts = [texts[i] for i in test_idx]
            test_probabilities, test_classes = fit_predict_proba(inner_texts, inner_labels, test_texts, seed)
            all_truth.extend(labels[i] for i in test_idx)
            all_preds.extend(route_predictions(test_probabilities, np.array(test_classes), threshold, test_texts))
    score = metrics(all_truth, all_preds)
    print(f"\n=== {name}: keyword rule + nested threshold + argmax (matched protocol) ===")
    print(score)
    print(f"outer-fold thresholds: mean={np.mean(thresholds):.3f} std={np.std(thresholds):.3f}")
    return score


def main() -> None:
    texts, labels = load_rows()
    groups = [template_key(text) for text in texts]
    print(f"rows={len(texts)} unique_templates={len(set(groups))}; seeds={SEEDS}")

    run_nested("Logistic Regression (shipped -- sanity check vs eval_classifier.py)", texts, labels, groups, fit_predict_lr)

    if XGBClassifier is None:
        print("\n=== XGBoost === SKIPPED (not installed)")
    else:
        run_nested("XGBoost", texts, labels, groups, fit_predict_xgb)

    if CatBoostClassifier is None:
        print("\n=== CatBoost (native text) === SKIPPED (not installed)")
    else:
        def fit_predict_cb(train_texts, train_labels, test_texts, seed):
            return fit_predict_catboost(train_texts, train_labels, test_texts, seed)

        run_nested("CatBoost (native text)", texts, labels, groups, fit_predict_cb)


if __name__ == "__main__":
    main()
