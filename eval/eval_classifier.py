"""Nested, template-grouped CV for routing; outputs confusion matrices and metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from triage.classifier import FRAUD_PATTERNS, ROUTES, build_model, load_rows, template_key  # noqa: E402

PRECISION_FLOOR = 0.60
SEEDS = (11, 23, 37, 51, 71)


def choose_threshold(y_true: list[str], probabilities: np.ndarray, classes: np.ndarray) -> float:
    """Maximize fraud recall subject to the fraud precision floor."""
    fraud_column = list(classes).index("fraud-report")
    fraud_scores = probabilities[:, fraud_column]
    candidates = sorted(set([0.0, *map(float, fraud_scores), 1.0]))
    feasible: list[tuple[float, float, float]] = []
    truth = np.asarray(y_true)
    for threshold in candidates:
        predicted = np.where(fraud_scores >= threshold, "fraud-report", "__other__")
        tp = int(np.sum((predicted == "fraud-report") & (truth == "fraud-report")))
        fp = int(np.sum((predicted == "fraud-report") & (truth != "fraud-report")))
        fn = int(np.sum((predicted != "fraud-report") & (truth == "fraud-report")))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision >= PRECISION_FLOOR:
            feasible.append((recall, precision, threshold))
    return max(feasible)[2] if feasible else 1.0


def route_predictions(probabilities: np.ndarray, classes: np.ndarray, threshold: float, texts: list[str] | None = None) -> list[str]:
    names = list(classes)
    result = []
    for index, row in enumerate(probabilities):
        if texts is not None and any(pattern.search(texts[index]) for pattern in FRAUD_PATTERNS):
            result.append("fraud-report")
        elif row[names.index("fraud-report")] >= threshold:
            result.append("fraud-report")
        else:
            result.append(names[int(row.argmax())])
    return result


def get_oof(texts: list[str], labels: list[str], groups: list[str], splitter: StratifiedGroupKFold) -> tuple[list[str], np.ndarray, np.ndarray]:
    predictions = [""] * len(texts)
    probabilities: list[np.ndarray | None] = [None] * len(texts)
    classes: np.ndarray | None = None
    for train_idx, test_idx in splitter.split(texts, labels, groups):
        model = build_model(class_weight="balanced")
        model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        fold_probabilities = model.predict_proba([texts[i] for i in test_idx])
        classes = model.classes_
        for idx, scores in zip(test_idx, fold_probabilities):
            probabilities[int(idx)] = scores
            predictions[int(idx)] = str(classes[int(scores.argmax())])
    if classes is None or any(item is None for item in probabilities):
        raise RuntimeError("Cross-validation did not produce a prediction for every row")
    return predictions, np.vstack(probabilities), classes


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    # precision/recall_score's pos_label path only works for 2-class input; with 4 routes
    # present we score the fraud-report class via one-vs-rest (average=None, labels=[...]).
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=ROUTES, average="macro", zero_division=0),
        "fraud_precision": precision_score(y_true, y_pred, labels=["fraud-report"], average=None, zero_division=0)[0],
        "fraud_recall": recall_score(y_true, y_pred, labels=["fraud-report"], average=None, zero_division=0)[0],
    }


def main() -> None:
    texts, labels = load_rows()
    groups = [template_key(text) for text in texts]
    print(f"rows={len(texts)} unique_templates={len(set(groups))}; grouped stratified 5-fold CV, seeds={SEEDS}")
    for name, class_weight in (("baseline (argmax)", None), ("balanced (argmax)", "balanced")):
        all_truth: list[str] = []
        all_preds: list[str] = []
        per_seed_accuracy: list[float] = []
        for seed in SEEDS:
            seed_truth: list[str] = []
            seed_preds: list[str] = []
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_idx, test_idx in splitter.split(texts, labels, groups):
                model = build_model(class_weight=class_weight)
                model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
                fold_predictions = list(model.predict([texts[i] for i in test_idx]))
                seed_truth.extend(labels[i] for i in test_idx)
                seed_preds.extend(fold_predictions)
            per_seed_accuracy.append(accuracy_score(seed_truth, seed_preds))
            all_truth.extend(seed_truth)
            all_preds.extend(seed_preds)
        score = metrics(all_truth, all_preds)
        print(f"\n{name}: {score}")
        print(f"per-seed accuracy: {np.mean(per_seed_accuracy):.4f} ± {np.std(per_seed_accuracy, ddof=1):.4f}")
        print("confusion matrix (true rows, predicted columns; order=" + ", ".join(ROUTES) + ")")
        print(confusion_matrix(all_truth, all_preds, labels=ROUTES))

    # Nested grouped CV: choose a threshold only from each outer-training partition.
    all_truth = []
    all_preds = []
    selected_thresholds = []
    for seed in SEEDS:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in outer.split(texts, labels, groups):
            inner_texts = [texts[i] for i in train_idx]
            inner_labels = [labels[i] for i in train_idx]
            inner_groups = [groups[i] for i in train_idx]
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 100)
            _, inner_probs, inner_classes = get_oof(inner_texts, inner_labels, inner_groups, inner)
            threshold = choose_threshold(inner_labels, inner_probs, inner_classes)
            selected_thresholds.append(threshold)
            model = build_model(class_weight="balanced")
            model.fit(inner_texts, inner_labels)
            test_probabilities = model.predict_proba([texts[i] for i in test_idx])
            all_truth.extend(labels[i] for i in test_idx)
            all_preds.extend(route_predictions(test_probabilities, model.classes_, threshold, [texts[i] for i in test_idx]))
    tuned = metrics(all_truth, all_preds)
    print(f"\nbalanced + nested fraud threshold (precision floor={PRECISION_FLOOR:.2f}): {tuned}")
    print(f"outer-fold thresholds: {[round(value, 3) for value in selected_thresholds]}")
    print("confusion matrix (true rows, predicted columns; order=" + ", ".join(ROUTES) + ")")
    print(confusion_matrix(all_truth, all_preds, labels=ROUTES))
    print("Operational gate: validate the chosen threshold on a temporally later, human-reviewed set before launch.")


if __name__ == "__main__":
    main()
