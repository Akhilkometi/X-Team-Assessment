"""Evaluate dated document retrieval against hand-authored gold labels."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kbqa.answer import KBAnswerer  # noqa: E402
from kbqa.loader import load_documents  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate(input_path: Path, gold_path: Path, answerer: KBAnswerer) -> dict[str, object]:
    questions = read_csv(input_path)
    gold = {row["qid"]: row["doc_ids"].strip() for row in read_csv(gold_path)}
    if set(row["qid"] for row in questions) != set(gold):
        raise ValueError(f"Gold-label coverage mismatch for {input_path.name}")
    correct_documents = 0
    correct_family = 0
    stale = 0
    same_family_predictions = 0
    correct_abstentions = 0
    predicted_abstentions = 0
    answerable_total = sum(value != "NONE" for value in gold.values())
    expected_abstentions = sum(value == "NONE" for value in gold.values())
    family_lookup = {doc.doc_id: doc.family_id for doc in answerer.documents}
    for row in questions:
        expected = gold[row["qid"]]
        result = answerer.answer(row["question"], row["as_of"])
        is_expected_abstention = expected == "NONE"
        predicted_abstentions += int(result.abstained)
        correct_abstentions += int(is_expected_abstention and result.abstained)
        if is_expected_abstention:
            correct_documents += int(result.abstained)
        else:
            selected_doc = result.doc_ids[0] if result.doc_ids else None
            correct_documents += int(not result.abstained and selected_doc == expected)
            if not result.abstained and result.family_id == family_lookup[expected]:
                same_family_predictions += 1
                correct_family += 1
                stale += int(selected_doc != expected)
    total = len(questions)
    return {
        "questions": total,
        # Of the answerable questions, did we even find the right policy topic? Decomposes
        # top1_document_accuracy into "wrong topic" vs. "right topic, wrong dated version".
        "family_accuracy_on_answerable": correct_family / answerable_total if answerable_total else None,
        "top1_document_accuracy": correct_documents / total if total else 0.0,
        "stale_version_rate": stale / same_family_predictions if same_family_predictions else 0.0,
        "stale_version_count": stale,
        "same_family_resolutions": same_family_predictions,
        "abstention_precision": correct_abstentions / predicted_abstentions if predicted_abstentions else None,
        "abstention_recall": correct_abstentions / expected_abstentions if expected_abstentions else None,
        "predicted_abstentions": predicted_abstentions,
        "expected_unanswerable": expected_abstentions,
    }


def main() -> None:
    parser_paths = (ROOT / "questions.csv", ROOT / "eval" / "gold.csv")
    answerer = KBAnswerer(load_documents())
    for label, input_path, gold_path in (
        ("provided questions", parser_paths[0], parser_paths[1]),
        ("paraphrases/date shifts", ROOT / "eval" / "paraphrases.csv", ROOT / "eval" / "paraphrases.csv"),
    ):
        print(f"{label}: {evaluate(input_path, gold_path, answerer)}")


if __name__ == "__main__":
    main()
