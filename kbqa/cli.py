"""CSV batch entry point: python -m kbqa.cli --input ... --output ..."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from kbqa.answer import KBAnswerer
from kbqa.loader import load_documents
from kbqa.resolve import status_inconsistencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer customer questions from the dated support KB")
    parser.add_argument("--input", required=True, help="CSV containing qid, question, as_of")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--min-score", type=float, default=1.5)
    parser.add_argument("--min-margin", type=float, default=0.15)
    args = parser.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"qid", "question", "as_of"}
        if not required.issubset(reader.fieldnames or []):
            parser.error("Input CSV must contain qid, question, and as_of columns")
        rows = list(reader)
    documents = load_documents()
    for finding in status_inconsistencies(documents):
        print(
            f"warning: {finding.doc_id} has status={finding.status!r} but dates say "
            f"valid_today={finding.is_valid_as_of_today} — metadata may be stale",
            file=sys.stderr,
        )
    answerer = KBAnswerer(documents, min_score=args.min_score, min_margin=args.min_margin)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["qid", "answer", "doc_ids"])
        writer.writeheader()
        for row in rows:
            result = answerer.answer(row["question"], row["as_of"])
            writer.writerow({"qid": row["qid"], "answer": result.answer, "doc_ids": ";".join(result.doc_ids)})


if __name__ == "__main__":
    main()
