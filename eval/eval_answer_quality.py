"""Score extracted answer TEXT against hand-authored required facts, not just doc_id.

eval_retrieval.py scores whether answer() returns the right DOCUMENT. It cannot tell
you whether the sentence extracted from that document actually contains the fact the
question asked for -- a run can score perfectly on document selection while quietly
returning an irrelevant sentence from the right document. This closes that gap for the
30 answerable questions in questions.csv.

eval/gold_answers.csv: qid -> required_facts, a "|"-separated set of acceptable
case-insensitive substrings; the answer passes if AT LEAST ONE alternative in the set
appears in the returned answer text (alternatives cover e.g. "same day|16:00" when
either phrasing would demonstrate the fact was retrieved).

Honesty note: these facts were authored by reading the current (doc-verified-correct)
answer output, not independently blind beforehand -- so a clean run here is partly a
restatement of "the extraction already contains the obvious fact," not fully
independent validation. Its real value is as a regression guard: if the extraction
logic, synonym list, or KB content changes later, this will catch an answer that still
picks the right document but a worse sentence. Treat a failure here as meaningful; treat
a pass as "no known regression," not "answer text independently verified correct."

Usage:
    python -m eval.eval_answer_quality
"""
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


def main() -> None:
    questions = {row["qid"]: row for row in read_csv(ROOT / "questions.csv")}
    gold = read_csv(ROOT / "eval" / "gold_answers.csv")
    answerer = KBAnswerer(load_documents())

    passed, failed = 0, []
    for row in gold:
        qid = row["qid"]
        question = questions[qid]
        result = answerer.answer(question["question"], question["as_of"])
        alternatives = [f.strip().lower() for f in row["required_facts"].split("|")]
        answer_lower = result.answer.lower()
        if any(fact in answer_lower for fact in alternatives):
            passed += 1
        else:
            failed.append((qid, row["required_facts"], result.answer))

    total = len(gold)
    print(f"answer fact-coverage: {passed}/{total} ({passed / total:.1%})")
    if failed:
        print("\nfailed (expected fact not found in returned answer text):")
        for qid, expected, got in failed:
            print(f"  {qid}: expected one of [{expected}]")
            print(f"       got: {got!r}")


if __name__ == "__main__":
    main()
