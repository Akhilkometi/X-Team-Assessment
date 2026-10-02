"""Exploratory: LLM answer synthesis on top of the SAME deterministic retrieval/resolution.

Not part of the shipped pipeline. kbqa/answer.py and requirements.txt are untouched;
this script only reuses KBAnswerer to pick the document (already measured at 100% top-1
accuracy in eval_retrieval.py), then asks an LLM to synthesize the final answer text
from that document's full body, instead of the shipped sentence-overlap extraction.

This deliberately does NOT let the LLM choose the document or the policy version -- that
stays fully owned by the deterministic retrieve-then-resolve pipeline. The only thing
being compared is the last step: extractive sentence selection vs. LLM synthesis, given
identical, already-correct evidence. Scored against the same eval/gold_answers.csv used
for the extractive baseline, so the comparison is apples-to-apples.

Requires GEMINI_API_KEY in .env (or the environment). Not in requirements.txt -- only
needed to re-run this specific comparison. If no key is present, or the API call fails
for any reason (auth, quota, network), this prints why and exits cleanly without
crashing or touching the shipped implementation -- the brief's "documented fallback /
clear skip" allowance, not a hard dependency.

Usage:
    python -m eval.eval_llm_answer
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kbqa.answer import KBAnswerer  # noqa: E402
from kbqa.loader import load_documents  # noqa: E402

PROMPT_TEMPLATE = """Answer the customer's question using ONLY the evidence below. \
Be concise (1-2 sentences). Include specific numbers/dates/facts from the evidence \
verbatim where relevant. If the evidence does not actually answer the question, \
reply exactly: "Not found in evidence."

Evidence:
{evidence}

Question: {question}

Answer:"""


def _load_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip()
                return value or None
    return None


def main() -> None:
    key = _load_key()
    if not key:
        print("GEMINI_API_KEY not set (checked env and .env) -- skipping LLM comparison.")
        print("Deterministic extractive baseline is unaffected; see eval_answer_quality.py.")
        return

    try:
        from google import genai
    except ImportError:
        print("google-genai not installed -- skipping LLM comparison (pip install google-genai to re-run).")
        return

    client = genai.Client(api_key=key)

    questions = {row["qid"]: row for row in csv.DictReader((ROOT / "questions.csv").open(newline="", encoding="utf-8"))}
    gold = list(csv.DictReader((ROOT / "eval" / "gold_answers.csv").open(newline="", encoding="utf-8")))
    documents_by_id = {doc.doc_id: doc for doc in load_documents()}
    answerer = KBAnswerer(load_documents())

    # One live call first, to fail fast and clearly instead of partway through 30 calls.
    try:
        probe = client.models.generate_content(model="gemini-2.0-flash", contents="Reply with exactly: OK")
        if "OK" not in (probe.text or ""):
            raise RuntimeError(f"unexpected probe response: {probe.text!r}")
    except Exception as exc:  # noqa: BLE001 -- any failure here means "skip", not "crash"
        print(f"LLM unavailable ({type(exc).__name__}): {str(exc)[:300]}")
        print("Skipping LLM comparison -- deterministic extractive baseline is unaffected.")
        return

    passed, failed, latencies = 0, [], []
    examples = []
    for row in gold:
        qid = row["qid"]
        q = questions[qid]
        extractive = answerer.answer(q["question"], q["as_of"])
        if extractive.abstained or not extractive.doc_ids:
            continue  # LLM synthesis is only compared where deterministic retrieval resolved a document
        document = documents_by_id[extractive.doc_ids[0]]

        prompt = PROMPT_TEMPLATE.format(evidence=document.body, question=q["question"])
        start = time.monotonic()
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            llm_answer = (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            llm_answer = f"[ERROR: {type(exc).__name__}: {str(exc)[:200]}]"
        latencies.append(time.monotonic() - start)

        alternatives = [f.strip().lower() for f in row["required_facts"].split("|")]
        ok = any(fact in llm_answer.lower() for fact in alternatives)
        passed += int(ok)
        if not ok:
            failed.append((qid, row["required_facts"], llm_answer))
        if len(examples) < 5:
            examples.append((qid, q["question"], extractive.answer, llm_answer))

    total = len(latencies)
    print(f"LLM answer fact-coverage: {passed}/{total} ({passed / total:.1%})" if total else "no questions scored")
    if latencies:
        print(f"latency: mean={sum(latencies) / len(latencies):.2f}s  max={max(latencies):.2f}s  (single-threaded, sequential calls)")
    if failed:
        print("\nfailed (expected fact not found in LLM answer):")
        for qid, expected, got in failed:
            print(f"  {qid}: expected one of [{expected}]  -- got: {got!r}")
    print("\nside-by-side examples (extractive vs. LLM synthesis, same resolved document):")
    for qid, question, extractive_answer, llm_answer in examples:
        print(f"\n  {qid}: {question}")
        print(f"    extractive: {extractive_answer}")
        print(f"    llm:        {llm_answer}")


if __name__ == "__main__":
    main()
