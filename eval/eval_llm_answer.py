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

Two optional, independent backends -- neither is in requirements.txt, only needed to
re-run this specific comparison:

  1. Local Gemma via Hugging Face transformers (preferred: no billing dependency, runs
     on-device). Needs HF_TOKEN in .env (or the environment) with the gemma-2-2b-it
     license accepted on huggingface.co, plus `pip install transformers torch`.
  2. Gemini API (fallback, documented history: see README "LLM experiment" section for
     why this was blocked in-session -- org-policy key restriction, then prepaid-credit
     exhaustion on the Express-mode sandbox project). Needs GEMINI_API_KEY.

If neither backend is available, or a call fails for any reason (auth, quota, network,
missing deps), this prints why and exits cleanly without crashing or touching the shipped
implementation -- the brief's "documented fallback / clear skip" allowance, not a hard
dependency.

Usage:
    python -m eval.eval_llm_answer
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections.abc import Callable
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

GEMMA_MODEL_ID = "google/gemma-2-2b-it"
GEMINI_MODEL_ID = "gemini-3.8-flash"


def _load_env_var(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{name}="):
                value = line.split("=", 1)[1].strip()
                return value or None
    return None


def _build_gemma_generator() -> tuple[str, Callable[[str], str]] | None:
    """Local Gemma via transformers. Returns None (with a printed reason) if unavailable."""
    token = _load_env_var("HF_TOKEN")
    if not token:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers/torch not installed -- skipping local Gemma backend "
              "(pip install transformers torch to enable).")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, token=token)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL_ID, token=token, dtype=torch.float32)
        model.to(device)
    except Exception as exc:  # noqa: BLE001 -- any load failure means "skip", not "crash"
        print(f"Local Gemma unavailable ({type(exc).__name__}): {str(exc)[:300]}")
        return None

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        output = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return f"{GEMMA_MODEL_ID} (local HF, {device})", generate


def _build_gemini_generator() -> tuple[str, Callable[[str], str]] | None:
    """Gemini API fallback. Returns None (with a printed reason) if unavailable."""
    key = _load_env_var("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
    except ImportError:
        print("google-genai not installed -- skipping Gemini backend (pip install google-genai to enable).")
        return None

    client = genai.Client(api_key=key)

    def generate(prompt: str) -> str:
        response = client.models.generate_content(model=GEMINI_MODEL_ID, contents=prompt)
        return (response.text or "").strip()

    return GEMINI_MODEL_ID, generate


def _build_generator() -> tuple[str, Callable[[str], str]] | None:
    """Try local Gemma first (no billing dependency), then Gemini. None if neither works."""
    return _build_gemma_generator() or _build_gemini_generator()


def main() -> None:
    backend = _build_generator()
    if backend is None:
        print("No LLM backend available -- skipping LLM comparison.")
        print("Set HF_TOKEN (local Gemma) or GEMINI_API_KEY (.env or env) to enable.")
        print("Deterministic extractive baseline is unaffected; see eval_answer_quality.py.")
        return
    backend_name, generate = backend

    # One live call first, to fail fast and clearly instead of partway through 30 calls.
    try:
        probe = generate("Reply with exactly: OK")
        if "OK" not in probe:
            raise RuntimeError(f"unexpected probe response: {probe!r}")
    except Exception as exc:  # noqa: BLE001 -- any failure here means "skip", not "crash"
        print(f"LLM unavailable via {backend_name} ({type(exc).__name__}): {str(exc)[:300]}")
        print("Skipping LLM comparison -- deterministic extractive baseline is unaffected.")
        return

    print(f"Using backend: {backend_name}\n")

    questions = {row["qid"]: row for row in csv.DictReader((ROOT / "questions.csv").open(newline="", encoding="utf-8"))}
    gold = list(csv.DictReader((ROOT / "eval" / "gold_answers.csv").open(newline="", encoding="utf-8")))
    documents_by_id = {doc.doc_id: doc for doc in load_documents()}
    answerer = KBAnswerer(load_documents())

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
            llm_answer = generate(prompt)
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
