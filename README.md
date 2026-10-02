# Support Triage and Knowledge Answering — Engineering Assessment

## Executive decision

**Do not sign off on the handed-over classifier.** The reported 98.75% is not a trustworthy estimate of generalization: the vectorizer was fit before the split, and the random holdout leaked highly repetitive message templates across train and test. The supplied dataset has 400 rows but only about 77 normalized templates under the grouping used here. The operational risk is specifically fraud misses, not aggregate accuracy.

This implementation makes a small, auditable correction rather than introducing an LLM: a scikit-learn pipeline, template-grouped cross-validation, balanced class loss, nested fraud-threshold selection and a high-recall fraud keyword override. The current nested grouped-CV run reached **94.0% fraud recall at 60.7% precision**, below the proposed ≥95% recall goal. It is **not production-ready** until the threshold/policy is validated on temporally later, human-reviewed data and the recall target is met without breaching the precision floor.

Part B uses topic-first family retrieval, then resolves the date-valid document version. It returns short extractive evidence and abstains for low-confidence or explicitly unsupported questions. No generative model is required on the hot path.

## Setup and commands

Python 3.10+ is recommended.

```text
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

From this directory:

```text
python -m triage.predict --input messages.csv --output predictions.csv
python -m triage.predict --input messages.csv --output predictions.csv --fraud-threshold 0.25
python -m kbqa.cli --input questions.csv --output answers.csv
python -m kbqa.cli --input questions.csv --output answers.csv --min-score 1.5 --min-margin 0.15
python -m eval.eval_classifier
python -m eval.eval_retrieval
python -m pytest -q
```

The classifier CSV contract is `text,prediction`. The answer CSV contract is `qid,answer,doc_ids`; `doc_ids` is semicolon-separated and empty when the system abstains. The default KB directory and training file are resolved relative to the package, not the current directory.

## Part A — classifier evaluation and decision

### Why 98.75% is inflated

The original script fits TF-IDF before its random split, which leaks vocabulary/IDF information. More importantly, many rows differ only by greeting, sign-off, asset, amount or device substitutions. A random row split places near-copies on both sides and measures template recall more than unseen-message generalization. The small 80-row holdout also offers no stable fraud-recall estimate or per-class error accounting. Accuracy alone is not an appropriate safety metric for the least common, high-cost fraud class.

### Evaluation protocol and observed results

`eval/eval_classifier.py` performs five-fold `StratifiedGroupKFold` using normalized template keys; each vectorizer is inside the model pipeline and is fit only on the training fold. It repeats the protocol for five seeds. The measured output for this repository's normalization is **77 unique groups** (grouping choices can change this count). On the current run:

| System | Accuracy | Macro-F1 | Fraud precision | Fraud recall |
|---|---:|---:|---:|---:|
| Unweighted pipeline, argmax | 0.8375 | 0.8150 | 0.7742 | 0.6720 |
| Balanced pipeline, argmax | 0.8105 | 0.7924 | 0.5963 | 0.7680 |
| Balanced + nested threshold + keyword override | 0.8210 | 0.8126 | 0.6072 | 0.9400 |

The unweighted pipeline's accuracy averaged **0.8375** across the repeated held-out predictions; the script also prints the five-seed mean and standard deviation. Treat these as estimates from a small synthetic corpus, not as production guarantees. The provided narrative's separate grouped estimate (84.8% ± 1.5, macro-F1 0.83, fraud recall 0.69) is directionally consistent; exact figures depend on the template canonicalizer, split seeds, and aggregation convention. Neither supports the 98.75% claim.

Confusion matrices below use true labels as rows and predictions as columns in this fixed order: `account-access`, `transaction-dispute`, `fraud-report`, `general`.

Unweighted pipeline, argmax:

| True \\ Predicted | account-access | transaction-dispute | fraud-report | general |
|---|---:|---:|---:|---:|
| account-access | 404 | 14 | 27 | 55 |
| transaction-dispute | 47 | 381 | 10 | 12 |
| fraud-report | 38 | 3 | 168 | 41 |
| general | 28 | 38 | 12 | 722 |

Balanced + nested fraud threshold + keyword override:

| True \\ Predicted | account-access | transaction-dispute | fraud-report | general |
|---|---:|---:|---:|---:|
| account-access | 401 | 15 | 32 | 52 |
| transaction-dispute | 45 | 385 | 19 | 1 |
| fraud-report | 0 | 1 | 235 | 14 |
| general | 26 | 52 | 101 | 621 |

The unweighted model misses 80 of 250 repeated fraud held-out predictions (recall 0.672). Thresholding improves recall to 0.94 while retaining 0.607 precision, but adds false-positive fraud routing. That is a policy trade-off to validate with operations, not evidence that the target has been met.

### Implementation and production gate

- `Pipeline(TfidfVectorizer, LogisticRegression)` closes vectorizer leakage.
- `class_weight="balanced"` is used in the recall-oriented candidate.
- The evaluator chooses the fraud threshold inside each outer-training partition using inner grouped OOF probabilities; it maximizes fraud recall subject to a 0.60 precision floor. The prediction CLI estimates its default cutoff from grouped out-of-fold predictions across the full training set; the explicit flag overrides that value. The nested evaluator's 0.94 recall/0.607 precision estimate remains below the recall target, so this policy is **not approved for production**.
- An explicit safety override catches unauthorised/never-authorised, stolen, phishing, scam, hacked and drained language. It should be monitored and reviewed as a policy rule.
- The production primary metric is fraud-report recall ≥0.95 subject to fraud precision ≥0.60; macro-F1 is secondary. Add a temporally later holdout, route-specific precision/recall, confidence calibration, drift monitoring and a human escalation/fallback before launch.

## Part B — date-aware KB answering

### Architecture

1. **Load and validate** markdown front matter, ISO dates, IDs and lineage pointers. Invalid or missing metadata fails fast.
2. **Index by family** by walking each `supersedes` chain to its root. The BM25 corpus combines titles and bodies for every version, so renamed topics remain discoverable (for example withdrawal fees/network charges and staking/earn).
3. **Retrieve the topic** using dependency-free BM25 plus a small, explicit synonym vocabulary and targeted intent boosts for high-ambiguity notices/referrals/phishing.
4. **Apply an abstention gate** when the best family score is too low or too close to the runner-up. Known out-of-scope intents, including bank-initiated chargebacks, return an explicit abstention rather than borrowing a superficially similar in-scope answer.
5. **Resolve time after retrieval.** A document is valid when `effective_date ≤ as_of` and `valid_until` is absent or `as_of ≤ valid_until`. The `status` field is deliberately ignored because it describes a current snapshot. If an all-expired notice family is retrieved, the answer is “no active notice or promotion” as of that date and the notice IDs are returned.
6. **Answer extractively.** Return up to two relevant source sentences and the resolved document ID. There is no LLM call, preserving low latency, predictable cost, and source traceability for high request volume.

### Retrieval evaluation

Hand-labelled targets are in `eval/gold.csv`; `NONE` denotes the eight unsupported questions. The 15 manually authored paraphrases/date shifts are in `eval/paraphrases.csv`. Current BM25/gate results:

| Set | Top-1 document accuracy | Stale-version rate | Abstention precision | Abstention recall |
|---|---:|---:|---:|---:|
| Provided questions (38) | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| Paraphrases/date shifts (15) | 1.0000 | 0.0000 | N/A | N/A |

Top-1 accuracy counts correctly abstained `NONE` cases. Stale-version rate is wrong-version selections divided by resolved cases whose selected family matched the gold family. The paraphrase set currently contains no unanswerable examples, so abstention precision/recall are undefined; expand it with adversarial unsupported questions before treating those metrics as meaningful. These are small, hand-authored development sets, not an unbiased estimate of held-back performance. Review every miss and tune thresholds only on a separated development set.

### Traps and data-quality caveats

- **q09:** current answer is “There is no minimum charge” from kb-013. Older fee versions mention $1 or $2 minimums.
- **Historical versioning:** q11 (2025-06-01) uses kb-012; q12 (2026-03-01) uses kb-031's 60-day window; q13 uses kb-041's older rate; q14 uses kb-021's former SMS recovery. In particular, q12 must not use the 30-day kb-032 policy.
- **Expired notices/promotions:** q15/q17 and q16/q18 are answered “no active notice or promotion” for their respective retrieved families, with source IDs preserved.
- **kb-094:** metadata `valid_until: 2026-12-31` conflicts with its body shutdown date of 2027-01-15. The lifecycle window governs retrieval; the later date is stated as document content. Resolve this conflict with the KB owner before production.
- **kb-093:** metadata expires 2026-12-31 while the body says “until further notice.” The explicit metadata governs date selection; confirm the notice lifecycle with its owner.
- **q38:** kb-032 covers disputes raised with this service, not chargebacks initiated by the customer's own bank. This implementation abstains.

### Known limits and next steps

BM25, rules and a small synonym list are intentionally transparent, but they can miss new paraphrases and have hand-tuned intent logic. Before launch, add more paraphrased/unsupported/date-shifted evaluation cases, inspect score/margin calibration by topic, review temporal gaps/overlaps, track abstentions and stale-version failures, and involve KB owners in contradictory metadata. A later semantic retriever can be evaluated as a candidate behind the same family-resolution and abstention contracts; it must not bypass temporal validity or provenance.

## Part C — intentionally not implemented

I would not spend this take-home's three-hour budget implementing optional multimodal ingestion. The screenshots and voice notes add little to the core routing/KB value and create additional failure modes, data handling, testing and operational surface. If the product requirement is established, **on-device/local Tesseract OCR for screenshots** is the stronger first experiment: inexpensive, repeatable and keeps synthetic-or-real PII off a hosted vision service. Audio adds ASR cost, accent/noise error, transcription consent/retention questions and little incremental routing signal; defer it until measured customer value justifies that risk. Any future multimodal path should preserve provenance, redact/minimize sensitive content, allow correction, and fall back to human review.

## Scope & trade-offs

**Prioritized:** both required parts working end-to-end with honest, runnable evaluation over flashier but unverified claims — a leakage-aware CV protocol and a dated/family-aware retriever with an explicit abstention gate, each with a confusion matrix or gold-label score behind every number in this README, not just a one-line summary.

**Deliberately left out:** Part C (multimodal routing) — see above; a learned/embedding retriever (BM25 + synonyms + intent rules is auditable and sufficient at this corpus size, and swapping it in later doesn't require changing the family/temporal contracts); an LLM rewrite step on top of the extractive answers; CI config and containerization.

**With more time:** a temporally-later, human-reviewed holdout to validate the fraud threshold instead of only grouped CV on the same corpus; more adversarial/unsupported questions in the paraphrase set so abstention precision/recall are measured on more than zero negative cases; calibration and drift monitoring for the fraud score; resolving the kb-093/kb-094 metadata-vs-body date conflicts with a KB owner instead of just flagging them.

**Time:** built against the ~3-hour budget the brief sets, with AI-assisted implementation throughout (the brief explicitly expects this) — I can walk through and defend every decision above in the video.

## Repository map

- `triage/`: pipeline classifier, safety policy and CSV predictor.
- `kbqa/`: metadata loader, family retriever, temporal resolver and batch answerer.
- `eval/`: grouped classifier evaluation, retrieval metrics, document labels and paraphrase set.
- `tests/`: temporal, abstention and correctness checks.
- `baseline/`: original handover classifier and updated evaluation decision.
