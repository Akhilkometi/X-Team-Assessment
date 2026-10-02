# Support Ticket Router — Evaluation and Sign-off Decision

**Decision: do not sign off for production.** The handover's 98.75% random-holdout accuracy is inflated by near-duplicate templates crossing the split. The vectorizer was also fit before splitting. The 80-row holdout and accuracy-only report do not establish reliable fraud performance.

## Current grouped-CV evidence

Run `python -m eval.eval_classifier` from the repository root. It uses a `Pipeline` (so TF-IDF is fit within each fold), template-grouped `StratifiedGroupKFold`, five seeds and nested threshold selection. This repository's normalizer creates 77 groups from the 400 messages.

Latest run (five repetitions pooled):

| Candidate | Accuracy | Macro-F1 | Fraud precision | Fraud recall |
|---|---:|---:|---:|---:|
| Pipeline, unweighted, argmax | 0.8375 | 0.8150 | 0.7742 | 0.6720 |
| Pipeline, balanced, argmax | 0.8105 | 0.7924 | 0.5963 | 0.7680 |
| Balanced, nested threshold, fraud keyword override | 0.8210 | 0.8126 | 0.6072 | 0.9400 |

The first model's pooled confusion matrix (true labels as rows; order `account-access`, `transaction-dispute`, `fraud-report`, `general`):

| True \\ Predicted | account-access | transaction-dispute | fraud-report | general |
|---|---:|---:|---:|---:|
| account-access | 404 | 14 | 27 | 55 |
| transaction-dispute | 47 | 381 | 10 | 12 |
| fraud-report | 38 | 3 | 168 | 41 |
| general | 28 | 38 | 12 | 722 |

The recall-oriented candidate's pooled confusion matrix:

| True \\ Predicted | account-access | transaction-dispute | fraud-report | general |
|---|---:|---:|---:|---:|
| account-access | 401 | 15 | 32 | 52 |
| transaction-dispute | 45 | 385 | 19 | 1 |
| fraud-report | 0 | 1 | 235 | 14 |
| general | 26 | 52 | 101 | 621 |

The candidate improves fraud recall from 0.672 to 0.940, while precision is 0.607. It still misses the proposed ≥0.95 recall target. It is not approved to ship based on this dataset.

## Proposed acceptance criteria

- Primary: fraud-report recall ≥0.95, subject to fraud precision ≥0.60.
- Secondary: macro-F1, with per-route precision, recall and confusion matrix.
- Evaluation: grouped CV by normalized template plus a temporally later holdout; select any threshold inside training/development data only.
- Before launch: validate on human-reviewed, representative traffic; assess calibration and distribution shift; monitor false negatives, escalation volume and rule overrides; provide human review/fallback.

Threshold selection currently maximizes grouped inner-OOF fraud recall subject to a 0.60 precision floor. The CLI default threshold is an initial policy value, not a production calibration. The high-recall keyword safety net is explicit and auditable, but requires ongoing review.
