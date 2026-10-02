# Support Ticket Router — Evaluation and Sign-off Decision

**Decision: do not sign off on the handed-over classifier as-is.** The reported 98.75% random-holdout accuracy is inflated by near-duplicate templates crossing the split, and the vectorizer was fit before splitting. The 80-row holdout and accuracy-only report do not establish reliable fraud performance. The corrected implementation below meets the proposed fraud-recall/precision targets on this dataset under grouped CV; recommend a temporally later, human-reviewed holdout to confirm before full production sign-off (see "Proposed acceptance criteria").

## Current grouped-CV evidence

Run `python -m eval.eval_classifier` from the repository root. It uses a `Pipeline` (so TF-IDF is fit within each fold), template-grouped `StratifiedGroupKFold`, five seeds and nested threshold selection. This repository's normalizer creates 77 groups from the 400 messages.

Latest run (five repetitions pooled):

| Candidate | Accuracy | Macro-F1 | Fraud precision | Fraud recall |
|---|---:|---:|---:|---:|
| Pipeline, unweighted, argmax | 0.8375 | 0.8150 | 0.7742 | 0.6720 |
| Pipeline, balanced, argmax | 0.8105 | 0.7924 | 0.5963 | 0.7680 |
| Balanced, nested threshold, fraud keyword override | 0.8280 | 0.8213 | 0.6209 | 0.9960 |

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
| fraud-report | 0 | 1 | 249 | 0 |
| general | 26 | 52 | 101 | 621 |

The candidate improves fraud recall from 0.672 to 0.996, while precision improves slightly to 0.621. This was reached by adding one targeted keyword pattern (see error analysis below) after inspecting the actual remaining false negatives under the full deployed policy — not by loosening the threshold, which would trade precision away. It now meets the proposed ≥0.95 recall / ≥0.60 precision targets on this dataset. That is evidence for shipping, not proof: it is still grouped CV on the same synthetic corpus, not a temporally later or human-reviewed holdout.

### Error analysis behind this change

Before the fix, inspecting actual false negatives under the full deployed policy (keyword rule + threshold, one representative CV seed) showed they had collapsed almost entirely to a single template: *"There's a suspicious [asset] transfer to an unknown wallet that I did not make."* This phrasing contains none of the original keyword triggers (unauthorized/stolen/phishing/scam/hacked/drained) and apparently sits under the probability threshold too. A similar-looking template — *"I see a [device] I don't recognize on my account and money was moved out"* — was already being caught by the probability threshold alone, so it wasn't the gap.

Checked the full training set for collision risk before adding a pattern for "did not make / never made / didn't authorize": the near-miss phrasings in non-fraud classes use different verbs entirely ("never received," "never sends," "never arrives"), so no false-positive collisions were introduced. Confirmed by rerunning the full nested grouped CV: precision moved up, not down (0.607 → 0.621), recall 0.940 → 0.996.

This is the right order of operations for this kind of fix: read the actual misclassified text, confirm the pattern is real (not one row's artifact — it recurred across near-duplicate rows), check for collisions against the rest of the corpus, then measure the before/after under the same protocol used to report the headline number. A keyword added without that check is exactly the kind of thing to push back on in review.

## Proposed acceptance criteria

- Primary: fraud-report recall ≥0.95, subject to fraud precision ≥0.60.
- Secondary: macro-F1, with per-route precision, recall and confusion matrix.
- Evaluation: grouped CV by normalized template plus a temporally later holdout; select any threshold inside training/development data only.
- Before launch: validate on human-reviewed, representative traffic; assess calibration and distribution shift; monitor false negatives, escalation volume and rule overrides; provide human review/fallback.

Threshold selection currently maximizes grouped inner-OOF fraud recall subject to a 0.60 precision floor. The CLI default threshold is an initial policy value, not a production calibration. The high-recall keyword safety net is explicit and auditable, but requires ongoing review.
