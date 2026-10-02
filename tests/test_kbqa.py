from datetime import date
from pathlib import Path

import pytest

from kbqa.answer import KBAnswerer
from kbqa.loader import load_documents
from kbqa.resolve import is_valid, resolve_document

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def documents():
    return load_documents(ROOT / "kb")


@pytest.fixture(scope="module")
def answerer(documents):
    return KBAnswerer(documents)


def test_loader_builds_families_from_supersedes_chain(documents):
    assert len(documents) == 31
    assert {doc.family_id for doc in documents if doc.doc_id in {"kb-011", "kb-012", "kb-013"}} == {"kb-011"}


def test_date_boundaries_are_inclusive_and_ignore_status(documents):
    old_fee = next(doc for doc in documents if doc.doc_id == "kb-012")
    assert old_fee.status == "superseded"
    assert is_valid(old_fee, date(2026, 4, 30))
    assert not is_valid(old_fee, date(2026, 5, 1))
    assert resolve_document(documents, "kb-011", date(2025, 6, 1)).document.doc_id == "kb-012"


def test_historical_dispute_and_staking_versions(documents):
    assert resolve_document(documents, "kb-031", date(2026, 3, 1)).document.doc_id == "kb-031"
    assert resolve_document(documents, "kb-041", date(2026, 3, 1)).document.doc_id == "kb-041"
    assert resolve_document(documents, "kb-041", date(2026, 7, 28)).document.doc_id == "kb-042"


def test_expired_notices_answer_no_active_notice(answerer):
    result = answerer.answer("Are withdrawals currently paused for maintenance?", "2026-07-28")
    assert not result.abstained
    assert "no active" in result.answer.lower()
    assert result.doc_ids == ("kb-091",)


def test_expired_referral_is_not_represented_as_current(answerer):
    result = answerer.answer("Is the invite a friend promotion still running?", "2026-07-28")
    assert "no active" in result.answer.lower()
    assert result.doc_ids == ("kb-092",)


def test_unanswerable_bank_chargeback_abstains(answerer):
    result = answerer.answer("How do I dispute a chargeback my own bank raised against me?", "2026-07-28")
    assert result.abstained
    assert result.reason.startswith("out_of_scope")


def test_out_of_scope_questions_abstain(answerer):
    for question in (
        "Do you offer margin trading with 10x leverage?",
        "What is the direct phone number for your fraud team?",
        "Can I open a joint account with my spouse?",
        "Can I pay my electricity bill through the app?",
    ):
        assert answerer.answer(question, "2026-07-28").abstained


def test_q09_uses_latest_fee_version_and_says_no_minimum(answerer):
    result = answerer.answer("What is the minimum charge on a withdrawal?", "2026-07-28")
    assert result.doc_ids == ("kb-013",)
    assert "no minimum charge" in result.answer.lower()


def test_unknown_question_abstains(answerer):
    result = answerer.answer("What is the moon's preferred color?", "2026-07-28")
    assert result.abstained


def test_referral_intent_does_not_match_earn_rates(answerer):
    result = answerer.answer("How much referral bonus can I earn if I invite a friend today?", "2026-07-28")
    assert result.doc_ids == ("kb-092",)
    assert "no active" in result.answer.lower()


def test_slow_card_purchase_retrieves_degradation_notice(answerer):
    result = answerer.answer("Why is my card purchase taking so long to show up right now?", "2026-07-28")
    assert result.doc_ids == ("kb-093",)
    assert "30 minutes" in result.answer


def test_support_code_request_retrieves_phishing_policy(answerer):
    result = answerer.answer("Support emailed me asking for my one time code, is that legitimate?", "2026-07-28")
    assert result.doc_ids == ("kb-110",)
    assert "never ask" in result.answer.lower()


def test_invalid_as_of_is_rejected(answerer):
    with pytest.raises(ValueError, match="ISO date"):
        answerer.answer("What fee applies?", "not-a-date")
