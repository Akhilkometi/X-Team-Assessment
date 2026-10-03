from pathlib import Path

import pytest

pytest.importorskip("pytesseract", reason="Part C (mediaqa) is optional; pip install -r requirements-mediaqa.txt to run these")

from mediaqa.ocr import extract_text  # noqa: E402
from triage.classifier import DATA_PATH, fit_model, load_rows, predict_route  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "media" / "screenshots"


@pytest.fixture(scope="module")
def model():
    texts, labels = load_rows(DATA_PATH)
    return fit_model(texts, labels)


def test_ocr_extracts_readable_text_from_login_error():
    text = extract_text(SCREENSHOTS / "login-error.png").lower()
    assert "verification" in text
    assert "authenticat" in text


def test_ocr_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_text(SCREENSHOTS / "does-not-exist.png")


def test_login_error_screenshot_routes_to_account_access(model):
    text = extract_text(SCREENSHOTS / "login-error.png")
    assert predict_route(model, text) == "account-access"


def test_txn_failed_screenshot_routes_to_transaction_dispute(model):
    text = extract_text(SCREENSHOTS / "txn-failed.png")
    assert predict_route(model, text) == "transaction-dispute"


def test_phishing_sms_screenshot_routes_to_fraud_report(model):
    # This one is caught by the learned model's probability, not the keyword safety
    # net -- the SMS never uses words like "stolen", "scam" or "unauthorized".
    text = extract_text(SCREENSHOTS / "phishing-sms.png")
    assert predict_route(model, text) == "fraud-report"
