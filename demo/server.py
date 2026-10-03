"""Local demo server: python -m demo.server, then open http://127.0.0.1:5050

Presentation convenience only -- the actual deliverables are the two CSV batch
CLIs (triage.predict, kbqa.cli) the brief asks for. This just puts a browser
in front of the same code so it's easier to show working live.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kbqa.answer import KBAnswerer  # noqa: E402
from kbqa.loader import load_documents  # noqa: E402
from triage.classifier import (  # noqa: E402
    DATA_PATH,
    FRAUD_PATTERNS,
    estimate_fraud_threshold,
    fit_model,
    load_rows,
)

try:
    from mediaqa.ocr import extract_text as ocr_extract_text

    MEDIAQA_AVAILABLE = True
except ImportError:
    MEDIAQA_AVAILABLE = False

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Fit once at startup, not per request -- same pattern the CLIs already use, just cached.
_train_texts, _train_labels = load_rows(DATA_PATH)
_model = fit_model(_train_texts, _train_labels)
_fraud_threshold = estimate_fraud_threshold(_train_texts, _train_labels)
_answerer = KBAnswerer(load_documents())

SAMPLE_SCREENSHOTS = ["login-error.png", "phishing-sms.png", "txn-failed.png"]

EXAMPLE_MESSAGES = [
    {"label": "general", "text": "How does staking work and what rewards can I expect on Polygon? Please advise."},
    {"label": "account-access", "text": "Hello team, The app won't let me sign in, it just spins on the login screen. Thanks."},
    {"label": "transaction-dispute", "text": "Hey, My withdrawal of 2 ETH shows completed but I never received it, please reverse it. Appreciate any help."},
    {"label": "fraud-report", "text": "Urgent: My funds are gone, $880 disappeared overnight and I believe this is fraud."},
]
EXAMPLE_QUESTIONS = [
    {"question": "What fee do I pay to withdraw crypto from my account?", "as_of": "2026-07-28"},
    {"question": "What was the crypto withdrawal fee in June 2025?", "as_of": "2025-06-01"},
    {"question": "How many days did a customer have to raise a dispute in March 2026?", "as_of": "2026-03-01"},
    {"question": "Do you offer margin trading with 10x leverage?", "as_of": "2026-07-28"},
]


def _classify(text: str) -> dict:
    probabilities = _model.predict_proba([text])[0]
    classes = list(_model.classes_)
    keyword_hit = next((p.pattern for p in FRAUD_PATTERNS if p.search(text)), None)
    if keyword_hit:
        prediction = "fraud-report"
    elif probabilities[classes.index("fraud-report")] >= _fraud_threshold:
        prediction = "fraud-report"
    else:
        prediction = classes[int(probabilities.argmax())]
    return {
        "prediction": prediction,
        "probabilities": {route: round(float(p), 4) for route, p in zip(classes, probabilities)},
        "keyword_matched": keyword_hit,
        "fraud_threshold": round(_fraud_threshold, 4),
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        example_messages=EXAMPLE_MESSAGES,
        example_questions=EXAMPLE_QUESTIONS,
        mediaqa_available=MEDIAQA_AVAILABLE,
    )


@app.post("/api/route-text")
def route_text():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text must not be empty"}), 400
    return jsonify(_classify(text))


@app.post("/api/route-image")
def route_image():
    if not MEDIAQA_AVAILABLE:
        return jsonify({"error": "Part C (mediaqa) isn't installed -- pip install -r requirements-mediaqa.txt and ensure tesseract is on PATH"}), 503
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "no image uploaded"}), 400
    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        return jsonify({"error": f"unsupported file type {suffix!r}; use png/jpg/jpeg"}), 400

    tmp_path = ROOT / "demo" / f"_upload{suffix}"
    uploaded.save(tmp_path)
    try:
        text = ocr_extract_text(tmp_path).strip()
    except Exception as exc:  # noqa: BLE001 -- surface any OCR failure to the UI, don't crash the server
        return jsonify({"error": f"OCR failed: {type(exc).__name__}: {exc}"}), 422
    finally:
        tmp_path.unlink(missing_ok=True)

    if not text:
        return jsonify({"ocr_text": "", "prediction": None, "error": "OCR found no readable text in this image"}), 200
    result = _classify(text)
    result["ocr_text"] = text
    return jsonify(result)


@app.get("/media/<name>")
def sample_image_file(name: str):
    if name not in SAMPLE_SCREENSHOTS:
        return jsonify({"error": f"unknown sample {name!r}"}), 404
    return send_from_directory(ROOT / "media" / "screenshots", name)


@app.get("/api/sample-image/<name>")
def route_sample_image(name: str):
    if not MEDIAQA_AVAILABLE:
        return jsonify({"error": "Part C (mediaqa) isn't installed -- pip install -r requirements-mediaqa.txt and ensure tesseract is on PATH"}), 503
    if name not in SAMPLE_SCREENSHOTS:
        return jsonify({"error": f"unknown sample {name!r}"}), 404
    path = ROOT / "media" / "screenshots" / name
    try:
        text = ocr_extract_text(path).strip()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"OCR failed: {type(exc).__name__}: {exc}"}), 422
    if not text:
        return jsonify({"ocr_text": "", "prediction": None, "error": "OCR found no readable text in this image"}), 200
    result = _classify(text)
    result["ocr_text"] = text
    return jsonify(result)


@app.post("/api/answer")
def answer_question():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    as_of = (data.get("as_of") or "").strip()
    if not question:
        return jsonify({"error": "question must not be empty"}), 400
    try:
        date.fromisoformat(as_of)
    except ValueError:
        return jsonify({"error": f"as_of must be an ISO date (YYYY-MM-DD), got {as_of!r}"}), 400

    result = _answerer.answer(question, as_of)
    return jsonify(
        {
            "answer": result.answer,
            "doc_ids": list(result.doc_ids),
            "abstained": result.abstained,
            "reason": result.reason,
            "family_id": result.family_id,
        }
    )


def main() -> None:
    print("Demo server (presentation only, not a deliverable) starting at http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
