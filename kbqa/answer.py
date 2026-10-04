"""Extractive KB question answering with temporal resolution and abstention."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from kbqa.loader import Document
from kbqa.resolve import Resolution, resolve_family
from kbqa.retrieve import FamilyEmbeddingIndex, Retrieval, expanded_query_terms, tokenize


@dataclass(frozen=True)
class Answer:
    answer: str
    doc_ids: tuple[str, ...]
    family_id: str | None
    abstained: bool
    reason: str | None
    retrieval_score: float = 0.0
    retrieval_margin: float = 0.0


def _extract_sentences(document: Document, question: str, count: int = 2) -> str:
    body = re.sub(r"^#\s+[^\n]+\s*", "", document.body, count=1)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body.replace("\n", " ")) if part.strip()]
    if not sentences:
        return ""
    query = set(expanded_query_terms(question))
    ranked = []
    for index, sentence in enumerate(sentences):
        words = set(tokenize(sentence))
        overlap = len(words.intersection(query))
        ranked.append((overlap, -index, index, sentence))
    selected = sorted(sorted(ranked, reverse=True)[:count], key=lambda item: item[2])
    useful = [item[3] for item in selected if item[0] > 0]
    if not useful:
        useful = sentences[:1]
    return " ".join(useful)


def _answer_resolution(resolution: Resolution, question: str, as_of: date) -> Answer:
    if resolution.state == "expired_notice":
        doc_ids = tuple(document.doc_id for document in resolution.evidence)
        return Answer(f"There is no active notice or promotion in this document family as of {as_of.isoformat()}.", doc_ids, resolution.family_id, False, None)
    if resolution.state != "valid" or resolution.document is None:
        return Answer("", (), resolution.family_id, True, f"temporal_{resolution.state}")
    text = _extract_sentences(resolution.document, question)
    if not text:
        return Answer("", (), resolution.family_id, True, "no_extractable_evidence")
    return Answer(text, (resolution.document.doc_id,), resolution.family_id, False, None)


class KBAnswerer:
    def __init__(self, documents: list[Document], *, min_score: float = 0.30, min_margin: float = 0.02):
        if not documents:
            raise ValueError("At least one KB document is required")
        self.documents = documents
        self.index = FamilyEmbeddingIndex(documents)
        self.min_score = min_score
        self.min_margin = min_margin

    def answer(self, question: str, as_of: date | str) -> Answer:
        """Answer at family level, select the date-valid version, or abstain."""
        if not question.strip():
            return Answer("", (), None, True, "empty_question")
        try:
            requested_date = as_of if isinstance(as_of, date) else date.fromisoformat(as_of)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"as_of must be an ISO date, got {as_of!r}") from exc
        retrieval: Retrieval = self.index.search(question, min_score=self.min_score, min_margin=self.min_margin)
        if retrieval.hit is None:
            return Answer("", (), None, True, retrieval.abstain_reason, retrieval.top_score, retrieval.margin)
        family = self.index.families[retrieval.hit.family_id]
        resolution = resolve_family(family, requested_date)
        answer = _answer_resolution(resolution, question, requested_date)
        return Answer(answer.answer, answer.doc_ids, answer.family_id, answer.abstained, answer.reason, retrieval.top_score, retrieval.margin)
