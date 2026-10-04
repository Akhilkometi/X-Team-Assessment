"""Family-level dense (embedding) retrieval with a conservative abstention gate."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from sentence_transformers import SentenceTransformer

from kbqa.loader import Document, group_families

_TOKEN = re.compile(r"[a-z0-9]+", re.I)
_STOPWORDS = set("a an and are as at be by can do does for from how i in is it my of on or the their to was what when where which who why with your you".split())
_SYNONYMS = {
    "fee": {"fee", "fees", "charge", "charges", "cost", "costs"},
    "withdrawal": {"withdrawal", "withdrawals", "withdraw", "outbound", "send", "sending", "cash", "cashout"},
    "transfer": {"transfer", "transfers", "moving", "move", "send", "sending"},
    "staking": {"staking", "staked", "stake", "earn", "rewards", "reward", "yield"},
    "authenticator": {"authenticator", "2fa", "two", "factor", "verification", "code"},
    "recover": {"recover", "recovery", "restore", "restoring", "access", "login"},
    "dispute": {"dispute", "disputes", "challenge", "raise", "complaint"},
    "fraud": {"fraud", "unauthorized", "unauthorised", "theft", "scam", "phishing", "stolen"},
    "limits": {"limit", "limits", "ceiling", "cap", "maximum", "tier"},
    "bank": {"bank", "deposit", "funding", "clearing", "clear"},
    "supported": {"support", "supported", "available", "listing", "listed"},
    "alert": {"alert", "alerts", "notification", "notifications"},
    "card": {"card", "debit", "purchase", "funded"},
    "referral": {"referral", "refer", "invite", "invitation", "bonus", "friend"},
    "delay": {"slow", "slower", "slowly", "delay", "delayed", "long", "taking", "minutes", "appear", "show"},
}

# Explicitly unsupported intents prevent a nearby article from being mistaken for an answer.
_UNSUPPORTED = (
    (re.compile(r"\b(?:margin|leverage)\b", re.I), "margin trading is not covered by this knowledge base"),
    (re.compile(r"\b(?:joint account|account with my spouse)\b", re.I), "joint accounts are not covered by this knowledge base"),
    (re.compile(r"\b(?:loan|loans|borrow|crypto.backed)\b", re.I), "crypto-backed lending is not covered by this knowledge base"),
    (re.compile(r"\b(?:custodian|reserve assets|who audits)\b", re.I), "custody and reserve audits are not covered by this knowledge base"),
    (re.compile(r"\b(?:electricity bill|bill pay|pay my .* bill)\b", re.I), "bill payment is not covered by this knowledge base"),
    (re.compile(r"\b(?:holder dies|after .* dies|death|beneficiary after)\b", re.I), "beneficiary handling after death is not covered by this knowledge base"),
    (re.compile(r"\b(?:direct phone number|phone number for .* fraud team)\b", re.I), "no direct fraud-team phone number is documented"),
    (re.compile(r"\bchargeback\b", re.I), "bank-initiated chargebacks are outside the documented in-app transaction dispute process"),
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text) if token.lower() not in _STOPWORDS]


def expanded_query_terms(question: str) -> list[str]:
    terms = set(tokenize(question))
    for group in _SYNONYMS.values():
        if terms.intersection(group):
            terms.update(group)
    return sorted(terms)


@dataclass(frozen=True)
class SearchHit:
    family_id: str
    score: float
    title: str


@dataclass(frozen=True)
class Retrieval:
    hit: SearchHit | None
    top_score: float
    margin: float
    candidates: tuple[SearchHit, ...]
    abstain_reason: str | None = None


_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# BGE models are trained asymmetrically: queries need this instruction prefix for
# retrieval (it measurably improves ranking), passages/documents do not.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model -- shared across index instances
    so repeated KBAnswerer construction (e.g. in tests) doesn't reload it each time."""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _model_cache


class FamilyEmbeddingIndex:
    """Family-level dense retrieval: embed each document family's full text once at
    build time, then rank by cosine similarity to the embedded question. Runs fully
    local via sentence-transformers -- no API key, no network call at query time."""

    def __init__(self, documents: Iterable[Document], *, model: SentenceTransformer | None = None):
        self.families = group_families(list(documents))
        self._model = model or _get_model()
        self._titles: dict[str, str] = {}
        family_ids: list[str] = []
        texts: list[str] = []
        for family_id, versions in self.families.items():
            title = " ".join(document.title for document in versions)
            text = title + " " + " ".join(document.body for document in versions)
            self._titles[family_id] = versions[-1].title
            family_ids.append(family_id)
            texts.append(text)
        self._family_ids = family_ids
        self._embeddings = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def search(self, question: str, *, min_score: float = 0.30, min_margin: float = 0.02, limit: int = 5) -> Retrieval:
        for pattern, reason in _UNSUPPORTED:
            if pattern.search(question):
                return Retrieval(None, 0.0, 0.0, (), f"out_of_scope: {reason}")
        if not question.strip():
            return Retrieval(None, 0.0, 0.0, (), "empty_or_noninformative_query")
        query_embedding = self._model.encode([_QUERY_PREFIX + question], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self._embeddings @ query_embedding  # cosine similarity: both sides are unit-normalized
        order = sorted(range(len(self._family_ids)), key=lambda i: (-scores[i], self._family_ids[i]))
        hits = [SearchHit(self._family_ids[i], float(scores[i]), self._titles[self._family_ids[i]]) for i in order]
        candidates = tuple(hits[:limit])
        top_score = hits[0].score if hits else 0.0
        margin = top_score - (hits[1].score if len(hits) > 1 else 0.0)
        if not hits or top_score < min_score:
            return Retrieval(None, top_score, margin, candidates, "top_score_below_threshold")
        if margin < min_margin:
            return Retrieval(None, top_score, margin, candidates, "top_family_margin_too_small")
        return Retrieval(hits[0], top_score, margin, candidates)
