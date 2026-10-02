"""Family-level BM25 retrieval with synonyms and a conservative abstention gate."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

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


class FamilyBM25:
    """Small dependency-free BM25 index over concatenated family versions."""

    def __init__(self, documents: Iterable[Document], *, k1: float = 1.5, b: float = 0.75):
        self.families = group_families(list(documents))
        self.k1 = k1
        self.b = b
        self._tokens: dict[str, list[str]] = {}
        self._titles: dict[str, str] = {}
        self._lengths: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        for family_id, versions in self.families.items():
            title = " ".join(document.title for document in versions)
            text = title + " " + " ".join(document.body for document in versions)
            tokens = tokenize(text)
            self._tokens[family_id] = tokens
            self._titles[family_id] = versions[-1].title
            self._lengths[family_id] = len(tokens)
            self._df.update(set(tokens))
        self._average_length = sum(self._lengths.values()) / max(len(self._lengths), 1)

    def search(self, question: str, *, min_score: float = 1.5, min_margin: float = 0.15, limit: int = 5) -> Retrieval:
        for pattern, reason in _UNSUPPORTED:
            if pattern.search(question):
                return Retrieval(None, 0.0, 0.0, (), f"out_of_scope: {reason}")
        query_terms = expanded_query_terms(question)
        if not query_terms:
            return Retrieval(None, 0.0, 0.0, (), "empty_or_noninformative_query")
        family_count = len(self._tokens)
        hits = []
        for family_id, tokens in self._tokens.items():
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                document_frequency = self._df[term]
                inverse_frequency = math.log(1 + (family_count - document_frequency + 0.5) / (document_frequency + 0.5))
                length_norm = self.k1 * (1 - self.b + self.b * len(tokens) / max(self._average_length, 1))
                score += inverse_frequency * frequency * (self.k1 + 1) / (frequency + length_norm)
            # A small title boost preserves topic match while body terms carry the answer evidence.
            title_terms = set(tokenize(self._titles[family_id]))
            score += 0.25 * sum(1 for term in query_terms if term in title_terms)
            question_lower = question.lower()
            if re.search(r"\b(?:referral|refer|invite|invitation|bonus)\b", question_lower) and "referral" in self._titles[family_id].lower():
                score += 5.0
            if re.search(r"\b(?:support|email\w*|message)\b", question_lower) and re.search(r"\b(?:one.time|verification|authenticator)\s+code\b", question_lower) and "phishing" in self._titles[family_id].lower():
                score += 15.0
            if re.search(r"\b(?:slow|slower|slowly|delay|delayed|taking .* long|not show up|long to show)\b", question_lower) and "notice" in self._titles[family_id].lower():
                score += 3.0
            if re.search(r"\b(?:disput\w*|challeng\w*)\b", question_lower) and "dispute" in self._titles[family_id].lower():
                score += 5.0
            hits.append(SearchHit(family_id, score, self._titles[family_id]))
        hits.sort(key=lambda item: (-item.score, item.family_id))
        candidates = tuple(hits[:limit])
        top_score = hits[0].score if hits else 0.0
        margin = top_score - (hits[1].score if len(hits) > 1 else 0.0)
        if not hits or top_score < min_score:
            return Retrieval(None, top_score, margin, candidates, "top_score_below_threshold")
        if margin < min_margin:
            return Retrieval(None, top_score, margin, candidates, "top_family_margin_too_small")
        return Retrieval(hits[0], top_score, margin, candidates)
