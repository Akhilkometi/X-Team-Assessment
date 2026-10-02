"""Select the historically valid version within a retrieved document family."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from kbqa.loader import Document, group_families


@dataclass(frozen=True)
class Resolution:
    family_id: str
    document: Document | None
    state: str  # valid, expired_notice, not_yet_effective, or gap
    evidence: tuple[Document, ...] = ()


def is_valid(document: Document, as_of: date) -> bool:
    """Validity is inclusive; status is intentionally not consulted."""
    return document.effective_date <= as_of and (document.valid_until is None or as_of <= document.valid_until)


def resolve_family(family: list[Document], as_of: date) -> Resolution:
    """Resolve a family by dates, never by today's status snapshot."""
    if not family:
        raise ValueError("Cannot resolve an empty document family")
    family_id = family[0].family_id
    if any(document.family_id != family_id for document in family):
        raise ValueError("All documents must belong to the same family")
    valid = [document for document in family if is_valid(document, as_of)]
    if len(valid) > 1:
        # Overlapping validity windows are ambiguous and must not be silently guessed.
        latest_start = max(document.effective_date for document in valid)
        winners = [document for document in valid if document.effective_date == latest_start]
        if len(winners) != 1:
            raise ValueError(f"Overlapping active versions in family {family_id}")
        selected = winners[0]
    elif valid:
        selected = valid[0]
    else:
        notices = all(document.category.lower() == "notice" for document in family)
        all_effective = all(document.effective_date <= as_of for document in family)
        all_expired = all(document.valid_until is not None and as_of > document.valid_until for document in family)
        if notices and all_effective and all_expired:
            return Resolution(family_id, None, "expired_notice", tuple(family))
        if all(document.effective_date > as_of for document in family):
            return Resolution(family_id, None, "not_yet_effective")
        return Resolution(family_id, None, "gap")
    return Resolution(family_id, selected, "valid", (selected,))


def resolve_document(documents: list[Document], family_id: str, as_of: date) -> Resolution:
    families = group_families(documents)
    if family_id not in families:
        raise KeyError(f"Unknown document family: {family_id}")
    return resolve_family(families[family_id], as_of)


@dataclass(frozen=True)
class StatusInconsistency:
    doc_id: str
    status: str
    is_valid_as_of_today: bool


def status_inconsistencies(documents: list[Document], as_of: date | None = None) -> list[StatusInconsistency]:
    """Flag docs whose `status` contradicts what effective_date/valid_until say as of today.

    Dates remain authoritative for resolution (see `is_valid`/`resolve_family`); this is a
    separate, non-blocking consistency check so a stale or mistyped status becomes an
    observable data-quality signal instead of being silently ignored.
    """
    today = as_of if as_of is not None else date.today()
    findings = []
    for document in documents:
        currently_valid = is_valid(document, today)
        if document.status == "current" and not currently_valid:
            findings.append(StatusInconsistency(document.doc_id, document.status, currently_valid))
        elif document.status == "expired" and currently_valid:
            findings.append(StatusInconsistency(document.doc_id, document.status, currently_valid))
    return findings
