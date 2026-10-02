"""Load KB markdown documents and resolve supersedes lineage into families."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_KB_DIR = Path(__file__).resolve().parents[1] / "kb"
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)
_REQUIRED = ("doc_id", "title", "category", "version", "effective_date", "valid_until", "status", "supersedes", "superseded_by")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    category: str
    version: int
    effective_date: date
    valid_until: date | None
    status: str
    supersedes: str | None
    superseded_by: str | None
    body: str
    path: Path
    family_id: str = ""


def _optional_id(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def load_documents(kb_dir: str | Path = DEFAULT_KB_DIR) -> list[Document]:
    """Parse markdown/YAML files, validating metadata and lineage references."""
    documents: dict[str, Document] = {}
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER.match(raw)
        if not match:
            raise ValueError(f"Missing YAML front matter: {path}")
        metadata = yaml.safe_load(match.group(1)) or {}
        missing = [key for key in _REQUIRED if key not in metadata]
        if missing:
            raise ValueError(f"{path.name} missing metadata fields: {', '.join(missing)}")
        doc_id = str(metadata["doc_id"])
        if doc_id in documents:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        try:
            effective = date.fromisoformat(str(metadata["effective_date"]))
            valid_until = date.fromisoformat(str(metadata["valid_until"])) if metadata["valid_until"] else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid date metadata in {path.name}") from exc
        if valid_until is not None and valid_until < effective:
            raise ValueError(f"valid_until precedes effective_date in {path.name}")
        documents[doc_id] = Document(
            doc_id=doc_id,
            title=str(metadata["title"]).strip(),
            category=str(metadata["category"]).strip(),
            version=int(metadata["version"]),
            effective_date=effective,
            valid_until=valid_until,
            status=str(metadata["status"]).strip().lower(),
            supersedes=_optional_id(metadata["supersedes"]),
            superseded_by=_optional_id(metadata["superseded_by"]),
            body=match.group(2).strip(),
            path=path,
        )
    if not documents:
        raise ValueError(f"No markdown KB documents found in {kb_dir}")

    for document in documents.values():
        if document.supersedes and document.supersedes not in documents:
            raise ValueError(f"{document.doc_id} supersedes missing document {document.supersedes}")
        if document.superseded_by and document.superseded_by not in documents:
            raise ValueError(f"{document.doc_id} points to missing successor {document.superseded_by}")
        if document.supersedes and documents[document.supersedes].superseded_by not in (None, document.doc_id):
            raise ValueError(f"Inconsistent reverse lineage for {document.doc_id}")

    def root_id(doc_id: str) -> str:
        seen: set[str] = set()
        current = documents[doc_id]
        while current.supersedes:
            if current.doc_id in seen:
                raise ValueError(f"Cycle in supersedes chain at {current.doc_id}")
            seen.add(current.doc_id)
            current = documents[current.supersedes]
        if current.doc_id in seen:
            raise ValueError(f"Cycle in supersedes chain at {current.doc_id}")
        return current.doc_id

    return [
        Document(**{**doc.__dict__, "family_id": root_id(doc.doc_id)})
        for doc in documents.values()
    ]


def group_families(documents: list[Document]) -> dict[str, list[Document]]:
    families: dict[str, list[Document]] = {}
    for document in documents:
        families.setdefault(document.family_id, []).append(document)
    for versions in families.values():
        versions.sort(key=lambda item: (item.effective_date, item.version, item.doc_id))
    return families
