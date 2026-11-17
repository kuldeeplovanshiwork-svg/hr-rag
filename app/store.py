"""In-memory document store. The tenant filter lives here, in code."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from . import security
from .retrieval import split_passages
from .schemas import DuplicateDocumentError, IngestRequest, UnprocessableDocumentError


@dataclass
class Document:
    tenant_id: str
    doc_id: str
    title: str
    doc_type: str
    raw_text: str      # original, kept for audit, never indexed
    clean_text: str    # sanitised, the only text that is ever indexed or shown
    topic: Optional[str] = None
    status: str = "active"
    effective_date: Optional[str] = None
    min_role: str = "recruiter"
    security_flags: List[str] = field(default_factory=list)
    passages: List[Tuple[int, int]] = field(default_factory=list)


class DocumentStore:
    def __init__(self, passage_max_chars: int = 350):
        self._docs: Dict[Tuple[str, str], Document] = {}
        self.quarantined: List[dict] = []
        self._passage_max_chars = passage_max_chars

    def build_document(self, *, tenant_id: str, req: IngestRequest) -> Document:
        """Tenant is an explicit argument: never read from the request body or the text."""
        if not tenant_id:
            raise UnprocessableDocumentError("A document must belong to a tenant.")
        result = security.sanitize_text(req.text)
        if not result.text.strip():
            raise UnprocessableDocumentError(
                "Document has no usable text after removing instruction-like or stuffed content.",
                details=result.flags,
            )
        return Document(
            tenant_id=tenant_id, doc_id=req.doc_id, title=security.sanitize_title(req.title),
            doc_type=req.doc_type, raw_text=req.text, clean_text=result.text, topic=req.topic,
            status=req.status,
            effective_date=req.effective_date.isoformat() if req.effective_date else None,
            min_role=req.min_role, security_flags=result.flags,
            passages=split_passages(result.text, self._passage_max_chars),
        )

    def add(self, doc: Document, overwrite: bool = False) -> None:
        key = (doc.tenant_id, doc.doc_id)
        if key in self._docs and not overwrite:
            raise DuplicateDocumentError(f"Document '{doc.doc_id}' already exists for this tenant.")
        self._docs[key] = doc

    def get(self, tenant_id: str, doc_id: str) -> Optional[Document]:
        return self._docs.get((tenant_id, doc_id))

    def visible_to(self, principal) -> List[Document]:
        """THE tenant + role filter. Retrieval is only ever given this list."""
        return [d for d in self._docs.values() if security.can_read(principal, d)]

    def count(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------ bulk loading
    def _quarantine(self, rel: str, reason: str) -> None:
        self.quarantined.append({"file": rel, "reason": reason})

    def load_directory(self, data_dir: Path) -> None:
        """Load *.json files (each a list of records). In data/tenant_x/ the folder name IS the
        tenant. Anywhere else the record must carry its own tenant_id, otherwise it is
        quarantined (this is the bulk-import bug the scenario is about)."""
        data_dir = Path(data_dir)
        for path in sorted(data_dir.rglob("*.json")):
            rel = path.relative_to(data_dir).as_posix()
            folder = path.parent.name
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._quarantine(rel, "invalid_json")
                continue
            if not isinstance(records, list):
                records = [records]
            for rec in records:
                if not isinstance(rec, dict):
                    self._quarantine(rel, "invalid_record")
                    continue
                rec = dict(rec)
                tenant = rec.pop("tenant_id", None)
                if folder.startswith("tenant_"):
                    if tenant not in (None, folder):
                        self._quarantine(rel, "tenant_mismatch")
                        continue
                    tenant = folder
                if not tenant or not isinstance(tenant, str):
                    self._quarantine(rel, "missing_tenant_id")
                    continue
                try:
                    req = IngestRequest(**rec)
                    self.add(self.build_document(tenant_id=tenant, req=req), overwrite=True)
                except (ValidationError, UnprocessableDocumentError):
                    self._quarantine(rel, "invalid_record")
