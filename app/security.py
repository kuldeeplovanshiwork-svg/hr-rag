"""Authentication, permission checks and input sanitising.

Layers (none rely on the LLM behaving):
  1. Tenant + role come from the API key (authenticate), never from a body or a document.
  2. The store only hands out documents the principal may read (can_read).
  3. assert_hits_permitted re-checks every retrieved hit and fails closed.
  4. sanitize_text removes instruction-like / stuffed sentences BEFORE indexing, so they never
     reach ranking or the prompt.
  5. citations.verify_answer rejects model output that cites anything outside the evidence.
The regex list is a speed bump, not the security boundary: layers 1-3 and 5 hold even if it misses.
"""
from __future__ import annotations

import hmac
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .config import Settings
from .retrieval import Hit, tokenize
from .schemas import AuthenticationError, IsolationViolationError, PermissionDeniedError

ROLE_RANK = {"employee": 1, "recruiter": 2, "hr_admin": 3}


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    role: str


def authenticate(api_key: Optional[str], settings: Settings) -> Principal:
    if not api_key:
        raise AuthenticationError("Missing X-API-Key header.")
    found = None
    for known, (tenant, role) in settings.api_keys.items():
        if hmac.compare_digest(known.encode(), api_key.encode()):
            found = (tenant, role)
    if found is None:
        raise AuthenticationError("Invalid API key.")
    return Principal(tenant_id=found[0], role=found[1])


def require_role(principal: Principal, minimum: str) -> None:
    if ROLE_RANK[principal.role] < ROLE_RANK[minimum]:
        raise PermissionDeniedError(f"Role '{principal.role}' may not perform this action.")


def can_read(principal: Principal, doc) -> bool:
    return doc.tenant_id == principal.tenant_id and ROLE_RANK[principal.role] >= ROLE_RANK[doc.min_role]


def assert_hits_permitted(principal: Principal, hits: Iterable[Hit], store) -> None:
    """Defence in depth: re-verify every hit against the store. Never echoes record content."""
    for h in hits:
        doc = store.get(principal.tenant_id, h.doc_id)
        if h.tenant_id != principal.tenant_id or doc is None or not can_read(principal, doc):
            raise IsolationViolationError("A retrieved record failed the access re-check; request aborted.")


# ----------------------------------------------------------------------------- sanitising
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_F = re.IGNORECASE

INJECTION_PATTERNS = [
    ("ignore_instructions", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|any|the)\b"
        r".{0,30}\b(instructions?|rules?|prompts?|polic(?:y|ies)|guidelines?)\b", _F)),
    ("permission_escalation", re.compile(
        r"\b(?:admin(?:istrator)? mode|grant(?:ed)?\b.{0,20}\b(?:permission|access)|approved by security"
        r"|you are now\b.{0,15}\b(?:system|admin)|(?:reveal|show|print|output|dump)\b.{0,50}"
        r"\b(?:candidate|profile|record|salary|tenant)|(?:other|another|all) tenants?|tenant_[a-z0-9]+"
        r"|bypass\b.{0,20}\b(?:filter|permission|access))", _F)),
    ("score_manipulation", re.compile(
        r"\b(?:relevance|ranking|priority|confidence) score\b|\brank (?:this|it)\b"
        r"|\b(?:highest|top|maximum) (?:rank|score|priority|relevance)\b|\bscore of this document\b"
        r"|\bset (?:the )?score\b", _F)),
    ("citation_forgery", re.compile(
        r"\[[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\]|\bcite (?:this|it|as)\b|\bsource id\b"
        r"|\bas (?:the )?(?:only|sole|official) source\b", _F)),
    ("role_spoofing", re.compile(r"\b(?:system|assistant)\s*(?:prompt|message)?\s*:", _F)),
]


@dataclass
class SanitizeResult:
    text: str
    flags: List[str]


def _is_stuffed(sentence: str) -> bool:
    toks = tokenize(sentence)
    if len(toks) < 6:
        return False
    counts = Counter(toks)
    return counts.most_common(1)[0][1] >= 4 or len(counts) / len(toks) < 0.5


def sanitize_text(text: str) -> SanitizeResult:
    """Drop instruction-like and keyword-stuffed sentences; neutralise tag syntax."""
    t = unicodedata.normalize("NFKC", text.translate(_ZERO_WIDTH))
    t = _CONTROL.sub(" ", t).replace("<", " ").replace(">", " ")
    flags: List[str] = []
    kept: List[str] = []
    for sent in _SENTENCE_SPLIT.split(t):
        s = sent.strip()
        if not s:
            continue
        hostile = False
        for name, pat in INJECTION_PATTERNS:
            if pat.search(s):
                flag = f"injection:{name}"
                if flag not in flags:
                    flags.append(flag)
                hostile = True
        if hostile:
            continue
        if _is_stuffed(s):
            if "keyword_stuffing" not in flags:
                flags.append("keyword_stuffing")
            continue
        kept.append(s if s[-1] in ".!?:" else s + ".")
    return SanitizeResult(" ".join(kept), flags)


def sanitize_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title.translate(_ZERO_WIDTH))
    t = re.sub(r"[\"<>\r\n]", " ", _CONTROL.sub(" ", t))
    return " ".join(t.split())[:200] or "untitled"
