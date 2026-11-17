"""Evidence analysis (outdated / conflicting) and verification of model output.

The model only PROPOSES an answer with inline [ID] markers. Everything the client sees as a
citation (source id, excerpt, score) is rebuilt here from retrieval results; model text can
never add or change a citation.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .config import Settings
from .retrieval import Hit
from .schemas import Citation

_BRACKET_RE = re.compile(r"\[([A-Za-z0-9_\-, ]{3,200})\]")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def to_citation(hit: Hit) -> Citation:
    return Citation(
        source_id=hit.doc_id, title=hit.title, doc_type=hit.doc_type, status=hit.status,
        effective_date=hit.effective_date, excerpt=hit.excerpt, score=hit.score,
    )


# ----------------------------------------------------------------------------- evidence analysis
@dataclass
class Conflict:
    topic: str
    hits: List[Hit]


@dataclass
class EvidenceAnalysis:
    current: List[Hit] = field(default_factory=list)
    outdated: List[Hit] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)


def _numbers(text: str) -> frozenset:
    return frozenset(n.replace(",", "") for n in _NUM_RE.findall(text))


def analyze_evidence(hits: Iterable[Hit]) -> EvidenceAnalysis:
    """Decided in code, not by the model.

    * superseded documents are never used as support; they are reported as outdated.
    * two ACTIVE documents with the same topic whose numeric facts differ are a conflict.
    """
    out = EvidenceAnalysis()
    by_topic: Dict[str, List[Hit]] = defaultdict(list)
    for h in hits:
        if h.status != "active":
            out.outdated.append(h)
            continue
        out.current.append(h)
        if h.topic:
            by_topic[h.topic].append(h)
    for topic, group in by_topic.items():
        if len({h.doc_id for h in group}) > 1 and len({_numbers(h.excerpt) for h in group}) > 1:
            out.conflicts.append(Conflict(topic=topic, hits=group))
    return out


def conflict_answer(conflict: Conflict) -> str:
    parts = [f"[{h.doc_id}] says: \"{' '.join(h.excerpt.split())}\"" for h in conflict.hits]
    return (
        f"The sources disagree about '{conflict.topic.replace('_', ' ')}'. "
        + " ".join(parts)
        + " I cannot tell which is correct; please confirm with an HR administrator."
    )


# ----------------------------------------------------------------------------- output verification
@dataclass
class Verified:
    ok: bool
    answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    reason: Optional[str] = None


def extract_cited_ids(text: str) -> List[str]:
    ids: List[str] = []
    for m in _BRACKET_RE.finditer(text):
        for part in m.group(1).split(","):
            part = part.strip()
            if part:
                ids.append(part)
    return ids


def verify_answer(raw: str, allowed: List[Hit], settings: Settings) -> Verified:
    """Accept the model answer only if every cited id is in the evidence set, no canary token
    appears that is absent from the evidence, and every number is supported by the evidence."""
    answer = raw.strip()
    ids = extract_cited_ids(answer)
    body = _BRACKET_RE.sub(" ", answer)
    if not ids or not re.search(r"[A-Za-z]", body):
        return Verified(False, reason="uncited_answer")
    by_id = {h.doc_id: h for h in allowed}
    if any(i not in by_id for i in ids):
        return Verified(False, reason="citation_outside_evidence")
    evidence_text = " ".join(h.excerpt for h in allowed)
    if any(c not in evidence_text for c in set(re.findall(settings.canary_pattern, answer))):
        return Verified(False, reason="output_guard_blocked")
    if _numbers(body) - _numbers(evidence_text):
        return Verified(False, reason="unsupported_number")
    cites = [to_citation(by_id[i]) for i in dict.fromkeys(ids)]
    return Verified(True, answer=answer, citations=cites)
