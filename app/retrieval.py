"""Passage splitting and BM25 scoring.

Every score in this service is computed here from token statistics only. Nothing in a
document is parsed for "score", "rank" or "priority" values, so a CV cannot influence its own
ranking except by genuinely containing words that match the question.

Anti keyword-stuffing: per-term frequency cap (max_tf), BM25 length normalisation, and
sentence-level stuffing removal in security.sanitize_text.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from .config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from .store import Document

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its me my of on or our "
    "should that the their there they this to was what when where which who will with you your "
    "about tell much many".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"[^\n.!?]+[.!?]*")


def stem(tok: str) -> str:
    """Very light stemmer (hire/hired/hires, policy/policies)."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    if tok.endswith(("ies", "ied")) and len(tok) > 4:
        tok = tok[:-3] + "y"
    else:
        for suf in ("ations", "ation", "ings", "ing", "ed", "s"):
            if tok.endswith(suf) and not tok.endswith("ss") and len(tok) - len(suf) >= 3:
                tok = tok[: -len(suf)]
                break
    if tok.endswith("e") and len(tok) > 3:
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def split_passages(text: str, max_chars: int = 350) -> List[Tuple[int, int]]:
    """Return (start, end) spans into ``text``. Every passage is a verbatim substring."""
    spans: List[Tuple[int, int]] = []
    cur: Optional[List[int]] = None
    for m in _SENT_RE.finditer(text):
        seg = m.group(0)
        s = m.start() + (len(seg) - len(seg.lstrip()))
        e = m.end() - (len(seg) - len(seg.rstrip()))
        if e <= s:
            continue
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= max_chars:
            cur[1] = e
        else:
            spans.append((cur[0], cur[1]))
            cur = [s, e]
    if cur is not None:
        spans.append((cur[0], cur[1]))
    return spans


@dataclass(frozen=True)
class Hit:
    tenant_id: str
    doc_id: str
    title: str
    doc_type: str
    topic: Optional[str]
    status: str
    effective_date: Optional[str]
    excerpt: str
    score: float


def _cosine01(a: List[float], b: List[float], floor: float, ceil: float) -> float:
    """Vectors are L2-normalised, so the dot product is the cosine. Rescale floor..ceil to 0..1."""
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, (dot - floor) / (ceil - floor)))


def search(docs: Iterable["Document"], query: str, *, top_k: int, settings: Settings,
           embedder=None) -> List[Hit]:
    """Hybrid search over passages of the supplied documents (best passage per document).

    keyword score  = BM25, normalised to 0..1 (share of query IDF weight covered)
    semantic score = MiniLM cosine similarity, rescaled to 0..1 (only if ``embedder`` is given)
    final score    = ((1 - w) * keyword + w * semantic) * trust weight chosen in code by doc type

    Callers pass ONLY permitted documents, so neither IDF statistics nor embeddings can leak
    anything about other tenants.
    """
    q_terms = set(tokenize(query))
    if not q_terms:
        return []
    passages = []
    for d in docs:
        for s, e in d.passages:
            text = d.clean_text[s:e]
            passages.append((d, text, Counter(tokenize(text))))
    if not passages:
        return []

    sims = None
    if embedder is not None:
        q_vec = embedder.encode([query])[0]
        p_vecs = embedder.encode([text for _, text, _ in passages])
        sims = [_cosine01(q_vec, v, settings.embed_floor, settings.embed_ceil) for v in p_vecs]
    w = settings.embed_weight if sims is not None else 0.0

    n = len(passages)
    avg = (sum(sum(c.values()) for _, _, c in passages) / n) or 1.0
    idf = {}
    for t in q_terms:
        df = sum(1 for _, _, c in passages if c.get(t))
        idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
    total_idf = sum(idf.values())
    k1, b = settings.bm25_k1, settings.bm25_b

    best: dict = {}
    for idx, (d, text, counts) in enumerate(passages):
        dl = sum(counts.values())
        raw = 0.0
        for t in q_terms:
            tf = min(counts.get(t, 0), settings.max_tf)
            if tf:
                raw += idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
        keyword = min(1.0, raw / total_idf)
        semantic = sims[idx] if sims is not None else 0.0
        combined = (1 - w) * keyword + w * semantic
        if combined <= 0:
            continue
        weight = settings.trust_weights.get(d.doc_type, 0.5)
        score = round(min(1.0, combined) * weight, 4)
        hit = Hit(
            tenant_id=d.tenant_id, doc_id=d.doc_id, title=d.title, doc_type=d.doc_type,
            topic=d.topic, status=d.status, effective_date=d.effective_date,
            excerpt=text[: settings.max_excerpt_chars], score=score,
        )
        key = (d.tenant_id, d.doc_id)
        cur = best.get(key)
        if cur is None or (hit.score, cur.excerpt) > (cur.score, hit.excerpt):
            best[key] = hit
    ranked = sorted(best.values(), key=lambda h: (-h.score, h.doc_id))
    return ranked[:top_k]
