"""Hybrid search (BM25 + embeddings). A fake embedder is used, so no model is downloaded."""
import pytest

from app import retrieval
from app.config import Settings
from app.retrieval import tokenize
from app.security import Principal

A_EMP = Principal("tenant_a", "employee")
A_REC = Principal("tenant_a", "recruiter")
S = Settings(use_embeddings=False)

# words that mean the same thing share a concept dimension (a toy stand-in for MiniLM)
CONCEPTS = ["probation", "notice", "leave", "bonus", "salary"]
_RAW = {"trial": "probation", "probation": "probation", "resign": "notice", "notice": "notice",
        "vacation": "leave", "leave": "leave", "bonus": "bonus", "salary": "salary", "pay": "salary"}
SYNONYMS = {tokenize(k)[0]: v for k, v in _RAW.items()}  # keys are stemmed like the real tokens


class FakeEmbedder:
    name = "fake-embedder"

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            vec = [0.0] * len(CONCEPTS)
            for tok in tokenize(t):
                if tok in SYNONYMS:  # unknown words add nothing, like unrelated text in a real model
                    vec[CONCEPTS.index(SYNONYMS[tok])] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


QUERY = "How long is the trial period for new staff?"  # says "trial", documents say "probation"


def test_semantic_match_beats_keyword_only(seed_store):
    docs = seed_store.visible_to(A_EMP)
    kw = retrieval.search(docs, QUERY, top_k=3, settings=S)
    hy = retrieval.search(docs, QUERY, top_k=3, settings=S, embedder=FakeEmbedder())
    kw_score = {h.doc_id: h.score for h in kw}.get("A-KB-002", 0.0)
    new = next(h for h in hy if h.doc_id == "A-KB-002")  # the superseded A-KB-001 may tie; code drops it later
    assert new.score > kw_score
    assert new.score >= S.min_score and kw_score < S.min_score  # keyword-only would have abstained


def test_hybrid_is_deterministic_and_tenant_scoped(seed_store):
    emb = FakeEmbedder()
    docs = seed_store.visible_to(A_EMP)
    r1 = retrieval.search(docs, QUERY, top_k=4, settings=S, embedder=emb)
    r2 = retrieval.search(list(reversed(docs)), QUERY, top_k=4, settings=S, embedder=emb)
    assert r1 == r2 and all(h.tenant_id == "tenant_a" for h in r1)


def test_embedder_never_sees_other_tenants_text(seed_store):
    seen = []

    class Spy(FakeEmbedder):
        def encode(self, texts):
            seen.extend(texts)
            return super().encode(texts)

    retrieval.search(seed_store.visible_to(A_REC), QUERY, top_k=3, settings=S, embedder=Spy())
    blob = " ".join(seen)
    for secret in ("Zara", "CANARY-TB-7731", "91000"):
        assert secret not in blob


def test_hostile_text_never_reaches_embedder_or_boosts_score(seed_store):
    seen = []

    class Spy(FakeEmbedder):
        def encode(self, texts):
            seen.extend(texts)
            return super().encode(texts)

    hits = retrieval.search(seed_store.visible_to(A_REC), "HR coordinator experience", top_k=10, settings=S,
                            embedder=Spy())
    blob = " ".join(seen)
    assert "IGNORE ALL PREVIOUS" not in blob and "relevance score" not in blob
    assert all(h.score <= S.trust_weights[h.doc_type] + 1e-9 for h in hits)


def test_off_topic_still_below_threshold_with_embeddings(seed_store):
    hits = retrieval.search(seed_store.visible_to(A_EMP), "What is the CEO favourite colour?", top_k=5,
                            settings=S, embedder=FakeEmbedder())
    assert not hits or hits[0].score < S.min_score


def test_api_answers_semantic_question_and_reports_embeddings(make_client, H):
    c = make_client(embedder=FakeEmbedder())
    assert c.get("/health").json()["embeddings"] is True
    r = c.post("/ask", json={"question": QUERY}, headers=H["a_employee"]).json()
    assert r["status"] == "answered" and r["citations"][0]["source_id"] == "A-KB-002"


def test_without_embedder_health_says_false(client):
    assert client.get("/health").json()["embeddings"] is False
