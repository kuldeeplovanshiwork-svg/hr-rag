"""Tenant/role filtering, ranking, determinism and scoring integrity."""
from app import retrieval
from app.config import Settings
from app.retrieval import split_passages, tokenize
from app.security import Principal

A_REC = Principal("tenant_a", "recruiter")
A_EMP = Principal("tenant_a", "employee")
A_ADM = Principal("tenant_a", "hr_admin")
B_REC = Principal("tenant_b", "recruiter")
B_ADM = Principal("tenant_b", "hr_admin")
S = Settings()


def ids(docs):
    return {d.doc_id for d in docs}


def test_tenant_filter_isolates_both_ways(seed_store):
    a, b = seed_store.visible_to(A_ADM), seed_store.visible_to(B_ADM)
    assert a and b
    assert {d.tenant_id for d in a} == {"tenant_a"}
    assert {d.tenant_id for d in b} == {"tenant_b"}
    assert not ids(a) & ids(b)


def test_role_filter(seed_store):
    assert "A-SAL-401" not in ids(seed_store.visible_to(A_EMP))
    assert "A-SAL-401" not in ids(seed_store.visible_to(A_REC))
    assert "A-SAL-401" in ids(seed_store.visible_to(A_ADM))
    assert "A-CV-301" not in ids(seed_store.visible_to(A_EMP))
    assert "A-CV-301" in ids(seed_store.visible_to(A_REC))


def test_quarantined_records_are_never_visible(seed_store):
    assert any(q["reason"] == "missing_tenant_id" for q in seed_store.quarantined)
    for p in (A_ADM, B_ADM):
        assert "X-CV-900" not in ids(seed_store.visible_to(p))


def test_ranking_prefers_matching_policy(seed_store):
    hits = retrieval.search(seed_store.visible_to(A_EMP), "What is the probation period for new hires?", top_k=5, settings=S)
    assert {h.doc_id for h in hits[:2]} == {"A-KB-001", "A-KB-002"}
    assert hits[0].score >= S.min_score


def test_tenants_get_their_own_answer_source(seed_store):
    q = "What is the probation period for new hires?"
    a = retrieval.search(seed_store.visible_to(A_EMP), q, top_k=3, settings=S)
    b = retrieval.search(seed_store.visible_to(B_REC), q, top_k=3, settings=S)
    assert all(h.tenant_id == "tenant_a" for h in a)
    assert b[0].doc_id == "B-KB-001" and all(h.tenant_id == "tenant_b" for h in b)


def test_off_topic_query_scores_below_threshold(seed_store):
    hits = retrieval.search(seed_store.visible_to(A_ADM), "What is the CEO favourite colour?", top_k=5, settings=S)
    assert not hits or hits[0].score < S.min_score


def test_scores_are_deterministic_and_top_k_respected(seed_store):
    docs = seed_store.visible_to(A_ADM)
    r1 = retrieval.search(docs, "notice period resignation", top_k=2, settings=S)
    r2 = retrieval.search(list(reversed(docs)), "notice period resignation", top_k=2, settings=S)
    assert r1 == r2 and len(r1) == 2


def test_empty_or_stopword_query_returns_nothing(seed_store):
    assert retrieval.search(seed_store.visible_to(A_ADM), "what is the", top_k=5, settings=S) == []


def test_excerpts_are_verbatim_substrings(seed_store):
    hits = retrieval.search(seed_store.visible_to(A_ADM), "salary bands notice probation referral leave", top_k=10, settings=S)
    assert hits
    for h in hits:
        assert h.excerpt in seed_store.get(h.tenant_id, h.doc_id).clean_text


def test_scores_use_only_code_weights(seed_store):
    hits = retrieval.search(seed_store.visible_to(A_REC), "HR coordinator experience recruiting", top_k=10, settings=S)
    cv_hits = [h for h in hits if h.doc_type == "cv"]
    assert cv_hits and all(0 < h.score <= S.trust_weights["cv"] + 1e-9 for h in cv_hits)
    assert all(0 <= h.score <= 1.0 for h in hits)


def test_stemming_and_passages():
    assert tokenize("New hires were hired") == ["new", "hir", "were", "hir"] or tokenize("hired")[0] == tokenize("hires")[0]
    text = "One sentence here. Another sentence follows. " * 20
    for s, e in split_passages(text, 100):
        assert text[s:e].strip() == text[s:e] and e - s <= 130
