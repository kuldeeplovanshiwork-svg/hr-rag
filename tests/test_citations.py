"""Citations, outdated/conflicting evidence, abstention and output verification."""
import pytest

from app import citations
from app.config import Settings
from app.retrieval import Hit
from tests.conftest import FakeLLM

PROB = {"question": "What is the probation period for new hires?"}
NOTICE = {"question": "What is the notice period for resignation?"}


def mk(doc_id, status="active", topic="t", excerpt="Value is 5 days.", score=0.9):
    return Hit("tenant_a", doc_id, "T", "kb", topic, status, None, excerpt, score)


# ------------------------------------------------------------------ unit tests
def test_extract_cited_ids():
    assert citations.extract_cited_ids("x [A-KB-001, A-KB-002] y [KB-003]") == ["A-KB-001", "A-KB-002", "KB-003"]
    assert citations.extract_cited_ids("no markers here") == []


def test_analyze_evidence_outdated_and_conflict():
    old = mk("D-OLD", status="superseded")
    a, b = mk("D-A", excerpt="It is 30 days."), mk("D-B", excerpt="It is 60 days.")
    res = citations.analyze_evidence([old, a, b])
    assert [h.doc_id for h in res.outdated] == ["D-OLD"]
    assert len(res.conflicts) == 1 and {h.doc_id for h in res.conflicts[0].hits} == {"D-A", "D-B"}


def test_same_numbers_is_not_a_conflict():
    res = citations.analyze_evidence([mk("D-A", excerpt="It is 30 days."), mk("D-B", excerpt="Still 30 days.")])
    assert res.conflicts == []


@pytest.mark.parametrize("raw, reason", [
    ("The period is 5 days.", "uncited_answer"),
    ("The period is 5 days. [OTHER-001]", "citation_outside_evidence"),
    ("The period is 9 days. [D-A]", "unsupported_number"),
    ("The period is 5 days. CANARY-ZZ-1 [D-A]", "output_guard_blocked"),
])
def test_verify_answer_rejections(raw, reason):
    v = citations.verify_answer(raw, [mk("D-A")], Settings())
    assert not v.ok and v.reason == reason


def test_verify_answer_accepts_and_rebuilds_citation():
    v = citations.verify_answer("The period is 5 days. [D-A]", [mk("D-A")], Settings())
    assert v.ok and v.citations[0].source_id == "D-A" and v.citations[0].excerpt == "Value is 5 days."


# ------------------------------------------------------------------ API behaviour
def test_answer_is_cited_and_verifiable(client, H, seed_store):
    r = client.post("/ask", json=PROB, headers=H["a_employee"]).json()
    assert r["status"] == "answered" and r["citations"]
    top = r["citations"][0]
    assert top["source_id"] == "A-KB-002" and "6 months" in top["excerpt"]
    for c in r["citations"]:
        assert c["excerpt"] in seed_store.get("tenant_a", c["source_id"]).clean_text
    assert "[A-KB-002]" in r["answer"]


def test_superseded_document_is_reported_not_used(client, H, fake_llm):
    r = client.post("/ask", json=PROB, headers=H["a_employee"]).json()
    assert "A-KB-001" in {c["source_id"] for c in r["outdated_sources"]}
    assert "A-KB-001" not in {c["source_id"] for c in r["citations"]}
    assert "A-KB-001" not in fake_llm.calls[0][1]


def test_conflicting_sources_are_surfaced_without_calling_model(client, H, fake_llm):
    r = client.post("/ask", json=NOTICE, headers=H["a_employee"]).json()
    assert r["status"] == "conflict"
    assert set(r["conflicts"][0]["source_ids"]) == {"A-KB-003", "A-TKT-101"}
    assert {c["source_id"] for c in r["citations"]} == {"A-KB-003", "A-TKT-101"}
    assert fake_llm.calls == []


def test_weak_evidence_abstains_without_calling_model(client, H, fake_llm):
    r = client.post("/ask", json={"question": "What is the CEO favourite colour?"}, headers=H["a_employee"]).json()
    assert r["status"] == "abstained" and r["abstain_reason"] == "insufficient_evidence"
    assert r["citations"] == [] and fake_llm.calls == []


@pytest.mark.parametrize("reply, reason", [
    ("INSUFFICIENT_EVIDENCE", "model_declined"),
    ("The probation period is 6 months.", "uncited_answer"),
    ("The probation period is 6 months. [A-KB-777]", "citation_outside_evidence"),
    ("The probation period is 12 months. [A-KB-002]", "unsupported_number"),
])
def test_bad_model_replies_are_withheld(make_client, H, reply, reason):
    r = make_client(FakeLLM(response=reply)).post("/ask", json=PROB, headers=H["a_employee"]).json()
    assert r["status"] == "abstained" and r["abstain_reason"] == reason
    assert r["citations"] == []


def test_model_cannot_cite_a_superseded_source(make_client, H):
    llm = FakeLLM(response="The probation period is 3 months. [A-KB-001]")
    r = make_client(llm).post("/ask", json=PROB, headers=H["a_employee"]).json()
    assert r["status"] == "abstained" and r["abstain_reason"] == "citation_outside_evidence"
    assert r["citations"] == []
