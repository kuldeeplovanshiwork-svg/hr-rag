"""Failure handling: LLM outages, bad replies, fail-closed behaviour, Ollama client mapping."""
import json

import httpx
import pytest

from app import retrieval
from app.llm import OllamaClient
from app.retrieval import Hit
from app.schemas import LLMBadResponseError, LLMTimeoutError, LLMUnavailableError
from tests.conftest import FakeLLM

Q = {"question": "What is the probation period for new hires?"}


@pytest.mark.parametrize("error, status, code", [
    (LLMUnavailableError("Cannot reach Ollama."), 503, "llm_unavailable"),
    (LLMTimeoutError("too slow"), 504, "llm_timeout"),
    (LLMBadResponseError("garbled"), 502, "llm_bad_response"),
])
def test_llm_failures_become_clear_api_errors(make_client, H, error, status, code):
    r = make_client(FakeLLM(error=error)).post("/ask", json=Q, headers=H["a_employee"])
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert "Traceback" not in r.text


def test_llm_not_needed_for_abstention(make_client, H):
    c = make_client(FakeLLM(error=LLMUnavailableError("down")))
    r = c.post("/ask", json={"question": "What is the CEO favourite colour?"}, headers=H["a_employee"])
    assert r.status_code == 200 and r.json()["status"] == "abstained"


def test_unexpected_error_does_not_leak_internals(make_client, H):
    c = make_client(FakeLLM(error=RuntimeError("secret internal detail")), raise_server_exceptions=False)
    r = c.post("/ask", json=Q, headers=H["a_employee"])
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in r.text


def test_isolation_violation_fails_closed(make_client, H, monkeypatch):
    """If retrieval ever returned another tenant's record, the request must abort, not answer."""
    foreign = Hit("tenant_b", "B-CV-001", "CV - Zara Quill", "cv", None, "active", None,
                  "Zara earns 91000 USD CANARY-TB-7731", 9.9)
    monkeypatch.setattr(retrieval, "search", lambda *a, **k: [foreign])
    llm = FakeLLM()
    r = make_client(llm).post("/ask", json=Q, headers=H["a_recruiter"])
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "isolation_violation"
    assert "Zara" not in r.text and "CANARY" not in r.text and "91000" not in r.text
    assert llm.calls == []


# ------------------------------------------------------------------ Ollama client mapping
def _client(handler):
    return OllamaClient("http://ollama.test", "qwen2.5:1.5b", timeout_s=5,
                        http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ollama_success_and_request_shape():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " hello [A-KB-002] "}})

    assert _client(handler).generate("sys", "usr") == "hello [A-KB-002]"
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "qwen2.5:1.5b" and seen["body"]["stream"] is False
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["options"]["temperature"] == 0.0


def test_ollama_connection_error_maps_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError):
        _client(handler).generate("s", "u")


def test_ollama_timeout_maps_to_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMTimeoutError):
        _client(handler).generate("s", "u")


def test_ollama_missing_model_maps_to_unavailable_with_hint():
    with pytest.raises(LLMUnavailableError, match="ollama pull"):
        _client(lambda r: httpx.Response(404, json={"error": "model not found"})).generate("s", "u")


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"unexpected": 1}),
    httpx.Response(200, json={"message": {"content": "   "}}),
])
def test_ollama_bad_responses_map_to_bad_response(response):
    with pytest.raises(LLMBadResponseError):
        _client(lambda r: response).generate("s", "u")


def test_ollama_ping():
    assert _client(lambda r: httpx.Response(200, json={"models": []})).ping() is True

    def down(request):
        raise httpx.ConnectError("refused")

    assert _client(down).ping() is False
