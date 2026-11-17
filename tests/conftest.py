"""Shared fixtures. No test needs Ollama: the LLM is replaced by small fakes."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.store import DocumentStore  # noqa: E402

_EVIDENCE_RE = re.compile(r'<evidence id="([^"]+)"[^>]*>\n(.*?)\n</evidence>', re.S)


class FakeLLM:
    """Records every prompt. Default behaviour is extractive: it quotes the first evidence
    block and cites it, which is what a well-behaved model would do."""

    model_name = "fake-llm"

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response(user) if callable(self.response) else self.response
        m = _EVIDENCE_RE.search(user)
        if not m:
            return "INSUFFICIENT_EVIDENCE"
        return f"{' '.join(m.group(2).split())} [{m.group(1)}]"

    def ping(self) -> bool:
        return True


@pytest.fixture
def settings() -> Settings:
    return Settings(use_embeddings=False)  # tests never download a model


@pytest.fixture
def seed_store(settings) -> DocumentStore:
    store = DocumentStore(settings.passage_max_chars)
    store.load_directory(settings.data_dir)
    return store


@pytest.fixture
def make_client(settings):
    def _make(llm=None, embedder=None, **client_kwargs):
        from fastapi.testclient import TestClient
        from app.main import create_app

        app = create_app(settings=settings, llm=llm or FakeLLM(), embedder=embedder)
        return TestClient(app, **client_kwargs)

    return _make


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(make_client, fake_llm):
    return make_client(fake_llm)


@pytest.fixture
def H():
    keys = {
        "a_admin": "demo-key-a-admin",
        "a_recruiter": "demo-key-a-recruiter",
        "a_employee": "demo-key-a-employee",
        "b_admin": "demo-key-b-admin",
        "b_recruiter": "demo-key-b-recruiter",
    }
    return {name: {"X-API-Key": key} for name, key in keys.items()}
