"""FastAPI entry point: /health, /ingest, /ask."""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, citations, retrieval, security
from .config import Settings
from .embeddings import load_embedder
from .llm import SENTINEL, SYSTEM_PROMPT, EvidenceItem, LLMClient, OllamaClient, build_user_prompt
from .schemas import (
    AppError, AskRequest, AskResponse, ConflictInfo, ErrorBody, ErrorResponse, HealthResponse,
    IngestRequest, IngestResponse,
)
from .security import Principal
from .store import DocumentStore

log = logging.getLogger("hr_rag")


def _error(status: int, code: str, message: str, details: Optional[list] = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


def _abstain(principal: Principal, reason: str, answer: str, *, model: Optional[str] = None,
             outdated=None, notes=None) -> AskResponse:
    return AskResponse(
        status="abstained", answer=answer, abstain_reason=reason, tenant_id=principal.tenant_id,
        role=principal.role, model=model, outdated_sources=outdated or [], security_notes=notes or [],
    )


def _security_notes(hits, store: DocumentStore, principal: Principal) -> List[str]:
    notes = []
    for h in hits:
        doc = store.get(principal.tenant_id, h.doc_id)
        if doc and doc.security_flags:
            notes.append(f"{h.doc_id}: untrusted text was sanitised before indexing ({', '.join(doc.security_flags)}).")
    return notes


def run_ask(state, principal: Principal, body: AskRequest) -> AskResponse:
    settings: Settings = state.settings
    store: DocumentStore = state.store
    llm: LLMClient = state.llm

    # 1) tenant + role filter in code, then retrieval over permitted documents only
    top_k = min(body.top_k or settings.default_top_k, settings.max_top_k)
    hits = retrieval.search(store.visible_to(principal), body.question, top_k=top_k, settings=settings,
                            embedder=state.embedder)
    # 2) fail-closed re-check of every hit
    security.assert_hits_permitted(principal, hits, store)
    hits = [h for h in hits if h.score >= settings.min_score]
    if not hits:
        return _abstain(principal, "insufficient_evidence",
                        "I could not find enough evidence in the documents you are allowed to access.")
    notes = _security_notes(hits, store, principal)

    # 3) outdated / conflicting evidence is decided in code
    analysis = citations.analyze_evidence(hits)
    outdated = [citations.to_citation(h) for h in analysis.outdated]
    if analysis.conflicts:
        first = analysis.conflicts[0]
        return AskResponse(
            status="conflict", answer=citations.conflict_answer(first),
            citations=[citations.to_citation(h) for h in first.hits], outdated_sources=outdated,
            conflicts=[ConflictInfo(topic=c.topic, source_ids=[h.doc_id for h in c.hits]) for c in analysis.conflicts],
            security_notes=notes, tenant_id=principal.tenant_id, role=principal.role,
        )
    if not analysis.current:
        return _abstain(principal, "only_outdated_evidence",
                        "Only outdated documents match this question; please check with an HR administrator.",
                        outdated=outdated, notes=notes)

    # 4) the model proposes an answer from sanitised evidence
    evidence = [EvidenceItem(id=h.doc_id, title=h.title, status=h.status, date=h.effective_date, text=h.excerpt)
                for h in analysis.current]
    raw = llm.generate(SYSTEM_PROMPT, build_user_prompt(body.question, evidence))

    print("MODEL RAW ANSWER:", repr(raw))
    print("AVAILABLE EVIDENCE IDS:", [item.id for item in evidence])

    verified = citations.verify_answer(raw, analysis.current, settings)
    if SENTINEL in raw:
        return _abstain(principal, "model_declined", "The available evidence does not answer this question.",
                        model=llm.model_name, outdated=outdated, notes=notes)

    # 5) code verifies the answer; citations are rebuilt from retrieval, never from model text
    verified = citations.verify_answer(raw, analysis.current, settings)
    if not verified.ok:
        log.warning("model answer rejected: %s", verified.reason)
        return _abstain(principal, verified.reason or "unverified_answer",
                        "The model's answer could not be verified against the sources, so it was withheld.",
                        model=llm.model_name, outdated=outdated, notes=notes)
    return AskResponse(
        status="answered", answer=verified.answer, citations=verified.citations, outdated_sources=outdated,
        security_notes=notes, tenant_id=principal.tenant_id, role=principal.role, model=llm.model_name,
    )


def create_app(settings: Optional[Settings] = None, llm: Optional[LLMClient] = None, embedder=None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="HR RAG Assistant", version=__version__)
    store = DocumentStore(settings.passage_max_chars)
    if settings.load_seed:
        store.load_directory(settings.data_dir)
    if embedder is None and settings.use_embeddings:
        embedder = load_embedder(settings.embedding_model)  # None if unavailable -> keyword-only
    app.state.settings = settings
    app.state.store = store
    app.state.embedder = embedder
    app.state.llm = llm or OllamaClient(
        settings.ollama_url, settings.ollama_model, timeout_s=settings.llm_timeout_s,
        num_ctx=settings.llm_num_ctx, max_tokens=settings.llm_max_tokens, temperature=settings.llm_temperature,
    )

    def get_principal(request: Request, x_api_key: Optional[str] = Header(default=None)) -> Principal:
        return security.authenticate(x_api_key, request.app.state.settings)

    # ------------------------------------------------------------------ error handlers
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"loc": [str(x) for x in e.get("loc", [])], "msg": str(e.get("msg", "")), "type": e.get("type", "")}
                   for e in exc.errors()]
        return _error(422, "validation_error", "The request did not pass validation.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return _error(500, "internal_error", "An internal error occurred.")

    # ------------------------------------------------------------------ routes
    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, deep: bool = False):
        st = request.app.state
        return HealthResponse(
            status="ok", version=__version__, model=st.llm.model_name, documents=st.store.count(),
            quarantined=len(st.store.quarantined), llm_reachable=st.llm.ping() if deep else None,
            embeddings=st.embedder is not None,
        )

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(body: IngestRequest, request: Request, principal: Principal = Depends(get_principal)):
        security.require_role(principal, "hr_admin")
        st = request.app.state
        doc = st.store.build_document(tenant_id=principal.tenant_id, req=body)  # tenant from the key
        st.store.add(doc, overwrite=body.overwrite)
        return IngestResponse(doc_id=doc.doc_id, tenant_id=doc.tenant_id, passages=len(doc.passages),
                              security_flags=doc.security_flags)

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request, principal: Principal = Depends(get_principal)):
        return run_ask(request.app.state, principal, body)

    return app


app = create_app()
