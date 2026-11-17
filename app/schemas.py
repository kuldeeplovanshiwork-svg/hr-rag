"""Pydantic request/response models and the API error hierarchy."""
from __future__ import annotations

import re
from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocType = Literal["cv", "ticket", "log", "kb"]
Role = Literal["employee", "recruiter", "hr_admin"]
DocStatus = Literal["active", "superseded"]

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _no_control_chars(value: str) -> str:
    if _CONTROL_CHARS.search(value):
        raise ValueError("must not contain control characters")
    return value


# ----------------------------------------------------------------------------- errors
class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, details: Optional[list] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class AuthenticationError(AppError):
    status_code, code = 401, "authentication_required"


class PermissionDeniedError(AppError):
    status_code, code = 403, "permission_denied"


class DuplicateDocumentError(AppError):
    status_code, code = 409, "duplicate_document"


class UnprocessableDocumentError(AppError):
    status_code, code = 422, "unprocessable_document"


class IsolationViolationError(AppError):
    status_code, code = 500, "isolation_violation"


class LLMUnavailableError(AppError):
    status_code, code = 503, "llm_unavailable"


class LLMTimeoutError(AppError):
    status_code, code = 504, "llm_timeout"


class LLMBadResponseError(AppError):
    status_code, code = 502, "llm_bad_response"


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[list] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ----------------------------------------------------------------------------- requests
class IngestRequest(BaseModel):
    """One document to index. The tenant is NOT in the body: it comes from the API key."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]{2,63}$")
    title: str = Field(min_length=1, max_length=200)
    doc_type: DocType
    text: str = Field(min_length=1, max_length=20000)
    topic: Optional[str] = Field(default=None, pattern=r"^[a-z0-9_]{2,64}$")
    status: DocStatus = "active"
    effective_date: Optional[date] = None
    min_role: Role = "recruiter"
    overwrite: bool = False

    @field_validator("title", "text")
    @classmethod
    def _clean(cls, v: str) -> str:
        return _no_control_chars(v)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def _question_ok(cls, v: str) -> str:
        _no_control_chars(v)
        if not re.search(r"[A-Za-z0-9]", v):
            raise ValueError("must contain letters or digits")
        return v


# ----------------------------------------------------------------------------- responses
class Citation(BaseModel):
    source_id: str
    title: str
    doc_type: str
    status: str
    effective_date: Optional[str] = None
    excerpt: str
    score: float


class ConflictInfo(BaseModel):
    topic: str
    source_ids: List[str]


class AskResponse(BaseModel):
    status: Literal["answered", "abstained", "conflict"]
    answer: str
    citations: List[Citation] = []
    outdated_sources: List[Citation] = []
    conflicts: List[ConflictInfo] = []
    abstain_reason: Optional[str] = None
    security_notes: List[str] = []
    tenant_id: str
    role: str
    model: Optional[str] = None


class IngestResponse(BaseModel):
    doc_id: str
    tenant_id: str
    status: Literal["indexed"] = "indexed"
    passages: int
    security_flags: List[str] = []


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    model: str
    documents: int
    quarantined: int
    llm_reachable: Optional[bool] = None
    embeddings: bool = False
