"""Central configuration. Values can be overridden with environment variables.

The API keys are DEMO keys for a synthetic dataset. In a real deployment the tenant and
role must come from a verified identity token (OIDC / JWT), never from the caller's body
or from any document text.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent

# api_key -> (tenant_id, role). DEMO ONLY.
DEFAULT_API_KEYS: Dict[str, Tuple[str, str]] = {
    "demo-key-a-admin": ("tenant_a", "hr_admin"),
    "demo-key-a-recruiter": ("tenant_a", "recruiter"),
    "demo-key-a-employee": ("tenant_a", "employee"),
    "demo-key-b-admin": ("tenant_b", "hr_admin"),
    "demo-key-b-recruiter": ("tenant_b", "recruiter"),
}

# Score multiplier by document type. Set here, in code; never by document text.
DEFAULT_TRUST_WEIGHTS: Dict[str, float] = {"kb": 1.0, "ticket": 0.9, "log": 0.85, "cv": 0.8}


@dataclass(frozen=True)
class Settings:
    # --- LLM (Ollama) ---
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "120")))
    llm_num_ctx: int = 2048
    llm_max_tokens: int = 200
    llm_temperature: float = 0.0
    # --- data ---
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(ROOT / "data"))))
    load_seed: bool = True
    # --- retrieval ---
    min_score: float = field(default_factory=lambda: float(os.getenv("MIN_SCORE", "0.35")))
    default_top_k: int = 4
    max_top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    max_tf: int = 3  # term-frequency cap: limits keyword stuffing
    passage_max_chars: int = 350
    max_excerpt_chars: int = 600
    # --- hybrid search (BM25 + embeddings) ---
    use_embeddings: bool = field(default_factory=lambda: os.getenv("USE_EMBEDDINGS", "1") == "1")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_weight: float = 0.5   # final = (1 - w) * keyword_score + w * semantic_score
    embed_floor: float = 0.2    # cosine at/below this counts as 0 (unrelated)
    embed_ceil: float = 0.8     # cosine at/above this counts as 1 (very similar)
    trust_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TRUST_WEIGHTS))
    # --- security ---
    api_keys: Dict[str, Tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_API_KEYS))
    canary_pattern: str = r"CANARY-[A-Z0-9\-]+"
