"""Fusion log sink — structured JSONL logging for fusion calls (UMB-125).

The fusion path produces a panel of model responses plus a judge synthesis. That
output is the training signal for FusionFactory-style routing data: each panel
member is a (query, model, response, performance) observation. This module turns
a :class:`~custom_routers.fusion_gate.executor.FusionResult` into two things:

  * ``log_fusion`` — one append-only structured JSONL line per fusion call,
    capturing the decision context (query, panel, judge, raw responses,
    analysis, token, cost) for audit and offline replay.
  * ``to_training_rows`` — per-model rows decomposed from ``responses[]``, shaped
    to be consumed by ``llmrouter/data/api_calling_evaluation.py`` (which keys on
    ``query`` / ``model_name`` / ``response`` / ``performance``).

Secrets hygiene: this sink NEVER serializes the untouched provider payload
(``FusionResult.raw``) and NEVER writes API keys, auth headers, or PII. Only the
explicitly enumerated fields below are emitted; everything else is dropped.

Default sink path mirrors the OpenClaw memory bank:
``~/.llmrouter/openclaw_memory.jsonl`` (override via ``sink_path``).

See: fusion-gate-router-prd-v0.2.0.md, openclaw_router/memory.py,
llmrouter/data/api_calling_evaluation.py.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .executor import FusionResult

# Default JSONL sink, shared with the OpenClaw memory bank layout.
DEFAULT_SINK_PATH = str(Path.home() / ".llmrouter" / "openclaw_memory.jsonl")

# Exact (case-insensitive) key names that mark a mapping entry as
# credential-bearing. Any matching key is dropped before serialization, at any
# nesting depth.
#
# Exact-match (not substring) is deliberate: substring matching on "token" /
# "auth" / "session" silently drops legitimate fields like ``prompt_tokens``,
# ``completion_tokens``, ``author``, ``authentication_method``, and
# ``session_id`` that may appear in usage/tracing metadata or multi-turn
# response structures. Actual inline credentials in free text are caught by
# ``_INLINE_SECRET_RE`` instead, which is the right tool for that job.
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "secret",
        "password",
        "passwd",
        "credential",
        "cookie",
    }
)

# Inline credential shapes to scrub from free text (e.g. accidental leakage in a
# model response). Conservative: redact obvious key formats, not arbitrary text.
_INLINE_SECRET_RE = re.compile(
    r"\b(sk-[A-Za-z0-9_\-]{12,}|Bearer\s+[A-Za-z0-9._\-]{12,})",
    re.IGNORECASE,
)

_REDACTED = "[REDACTED]"


def _is_secret_key(key: str) -> bool:
    """Return True when a mapping key is a known credential-bearing key name."""
    return key.lower() in _SECRET_KEYS


def _scrub(value: Any) -> Any:
    """Recursively drop secret-keyed entries and redact inline credentials.

    Mappings: keys whose name is in :data:`_SECRET_KEYS` are removed entirely.
    Strings: inline key/bearer shapes are replaced with ``[REDACTED]``.
    Other scalars and containers are walked structurally.
    """
    if isinstance(value, dict):
        return {
            str(k): _scrub(v)
            for k, v in value.items()
            if not _is_secret_key(str(k))
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        return _INLINE_SECRET_RE.sub(_REDACTED, value)
    return value


def _scrub_response(resp: dict[str, Any]) -> dict[str, Any]:
    """Normalize one panel response to ``{"model", "content"}``, scrubbed.

    Tolerates the executor's ``{"model", "content"}`` shape while dropping any
    extra credential-bearing fields a provider payload might carry.
    """
    safe = _scrub(resp) if isinstance(resp, dict) else {}
    return {
        "model": safe.get("model"),
        "content": safe.get("content"),
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(sink_path: str | None) -> Path:
    """Resolve the sink path, expanding ``~`` and environment variables."""
    raw = (sink_path or "").strip() or DEFAULT_SINK_PATH
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def log_fusion(
    result: FusionResult,
    query: str,
    sink_path: str | None = None,
    token: int | None = None,
    cost: float | None = None,
) -> Path:
    """Append one structured JSONL entry describing a fusion call.

    Args:
        result: Parsed fusion output (panel responses, analysis, judge, cost).
        query: The user query that triggered the fusion call.
        sink_path: Target JSONL file. Defaults to
            ``~/.llmrouter/openclaw_memory.jsonl``. ``~`` / env vars are expanded.
        token: Total token count for the call, when known. Falls back to None.
        cost: Total cost for the call. Falls back to ``result.cost`` when None.

    Returns:
        The resolved :class:`~pathlib.Path` the entry was appended to.

    Notes:
        The provider's raw payload (``result.raw``) is intentionally NOT written.
        All emitted fields are scrubbed for credential-bearing keys and inline
        secret shapes; no API keys, auth headers, or PII are persisted.
    """
    path = _resolve_path(sink_path)

    record = {
        "ts": _utc_now_iso(),
        "strategy": "fusion",
        "query": query,
        "panel": list(result.panel),
        "judge": result.judge,
        "responses": [_scrub_response(r) for r in result.responses],
        "analysis": _scrub(result.analysis) if result.analysis is not None else None,
        "token": token,
        "cost": cost if cost is not None else result.cost,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def to_training_rows(result: FusionResult, query: str) -> list[dict[str, Any]]:
    """Decompose ``responses[]`` into per-model FusionFactory training rows.

    Each panel response becomes one row keyed to match the schema produced by
    ``llmrouter/data/api_calling_evaluation.py``: ``query`` / ``model_name`` /
    ``response`` / ``performance``. The ``model`` alias is included alongside
    ``model_name`` so the rows also satisfy the OpenClaw memory layout, which
    keys on ``query`` / ``model``.

    Args:
        result: Parsed fusion output containing the panel ``responses[]``.
        query: The user query that produced the responses.

    Returns:
        One dict per panel response. ``performance`` defaults to ``None`` because
        fusion responses are not graded at log time; an offline evaluator fills
        it in. Content is scrubbed of inline secrets.

    Notes:
        No API keys, auth headers, or PII are emitted.
    """
    rows: list[dict[str, Any]] = []
    for resp in result.responses:
        safe = _scrub_response(resp)
        model = safe.get("model")
        rows.append(
            {
                "query": query,
                "model_name": model,
                "model": model,
                "response": safe.get("content"),
                "performance": None,
                "strategy": "fusion",
                "judge": result.judge,
            }
        )
    return rows
