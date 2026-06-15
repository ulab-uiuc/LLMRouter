"""Offline unit tests for FusionExecutor.run (UMB-120).

All tests mock the HTTP layer — no live network/API calls are made. The HTTP
seam is patched at ``requests.post`` (the executor prefers ``requests`` when it
imports successfully), so the request body can be inspected and the response
faked.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

import pytest

# Import the executor module directly by file path so these tests stay offline
# and free of the package __init__ (which imports torch via router.py). The
# executor itself has no torch dependency. The module is registered in
# sys.modules before execution so its dataclasses resolve field types.
_EXECUTOR_PATH = os.path.join(os.path.dirname(__file__), "..", "executor.py")
_spec = importlib.util.spec_from_file_location("fusion_gate_executor", _EXECUTOR_PATH)
assert _spec is not None and _spec.loader is not None
_executor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _executor
_spec.loader.exec_module(_executor)

CostCeilingExceeded = _executor.CostCeilingExceeded
FusionExecutor = _executor.FusionExecutor
FusionResult = _executor.FusionResult

PANEL = ["model-a", "model-b", "model-c"]
JUDGE = "judge-model"

# Per-model unit prices mirroring default_llm.json's input_price/output_price.
LLM_DATA: dict[str, dict[str, Any]] = {
    "model-a": {"input_price": 0.20, "output_price": 0.20},
    "model-b": {"input_price": 0.60, "output_price": 0.60},
    "model-c": {"input_price": 0.90, "output_price": 0.90},
    "judge-model": {"input_price": 1.20, "output_price": 1.20},
}

API_KEYS = {"OpenRouter": "sk-test-key"}


class _FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - mirror requests API
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _make_executor(cost_ceiling: float | None = None) -> FusionExecutor:
    return FusionExecutor(
        llm_data=LLM_DATA,
        judge=JUDGE,
        cost_ceiling=cost_ceiling,
    )


def _patch_post(monkeypatch, payload: dict[str, Any], captured: dict[str, Any]):
    """Patch requests.post to capture the body and return ``payload``."""

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):  # noqa: A002
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        captured["timeout"] = timeout
        return _FakeResponse(payload)

    import requests

    monkeypatch.setattr(requests, "post", fake_post)


def test_happy_path_parses_responses_and_analysis(monkeypatch):
    payload = {
        "status": "ok",
        "answer": "Fused answer.",
        "analysis": {
            "consensus": "All models agree X.",
            "contradictions": ["b disagrees on Y"],
            "blind_spots": ["none flagged Z"],
        },
        "responses": [
            {"model": "model-a", "content": "answer from a"},
            {"model": "model-b", "content": "answer from b"},
            {"model": "model-c", "content": "answer from c"},
        ],
        "cost": 1.23,
    }
    captured: dict[str, Any] = {}
    _patch_post(monkeypatch, payload, captured)

    result = _make_executor().run("What is 2+2?", PANEL, api_keys=API_KEYS)

    assert isinstance(result, FusionResult)
    assert result.answer == "Fused answer."
    assert result.analysis == {
        "consensus": "All models agree X.",
        "contradictions": ["b disagrees on Y"],
        "blind_spots": ["none flagged Z"],
    }
    assert [r["model"] for r in result.responses] == PANEL
    assert result.responses[0]["content"] == "answer from a"
    assert result.panel == PANEL
    assert result.judge == JUDGE
    assert result.cost == 1.23
    assert result.raw == payload


def test_request_body_uses_required_tool_choice_and_panel(monkeypatch):
    payload = {
        "status": "ok",
        "answer": "ok",
        "analysis": {"consensus": "c", "contradictions": [], "blind_spots": []},
        "responses": [{"model": "model-a", "content": "x"}],
    }
    captured: dict[str, Any] = {}
    _patch_post(monkeypatch, payload, captured)

    _make_executor().run("q", PANEL, api_keys=API_KEYS)

    body = captured["body"]
    assert body["tool_choice"] == "required"
    assert body["messages"] == [{"role": "user", "content": "q"}]

    tool = body["tools"][0]
    assert tool["type"] == "openrouter:fusion"
    assert tool["parameters"]["analysis_models"] == PANEL
    assert tool["parameters"]["model"] == JUDGE

    # The Authorization header carries the key but the body never does.
    assert captured["headers"]["Authorization"] == "Bearer sk-test-key"
    assert "sk-test-key" not in json.dumps(body)


def test_judge_failure_falls_back_without_crashing(monkeypatch):
    # status "ok" but analysis omitted -> synthesize from responses[].
    payload = {
        "status": "ok",
        "responses": [
            {"model": "model-a", "content": "partial a"},
            {"model": "model-b", "content": "partial b"},
        ],
    }
    captured: dict[str, Any] = {}
    _patch_post(monkeypatch, payload, captured)

    result = _make_executor().run("q", PANEL, api_keys=API_KEYS)

    assert result.analysis is None
    assert result.answer == "partial a\n\npartial b"
    assert [r["model"] for r in result.responses] == ["model-a", "model-b"]


def test_project_cost_is_per_query_dollars():
    """project_cost returns an estimated per-query DOLLAR cost, not a unit-price proxy.

    Prices in LLM_DATA are per-million-token. For each member,
    (input_price*prompt_tokens + output_price*completion_tokens)/1e6, with
    prompt_tokens estimated from the query (max(1, len(query)//4)) and
    completion_tokens = est_completion_tokens (default 512).
    """
    executor = _make_executor()
    query = "x" * 400  # 400 chars -> ~100 prompt tokens
    projected = executor.project_cost(PANEL, JUDGE, query=query)

    prompt_toks = max(1, len(query) // 4)  # 100
    completion_toks = 512
    expected = 0.0
    for name in PANEL + [JUDGE]:
        info = LLM_DATA[name]
        expected += (info["input_price"] * prompt_toks + info["output_price"] * completion_toks) / 1e6

    assert projected == pytest.approx(expected)
    # Dollar-scale: a realistic per-query cost is well under a dollar here.
    assert 0.0 < projected < 0.01


def test_cost_ceiling_aborts_before_http_call(monkeypatch):
    # Sentinel post that fails the test if the network layer is reached.
    def boom(*args, **kwargs):
        raise AssertionError("HTTP call must not happen when cost ceiling exceeded")

    import requests

    monkeypatch.setattr(requests, "post", boom)

    # A realistic per-query DOLLAR projection (~$0.0015 for this panel+judge) must
    # trip a tight dollar ceiling. The ceiling is now interpreted as dollars/query.
    executor = _make_executor(cost_ceiling=0.0005)

    with pytest.raises(CostCeilingExceeded) as exc:
        executor.run("q", PANEL, judge=JUDGE, api_keys=API_KEYS)

    assert exc.value.ceiling == 0.0005
    assert exc.value.projected > 0.0005
    # Sanity: the projection is dollar-scale, not the old unit-price-sum proxy (~5.8).
    assert exc.value.projected < 0.01


def test_realistic_cost_ceiling_allows_when_under_cap(monkeypatch):
    """A realistic $0.05/query ceiling does NOT abort this cheap panel."""
    payload = {
        "status": "ok",
        "answer": "ok",
        "analysis": {"consensus": "c", "contradictions": [], "blind_spots": []},
        "responses": [{"model": "model-a", "content": "x"}],
    }
    captured: dict[str, Any] = {}
    _patch_post(monkeypatch, payload, captured)

    executor = _make_executor(cost_ceiling=0.05)
    result = executor.run("q", PANEL, judge=JUDGE, api_keys=API_KEYS)
    assert result.answer == "ok"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
