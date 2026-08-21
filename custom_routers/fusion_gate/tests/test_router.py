"""Offline integration tests for ``FusionGateRouter`` (UMB-121/123/124).

These exercise the router end-to-end through a temp YAML config and a tiny
in-memory candidate file. Fully offline:

  * no large data files — the temp config references only ``llm_data`` (no
    routing_data), so MetaRouter's DataLoader loads nothing heavy;
  * no network — the only fusion-execution test monkeypatches ``requests.post``;
  * ``--route-only`` / routing paths make NO HTTP call and spend nothing.

Coverage:
  - all six config keys are read and respected (threshold, k, judge,
    provider/base_url, panel_preset, cost_ceiling)
  - cost_ceiling downgrades fusion -> single (abort) with no spend
  - route_single is spend-free (the route-only contract): a decision dict, no
    executor.run invocation
  - panel varies by query type
  - all three tiers (single / budget_fusion / fusion) are reachable
  - fuse() logs every call via fusion_log, scrubbed and raw-payload-free
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Make the repo importable as a package root so ``custom_routers`` resolves.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# router.py imports torch eagerly (MetaRouter subclasses nn.Module). When torch
# is absent this module must SKIP cleanly rather than fail collection — otherwise
# it interrupts the whole suite and the four torch-free modules never run. See
# tests/conftest.py and fusion_gate/__init__.py for the torch-free design.
pytest.importorskip("torch")

from custom_routers.fusion_gate.router import FusionGateRouter  # noqa: E402

# Tiny candidate set (default_llm.json shape). Distinct prices/sizes so panel
# ordering and cost projection are deterministic.
LLM_DATA: dict[str, dict[str, Any]] = {
    "cheap-7b": {
        "size": "7B",
        "feature": "fast and efficient small model",
        "input_price": 0.20,
        "output_price": 0.20,
        "model": "vendor/cheap-7b",
        "service": "OpenRouter",
    },
    "mid-49b": {
        "size": "49B",
        "feature": "powerful high-accuracy model for complex tasks",
        "input_price": 0.90,
        "output_price": 0.90,
        "model": "vendor/mid-49b",
        "service": "OpenRouter",
    },
    "big-141b": {
        "size": "141B",
        "feature": "advanced large-scale model with exceptional performance",
        "input_price": 1.20,
        "output_price": 1.20,
        "model": "vendor/big-141b",
        "service": "OpenRouter",
    },
}


def _write_config(
    tmp_path: Path,
    *,
    threshold: float = 0.5,
    budget_threshold: float | None = 0.3,
    k: int = 2,
    judge: str | None = None,
    panel_preset: str = "Quality",
    cost_ceiling: float | None = None,
    log_sink_path: str | None = None,
) -> str:
    """Write a tiny llm_data JSON + a router YAML; return the YAML path."""
    llm_path = tmp_path / "llm.json"
    llm_path.write_text(json.dumps(LLM_DATA), encoding="utf-8")

    hparam_lines = [
        f"  threshold: {threshold}",
        f"  k: {k}",
        f"  judge: {('null' if judge is None else judge)}",
        f"  panel_preset: '{panel_preset}'",
        f"  cost_ceiling: {('null' if cost_ceiling is None else cost_ceiling)}",
        "  provider: 'OpenRouter'",
        "  base_url: 'https://openrouter.ai/api/v1'",
        f"  budget_threshold: {('null' if budget_threshold is None else budget_threshold)}",
    ]
    if log_sink_path is not None:
        hparam_lines.append(f"  log_sink_path: '{log_sink_path}'")

    yaml_text = (
        "data_path:\n"
        f"  llm_data: '{llm_path}'\n"
        "hparam:\n" + "\n".join(hparam_lines) + "\n"
        "api_endpoint: 'https://openrouter.ai/api/v1'\n"
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    return str(cfg_path)


def _router(tmp_path: Path, **kwargs) -> FusionGateRouter:
    return FusionGateRouter(_write_config(tmp_path, **kwargs))


# ----------------------------------------------------- config keys (UMB-121)


def test_all_six_config_keys_are_read(tmp_path):
    r = _router(
        tmp_path,
        threshold=0.42,
        k=3,
        judge="big-141b",
        panel_preset="Budget",
        cost_ceiling=7.5,
    )
    assert r.threshold == 0.42
    assert r.k == 3
    assert r.judge == "big-141b"
    assert r.panel_preset == "Budget"
    assert r.cost_ceiling == 7.5
    # provider/base_url pair (the 6th key).
    assert r.provider == "OpenRouter"
    assert r.base_url == "https://openrouter.ai/api/v1"
    # base_url is threaded into the executor's endpoint.
    assert r.executor.api_endpoint == "https://openrouter.ai/api/v1"
    # threshold/k/judge are threaded into the gate/executor.
    assert r.gate.threshold == 0.42
    assert r.executor.judge == "big-141b"
    assert r.executor.cost_ceiling == 7.5


# --------------------------------------------------- spend-free route-only


def test_route_single_is_spend_free(tmp_path, monkeypatch):
    """route_single never invokes the executor (no API call / no spend)."""
    r = _router(tmp_path, threshold=0.0)  # force fusion for any query

    def boom(*args, **kwargs):
        raise AssertionError("routing must not call the executor / network")

    monkeypatch.setattr(r.executor, "run", boom)

    decision = r.route_single({"query": "Solve this hard logic puzzle step by step"})
    assert decision["strategy"] == "fusion"
    assert "panel" in decision and decision["panel"]
    assert "judge" in decision
    # Carries a model_name label for CLI route_query compatibility, but no call.
    assert decision["model_name"] is not None


def test_route_only_decision_shape_for_single(tmp_path):
    r = _router(tmp_path, threshold=0.99, budget_threshold=0.98)  # force single
    decision = r.route_single({"query": "Hi"})
    assert decision["strategy"] == "single"
    assert decision["tier"] == "single"
    assert decision["model_name"] in LLM_DATA
    assert decision["predicted_llm"] == decision["model_name"]


# -------------------------------------------------- cost_ceiling (UMB-121)


def test_cost_ceiling_downgrades_fusion_to_single(tmp_path):
    """A realistic per-query DOLLAR ceiling aborts fusion -> single, no spend.

    The projected cost is now an estimated dollar amount (~$0.001/query for this
    k=2 panel), so a tight $0.0005 ceiling trips the guard. This guards against
    the regression where the dollar projection silently no-op'd a sub-$1 ceiling.
    """
    r = _router(tmp_path, threshold=0.0, k=2, cost_ceiling=0.0005)
    decision = r.route_single({"query": "Solve this hard logic puzzle step by step"})

    assert decision["strategy"] == "single"
    assert decision["tier"] == "single"
    assert decision["downgraded_from"] in ("budget_fusion", "fusion")
    assert decision["projected_cost"] > 0.0005
    # Dollar-scale projection, not the old unit-price-sum proxy.
    assert decision["projected_cost"] < 0.01
    assert decision["model_name"] in LLM_DATA  # cheapest single fallback


def test_cost_ceiling_allows_fusion_when_under_cap(tmp_path):
    # A realistic $0.05/query ceiling comfortably clears this cheap k=2 panel.
    r = _router(tmp_path, threshold=0.0, k=2, cost_ceiling=0.05)
    decision = r.route_single({"query": "Solve this hard logic puzzle step by step"})
    assert decision["strategy"] == "fusion"
    assert decision["projected_cost"] <= 0.05


# -------------------------------------------------- panel varies by query


def test_panel_varies_by_query_type(tmp_path):
    """The capability-scored panel changes with the query category (UMB-123).

    Inject per-category routing performance so cheap-7b is best at code and
    big-141b is best at reasoning; a code query and a reasoning query must then
    produce different panels.
    """
    r = _router(tmp_path, threshold=0.0, k=2)
    # Swap in a capability scorer backed by category-discriminating routing data.
    from custom_routers.fusion_gate.capability import CapabilityScorer

    routing_rows = [
        {"task_name": "humaneval-code", "model_name": "cheap-7b", "performance": 0.95},
        {"task_name": "humaneval-code", "model_name": "mid-49b", "performance": 0.30},
        {"task_name": "humaneval-code", "model_name": "big-141b", "performance": 0.20},
        {"task_name": "agentverse-logicgrid", "model_name": "cheap-7b", "performance": 0.10},
        {"task_name": "agentverse-logicgrid", "model_name": "mid-49b", "performance": 0.50},
        {"task_name": "agentverse-logicgrid", "model_name": "big-141b", "performance": 0.98},
    ]
    r.capability = CapabilityScorer(llm_data=LLM_DATA, routing_data=routing_rows)

    code_panel = r.route_single({"query": "Write a function to fix this bug in my code"})["panel"]
    reasoning_panel = r.route_single(
        {"query": "Solve this logic puzzle, reason step by step"}
    )["panel"]

    assert code_panel[0] == "cheap-7b"
    assert reasoning_panel[0] == "big-141b"
    assert code_panel != reasoning_panel


def test_preset_fallback_when_capability_unavailable(tmp_path):
    """When capability scoring yields no panel, the configured preset drives it."""
    quality = _router(tmp_path, threshold=0.0, k=2, panel_preset="Quality")
    budget = _router(tmp_path, threshold=0.0, k=2, panel_preset="Budget")
    # Simulate "capability data unavailable" by making select_panel return None;
    # the router must then fall back to the configured preset (resolved by price).
    quality.capability.select_panel = lambda query, k: None
    budget.capability.select_panel = lambda query, k: None

    q_panel = quality.route_single({"query": "general question"})["panel"]
    b_panel = budget.route_single({"query": "general question"})["panel"]

    assert q_panel[0] == "big-141b"  # Quality preset -> most capable first
    assert b_panel[0] == "cheap-7b"  # Budget preset -> cheapest first
    assert q_panel != b_panel


# --------------------------------------------------- three tiers (UMB-124)


def test_all_three_tiers_reachable(tmp_path):
    """single / budget_fusion / fusion are all reachable via the two thresholds."""
    r = _router(tmp_path, threshold=0.6, budget_threshold=0.2, k=2, cost_ceiling=None)

    # Easy query -> low difficulty -> single.
    easy = r.route_single({"query": "Hi"})
    assert easy["tier"] == "single"

    # Force exact difficulties through the gate to land in each band.
    r.gate.estimator = lambda _e: 0.4  # between budget_threshold and threshold
    mid = r.route_single({"query": "x", "embedding": [0.0]})
    assert mid["tier"] == "budget_fusion"

    r.gate.estimator = lambda _e: 0.9  # above threshold
    hard = r.route_single({"query": "x", "embedding": [0.0]})
    assert hard["tier"] == "fusion"


def test_budget_tier_uses_budget_preset_on_fallback(tmp_path):
    """Mid-difficulty (budget_fusion) falls back to the cheap Budget panel.

    Even with panel_preset='Quality', the budget tier's fallback preset is Budget
    so the cheap panel is used when capability data is unavailable (UMB-124).
    """
    r = _router(tmp_path, threshold=0.6, budget_threshold=0.2, k=2, panel_preset="Quality")
    r.capability.select_panel = lambda query, k: None  # force preset fallback
    r.gate.estimator = lambda _e: 0.4  # land in the budget_fusion band

    decision = r.route_single({"query": "x", "embedding": [0.0]})
    assert decision["tier"] == "budget_fusion"
    assert decision["panel"][0] == "cheap-7b"  # Budget preset -> cheapest first


# ----------------------------------------------- fuse() logs (UMB-125 wiring)


def test_fuse_logs_every_call_scrubbed(tmp_path, monkeypatch):
    sink = tmp_path / "fusion_log.jsonl"
    r = _router(tmp_path, threshold=0.0, k=2, log_sink_path=str(sink))

    payload = {
        "status": "ok",
        "answer": "Fused answer.",
        "analysis": {"consensus": "agree", "contradictions": [], "blind_spots": []},
        "responses": [
            {"model": "big-141b", "content": "leaked sk-abcdefghijklmnop secret"},
            {"model": "mid-49b", "content": "ok"},
        ],
        "cost": 2.0,
    }

    # Patch the executor's HTTP seam directly so the test is robust whether or
    # not `requests` is installed (the executor falls back to urllib otherwise).
    # No real network call is ever made.
    monkeypatch.setattr(
        r.executor, "_post_chat_completions", lambda body, api_key: payload
    )

    decision = r.route_single({"query": "Solve this hard logic puzzle step by step"})
    result = r.fuse(decision, api_keys={"OpenRouter": "sk-test-key"})

    assert result.answer == "Fused answer."
    assert sink.exists()

    lines = sink.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["strategy"] == "fusion"
    assert record["query"].startswith("Solve this hard logic puzzle")
    # Raw provider payload must NOT be persisted, and inline secrets scrubbed.
    assert "raw" not in record
    serialized = json.dumps(record)
    assert "sk-abcdefghijklmnop" not in serialized
    assert "sk-test-key" not in serialized


def test_fuse_returns_result_when_log_sink_unwritable(tmp_path, monkeypatch):
    """A logging failure must NOT destroy the already-computed FusionResult.

    The sink directory is made unwritable so ``log_fusion`` raises an OSError on
    write; ``fuse()`` must swallow it and still return the result.
    """
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    sink = locked_dir / "fusion_log.jsonl"
    r = _router(tmp_path, threshold=0.0, k=2, log_sink_path=str(sink))

    payload = {
        "status": "ok",
        "answer": "Fused answer.",
        "analysis": {"consensus": "agree", "contradictions": [], "blind_spots": []},
        "responses": [{"model": "big-141b", "content": "ok"}],
        "cost": 2.0,
    }
    monkeypatch.setattr(
        r.executor, "_post_chat_completions", lambda body, api_key: payload
    )

    # Revoke write/execute so creating the file under it fails with OSError.
    os.chmod(locked_dir, 0o500)
    try:
        decision = r.route_single({"query": "Solve this hard logic puzzle step by step"})
        result = r.fuse(decision, api_keys={"OpenRouter": "sk-test-key"})
    finally:
        # Restore perms so tmp_path cleanup can remove the tree.
        os.chmod(locked_dir, 0o700)

    # The result survives even though the log write failed.
    assert result.answer == "Fused answer."
    assert not sink.exists()


def test_fuse_rejects_non_fusion_result(tmp_path):
    r = _router(tmp_path, threshold=0.99, budget_threshold=0.98)
    decision = r.route_single({"query": "Hi"})
    assert decision["strategy"] == "single"
    with pytest.raises(ValueError):
        r.fuse(decision)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
