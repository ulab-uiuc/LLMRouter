"""Offline unit tests for ``CapabilityScorer`` (UMB-123).

Fully offline: no network, no torch, no trained model, and no large data files.
``capability.py`` is loaded directly by file path (like ``test_gate.py``) so the
package ``__init__`` — which pulls in ``router.py``/torch — is never imported.

Coverage:
  - panel membership VARIES by query type (code/math/reasoning vs general) when
    backed by per-category routing performance
  - top-k respected; k clamped against the candidate set
  - preset fallback (Quality vs Budget) resolves by price
  - ``select_panel`` returns None (-> preset fallback) when no capability data
    and llm_data carries no usable prior
  - task_name -> category bucketing
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

_CAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "capability.py"))
_spec = importlib.util.spec_from_file_location("fusion_gate_capability", _CAP_PATH)
assert _spec is not None and _spec.loader is not None
_cap_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cap_mod
_spec.loader.exec_module(_cap_mod)

CapabilityScorer = _cap_mod.CapabilityScorer

# Candidate set mirroring default_llm.json shape (size / feature / prices).
LLM_DATA: dict[str, dict[str, Any]] = {
    "cheap-7b": {
        "size": "7B",
        "feature": "fast and efficient small model",
        "input_price": 0.20,
        "output_price": 0.20,
    },
    "mid-49b": {
        "size": "49B",
        "feature": "powerful high-accuracy model for complex tasks",
        "input_price": 0.90,
        "output_price": 0.90,
    },
    "big-141b": {
        "size": "141B",
        "feature": "advanced large-scale model with exceptional performance",
        "input_price": 1.20,
        "output_price": 1.20,
    },
    "moe-45b": {
        "size": "45B",
        "feature": "mixture of experts optimized for creative generation",
        "input_price": 0.60,
        "output_price": 0.60,
    },
}

# Routing rows that make different models best at different categories so the
# panel is forced to vary by query type. cheap-7b dominates "code", big-141b
# dominates "reasoning"/"math".
ROUTING_ROWS = [
    {"task_name": "humaneval-code", "model_name": "cheap-7b", "performance": 0.95},
    {"task_name": "humaneval-code", "model_name": "mid-49b", "performance": 0.30},
    {"task_name": "humaneval-code", "model_name": "big-141b", "performance": 0.20},
    {"task_name": "humaneval-code", "model_name": "moe-45b", "performance": 0.40},
    {"task_name": "agentverse-logicgrid", "model_name": "cheap-7b", "performance": 0.10},
    {"task_name": "agentverse-logicgrid", "model_name": "mid-49b", "performance": 0.50},
    {"task_name": "agentverse-logicgrid", "model_name": "big-141b", "performance": 0.98},
    {"task_name": "agentverse-logicgrid", "model_name": "moe-45b", "performance": 0.40},
]


def _scorer(routing_data: Any = None) -> CapabilityScorer:
    return CapabilityScorer(llm_data=LLM_DATA, routing_data=routing_data or ROUTING_ROWS)


# ----------------------------------------------------------- query classification


def test_classify_query_categories():
    s = _scorer()
    assert s.classify_query("Write a python function to debug this code") == "code"
    assert s.classify_query("Compute the integral and prove the theorem") == "math"
    assert s.classify_query("Solve this logic puzzle step by step") == "reasoning"
    assert s.classify_query("What is the capital of France?") == "general"
    assert s.classify_query("") == "general"


# ------------------------------------------------------------- panel variation


def test_panel_varies_by_query_type():
    """A code query and a reasoning query must yield different panels."""
    s = _scorer()
    code_panel = s.select_panel("Write a function to fix this bug in my code", k=2)
    reasoning_panel = s.select_panel("Solve this logic puzzle, reason step by step", k=2)

    assert code_panel is not None and reasoning_panel is not None
    # cheap-7b is best at code; big-141b is best at reasoning.
    assert code_panel[0] == "cheap-7b"
    assert reasoning_panel[0] == "big-141b"
    assert code_panel != reasoning_panel


def test_top_k_respected_and_clamped():
    s = _scorer()
    assert len(s.select_panel("debug this code", k=2)) == 2
    # k larger than candidate count returns all candidates, not an error.
    full = s.select_panel("debug this code", k=99)
    assert len(full) == len(LLM_DATA)
    # k <= 0 -> None (preset fallback trigger).
    assert s.select_panel("debug this code", k=0) is None


# ------------------------------------------------------------------ fallback


def test_select_panel_returns_none_without_any_capability_signal():
    """No routing data AND no llm_data prior => None (preset fallback)."""
    s = CapabilityScorer(llm_data={}, routing_data=None)
    assert s.select_panel("anything", k=3) is None


def test_preset_panel_quality_vs_budget_by_price():
    s = _scorer()
    quality = s.preset_panel("Quality", k=2)
    budget = s.preset_panel("Budget", k=2)

    # Quality favors most-capable (highest price proxy) first.
    assert quality[0] == "big-141b"
    # Budget favors cheapest first.
    assert budget[0] == "cheap-7b"
    assert quality != budget


def test_static_prior_used_when_routing_data_absent():
    """Without routing data, scoring still differentiates via the llm_data prior."""
    s = CapabilityScorer(llm_data=LLM_DATA, routing_data=None)
    panel = s.select_panel("general knowledge question", k=2)
    assert panel is not None
    # Largest/most-capable model ranks first via the size/feature prior.
    assert panel[0] == "big-141b"


# ---------------------------------------------------------- task bucketing


def test_task_name_to_category_bucketing():
    s = _scorer()
    assert s._task_to_category("humaneval-code") == "code"
    assert s._task_to_category("agentverse-logicgrid") == "reasoning"
    assert s._task_to_category("gsm8k") == "math"
    assert s._task_to_category("trivia-qa") == "general"
    assert s._task_to_category(None) == "general"


def test_dataframe_like_routing_data_is_accepted():
    """A pandas-like object exposing to_dict(orient='records') is consumed."""

    class _FakeDF:
        def __init__(self, rows):
            self._rows = rows

        def to_dict(self, orient="records"):  # noqa: D401 - mirror pandas API
            assert orient == "records"
            return self._rows

    s = CapabilityScorer(llm_data=LLM_DATA, routing_data=_FakeDF(ROUTING_ROWS))
    code_panel = s.select_panel("debug this code", k=1)
    assert code_panel == ["cheap-7b"]


# ----------------------------------------------------------------- runner


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:  # pragma: no cover - reporting path
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - reporting path
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
