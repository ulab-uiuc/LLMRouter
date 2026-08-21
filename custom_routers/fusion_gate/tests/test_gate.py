"""Offline unit tests for ``RouteGate`` (UMB-119).

These tests run fully offline with no network, no torch, and no trained model:
the lexical fallback is exercised directly, and the injected-estimator path is
driven by a plain Python stub. Compatible with pytest (``pytest test_gate.py``)
and also runnable standalone (``python test_gate.py``) since pytest is not a
hard dependency of this repo.

Coverage:
  - single tier for an easy / high-confidence query
  - fusion tier for a hard / low-confidence query
  - high_stakes override forces fusion regardless of difficulty
  - the threshold is config-driven (passing a different threshold flips the tier)
  - the cheapest model is selected on the single path
  - the injected estimator overrides the lexical heuristic when an embedding is present
  - confidence rises with distance from the threshold
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Load gate.py by file path rather than importing the package. The package
# __init__ pulls in router.py, which depends on torch; the gate has no such
# dependency, so loading the module directly keeps these tests torch-free and
# fully offline. The module is registered in sys.modules before execution so
# dataclass field-annotation resolution (PEP 563) can find its namespace.
_GATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gate.py"))
_spec = importlib.util.spec_from_file_location("fusion_gate_gate", _GATE_PATH)
_gate_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _gate_mod
_spec.loader.exec_module(_gate_mod)

GateDecision = _gate_mod.GateDecision
RouteGate = _gate_mod.RouteGate

# Minimal candidate set mirroring default_llm.json shape (name -> prices).
LLM_DATA = {
    "cheap-7b": {"input_price": 0.20, "output_price": 0.20},
    "mid-49b": {"input_price": 0.90, "output_price": 0.90},
    "big-141b": {"input_price": 1.20, "output_price": 1.20},
}


def _gate(threshold: float = 0.5, estimator=None) -> RouteGate:
    return RouteGate(llm_data=LLM_DATA, threshold=threshold, estimator=estimator)


# --------------------------------------------------------------------- tiers


def test_easy_query_routes_single_high_confidence():
    """A short, simple question stays on the cheap single path with high confidence."""
    gate = _gate(threshold=0.5)
    decision = gate.decide({"query": "What is the capital of France?"})

    assert isinstance(decision, GateDecision)
    assert decision.tier == "single"
    assert decision.model_name == "cheap-7b"  # cheapest capable model
    assert decision.panel == []
    assert decision.difficulty < 0.5
    assert decision.confidence > 0.5  # clearly below threshold => confident


def test_hard_query_routes_fusion_low_confidence_near_threshold():
    """A long, multi-part code/math query escalates to fusion."""
    gate = _gate(threshold=0.5)
    query = (
        "Write a function to compute the integral of a matrix, then prove its "
        "complexity, and also debug this regex; how do these interact with the "
        "algorithm above and what is the derivative? " * 2
    )
    decision = gate.decide({"query": query})

    assert decision.tier == "fusion"
    assert decision.model_name is None  # single-only model_name; None for fusion
    assert decision.difficulty >= 0.5


def test_high_stakes_forces_fusion_even_for_easy_query():
    """high_stakes overrides difficulty and forces fusion at full confidence."""
    gate = _gate(threshold=0.5)
    decision = gate.decide({"query": "Hi", "high_stakes": True})

    assert decision.tier == "fusion"
    assert decision.model_name is None
    assert decision.confidence == 1.0  # caller override => fully confident in fusing


def test_threshold_is_config_driven():
    """The same query flips tier purely based on the injected threshold."""
    query = {"query": "Explain how a hash map handles collisions and resizing."}

    # Difficulty for this query is some fixed value d in (0, 1). Compute it once.
    d = _gate()._lexical_difficulty(query["query"])
    assert 0.0 < d < 1.0  # guard: must straddle for the test to be meaningful

    lenient = _gate(threshold=d + 0.1).decide(query)  # threshold above d => single
    strict = _gate(threshold=d - 0.1).decide(query)    # threshold below d => fusion

    assert lenient.tier == "single"
    assert strict.tier == "fusion"


# ------------------------------------------------------------ model selection


def test_single_path_selects_cheapest_model():
    """Single path always returns the lowest-cost candidate."""
    gate = _gate(threshold=0.99)  # force single for almost anything
    decision = gate.decide({"query": "easy"})
    assert decision.tier == "single"
    assert decision.model_name == "cheap-7b"


# ------------------------------------------------------------- estimator path


def test_injected_estimator_overrides_lexical_heuristic():
    """When an embedding + estimator are provided, the estimator decides difficulty."""
    # Estimator ignores the embedding and returns a fixed hard score.
    hard_estimator = lambda _embedding: 0.95  # noqa: E731
    gate = _gate(threshold=0.5, estimator=hard_estimator)

    # Query text alone would be "easy"; the estimator should override it.
    decision = gate.decide({"query": "easy", "embedding": [0.0, 0.0, 0.0]})
    assert decision.tier == "fusion"
    assert abs(decision.difficulty - 0.95) < 1e-9


def test_estimator_output_with_item_is_coerced():
    """A tensor-like return (exposing .item()) is coerced to a float scalar."""

    class _ScalarLike:
        def __init__(self, value: float):
            self._value = value

        def item(self) -> float:
            return self._value

    gate = _gate(threshold=0.5, estimator=lambda _e: _ScalarLike(0.10))
    decision = gate.decide({"query": "anything", "embedding": [1.0]})
    assert decision.tier == "single"
    assert abs(decision.difficulty - 0.10) < 1e-9


def test_estimator_ignored_without_embedding():
    """No embedding => lexical fallback runs even if an estimator is wired in."""
    gate = _gate(threshold=0.5, estimator=lambda _e: 0.99)
    decision = gate.decide({"query": "What is 2 plus 2?"})  # no embedding key
    # Falls back to lexical heuristic (easy) rather than the estimator's 0.99.
    assert decision.tier == "single"
    assert decision.difficulty < 0.5


# ----------------------------------------------------------------- confidence


def test_confidence_increases_with_distance_from_threshold():
    """Confidence is monotonic in the absolute margin to the threshold."""
    gate = _gate(threshold=0.5)
    near = gate._confidence(0.5)   # at the boundary
    mid = gate._confidence(0.7)
    far = gate._confidence(1.0)
    assert near < mid < far
    assert near == 0.0
    assert far == 1.0


def test_lexical_difficulty_is_deterministic_and_bounded():
    """The heuristic is pure and always returns a value in [0, 1]."""
    gate = _gate()
    samples = ["", "hi", "Explain the algorithm.", "a" * 5000, "1. x? 2. y? and z?"]
    for text in samples:
        d = gate._lexical_difficulty(text)
        assert 0.0 <= d <= 1.0
        assert gate._lexical_difficulty(text) == d  # deterministic


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
