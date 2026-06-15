"""Fast offline tests for the eval + retrain harness (UMB-122/124/126).

These run the MOCK harness end-to-end against the bundled fixtures — zero spend,
no network. The harness modules are imported by file path (like test_executor.py)
so importing them never triggers the package __init__ (which pulls in torch via
router.py); the harness itself is torch-free.

Run: ``pytest custom_routers/fusion_gate/tests/test_eval_harness.py`` or, with no
pytest installed, ``python custom_routers/fusion_gate/tests/test_eval_harness.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_EVAL_DIR = Path(_REPO_ROOT) / "custom_routers" / "fusion_gate" / "eval"
_FIXTURES = _EVAL_DIR / "fixtures"


def _ensure_namespace_packages() -> None:
    """Register lightweight package stubs so the eval modules' relative imports
    (``from .eval_harness import ...``) resolve WITHOUT executing
    ``custom_routers/fusion_gate/__init__.py`` — which imports torch via
    router.py. The harness is deliberately torch-free, so we route around it."""
    for pkg_name, pkg_dir in (
        ("custom_routers", Path(_REPO_ROOT) / "custom_routers"),
        ("custom_routers.fusion_gate", _EVAL_DIR.parent),
        ("custom_routers.fusion_gate.eval", _EVAL_DIR),
    ):
        if pkg_name not in sys.modules:
            mod = types.ModuleType(pkg_name)
            mod.__path__ = [str(pkg_dir)]  # mark as a package
            sys.modules[pkg_name] = mod


def _import_eval_module(name: str, filename: str):
    """Import an eval module by path under its package alias."""
    full = f"custom_routers.fusion_gate.eval.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, str(_EVAL_DIR / filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace_packages()
eval_harness = _import_eval_module("eval_harness", "eval_harness.py")
retrain = _import_eval_module("retrain", "retrain.py")

DATASET = str(_FIXTURES / "hard_slice.jsonl")
LLM = str(_FIXTURES / "llm_candidates.json")
ROUTING = str(_FIXTURES / "routing_data.jsonl")


def _build_harness(**overrides):
    kwargs = dict(
        dataset_path=DATASET, llm_path=LLM, routing_path=ROUTING,
        threshold=0.5, budget_threshold=0.3, k=2, judge=None, panel_preset="Quality",
    )
    kwargs.update(overrides)
    return eval_harness.build_mock_harness(**kwargs)


# --------------------------------------------------------- harness end-to-end


def test_harness_runs_all_three_arms_offline():
    harness, dataset, _ = _build_harness()
    arms = harness.run_all()

    assert set(arms) == {"always_route", "always_fuse", "fusion_gate"}
    for name, r in arms.items():
        assert r.n == len(dataset)
        assert 0.0 <= r.quality <= 1.0
        assert r.blended_cost >= 0.0
        assert 0.0 <= r.escalation_p <= 1.0

    # always_route never escalates; always_fuse always escalates.
    assert arms["always_route"].escalation_p == 0.0
    assert arms["always_fuse"].escalation_p == 1.0


def test_blended_cost_ordering():
    """Fusion costs more per query than a single model; gate sits in between or below fuse."""
    harness, _, _ = _build_harness()
    arms = harness.run_all()
    assert arms["always_route"].blended_cost < arms["always_fuse"].blended_cost
    assert arms["fusion_gate"].blended_cost <= arms["always_fuse"].blended_cost


def test_m1_m2_m3_verdicts_present_and_pass_on_fixtures():
    harness, dataset, dataset_path = _build_harness()
    arms = harness.run_all()
    verdicts = eval_harness.compute_verdicts(arms)

    # M1: gate quality >= 95% of always-fuse quality.
    assert verdicts["M1"]["pass"], verdicts["M1"]
    # M2: blended cost <= 1.6x always-route.
    assert verdicts["M2"]["pass"], verdicts["M2"]
    # M3: gate-precision is measured (escalated queries that beat best single).
    assert verdicts["M3"]["gate_precision"] is not None
    assert verdicts["M3"]["n_escalated"] >= 1


def test_gate_precision_counts_only_improvements():
    """M3 counts an escalation only when fusion is right AND best single is wrong."""
    harness, dataset, _ = _build_harness()
    arm = harness.run_fusion_gate()
    # On the fixtures, fusion is designed to be correct where the majority single
    # answer is wrong on at least some escalated queries.
    assert arm.n_escalated_improved >= 1
    assert arm.n_escalated_improved <= arm.n_escalated


def test_report_and_csv_are_written(tmp_path):
    rc = eval_harness.main([
        "--mock", "--dataset", DATASET, "--llm", LLM, "--routing", ROUTING,
        "--out", str(tmp_path), "--with-retrain",
    ])
    assert rc == 0
    csv_path = tmp_path / "results.csv"
    md_path = tmp_path / "results.md"
    assert csv_path.exists()
    assert md_path.exists()

    md = md_path.read_text(encoding="utf-8")
    # The report must explicitly flag mock provenance and the keyed live path.
    assert "MOCK fixtures" in md
    assert "keyed live run" in md.lower()
    assert "M1" in md and "M2" in md and "M3" in md
    # --with-retrain appends the retrain delta block.
    assert "Retrain" in md and "before vs after" in md

    # CSV has a header + three arm rows.
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 4
    assert rows[0].startswith("arm,")


def test_live_mode_is_blocked():
    """The offline harness refuses --live so a stray run cannot spend."""
    import pytest as _pytest  # only used when pytest is present

    with _pytest.raises(SystemExit):
        eval_harness.main(["--live", "--dataset", DATASET, "--llm", LLM])


# --------------------------------------------------------- retrain loop


def test_retrain_measures_m3_before_and_after_offline():
    dataset = eval_harness.load_jsonl(DATASET)
    llm_data = eval_harness.load_llm_candidates(LLM)
    base_routing = eval_harness.load_jsonl(ROUTING)

    log = retrain.synthesize_fusion_log(dataset, llm_data, base_routing, k=3, judge=None)
    assert log, "expected a non-empty synthesized fusion log"
    # Log entries are in fusion_log shape.
    assert all("responses" in e and "query" in e for e in log)

    result = retrain.run_retrain(dataset, llm_data, base_routing, log=log, k=3, judge=None)

    # Routing table is strictly augmented by the replayed responses.
    assert result["n_augmented_routing_rows"] > result["n_base_routing_rows"]
    # Both M3 measurements are produced and the delta is reported.
    assert result["m3_before"] is not None
    assert result["m3_after"] is not None
    assert result["m3_delta"] == result["m3_after"] - result["m3_before"]
    # Threshold is refit within bounds.
    assert 0.1 <= result["threshold_after"] <= 0.9


def test_retrain_block_renders_with_mock_flag():
    dataset = eval_harness.load_jsonl(DATASET)
    block = retrain.mock_retrain_report_block(
        dataset=dataset, llm_path=LLM, routing_path=ROUTING, k=3, judge=None
    )
    assert "Retrain" in block
    assert "MOCK fixtures" in block
    assert "M3 gate-precision" in block


# --------------------------------------------------------- manual runner


def _run_all_manually() -> int:
    """Run every test_* with no pytest (env without pytest installed)."""
    import tempfile

    failures = 0
    for fn_name, fn in sorted(globals().items()):
        if not fn_name.startswith("test_") or not callable(fn):
            continue
        try:
            if fn_name == "test_report_and_csv_are_written":
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            elif fn_name == "test_live_mode_is_blocked":
                # Reproduce the SystemExit assertion without pytest.
                try:
                    eval_harness.main(["--live", "--dataset", DATASET, "--llm", LLM])
                except SystemExit:
                    pass
                else:
                    raise AssertionError("--live should SystemExit")
            else:
                fn()
            print(f"PASS {fn_name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn_name}: {type(exc).__name__}: {exc}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all_manually() else 0)
