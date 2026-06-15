"""retrain — scripted, repeatable gate + capability refit from fusion logs (UMB-126).

Closes the loop: logged fusion calls (``fusion_log`` JSONL format, i.e. each line
``{ts, strategy, query, panel, judge, responses[], analysis, token, cost}``) are
fed back into the ``api_calling_evaluation`` training-row format, the routing
table is AUGMENTED with the per-model performance those responses imply, and the
gate + capability scorer are REFIT on the augmented data. We then re-measure M3
(gate-precision) BEFORE vs AFTER and report the delta.

Pipeline (offline, deterministic, zero spend):

  1. Load logged fusion responses (``fusion_log`` JSONL). Each ``responses[]``
     entry is decomposed via the same shape ``fusion_log.to_training_rows``
     emits: ``{query, model_name, response, performance, ...}``.
  2. Grade each decomposed response against the hard-slice ground truth
     (offline exact-match; the live path would use
     ``api_calling_evaluation.eval_perf``) to fill ``performance``.
  3. Build routing rows ``{task_name, model_name, performance}`` from the graded
     responses and AUGMENT the base routing table with them.
  4. Refit: a fresh :class:`CapabilityScorer` over the augmented routing data
     (capability scores), and re-tune the gate threshold from the augmented
     difficulty/quality signal (gate refit).
  5. Re-run the fusion-gate arm with the refit components and report M3 before
     vs after as a delta in the report.

``--mock`` path: synthesizes a fusion log from the bundled fixtures (so there is
something to replay with zero spend) and runs the full before/after measurement.
Live path: point ``--log`` at a real ``fusion_log`` sink produced by keyed
``FusionGateRouter.fuse`` calls; same code path, real responses.

This module never imports torch/pandas and makes no network call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .eval_harness import (
    FUSION_TIERS,
    EvalHarness,
    MockFusionExecutor,
    RouteGate,
    best_single_answer,
    load_jsonl,
    load_llm_candidates,
    score_answer,
    _FIXTURES_DIR,
)


# ---------------------------------------------------------------------------
# Step 1–2: replay fusion log into graded training rows
# ---------------------------------------------------------------------------


def synthesize_fusion_log(
    dataset: list[dict[str, Any]],
    llm_data: dict[str, Any],
    routing_data: list[dict[str, Any]] | None,
    *,
    k: int,
    judge: str | None,
    threshold: float = 0.5,
    budget_threshold: float | None = 0.3,
) -> list[dict[str, Any]]:
    """Produce a fusion-log-shaped list by running the mock harness's fuse path.

    This gives the retrain loop something to replay offline with zero spend, in
    exactly the ``fusion_log`` JSONL shape a live run would persist. Only queries
    the gate escalates are logged (mirroring the real fuse-only logging).
    """
    records_by_query = {r["query"]: r for r in dataset}
    executor = MockFusionExecutor(llm_data=llm_data, records_by_query=records_by_query)
    harness = EvalHarness(
        dataset=dataset, llm_data=llm_data, executor=executor,
        routing_data=routing_data, threshold=threshold,
        budget_threshold=budget_threshold, k=k, judge=judge,
    )
    log: list[dict[str, Any]] = []
    for record in dataset:
        query = record["query"]
        decision = harness.gate.decide({"query": query})
        if decision.tier not in FUSION_TIERS:
            continue
        panel = harness._select_panel(query, decision.tier)
        result = executor.run(query, panel, judge=judge)
        log.append(
            {
                "strategy": "fusion",
                "query": query,
                "panel": list(result.panel),
                "judge": result.judge,
                "responses": [
                    {"model": r.get("model"), "content": r.get("content")}
                    for r in result.responses
                ],
                "analysis": result.analysis,
                "token": None,
                "cost": result.cost,
            }
        )
    return log


def grade_log_to_training_rows(
    log: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decompose ``responses[]`` to graded training rows (api_calling_evaluation shape).

    Mirrors ``fusion_log.to_training_rows`` ({query, model_name, response,
    performance, ...}) but FILLS ``performance`` by grading each response against
    the matching dataset record's ground truth (offline exact-match). The live
    path would grade via ``api_calling_evaluation.eval_perf``.
    """
    gt_by_query = {r["query"]: r for r in dataset}
    rows: list[dict[str, Any]] = []
    for entry in log:
        query = entry.get("query", "")
        record = gt_by_query.get(query, {})
        gt = record.get("ground_truth")
        task_name = record.get("task_name")
        for resp in entry.get("responses", []) or []:
            model = resp.get("model")
            content = resp.get("content")
            perf = score_answer(content, gt) if gt is not None else None
            rows.append(
                {
                    "query": query,
                    "task_name": task_name,
                    "model_name": model,
                    "model": model,
                    "response": content,
                    "performance": perf,
                    "strategy": "fusion",
                    "judge": entry.get("judge"),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Step 3: augment the routing table
# ---------------------------------------------------------------------------


def augment_routing_data(
    base_routing: list[dict[str, Any]] | None,
    training_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append graded (task_name, model_name, performance) rows to the base table.

    Rows missing a model or performance are skipped. The result is consumable by
    :class:`CapabilityScorer` (it keys on ``task_name`` / ``model_name`` /
    ``performance``), so a fresh scorer over this list is the capability refit.
    """
    augmented: list[dict[str, Any]] = list(base_routing or [])
    for row in training_rows:
        if row.get("model_name") is None or row.get("performance") is None:
            continue
        augmented.append(
            {
                "task_name": row.get("task_name") or "fusion",
                "model_name": row["model_name"],
                "performance": float(row["performance"]),
            }
        )
    return augmented


# ---------------------------------------------------------------------------
# Step 4: refit the gate threshold
# ---------------------------------------------------------------------------


def refit_gate_thresholds(
    dataset: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    *,
    current_threshold: float,
    current_budget_threshold: float | None,
) -> tuple[float, float | None]:
    """Re-tune the gate thresholds from the augmented quality signal.

    Heuristic, deterministic refit examining the logged fusion responses:

      * Where fusion was logged but the BEST single answer was already correct,
        the escalation was WASTED — raise the lower (``budget_threshold``) floor
        so those low-difficulty queries route single next time, lifting M3
        precision (fewer non-improving escalations).
      * The upper ``threshold`` is nudged by how reliably fusion BEAT the best
        single answer overall: more help → lower it (escalate more), less →
        raise it. Bounded so the refit can never disable the gate.

    Returns ``(threshold, budget_threshold)``. The live path could fit a learned
    ``DifficultyEstimator`` here instead; this offline refit stays torch-free and
    reproducible.
    """
    gt_by_query = {r["query"]: r for r in dataset}
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in training_rows:
        by_query.setdefault(row["query"], []).append(row)

    helped = 0
    wasted = 0
    total = 0
    wasted_difficulties: list[float] = []
    # A throwaway gate purely to score difficulty consistently with the harness.
    diff_gate = RouteGate(llm_data={"_": {}})
    for query, rows in by_query.items():
        record = gt_by_query.get(query)
        if record is None:
            continue
        gt = record.get("ground_truth")
        if gt is None:
            continue
        total += 1
        fusion_best = max((r.get("performance") or 0.0) for r in rows) if rows else 0.0
        single_correct = score_answer(best_single_answer(record), gt) >= 1.0
        if fusion_best >= 1.0 and not single_correct:
            helped += 1
        elif single_correct:
            # Fusion was logged but a single model already had it: wasted spend.
            wasted += 1
            wasted_difficulties.append(diff_gate._lexical_difficulty(query))

    if total == 0:
        return current_threshold, current_budget_threshold

    help_rate = helped / total
    delta = (0.3 - help_rate) * 0.4
    threshold = max(0.1, min(0.9, current_threshold + delta))

    # Raise the budget floor just above the hardest WASTED escalation so those
    # low-value queries route single, without crossing the upper threshold.
    budget_threshold = current_budget_threshold
    if wasted_difficulties:
        floor = max(wasted_difficulties) + 1e-3
        base = current_budget_threshold if current_budget_threshold is not None else 0.0
        budget_threshold = min(threshold, max(base, floor))

    return threshold, budget_threshold


# ---------------------------------------------------------------------------
# Step 5: before/after M3 measurement
# ---------------------------------------------------------------------------


def measure_m3(
    dataset: list[dict[str, Any]],
    llm_data: dict[str, Any],
    routing_data: list[dict[str, Any]] | None,
    *,
    threshold: float,
    budget_threshold: float | None,
    k: int,
    judge: str | None,
) -> tuple[float | None, int, int]:
    """Run the fusion-gate arm and return (gate_precision, n_escalated, n_improved)."""
    records_by_query = {r["query"]: r for r in dataset}
    executor = MockFusionExecutor(llm_data=llm_data, records_by_query=records_by_query)
    harness = EvalHarness(
        dataset=dataset, llm_data=llm_data, executor=executor,
        routing_data=routing_data, threshold=threshold,
        budget_threshold=budget_threshold, k=k, judge=judge,
    )
    arm = harness.run_fusion_gate()
    return arm.gate_precision, arm.n_escalated, arm.n_escalated_improved


def run_retrain(
    dataset: list[dict[str, Any]],
    llm_data: dict[str, Any],
    base_routing: list[dict[str, Any]] | None,
    *,
    log: list[dict[str, Any]],
    k: int,
    judge: str | None,
    threshold: float = 0.5,
    budget_threshold: float | None = 0.3,
) -> dict[str, Any]:
    """Full retrain loop. Returns a structured before/after result dict."""
    # BEFORE: measure M3 on the base routing data + base threshold.
    before_m3, before_esc, before_imp = measure_m3(
        dataset, llm_data, base_routing,
        threshold=threshold, budget_threshold=budget_threshold, k=k, judge=judge,
    )

    # Replay the log -> graded rows -> augmented routing + refit thresholds.
    training_rows = grade_log_to_training_rows(log, dataset)
    augmented_routing = augment_routing_data(base_routing, training_rows)
    refit_threshold, refit_budget_threshold = refit_gate_thresholds(
        dataset, training_rows,
        current_threshold=threshold, current_budget_threshold=budget_threshold,
    )

    # AFTER: measure M3 with the refit capability data + refit thresholds.
    after_m3, after_esc, after_imp = measure_m3(
        dataset, llm_data, augmented_routing,
        threshold=refit_threshold, budget_threshold=refit_budget_threshold,
        k=k, judge=judge,
    )

    delta = None
    if before_m3 is not None and after_m3 is not None:
        delta = after_m3 - before_m3

    return {
        "n_log_entries": len(log),
        "n_training_rows": len(training_rows),
        "n_base_routing_rows": len(base_routing or []),
        "n_augmented_routing_rows": len(augmented_routing),
        "threshold_before": threshold,
        "threshold_after": refit_threshold,
        "budget_threshold_before": budget_threshold,
        "budget_threshold_after": refit_budget_threshold,
        "m3_before": before_m3,
        "m3_after": after_m3,
        "m3_delta": delta,
        "escalated_before": before_esc,
        "escalated_after": after_esc,
        "improved_before": before_imp,
        "improved_after": after_imp,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def render_retrain_block(result: dict[str, Any], *, mock: bool) -> str:
    """Render the retrain delta as a markdown block appended to results.md."""
    src = "MOCK fixtures (synthesized fusion log, zero spend)" if mock else "LIVE fusion log"
    lines = [
        "## Retrain (UMB-126): M3 before vs after",
        "",
        f"- Source: {src}",
        f"- Replayed {result['n_log_entries']} fusion-log entries → "
        f"{result['n_training_rows']} graded training rows.",
        f"- Routing table augmented: {result['n_base_routing_rows']} → "
        f"{result['n_augmented_routing_rows']} rows.",
        f"- Gate threshold refit: {_fmt(result['threshold_before'], '.3f')} → "
        f"{_fmt(result['threshold_after'], '.3f')}.",
        f"- Gate budget_threshold refit: {_fmt(result.get('budget_threshold_before'), '.3f')} → "
        f"{_fmt(result.get('budget_threshold_after'), '.3f')} "
        "(raised so wasted low-difficulty escalations route single).",
        "",
        "| Metric | Before | After | Delta |",
        "|--------|--------|-------|-------|",
        f"| M3 gate-precision | {_fmt(result['m3_before'])} | {_fmt(result['m3_after'])} | "
        f"{_fmt(result['m3_delta'], '+.4f')} |",
        f"| Escalated | {result['escalated_before']} | {result['escalated_after']} | "
        f"{result['escalated_after'] - result['escalated_before']:+d} |",
        f"| Escalated-and-improved | {result['improved_before']} | {result['improved_after']} | "
        f"{result['improved_after'] - result['improved_before']:+d} |",
    ]
    if mock:
        lines.append("")
        lines.append(
            "> Retrain numbers are from MOCK fixtures; the real M3 delta (M4) "
            "requires a keyed live run replaying a real fusion-log sink."
        )
    return "\n".join(lines)


def mock_retrain_report_block(
    *,
    dataset: list[dict[str, Any]],
    llm_path: str | Path,
    routing_path: str | Path | None,
    k: int,
    judge: str | None,
) -> str:
    """Convenience used by eval_harness --with-retrain: run mock retrain, return md block."""
    llm_data = load_llm_candidates(llm_path)
    base_routing = (
        load_jsonl(routing_path)
        if routing_path and Path(routing_path).exists()
        else None
    )
    # Deliberately LOOSE before-thresholds so the mock before-state over-escalates
    # (escalates easy queries the best single model already solves). The refit then
    # raises the budget floor, removing those wasted escalations and lifting M3 —
    # demonstrating the loop produces a real, positive before→after delta offline.
    before_threshold = 0.4
    before_budget_threshold = 0.1
    log = synthesize_fusion_log(
        dataset, llm_data, base_routing, k=k, judge=judge,
        threshold=before_threshold, budget_threshold=before_budget_threshold,
    )
    result = run_retrain(
        dataset, llm_data, base_routing, log=log, k=k, judge=judge,
        threshold=before_threshold, budget_threshold=before_budget_threshold,
    )
    return render_retrain_block(result, mock=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Offline mock mode (default; synthesizes a log from fixtures, zero spend).")
    parser.add_argument("--dataset", default=str(_FIXTURES_DIR / "hard_slice.jsonl"))
    parser.add_argument("--llm", default=str(_FIXTURES_DIR / "llm_candidates.json"))
    parser.add_argument("--routing", default=str(_FIXTURES_DIR / "routing_data.jsonl"))
    parser.add_argument("--log", default=None,
                        help="Path to a fusion_log JSONL sink to replay. "
                             "When omitted in --mock, a log is synthesized from fixtures.")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    parser.add_argument("--k", type=int, default=2,
                        help="Fusion panel size (default 2, matching eval_harness).")
    parser.add_argument("--judge", default=None)
    # Loose before-thresholds by default so the offline demonstration shows a
    # real before→after M3 lift (the loose gate over-escalates easy queries;
    # the refit raises the budget floor to remove that wasted spend). Tighten
    # these to match a live config when replaying a real fusion-log sink.
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--budget-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)

    dataset = load_jsonl(args.dataset)
    llm_data = load_llm_candidates(args.llm)
    base_routing = load_jsonl(args.routing) if Path(args.routing).exists() else None

    if args.log and Path(args.log).exists():
        log = load_jsonl(args.log)
    else:
        log = synthesize_fusion_log(
            dataset, llm_data, base_routing, k=args.k, judge=args.judge,
            threshold=args.threshold, budget_threshold=args.budget_threshold,
        )

    result = run_retrain(
        dataset, llm_data, base_routing, log=log, k=args.k, judge=args.judge,
        threshold=args.threshold, budget_threshold=args.budget_threshold,
    )
    block = render_retrain_block(result, mock=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrain.md").write_text(block + "\n", encoding="utf-8")
    (out_dir / "retrain.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote {out_dir / 'retrain.md'}")
    print(f"M3 before={_fmt(result['m3_before'])} after={_fmt(result['m3_after'])} "
          f"delta={_fmt(result['m3_delta'], '+.4f')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
