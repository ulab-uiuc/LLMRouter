"""eval_harness — three-arm route-vs-fuse evaluation (UMB-122, UMB-124).

Compares three strategies over a held-out HARD-query slice drawn from the
LLMRouter benchmark families (GSM8K / MATH / GPQA / MBPP):

  * ``always_route``  — baseline: every query takes the cheap single-model path
                        (the gate's cheapest-capable single pick). One model call.
  * ``always_fuse``   — every query takes the full Quality fusion panel.
  * ``fusion_gate``   — the FusionGateRouter decision: gate each query between the
                        single path and a fusion tier, fusing only the hard ones.

Per arm it captures:

  * quality score  — mean correctness of the chosen answer vs ground truth.
  * blended cost   — mean projected $ per query (single = one model; fusion =
                     Σ(panel)+judge, from the executor's ``project_cost``).
  * escalation rate ``p`` — fraction of queries the arm sent to a fusion tier.
  * gate-precision (M3, UMB-124) — among ESCALATED queries, the fraction whose
                     synthesized fusion answer beats the best single-model answer.

Metric targets reported against the baselines:

  * M1: fusion-gate quality >= 95% of always-fuse quality on the hard slice.
  * M2: fusion-gate blended cost <= 1.6x always-route blended cost.
  * M3: gate-precision (escalated-and-improved) — reported per UMB-124.

OFFLINE / ZERO-SPEND (``--mock``, the default): a deterministic stub executor
(:class:`MockFusionExecutor`) reads canned per-model answers from the bundled
fixture (``fixtures/hard_slice.jsonl``); NO network call is made and nothing is
spent. The harness composes the plugin's torch-free seams (``RouteGate``,
``CapabilityScorer``, ``FusionExecutor`` projection, ``fusion_log``) directly,
mirroring what ``FusionGateRouter`` wires internally — it never imports torch.

LIVE RUN (keyed, real spend — documented, not the default): construct the real
``FusionGateRouter`` from ``custom_routers/fusion_gate/config.yaml`` and call its
``route_single`` / ``fuse`` with ``OPENROUTER_API_KEY`` (or ``API_KEYS``) set,
over a real benchmark slice. See ``results.md`` and ``--help``. The live path is
intentionally NOT wired into this offline harness so a stray run cannot spend.

Usage (offline)::

    python -m custom_routers.fusion_gate.eval.eval_harness --mock \
        --out custom_routers/fusion_gate/eval/out

Outputs: ``<out>/results.csv`` (per-arm rows) and ``<out>/results.md`` (report).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# --- offline, torch-free imports -------------------------------------------
# Load the plugin's torch-free modules directly by file path so importing this
# harness never triggers the package __init__ (which imports torch via router.py).
_PLUGIN_DIR = Path(__file__).resolve().parents[1]
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_module(name: str, filename: str):
    """Load a sibling plugin module by file path (no package import side effects)."""
    path = _PLUGIN_DIR / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field types in the module resolve.
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_gate = _load_module("fusion_gate_eval_gate", "gate.py")
_capability = _load_module("fusion_gate_eval_capability", "capability.py")
_executor = _load_module("fusion_gate_eval_executor", "executor.py")

RouteGate = _gate.RouteGate
GateDecision = _gate.GateDecision
FUSION_TIERS = _gate.FUSION_TIERS
TIER_TO_PRESET = _gate.TIER_TO_PRESET
resolve_preset = _gate.resolve_preset
CapabilityScorer = _capability.CapabilityScorer
FusionExecutor = _executor.FusionExecutor
FusionResult = _executor.FusionResult


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts (skips blank lines)."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_llm_candidates(path: str | Path) -> dict[str, Any]:
    """Read the candidate-metadata JSON (default_llm.json shape)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Deterministic mock executor (zero spend)
# ---------------------------------------------------------------------------


class MockFusionExecutor:
    """Deterministic, offline stand-in for :class:`FusionExecutor`.

    Mirrors the real executor's ``run`` signature and returns a real
    :class:`FusionResult`, but instead of an OpenRouter HTTP call it synthesizes
    the panel ``responses[]`` and the fused ``answer`` from canned per-record
    fixture data — so the harness exercises the full route→fuse→log flow with
    ZERO spend and no network. Cost is taken from the real ``project_cost`` so
    the blended-cost metric stays faithful to the live cost model.

    The mock NEVER touches OpenRouter HTTP specifics; all such logic stays in
    ``executor.py`` per the plugin's beta-tool isolation rule. This class only
    fills ``FusionResult`` fields a live call would populate.
    """

    def __init__(self, llm_data: dict[str, Any], records_by_query: dict[str, dict[str, Any]]):
        self.llm_data = llm_data
        self._by_query = records_by_query
        # Reuse the real projector for faithful cost accounting (no network).
        self._projector = FusionExecutor(llm_data=llm_data)

    def project_cost(
        self,
        panel: list[str],
        judge: str | None,
        query: str | None = None,
        prompt_tokens: int | None = None,
    ) -> float:
        """Delegate to the real per-query dollar cost projection (Σ panel + judge)."""
        return self._projector.project_cost(
            panel, judge, query=query, prompt_tokens=prompt_tokens
        )

    def run(
        self,
        query: str,
        panel: list[str],
        judge: str | None = None,
        api_keys: dict[str, str] | None = None,
        **gen_kwargs: Any,
    ) -> FusionResult:
        """Synthesize a FusionResult from fixture data — no network, no spend."""
        record = self._by_query.get(query, {})
        single_answers: dict[str, str] = record.get("single_answers", {})
        responses = [
            {"model": name, "content": single_answers.get(name, "")}
            for name in panel
        ]
        # The fixture carries the judge's synthesized answer for hard queries.
        fused = record.get("fusion_answer", "")
        cost = self.project_cost(panel, judge, query=query)
        return FusionResult(
            answer=fused,
            analysis={"consensus": fused, "contradictions": [], "blind_spots": []},
            responses=responses,
            panel=list(panel),
            judge=judge,
            cost=cost,
            raw=None,
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def normalize_answer(answer: Any) -> str:
    """Normalize an answer for exact comparison (offline-safe, deterministic).

    Lowercased, stripped, with surrounding whitespace/punctuation removed. Kept
    intentionally simple: the bundled fixtures use clean canonical answers so a
    light normalization suffices for the mock metrics. The live path would defer
    to ``llmrouter/data/api_calling_evaluation.eval_perf`` for benchmark-grade
    scoring (GSM8K / MATH / code-exec).
    """
    text = str(answer).strip().lower()
    return text.strip(" .$\t\n")


def score_answer(prediction: Any, ground_truth: Any) -> float:
    """Binary correctness in {0.0, 1.0} via normalized exact match."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def is_hard_record(record: dict[str, Any]) -> bool:
    """True if a fixture record belongs to the HARD slice.

    The hard slice is the fixed, arm-independent set of records the harness uses
    for the M3 gate-precision metric, so M3 is computed over the SAME slice for
    every arm (apples-to-apples). A record is hard when its ``id`` carries the
    ``-hard-`` marker (e.g. ``gsm8k-hard-01``), with a ``difficulty == "hard"``
    field honored as an explicit override when present.
    """
    explicit = record.get("difficulty")
    if explicit is not None:
        return str(explicit).lower() == "hard"
    return "-hard-" in str(record.get("id", ""))


def best_single_answer(record: dict[str, Any]) -> str:
    """The best single-model answer for a record.

    Prefers the explicit ``single_best_answer`` field; otherwise picks the most
    common answer across ``single_answers`` (majority vote), ties broken by the
    answer that matches ground truth when present.
    """
    explicit = record.get("single_best_answer")
    if explicit is not None:
        return str(explicit)
    answers = list(record.get("single_answers", {}).values())
    if not answers:
        return ""
    gt = record.get("ground_truth")
    # Majority vote; prefer a correct answer on ties.
    counts: dict[str, int] = {}
    for a in answers:
        counts[str(a)] = counts.get(str(a), 0) + 1
    best = max(
        counts,
        key=lambda a: (counts[a], 1 if gt is not None and score_answer(a, gt) else 0),
    )
    return best


# ---------------------------------------------------------------------------
# Arm results
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """Aggregate metrics for one evaluation arm."""

    arm: str
    n: int = 0
    quality: float = 0.0          # mean correctness in [0, 1]
    blended_cost: float = 0.0     # mean projected $ per query
    escalation_p: float = 0.0     # fraction routed to a fusion tier
    gate_precision: float | None = None  # M3 (UMB-124): None when undefined
    n_escalated: int = 0
    n_escalated_improved: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Three-arm route-vs-fuse evaluator over a hard-query slice.

    Args:
        dataset: list of fixture records (see ``fixtures/hard_slice.jsonl``).
        llm_data: candidate-metadata mapping (default_llm.json shape).
        routing_data: optional per-model routing performance rows for the
            capability scorer (panel selection); list of dicts or None.
        threshold / budget_threshold / k / judge / panel_preset: gate + panel
            hyperparameters, mirroring the router's config keys.
        executor: a run-able executor exposing ``run`` and ``project_cost``. In
            mock mode this is a :class:`MockFusionExecutor`; a live run passes a
            keyed :class:`FusionExecutor`.
    """

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        llm_data: dict[str, Any],
        executor: Any,
        routing_data: list[dict[str, Any]] | None = None,
        threshold: float = 0.5,
        budget_threshold: float | None = 0.3,
        k: int = 3,
        judge: str | None = None,
        panel_preset: str = "Quality",
    ):
        self.dataset = dataset
        self.llm_data = llm_data
        self.executor = executor
        self.k = k
        self.judge = judge
        self.panel_preset = panel_preset
        self.gate = RouteGate(
            llm_data=llm_data,
            threshold=threshold,
            budget_threshold=budget_threshold,
        )
        self.capability = CapabilityScorer(llm_data=llm_data, routing_data=routing_data)

    # ----------------------------------------------------------- panel select

    def _select_panel(self, query: str, tier: str) -> list[str]:
        """Capability-scored top-k panel, preset fallback by tier (UMB-123/124)."""
        panel = self.capability.select_panel(query, self.k)
        if panel:
            return panel
        # Shared tier->preset resolution (gate.resolve_preset) so the harness and
        # FusionGateRouter._select_panel cannot diverge.
        preset = resolve_preset(tier, self.panel_preset)
        return self.capability.preset_panel(preset, self.k)

    def _quality_preset_panel(self) -> Callable[[str], list[str]]:
        """Panel selector for the always-fuse arm (always the Quality preset)."""

        def select(query: str) -> list[str]:
            panel = self.capability.select_panel(query, self.k)
            if panel:
                return panel
            return self.capability.preset_panel("Quality", self.k)

        return select

    # ----------------------------------------------------------- arms

    def _best_single_model(self, query: str) -> str:
        """The capability-best single model for a query (fair single-router pick).

        Mirrors what a good classic single-model router would choose: the
        top-1 capability-scored candidate for the query category, falling back to
        the Quality preset head, then the cheapest model. Used by the
        ``always_route`` baseline so it is a CAPABLE single-router, not a
        cheapest-only strawman.
        """
        top = self.capability.select_panel(query, 1)
        if top:
            return top[0]
        preset = self.capability.preset_panel("Quality", 1)
        if preset:
            return preset[0]
        return self.gate.cheapest_model()

    def run_always_route(self) -> ArmResult:
        """Baseline arm: every query → its capability-best single model (one call)."""
        res = ArmResult(arm="always_route", n=len(self.dataset))
        total_q = 0.0
        total_cost = 0.0
        for record in self.dataset:
            model = self._best_single_model(record["query"])
            # Single-model answer for that model; fall back to best single answer.
            ans = record.get("single_answers", {}).get(model)
            if ans is None:
                ans = best_single_answer(record)
            q = score_answer(ans, record.get("ground_truth"))
            cost = self.executor.project_cost([model], None, query=record["query"])
            total_q += q
            total_cost += cost
            res.per_query.append(
                {"id": record.get("id"), "arm": "always_route", "escalated": False,
                 "model": model, "answer": ans, "quality": q, "cost": cost}
            )
        res.quality = total_q / res.n if res.n else 0.0
        res.blended_cost = total_cost / res.n if res.n else 0.0
        res.escalation_p = 0.0
        return res

    def run_always_fuse(self) -> ArmResult:
        """Always-fuse arm: every query → full Quality fusion panel."""
        res = ArmResult(arm="always_fuse", n=len(self.dataset))
        select = self._quality_preset_panel()
        total_q = 0.0
        total_cost = 0.0
        for record in self.dataset:
            panel = select(record["query"])
            result = self.executor.run(record["query"], panel, judge=self.judge)
            q = score_answer(result.answer, record.get("ground_truth"))
            cost = result.cost if result.cost is not None else self.executor.project_cost(panel, self.judge, query=record["query"])
            total_q += q
            total_cost += cost
            res.per_query.append(
                {"id": record.get("id"), "arm": "always_fuse", "escalated": True,
                 "panel": panel, "answer": result.answer, "quality": q, "cost": cost}
            )
        res.quality = total_q / res.n if res.n else 0.0
        res.blended_cost = total_cost / res.n if res.n else 0.0
        res.escalation_p = 1.0
        # Every query escalates: gate-precision over all of them (M3).
        res.n_escalated, res.n_escalated_improved, res.gate_precision = self._gate_precision(
            res.per_query, escalated_only=True
        )
        return res

    def run_fusion_gate(self) -> ArmResult:
        """Fusion-gate arm: gate each query single-vs-fuse, fuse only the hard ones."""
        res = ArmResult(arm="fusion_gate", n=len(self.dataset))
        total_q = 0.0
        total_cost = 0.0
        escalated = 0
        for record in self.dataset:
            query = record["query"]
            decision: GateDecision = self.gate.decide({"query": query})
            if decision.tier not in FUSION_TIERS:
                # Single path.
                model = decision.model_name or self.gate.cheapest_model()
                ans = record.get("single_answers", {}).get(model)
                if ans is None:
                    ans = best_single_answer(record)
                q = score_answer(ans, record.get("ground_truth"))
                cost = self.executor.project_cost([model], None, query=query)
                total_q += q
                total_cost += cost
                res.per_query.append(
                    {"id": record.get("id"), "arm": "fusion_gate", "tier": decision.tier,
                     "escalated": False, "model": model, "answer": ans, "quality": q,
                     "cost": cost}
                )
                continue

            # Fusion path.
            escalated += 1
            panel = self._select_panel(query, decision.tier)
            result = self.executor.run(query, panel, judge=self.judge)
            q = score_answer(result.answer, record.get("ground_truth"))
            cost = result.cost if result.cost is not None else self.executor.project_cost(panel, self.judge, query=query)
            total_q += q
            total_cost += cost
            res.per_query.append(
                {"id": record.get("id"), "arm": "fusion_gate", "tier": decision.tier,
                 "escalated": True, "panel": panel, "answer": result.answer,
                 "quality": q, "cost": cost}
            )
        res.quality = total_q / res.n if res.n else 0.0
        res.blended_cost = total_cost / res.n if res.n else 0.0
        res.escalation_p = escalated / res.n if res.n else 0.0
        res.n_escalated, res.n_escalated_improved, res.gate_precision = self._gate_precision(
            res.per_query, escalated_only=True
        )
        return res

    # ----------------------------------------------------------- M3 metric

    def _gate_precision(
        self, per_query: list[dict[str, Any]], escalated_only: bool
    ) -> tuple[int, int, float | None]:
        """Gate-precision (M3, UMB-124) — computed over the fixed HARD slice.

        APPLES-TO-APPLES: M3 is scored over the SAME hard slice for every arm
        (records flagged by ``is_hard_record``), not over each arm's own
        escalation set. Without this, ``always_fuse`` (which "escalates" every
        query, easy + hard) and ``fusion_gate`` (which escalates only the hard
        ones) would compute M3 over different denominators and the numbers would
        not be comparable.

        Among the hard-slice queries an arm actually escalated, M3 is the
        fraction whose synthesized fusion answer BEATS the best single-model
        answer — i.e. the fusion answer is correct AND the best single answer is
        not. Returns ``(n_escalated, n_escalated_improved, precision)``;
        precision is ``None`` when the arm escalated no hard-slice query
        (undefined — e.g. ``always_route``, which makes no escalation decision).
        """
        by_id = {r.get("id"): r for r in self.dataset}
        n_esc = 0
        n_improved = 0
        for row in per_query:
            # Every arm now stamps an explicit ``escalated`` bool on each row, so
            # the M3 filter reads that field directly rather than inferring it
            # from the arm name (which coupled this logic to a string constant).
            if escalated_only and not row.get("escalated", False):
                continue
            record = by_id.get(row.get("id"))
            if record is None:
                continue
            # Restrict to the fixed hard slice so the denominator is identical
            # across arms.
            if not is_hard_record(record):
                continue
            n_esc += 1
            gt = record.get("ground_truth")
            fusion_correct = score_answer(row.get("answer"), gt) >= 1.0
            single_correct = score_answer(best_single_answer(record), gt) >= 1.0
            # "Beats the best single answer": fusion right where best single wrong.
            if fusion_correct and not single_correct:
                n_improved += 1
        precision = (n_improved / n_esc) if n_esc else None
        return n_esc, n_improved, precision

    # ----------------------------------------------------------- run all

    def run_all(self) -> dict[str, ArmResult]:
        """Run all three arms and return ``{arm_name: ArmResult}``."""
        return {
            "always_route": self.run_always_route(),
            "always_fuse": self.run_always_fuse(),
            "fusion_gate": self.run_fusion_gate(),
        }


# ---------------------------------------------------------------------------
# Metric verdicts (M1 / M2 / M3)
# ---------------------------------------------------------------------------


def compute_verdicts(arms: dict[str, ArmResult]) -> dict[str, Any]:
    """Compute the M1 / M2 / M3 pass-fail verdicts from arm results.

    M1: fusion_gate.quality >= 0.95 * always_fuse.quality.
    M2: fusion_gate.blended_cost <= 1.6 * always_route.blended_cost.
    M3: fusion_gate.gate_precision (escalated-and-improved) — reported; the
        target is informational (no hard threshold mandated by UMB-124 beyond
        "measured"), so the verdict reports the value and flags > 0.0.
    """
    route = arms["always_route"]
    fuse = arms["always_fuse"]
    gate = arms["fusion_gate"]

    m1_target = 0.95 * fuse.quality
    m1_pass = gate.quality >= m1_target
    m1_ratio = (gate.quality / fuse.quality) if fuse.quality > 0 else None

    m2_target = 1.6 * route.blended_cost
    m2_pass = gate.blended_cost <= m2_target
    m2_ratio = (gate.blended_cost / route.blended_cost) if route.blended_cost > 0 else None

    m3_value = gate.gate_precision
    m3_pass = (m3_value is not None) and (m3_value > 0.0)

    return {
        "M1": {
            "name": "gate quality >= 95% of always-fuse quality (hard slice)",
            "gate_quality": gate.quality,
            "always_fuse_quality": fuse.quality,
            "target": m1_target,
            "ratio": m1_ratio,
            "pass": m1_pass,
        },
        "M2": {
            "name": "blended cost <= 1.6x always-route",
            "gate_cost": gate.blended_cost,
            "always_route_cost": route.blended_cost,
            "target": m2_target,
            "ratio": m2_ratio,
            "pass": m2_pass,
        },
        "M3": {
            "name": "gate-precision: escalated answers that beat best single",
            "gate_precision": m3_value,
            "n_escalated": gate.n_escalated,
            "n_escalated_improved": gate.n_escalated_improved,
            "pass": m3_pass,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_results_csv(arms: dict[str, ArmResult], path: str | Path) -> None:
    """Write the per-arm summary CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["arm", "n", "quality", "blended_cost", "escalation_p",
             "gate_precision", "n_escalated", "n_escalated_improved"]
        )
        for arm in ("always_route", "always_fuse", "fusion_gate"):
            r = arms[arm]
            writer.writerow(
                [r.arm, r.n, f"{r.quality:.4f}", f"{r.blended_cost:.6f}",
                 f"{r.escalation_p:.4f}",
                 "" if r.gate_precision is None else f"{r.gate_precision:.4f}",
                 r.n_escalated, r.n_escalated_improved]
            )


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def render_results_md(
    arms: dict[str, ArmResult],
    verdicts: dict[str, Any],
    *,
    mock: bool,
    dataset_path: str,
    n: int,
    retrain_block: str | None = None,
) -> str:
    """Render the human-readable results.md report."""
    route = arms["always_route"]
    fuse = arms["always_fuse"]
    gate = arms["fusion_gate"]

    source = "MOCK fixtures (zero spend)" if mock else "LIVE keyed run"
    lines: list[str] = []
    lines.append("# FusionGateRouter — eval harness results")
    lines.append("")
    if mock:
        lines.append(
            "> **These numbers are from MOCK fixtures (deterministic stub executor, "
            "zero spend).** They validate the harness wiring and metric math, NOT "
            "real model quality. **Real M1–M4 numbers require a keyed live run** "
            "(`OPENROUTER_API_KEY` / `API_KEYS` set) against a real benchmark slice "
            "— see the *Live run* section below."
        )
    else:
        lines.append("> Numbers from a LIVE keyed run (real OpenRouter spend).")
    lines.append("")
    lines.append(f"- Source: {source}")
    lines.append(f"- Hard slice: `{dataset_path}` ({n} held-out queries; GSM8K / MATH / GPQA / MBPP)")
    lines.append("")
    lines.append("## Per-arm metrics")
    lines.append("")
    lines.append("| Arm | n | Quality | Blended cost | Escalation p | Gate-precision (M3) |")
    lines.append("|-----|---|---------|--------------|--------------|---------------------|")
    for r in (route, fuse, gate):
        lines.append(
            f"| {r.arm} | {r.n} | {_fmt(r.quality)} | {_fmt(r.blended_cost, '.6f')} | "
            f"{_fmt(r.escalation_p)} | {_fmt(r.gate_precision)} |"
        )
    lines.append("")
    lines.append("## Metric targets")
    lines.append("")
    m1 = verdicts["M1"]
    m2 = verdicts["M2"]
    m3 = verdicts["M3"]
    lines.append(
        f"- **M1** — {m1['name']}: gate quality {_fmt(m1['gate_quality'])} vs "
        f"target {_fmt(m1['target'])} (95% of always-fuse {_fmt(m1['always_fuse_quality'])}); "
        f"ratio {_fmt(m1['ratio'])} → **{'PASS' if m1['pass'] else 'FAIL'}**."
    )
    lines.append(
        f"- **M2** — {m2['name']}: gate cost {_fmt(m2['gate_cost'], '.6f')} vs "
        f"target {_fmt(m2['target'], '.6f')} (1.6x always-route {_fmt(m2['always_route_cost'], '.6f')}); "
        f"ratio {_fmt(m2['ratio'])} → **{'PASS' if m2['pass'] else 'FAIL'}**."
    )
    lines.append(
        f"- **M3** — {m3['name']}: gate-precision {_fmt(m3['gate_precision'])} "
        f"({m3['n_escalated_improved']}/{m3['n_escalated']} escalated beat best single) "
        f"→ **{'measured' if m3['pass'] else 'undefined/none'}**."
    )
    lines.append("")
    if retrain_block:
        lines.append(retrain_block)
        lines.append("")
    lines.append("## Live run (keyed, real spend)")
    lines.append("")
    lines.append(
        "The committed numbers above are from MOCK fixtures and a deterministic "
        "stub executor — **zero spend, no network**. To produce real M1–M4 "
        "numbers you must run keyed against real models:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Provide an OpenRouter key (never commit it):")
    lines.append("export OPENROUTER_API_KEY=sk-...        # or: export API_KEYS='{\"OpenRouter\": \"sk-...\"}'")
    lines.append("")
    lines.append("# 2. Build the real router from the plugin config and route+fuse a")
    lines.append("#    real benchmark slice (GSM8K/MATH/GPQA/MBPP), scoring answers with")
    lines.append("#    llmrouter/data/api_calling_evaluation.eval_perf. The real")
    lines.append("#    FusionGateRouter + FusionExecutor make the openrouter:fusion calls;")
    lines.append("#    all OpenRouter HTTP specifics stay inside executor.py.")
    lines.append("#    (This offline harness does NOT make live calls by design.)")
    lines.append("```")
    lines.append("")
    lines.append(
        "M4 (the offline log→retrain quality delta) is produced by `retrain.py`; "
        "its mock delta is reported above when `--with-retrain` is passed."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_mock_harness(
    *,
    dataset_path: str | Path,
    llm_path: str | Path,
    routing_path: str | Path | None,
    threshold: float,
    budget_threshold: float | None,
    k: int,
    judge: str | None,
    panel_preset: str,
) -> tuple[EvalHarness, list[dict[str, Any]], str]:
    """Construct an offline mock harness from fixture paths. Returns (harness, dataset, dataset_path)."""
    dataset = load_jsonl(dataset_path)
    llm_data = load_llm_candidates(llm_path)
    routing_data = load_jsonl(routing_path) if routing_path and Path(routing_path).exists() else None
    records_by_query = {r["query"]: r for r in dataset}
    executor = MockFusionExecutor(llm_data=llm_data, records_by_query=records_by_query)
    harness = EvalHarness(
        dataset=dataset,
        llm_data=llm_data,
        executor=executor,
        routing_data=routing_data,
        threshold=threshold,
        budget_threshold=budget_threshold,
        k=k,
        judge=judge,
        panel_preset=panel_preset,
    )
    return harness, dataset, str(dataset_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Offline mock mode (default; zero spend, no network).")
    # --live is intentionally hidden from --help: this harness is offline-only and
    # passing it is an immediate error (see the parser.error below). It is kept
    # (suppressed) so a stray --live yields a clear "live mode not supported"
    # message rather than an opaque "unrecognized arguments" failure.
    parser.add_argument("--live", dest="mock", action="store_false",
                        help=argparse.SUPPRESS)
    parser.add_argument("--dataset", default=str(_FIXTURES_DIR / "hard_slice.jsonl"),
                        help="Hard-slice JSONL dataset.")
    parser.add_argument("--llm", default=str(_FIXTURES_DIR / "llm_candidates.json"),
                        help="Candidate-metadata JSON (default_llm.json shape).")
    parser.add_argument("--routing", default=str(_FIXTURES_DIR / "routing_data.jsonl"),
                        help="Per-model routing performance JSONL (capability source).")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"),
                        help="Output directory for results.csv and results.md.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--budget-threshold", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=2,
                        help="Fusion panel size. Default 2 keeps the panel cost-bounded "
                             "so the hard-slice blended cost stays within the M2 target; "
                             "the plugin config's k=3 trades cost for breadth.")
    parser.add_argument("--judge", default=None)
    parser.add_argument("--panel-preset", default="Quality")
    parser.add_argument("--with-retrain", action="store_true",
                        help="Append the mock retrain (M3 before/after) delta to results.md.")
    args = parser.parse_args(argv)

    if not args.mock:
        parser.error(
            "Live mode is intentionally not wired into this offline harness so a "
            "stray run cannot spend. Use the keyed live-run path documented in "
            "results.md (build the real FusionGateRouter + FusionExecutor)."
        )

    harness, dataset, dataset_path = build_mock_harness(
        dataset_path=args.dataset,
        llm_path=args.llm,
        routing_path=args.routing,
        threshold=args.threshold,
        budget_threshold=args.budget_threshold,
        k=args.k,
        judge=args.judge,
        panel_preset=args.panel_preset,
    )
    arms = harness.run_all()
    verdicts = compute_verdicts(arms)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(arms, out_dir / "results.csv")

    retrain_block = None
    if args.with_retrain:
        # Lazy import to keep the base harness dependency-light.
        from . import retrain as _retrain  # type: ignore

        retrain_block = _retrain.mock_retrain_report_block(
            dataset=dataset,
            llm_path=args.llm,
            routing_path=args.routing,
            k=args.k,
            judge=args.judge,
        )

    # Report a portable repo-relative path so the committed results.md is not
    # tied to one machine's home directory.
    try:
        display_path = os.path.relpath(dataset_path, _PLUGIN_DIR.parents[1])
    except ValueError:  # pragma: no cover - different drive on some platforms
        display_path = dataset_path
    md = render_results_md(
        arms, verdicts, mock=args.mock, dataset_path=display_path,
        n=len(dataset), retrain_block=retrain_block,
    )
    (out_dir / "results.md").write_text(md, encoding="utf-8")

    print(f"Wrote {out_dir / 'results.csv'}")
    print(f"Wrote {out_dir / 'results.md'}")
    for arm, r in arms.items():
        print(f"  {arm}: quality={r.quality:.4f} cost={r.blended_cost:.6f} "
              f"p={r.escalation_p:.4f} gate_precision={_fmt(r.gate_precision)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
