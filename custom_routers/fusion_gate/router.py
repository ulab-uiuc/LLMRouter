"""FusionGateRouter — route-vs-fuse meta-router for LLMRouter.

Integration entry point (UMB-118/121/123/124). Implements the MetaRouter
contract (``route_single`` / ``route_batch``) and owns:

  - UMB-121: reading and respecting all six config keys (``threshold``, ``k``,
    ``judge``, ``provider``/``base_url``, ``panel_preset``, ``cost_ceiling``),
    plus a spend-free ``--route-only`` path and the cost guard.
  - UMB-123: capability-scored panel selection via ``CapabilityScorer``, with a
    preset (Quality / Budget) fallback when capability data is unavailable.
  - UMB-124: a three-tier dial — ``single`` -> ``budget_fusion`` (cheap panel)
    -> ``fusion`` (full Quality panel) — threaded from the gate through to panel
    selection.

The router's only job is to DECIDE, per query, between:
  - the cheap SINGLE-model path (classic LLMRouter routing),
  - a cheap BUDGET-fusion panel, or
  - the full QUALITY-fusion panel (OpenRouter ``openrouter:fusion`` server tool).

Routing never spends: ``route_single`` returns a decision dict (no API call).
Spend happens only in ``fuse()``, which is invoked separately and logs every
call via ``fusion_log.log_fusion``.

See: fusion-gate-router-prd-v0.2.0.md
"""

from __future__ import annotations

import sys
from typing import Any

# torch is a GENUINE transitive requirement of this module: MetaRouter subclasses
# torch.nn.Module, so importing FusionGateRouter requires torch even though the
# gate itself runs no inference. This import is intentionally eager (not lazy) so
# the dependency fails loudly at import time rather than mid-route. Torch-free
# unit tests therefore load gate.py / executor.py / fusion_log.py by file path,
# never through this module — see custom_routers/fusion_gate/tests/.
import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter

from .capability import CapabilityScorer
from .executor import CostCeilingExceeded, FusionExecutor, FusionResult
from .fusion_log import log_fusion
from .gate import FUSION_TIERS, GateDecision, RouteGate, resolve_preset


class FusionGateRouter(MetaRouter):
    """Meta-router that gates each query between single routing and fusion tiers.

    Decision contract returned by ``route_single``:

      single path → {"query", "strategy": "single", "tier": "single",
                     "model_name", "predicted_llm", "difficulty", "confidence"}
      fusion path → {"query", "strategy": "fusion", "tier": "budget_fusion"|"fusion",
                     "panel": [...], "judge": ..., "model_name", "predicted_llm",
                     "difficulty", "confidence", "projected_cost"}

    Both shapes carry ``strategy`` and ``tier`` so downstream code can branch, and
    both carry ``model_name`` for drop-in compatibility with the CLI's
    ``route_query`` (which keys on ``model_name`` / ``predicted_llm``). The fusion
    path's ``model_name`` is the judge (or panel head) purely as a label — no API
    call is made during routing.
    """

    def __init__(self, yaml_path: str):
        # MetaRouter manages config/LLM-candidate loading. A simple gate needs no
        # trainable model, so Identity stands in until a learned gate lands.
        model = nn.Identity()
        super().__init__(model=model, yaml_path=yaml_path)

        # Available candidate LLMs (name -> metadata dict from default_llm.json).
        self.llm_names: list[str] = list(self.llm_data.keys())

        # ------------------------------------------------------ config (UMB-121)
        # All six config keys are read here and respected downstream. `.get`
        # keeps construction robust if a key is omitted.
        hparam: dict[str, Any] = self.cfg.get("hparam", {}) or {}
        self.threshold: float = float(hparam.get("threshold", 0.5))
        self.k: int = int(hparam.get("k", 3))
        self.judge: str | None = hparam.get("judge")
        self.panel_preset: str = hparam.get("panel_preset", "Quality")
        cost_ceiling = hparam.get("cost_ceiling")
        self.cost_ceiling: float | None = (
            float(cost_ceiling) if cost_ceiling is not None else None
        )
        # Per-completion output-token estimate feeding the dollar cost projection.
        self.est_completion_tokens: int = int(hparam.get("est_completion_tokens", 512))
        # Three-tier dial (UMB-124): optional lower boundary for the middle tier.
        budget_threshold = hparam.get("budget_threshold")
        self.budget_threshold: float | None = (
            float(budget_threshold) if budget_threshold is not None else None
        )

        # Provider / base_url (UMB-121). ``base_url`` is the OpenRouter endpoint
        # for the beta server tool; ``provider`` is informational and resolved by
        # the executor for key lookup. ``base_url`` takes precedence over the
        # legacy top-level ``api_endpoint``.
        self.provider: str | None = hparam.get("provider") or self.cfg.get("provider")
        self.base_url: str | None = (
            hparam.get("base_url")
            or self.cfg.get("base_url")
            or self.cfg.get("api_endpoint")
        )

        # Optional JSONL sink for fusion logging (defaults inside fusion_log).
        self.log_sink_path: str | None = hparam.get("log_sink_path")

        # ----------------------------------------------------------- seams
        self.gate = RouteGate(
            llm_data=self.llm_data,
            threshold=self.threshold,
            budget_threshold=self.budget_threshold,
        )
        # Capability scorer (UMB-123) sources per-model performance from the
        # routing data the DataLoader attached (DataFrame or None). Prefer the
        # train split (richer); fall back to test split.
        routing_data = getattr(self, "routing_data_train", None)
        if routing_data is None:
            routing_data = getattr(self, "routing_data_test", None)
        self.capability = CapabilityScorer(
            llm_data=self.llm_data,
            routing_data=routing_data,
        )
        self.executor = FusionExecutor(
            llm_data=self.llm_data,
            judge=self.judge,
            panel_preset=self.panel_preset,
            cost_ceiling=self.cost_ceiling,
            api_endpoint=self.base_url,
            est_completion_tokens=self.est_completion_tokens,
        )

    # ------------------------------------------------------------------ routing

    def route_single(self, query_input: dict) -> dict:
        """Route one query: decide tier, then select the panel if fusing.

        SPEND-FREE: this only computes a decision. No OpenRouter call is made
        here, so ``--route-only`` (and the normal CLI route step) never spend.
        For the fusion tiers the intended panel/judge and a projected cost are
        included so callers can audit the plan before invoking ``fuse()``.
        """
        query = query_input["query"]

        decision: GateDecision = self.gate.decide(query_input)

        if decision.tier not in FUSION_TIERS:
            return {
                "query": query,
                "strategy": "single",
                "tier": decision.tier,
                "model_name": decision.model_name,
                "predicted_llm": decision.model_name,
                "difficulty": decision.difficulty,
                "confidence": decision.confidence,
            }

        # Fusion tier: select the panel (UMB-123/124). The judge is the
        # config-driven slug (None => the executor uses the outer model).
        panel = self._select_panel(query_input, decision)
        judge = self.judge

        # Cost guard (UMB-121): when the projected Σ(panel)+judge exceeds the
        # ceiling, abort fusion by DOWNGRADING to the cheap single path rather
        # than spending. The downgrade is reported via ``tier``/``downgraded``.
        projected = self.executor.project_cost(panel, judge, query=query)
        if self.cost_ceiling is not None and projected > self.cost_ceiling:
            fallback_model = self.gate.cheapest_model()
            return {
                "query": query,
                "strategy": "single",
                "tier": "single",
                "downgraded_from": decision.tier,
                "model_name": fallback_model,
                "predicted_llm": fallback_model,
                "difficulty": decision.difficulty,
                "confidence": decision.confidence,
                "projected_cost": projected,
                "cost_ceiling": self.cost_ceiling,
            }

        return {
            "query": query,
            "strategy": "fusion",
            "tier": decision.tier,
            "panel": panel,
            "judge": judge,
            # Label only (the judge/outer model); no API call is made in routing.
            "model_name": judge or (panel[0] if panel else None),
            "predicted_llm": judge or (panel[0] if panel else None),
            "difficulty": decision.difficulty,
            "confidence": decision.confidence,
            "projected_cost": projected,
        }

    def route_batch(self, batch: list) -> list:
        """Route multiple queries."""
        return [self.route_single(q) for q in batch]

    # ----------------------------------------------------------------- internals

    def _select_panel(self, query_input: dict, decision: GateDecision) -> list[str]:
        """Pick the fusion panel (maps to the tool's ``analysis_models``).

        UMB-123/124: capability-scored, tier-aware selection.
          - The capability scorer (UMB-123) scores candidates for the query's
            category and returns the top-k; panel membership therefore varies by
            query type (code/math/reasoning vs general).
          - The tier (UMB-124) selects the fallback preset when capability data
            is unavailable: ``budget_fusion`` -> Budget, ``fusion`` -> Quality.
          - A panel pre-selected on the gate decision wins, if present.
        """
        if decision.panel:
            return decision.panel

        query = query_input.get("query", "")
        panel = self.capability.select_panel(query, self.k)
        if panel:
            return panel

        # Fallback: capability data unavailable for this query -> preset panel.
        # The tier dictates the preset: the mid ``budget_fusion`` tier maps to the
        # cheap Budget panel; any other tier (the top ``fusion`` tier) resolves to
        # the router's configured ``panel_preset``. Resolution is delegated to
        # ``gate.resolve_preset`` — the single source of truth shared with the
        # eval harness so the two cannot silently diverge.
        preset = resolve_preset(decision.tier, self.panel_preset)
        return self.capability.preset_panel(preset, self.k)

    # --------------------------------------------------------------- execution

    def fuse(self, route_result: dict, **gen_kwargs: Any) -> FusionResult:
        """Execute a fusion decision via the FusionExecutor (UMB-120).

        Kept separate from ``route_single`` so ``--route-only`` (UMB-121) can
        return the decision without ever calling this. Every fusion call is
        logged via ``fusion_log.log_fusion`` (UMB-125) — secret-scrubbed,
        raw-payload-free.

        Raises:
            ValueError: if called on a non-fusion route result.
            CostCeilingExceeded: re-raised from the executor's pre-call guard.
        """
        if route_result.get("strategy") != "fusion":
            raise ValueError("fuse() called on a non-fusion route result")

        query = route_result["query"]
        result = self.executor.run(
            query=query,
            panel=route_result["panel"],
            judge=route_result.get("judge"),
            **gen_kwargs,
        )

        # Log every fusion call (audit + training signal). A logging failure
        # (disk-full, permission-denied, NFS timeout) must NEVER destroy the
        # already-computed result, so the append is best-effort: any exception is
        # swallowed (reported to stderr) and the result is returned regardless.
        # The sink is append-only and secret-scrubbed inside fusion_log.
        try:
            log_fusion(result, query=query, sink_path=self.log_sink_path, cost=result.cost)
        except Exception as exc:  # noqa: BLE001 - logging must not lose the result
            # Report the failure type only; never echo the path/secret-bearing detail.
            print(
                f"fusion_log: failed to persist fusion call ({type(exc).__name__}); "
                "continuing without losing the result.",
                file=sys.stderr,
            )
        return result
