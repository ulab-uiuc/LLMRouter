"""RouteGate — the single-vs-fusion decision (UMB-119).

The gate decides, per query, whether to take the cheap SINGLE-model path or
escalate to the FUSION path. The decision is driven by two scalars in [0, 1]:

  difficulty  — how hard the query is (higher => more likely to fuse)
  confidence  — how sure the gate is in its single-vs-fusion call

Difficulty estimation follows LLMRouter's ``ThresholdRouter``
(``custom_routers/thresholdrouter/router.py``) two-mode design:

  1. Injected estimator (preferred). When the caller supplies a query embedding
     via ``query_input['embedding']`` AND an estimator is wired in, the gate
     defers to the learned ``DifficultyEstimator``. The estimator is duck-typed
     (any callable ``embedding -> score``) so this module needs no torch import
     and stays unit-testable with no trained model present.

  2. Lexical fallback (always available). A deterministic, documented heuristic
     over the raw query text — length, code/math markers, multi-part questions.
     This guarantees the gate runs end-to-end with no embedding and no model.

The estimator and the lexical heuristic are kept as separate methods so each is
independently unit-testable. ``GateDecision`` is extended additively only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

# Three-tier dial (UMB-124). The middle tier escalates a mid-difficulty query to
# a *cheap* Budget fusion panel; the top tier escalates the hardest queries to
# the full Quality fusion panel. ``"fusion"`` is retained as an alias-free
# explicit Quality tier so existing callers/tests that branch on
# ``tier == "fusion"`` keep working. Extension is additive: a new value is added
# between the existing ones, not a rename.
Tier = Literal["single", "budget_fusion", "fusion"]

# Tiers that take the FUSION path (panel + judge) rather than single routing.
FUSION_TIERS: frozenset[str] = frozenset({"budget_fusion", "fusion"})

# Maps a fusion tier to the panel preset used when capability data is missing.
# Only the mid tier has a fixed preset here: ``budget_fusion`` always falls back
# to the cheap Budget panel. The top ``fusion`` tier is intentionally absent so
# ``TIER_TO_PRESET.get(tier, self.panel_preset)`` resolves it to the router's
# configured ``panel_preset`` (Quality by default) — a hardcoded "fusion":
# "Quality" entry here would be dead data, always overridden by panel_preset.
TIER_TO_PRESET: dict[str, str] = {
    "budget_fusion": "Budget",
}


def resolve_preset(tier: str, default_preset: str) -> str:
    """Resolve the panel preset for ``tier``, falling back to ``default_preset``.

    Single source of truth for the tier->preset mapping shared by
    ``FusionGateRouter._select_panel`` and the eval harness, so the two cannot
    silently diverge. The mid ``budget_fusion`` tier maps to the cheap Budget
    panel via ``TIER_TO_PRESET``; every other tier (notably the top ``fusion``
    tier, deliberately absent from ``TIER_TO_PRESET``) resolves to the caller's
    configured ``default_preset`` (typically ``panel_preset``, Quality by
    default).

    Args:
        tier: The gate-decided tier (e.g. ``"budget_fusion"`` / ``"fusion"``).
        default_preset: Preset to use when ``tier`` has no fixed mapping.

    Returns:
        The preset name (e.g. ``"Budget"`` / ``"Quality"``).
    """
    return TIER_TO_PRESET.get(tier, default_preset)

# --- lexical heuristic tuning constants (documented, deterministic) ----------
# Difficulty is a weighted blend of independent signals, each normalized to
# [0, 1], then clamped. Weights sum to 1.0 so difficulty stays in [0, 1].
_LENGTH_SATURATION_CHARS = 400.0  # query length at which the length signal hits 1.0
_LENGTH_WEIGHT = 0.40
_CODE_MATH_WEIGHT = 0.35
_MULTIPART_WEIGHT = 0.25
_MULTIPART_SATURATION = 3.0  # number of sub-questions at which the signal hits 1.0

# Markers that indicate code or math content (case-insensitive substring match
# for words; symbol matches are literal). Deliberately conservative.
_CODE_MATH_KEYWORDS = (
    "code",
    "function",
    "algorithm",
    "compile",
    "debug",
    "regex",
    "integral",
    "derivative",
    "theorem",
    "proof",
    "equation",
    "matrix",
    "complexity",
)
_CODE_MATH_SYMBOLS = ("```", "def ", "class ", "{", "}", ";", "=>", "==", "->", "\\", "^", "∫", "∑", "√")


@dataclass
class GateDecision:
    """Result of the gate.

    tier        : "single" or "fusion"
    model_name  : chosen model when tier == "single" (None for fusion)
    panel       : optional pre-selected panel for fusion (UMB-123 may fill this);
                  when empty, the router/executor fall back to the preset
    difficulty  : estimated difficulty in [0, 1] (for logging / threshold tuning)
    confidence  : router confidence in [0, 1]
    """

    tier: Tier
    model_name: str | None = None
    panel: list[str] = field(default_factory=list)
    difficulty: float = 0.0
    confidence: float = 1.0


class RouteGate:
    """Route-vs-fuse gate.

    Args:
        llm_data: name -> candidate-metadata mapping (from default_llm.json).
        threshold: difficulty cutoff to escalate single -> fusion. Sourced from
            the router YAML and injected by ``FusionGateRouter``; never hardcoded
            here beyond a permissive default for standalone construction.
        estimator: optional learned difficulty estimator. Any callable mapping a
            query embedding to a difficulty score in [0, 1]. Duck-typed so no
            torch dependency is introduced; a scalar is extracted via ``.item()``
            when the return value exposes it (e.g. a 0-d tensor).
    """

    def __init__(
        self,
        llm_data: dict[str, Any],
        threshold: float = 0.5,
        estimator: Callable[[Any], Any] | None = None,
        budget_threshold: float | None = None,
    ):
        self.llm_data = llm_data
        self.llm_names = list(llm_data.keys())
        self.threshold = threshold
        self.estimator = estimator
        # Three-tier dial (UMB-124). ``budget_threshold`` is the LOWER boundary:
        #   difficulty <  budget_threshold              -> single
        #   budget_threshold <= difficulty < threshold  -> budget_fusion (cheap)
        #   difficulty >= threshold                      -> fusion (Quality)
        # When ``budget_threshold`` is None (or >= threshold) the middle tier is
        # disabled and the gate degrades to the original two-tier single/fusion
        # behavior, so existing two-threshold-free configs are unaffected.
        if budget_threshold is None or budget_threshold >= threshold:
            self.budget_threshold = threshold
        else:
            self.budget_threshold = budget_threshold

    def decide(self, query_input: dict) -> GateDecision:
        """Decide single-vs-fusion for one query.

        Rules (three-tier dial, UMB-124):
          - ``high_stakes`` forces the full Quality ``fusion`` tier regardless of
            difficulty (max confidence in the fusion call, since the caller has
            overridden the gate).
          - otherwise estimate difficulty (injected estimator if an embedding is
            present, else the lexical fallback) and place it against the two
            thresholds:
              * difficulty >= ``threshold``            => ``fusion`` (Quality panel)
              * ``budget_threshold`` <= difficulty     => ``budget_fusion`` (cheap)
              * difficulty < ``budget_threshold``      => ``single`` (cheapest model)
          - confidence is derived from the margin to ``threshold`` (see
            ``_confidence``). When the middle tier is disabled
            (``budget_threshold == threshold``) this collapses to single/fusion.
        """
        if query_input.get("high_stakes"):
            # Caller override: fuse, and report difficulty if we can still compute
            # it for logging, but the decision itself is forced with full confidence.
            difficulty = self._difficulty(query_input)
            return GateDecision(
                tier="fusion",
                difficulty=difficulty,
                confidence=1.0,
            )

        difficulty = self._difficulty(query_input)
        confidence = self._confidence(difficulty)

        if difficulty >= self.threshold:
            return GateDecision(
                tier="fusion",
                difficulty=difficulty,
                confidence=confidence,
            )

        if difficulty >= self.budget_threshold:
            return GateDecision(
                tier="budget_fusion",
                difficulty=difficulty,
                confidence=confidence,
            )

        return GateDecision(
            tier="single",
            model_name=self._cheapest_model(),
            difficulty=difficulty,
            confidence=confidence,
        )

    # ----------------------------------------------------------- difficulty

    def _difficulty(self, query_input: dict) -> float:
        """Estimate difficulty in [0, 1] for a query.

        Prefers the injected estimator when both an embedding and an estimator
        are available; otherwise falls back to the deterministic lexical
        heuristic over the raw query text.
        """
        embedding = query_input.get("embedding")
        if embedding is not None and self.estimator is not None:
            return self._estimate_with_model(embedding)
        return self._lexical_difficulty(query_input.get("query", ""))

    def _estimate_with_model(self, embedding: Any) -> float:
        """Run the injected estimator and coerce its output to a clamped float.

        The estimator is duck-typed: any callable returning either a Python float
        or an object exposing ``.item()`` (e.g. a 0-d / 1-element tensor). This
        mirrors ``ThresholdRouter._estimate_difficulty`` without importing torch.
        """
        score = self.estimator(embedding)
        if hasattr(score, "item"):
            score = score.item()
        return self._clamp(float(score))

    def _lexical_difficulty(self, query: str) -> float:
        """Deterministic lexical difficulty heuristic (no model required).

        Blends three normalized signals:
          - length      : ``len(query) / 400`` clamped to 1.0.
          - code/math    : 1.0 if any code/math keyword or symbol is present,
                           else 0.0.
          - multi-part   : count of sub-questions (``?`` plus enumerated/"and"-joined
                           clauses) normalized by ``_MULTIPART_SATURATION``.

        The blend is a fixed-weight convex combination, so the result is always
        in [0, 1] and fully reproducible. Kept pure (text in, float out) for
        unit testing.
        """
        if not query:
            return 0.0

        length_signal = self._clamp(len(query) / _LENGTH_SATURATION_CHARS)
        code_math_signal = 1.0 if self._has_code_or_math(query) else 0.0
        multipart_signal = self._clamp(self._count_subquestions(query) / _MULTIPART_SATURATION)

        difficulty = (
            _LENGTH_WEIGHT * length_signal
            + _CODE_MATH_WEIGHT * code_math_signal
            + _MULTIPART_WEIGHT * multipart_signal
        )
        return self._clamp(difficulty)

    @staticmethod
    def _has_code_or_math(query: str) -> bool:
        """True if the query contains a code/math keyword or symbol."""
        lowered = query.lower()
        if any(keyword in lowered for keyword in _CODE_MATH_KEYWORDS):
            return True
        return any(symbol in query for symbol in _CODE_MATH_SYMBOLS)

    @staticmethod
    def _count_subquestions(query: str) -> int:
        """Count distinct sub-questions / parts in a query.

        Heuristic: the larger of (a) the number of '?' characters and
        (b) 1 + the number of enumerated parts or coordinating " and "/" ; "
        separators. A single simple question therefore counts as 1.
        """
        question_marks = query.count("?")
        # Enumerated parts: "1.", "2)", "- ", or coordinating separators.
        enumerations = len(re.findall(r"(?:\b\d+[.)]\s)|(?:\s;\s)|(?:\sand\s)", query))
        parts = max(question_marks, 1 + enumerations)
        return parts

    # ----------------------------------------------------------- confidence

    def _confidence(self, difficulty: float) -> float:
        """Derive confidence in [0, 1] from the margin to the threshold.

        Intuition: the gate is most confident when difficulty sits far from the
        threshold (clearly easy or clearly hard) and least confident right at the
        boundary. We normalize the absolute margin by the larger side of the
        threshold split so both an easy and a hard query can reach full
        confidence at the extremes.
        """
        margin = abs(difficulty - self.threshold)
        span = max(self.threshold, 1.0 - self.threshold)
        if span <= 0.0:
            return 1.0
        return self._clamp(margin / span)

    # ----------------------------------------------------------- selection

    def cheapest_model(self) -> str:
        """Pick the lowest-cost candidate as the single-path default (public API).

        Public entry point for callers outside this class (the router's downgrade
        guard, the eval harness). Delegates to the private implementation.
        """
        return self._cheapest_model()

    def _cheapest_model(self) -> str:
        """Pick the lowest-cost candidate as the single-path default."""

        def cost(name: str) -> float:
            info = self.llm_data.get(name, {})
            # default_llm.json uses input_price / output_price; fall back gracefully
            return float(info.get("input_price", 0.0)) + float(info.get("output_price", 0.0))

        return min(self.llm_names, key=cost) if self.llm_names else ""

    # ----------------------------------------------------------- utilities

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        """Clamp ``value`` into the closed interval [low, high]."""
        return max(low, min(high, value))
