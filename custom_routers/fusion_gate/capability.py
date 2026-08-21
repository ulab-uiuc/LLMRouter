"""CapabilityScorer — capability-scored fusion panel selection (UMB-123).

Scores each candidate model against a query and returns the top-k panel that
maps to the OpenRouter ``openrouter:fusion`` tool's ``analysis_models``. Panel
membership varies by **query type**: a code/math/reasoning query and a general
query draw on different per-category performance, so they generally produce
different panels.

Capability source (offline, no network):
  - LLMRouter per-model routing performance — the ``routing_data_*`` tables that
    ``MetaRouter``'s ``DataLoader`` attaches to the router (a pandas DataFrame or
    a list of row dicts). Each row carries ``task_name`` / ``model_name`` /
    ``performance``; we bucket ``task_name`` into a small set of query
    categories and aggregate mean performance per ``(category, model)``.
  - The ``feature`` text and prices in ``default_llm.json`` provide a deterministic
    secondary signal (capability prior from model size/feature wording, lightly
    cost-penalized) so scoring still differentiates models for categories the
    routing table does not cover.

Fallback contract (UMB-123): when no usable capability data is available for a
query's category, ``select_panel`` returns ``None`` so the caller falls back to
the configured ``panel_preset`` (Quality / Budget). The presets are also defined
here so the router/executor share one source of truth.

The scorer is pure data-in / list-out and imports no torch, keeping it fully
offline and unit-testable with small in-memory fixtures.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal

QueryCategory = Literal["code", "math", "reasoning", "general"]

# --- query-type detection (deterministic, documented) ------------------------
# Mirrors the gate's code/math markers but resolves a single *category* label so
# panel selection can be category-specific. Order matters: the first matching
# category wins, with "general" as the catch-all.
_CODE_KEYWORDS = (
    "code",
    "function",
    "compile",
    "debug",
    "regex",
    "program",
    "python",
    "javascript",
    "bug",
)
_CODE_SYMBOLS = ("```", "def ", "class ", "{", "}", ";", "=>", "->")
_MATH_KEYWORDS = (
    "integral",
    "derivative",
    "theorem",
    "proof",
    "equation",
    "matrix",
    "algebra",
    "calculus",
    "probability",
    "geometry",
)
_MATH_SYMBOLS = ("∫", "∑", "√", "^", "\\")
_REASONING_KEYWORDS = (
    "algorithm",
    "complexity",
    "reason",
    "logic",
    "deduce",
    "puzzle",
    "explain why",
    "step by step",
    "strategy",
    "plan",
)

# Mapping from routing-data ``task_name`` substrings to a query category. The
# example routing data uses task names like ``agentverse-logicgrid``; this lets
# the per-category aggregation align with the query-type detector above.
_TASK_CATEGORY_PATTERNS: tuple[tuple[str, QueryCategory], ...] = (
    ("logic", "reasoning"),
    ("reason", "reasoning"),
    ("grid", "reasoning"),
    ("puzzle", "reasoning"),
    ("math", "math"),
    ("gsm", "math"),
    ("arithmetic", "math"),
    ("algebra", "math"),
    ("code", "code"),
    ("humaneval", "code"),
    ("mbpp", "code"),
    ("program", "code"),
)

# Built-in presets used as the fallback panel when capability data is missing.
# These are *labels*, not model names: the scorer resolves them against the
# candidate set by price (cheapest-N for Budget, most-capable-N for Quality).
PRESET_QUALITY = "Quality"
PRESET_BUDGET = "Budget"


class CapabilityScorer:
    """Score candidate models per query and pick a top-k fusion panel.

    Args:
        llm_data: name -> candidate-metadata mapping (from default_llm.json),
            carrying ``feature`` text and ``input_price`` / ``output_price``.
        routing_data: optional per-model performance source — a pandas DataFrame
            or an iterable of row dicts with ``task_name`` / ``model_name`` /
            ``performance`` keys. When ``None`` or empty, capability scoring
            falls back to the static prior derived from ``llm_data``.
    """

    def __init__(
        self,
        llm_data: dict[str, Any],
        routing_data: Any = None,
    ):
        self.llm_data = llm_data
        self.llm_names = list(llm_data.keys())
        # category -> {model_name -> mean performance in roughly [0, 1]}
        self._perf_by_category: dict[QueryCategory, dict[str, float]] = (
            self._aggregate_performance(routing_data)
        )

    # ----------------------------------------------------------- public API

    def select_panel(self, query: str, k: int) -> list[str] | None:
        """Return the capability-scored top-k panel for ``query``.

        The query is classified into a category (code/math/reasoning/general);
        models are scored for that category and the top ``k`` by score are
        returned. Returns ``None`` when no usable capability data exists for the
        category, signalling the caller to fall back to ``panel_preset``.

        Args:
            query: Raw query text.
            k: Panel size (maps to the fusion tool's ``analysis_models`` length).

        Returns:
            A list of up to ``k`` candidate model names, or ``None`` to trigger
            the preset fallback.
        """
        if k <= 0 or not self.llm_names:
            return None

        category = self.classify_query(query)
        scores = self._score_models(category)
        if scores is None:
            return None

        ranked = sorted(
            self.llm_names,
            key=lambda name: (scores.get(name, 0.0), name),
            reverse=True,
        )
        return ranked[:k]

    def preset_panel(self, preset: str, k: int) -> list[str]:
        """Resolve a named preset (Quality / Budget) to a top-k panel.

        Quality => the ``k`` most-capable candidates (price-as-capability proxy,
        descending). Budget => the ``k`` cheapest candidates. Any unrecognized
        preset is treated as Quality. Used as the fallback when capability data
        is unavailable.
        """
        if k <= 0 or not self.llm_names:
            return []

        by_price_desc = sorted(
            self.llm_names, key=lambda name: (self._price(name), name), reverse=True
        )
        if str(preset).lower() == PRESET_BUDGET.lower():
            cheapest = sorted(self.llm_names, key=lambda name: (self._price(name), name))
            return cheapest[:k]
        return by_price_desc[:k]

    def classify_query(self, query: str) -> QueryCategory:
        """Classify a query into a coarse capability category.

        Deterministic precedence: code, then math, then reasoning, else general.
        Kept pure (text in, label out) for unit testing.
        """
        if not query:
            return "general"
        lowered = query.lower()

        if self._matches(lowered, query, _CODE_KEYWORDS, _CODE_SYMBOLS):
            return "code"
        if self._matches(lowered, query, _MATH_KEYWORDS, _MATH_SYMBOLS):
            return "math"
        if any(keyword in lowered for keyword in _REASONING_KEYWORDS):
            return "reasoning"
        return "general"

    # ----------------------------------------------------------- scoring

    def _score_models(self, category: QueryCategory) -> dict[str, float] | None:
        """Build a per-model score map for a category, or ``None`` if unusable.

        Combines two signals:
          1. Empirical per-category performance from the routing data (primary).
          2. A static prior from ``llm_data`` (feature/size wording, lightly
             cost-penalized) so models absent from the routing table for this
             category still rank relative to one another.

        Returns ``None`` only when *neither* signal yields any differentiation
        (no routing data for the category AND no llm_data prior), which is the
        fallback trigger for ``select_panel``.
        """
        empirical = self._perf_by_category.get(category, {})
        prior = self._static_prior()

        if not empirical and not prior:
            return None

        scores: dict[str, float] = {}
        for name in self.llm_names:
            emp = empirical.get(name)
            pri = prior.get(name, 0.0)
            if emp is not None:
                # Empirical performance dominates; the prior breaks ties and
                # ranks models the routing table did not cover for this category.
                scores[name] = 0.8 * emp + 0.2 * pri
            else:
                scores[name] = pri
        return scores

    def _static_prior(self) -> dict[str, float]:
        """Capability prior in [0, 1] from llm_data feature text and price.

        Heuristic and deterministic: larger / more-capable wording and higher
        price correlate with capability in the candidate set, but cost is lightly
        penalized so two models with similar capability favor the cheaper one.
        Returns an empty map when ``llm_data`` is empty.
        """
        if not self.llm_names:
            return {}

        prices = [self._price(name) for name in self.llm_names]
        max_price = max(prices) if prices else 0.0

        prior: dict[str, float] = {}
        for name in self.llm_names:
            info = self.llm_data.get(name, {})
            capability = self._feature_capability(info)
            price = self._price(name)
            # Normalize price to [0, 1]; subtract a small cost penalty.
            norm_price = (price / max_price) if max_price > 0 else 0.0
            prior[name] = self._clamp(capability - 0.1 * norm_price)
        return prior

    def _feature_capability(self, info: dict[str, Any]) -> float:
        """Estimate capability in [0, 1] from a candidate's size/feature text.

        Uses the model ``size`` (parameter count) when parseable, else falls
        back to capability-suggestive wording in the ``feature`` blurb. Both are
        normalized into [0, 1]; deterministic and offline.
        """
        size_score = self._size_score(info.get("size"))
        if size_score is not None:
            return size_score

        feature = str(info.get("feature", "")).lower()
        strong_markers = (
            "powerful",
            "high-accuracy",
            "exceptional",
            "advanced",
            "complex",
            "large-scale",
        )
        hits = sum(1 for marker in strong_markers if marker in feature)
        return self._clamp(hits / 3.0)

    @staticmethod
    def _size_score(size: Any) -> float | None:
        """Parse a parameter-count string (e.g. ``"49B"``) into a [0, 1] score.

        Normalized by a 200B saturation point so the example candidate set
        (7B..141B) spreads across the range. Returns ``None`` when unparseable.
        """
        if size is None:
            return None
        match = re.match(r"\s*([\d.]+)\s*([bBmM]?)", str(size))
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        unit = match.group(2).lower()
        billions = value / 1000.0 if unit == "m" else value
        return CapabilityScorer._clamp(billions / 200.0)

    # ----------------------------------------------------------- aggregation

    def _aggregate_performance(
        self, routing_data: Any
    ) -> dict[QueryCategory, dict[str, float]]:
        """Aggregate mean performance per (category, model) from routing data.

        Accepts a pandas DataFrame or an iterable of row dicts. Rows missing the
        required keys are skipped. ``task_name`` is bucketed into a query
        category; ``performance`` values are averaged per (category, model).
        Returns an empty mapping when no usable rows are present.
        """
        rows = self._iter_rows(routing_data)
        # category -> model -> [running_sum, count]
        accum: dict[QueryCategory, dict[str, list[float]]] = {}

        for row in rows:
            model = row.get("model_name")
            perf = row.get("performance")
            task = row.get("task_name")
            if model is None or perf is None:
                continue
            try:
                perf_value = float(perf)
            except (TypeError, ValueError):
                continue
            category = self._task_to_category(task)
            bucket = accum.setdefault(category, {})
            entry = bucket.setdefault(str(model), [0.0, 0.0])
            entry[0] += perf_value
            entry[1] += 1.0

        result: dict[QueryCategory, dict[str, float]] = {}
        for category, models in accum.items():
            result[category] = {
                name: (total / count) if count else 0.0
                for name, (total, count) in models.items()
            }
        return result

    @staticmethod
    def _iter_rows(routing_data: Any) -> Iterable[dict[str, Any]]:
        """Yield row dicts from a DataFrame or an iterable of dicts.

        DataFrames are detected by duck-typing ``to_dict`` (pandas) so this
        module never imports pandas. Anything else is treated as an iterable of
        mapping-like rows; non-mappings are ignored.
        """
        if routing_data is None:
            return []
        # pandas DataFrame: convert to list-of-dicts without importing pandas.
        if hasattr(routing_data, "to_dict"):
            try:
                return routing_data.to_dict(orient="records")
            except TypeError:
                return []
        if isinstance(routing_data, dict):
            return []
        try:
            return [row for row in routing_data if isinstance(row, dict)]
        except TypeError:
            return []

    @staticmethod
    def _task_to_category(task_name: Any) -> QueryCategory:
        """Bucket a routing-data ``task_name`` into a query category."""
        if not task_name:
            return "general"
        lowered = str(task_name).lower()
        for pattern, category in _TASK_CATEGORY_PATTERNS:
            if pattern in lowered:
                return category
        return "general"

    # ----------------------------------------------------------- utilities

    def _price(self, name: str) -> float:
        """Per-model unit price (input + output) from llm_data."""
        info = self.llm_data.get(name, {})
        return float(info.get("input_price", 0.0)) + float(info.get("output_price", 0.0))

    @staticmethod
    def _matches(
        lowered: str, raw: str, keywords: tuple[str, ...], symbols: tuple[str, ...]
    ) -> bool:
        """True when any keyword (lowercased) or raw symbol is present."""
        if any(keyword in lowered for keyword in keywords):
            return True
        return any(symbol in raw for symbol in symbols)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        """Clamp ``value`` into [low, high]."""
        return max(low, min(high, value))
