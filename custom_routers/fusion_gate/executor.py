"""FusionExecutor — isolates the OpenRouter `openrouter:fusion` call (UMB-120).

SCAFFOLD ONLY. This is the single blast point for the beta server-tool API:
all OpenRouter-specific request/response handling lives here and nowhere else,
so upstream changes touch one file. UMB-120 implements `run`; UMB-128 may add a
local fan-out path behind the same interface.

OpenRouter call shape (for the implementer):
    POST {api_endpoint or https://openrouter.ai/api/v1}/chat/completions
    body: {
      "model": <outer model>,
      "messages": [{"role": "user", "content": query}],
      "tools": [{"type": "openrouter:fusion",
                 "parameters": {"analysis_models": panel, "model": judge}}],
      "tool_choice": "required"   # gate already decided to fuse
    }
Result tool payload: { status, analysis?, responses: [{model, content}, ...] }
  - judge may fail → status "ok" with `analysis` omitted; fall back to writing
    the answer from `responses[]`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"

# Provider key used to resolve the OpenRouter credential from an API_KEYS dict.
OPENROUTER_PROVIDER = "OpenRouter"

# Default per-completion output-token estimate used by project_cost when no
# explicit completion-token count is supplied. Overridable via the
# ``est_completion_tokens`` hparam.
DEFAULT_EST_COMPLETION_TOKENS = 512

# Roughly four characters per token — the standard heuristic for estimating
# prompt token count from raw query text.
_CHARS_PER_TOKEN = 4

# OpenRouter server-tool identifier (BETA). Confined to this module.
FUSION_TOOL_TYPE = "openrouter:fusion"


class CostCeilingExceeded(Exception):
    """Raised when the projected fusion cost exceeds the configured ceiling.

    Carries the projected per-query DOLLAR cost and the ceiling (also in dollars)
    so callers can log/report the abort without re-projecting. Raised BEFORE any
    HTTP call is made.
    """

    def __init__(self, projected: float, ceiling: float):
        self.projected = projected
        self.ceiling = ceiling
        super().__init__(
            f"Projected fusion cost ${projected:.6f} exceeds cost_ceiling "
            f"${ceiling:.6f} per query; aborting before the OpenRouter call."
        )


class FusionExecutorError(Exception):
    """Raised on an unrecoverable OpenRouter fusion response (transport/parse)."""


@dataclass
class FusionResult:
    """Parsed output of a fusion call.

    answer      : final synthesized answer (judge output, or fallback from panel)
    analysis    : structured analysis JSON (consensus/contradictions/blind_spots),
                  or None when the judge failed
    responses   : raw per-model responses [{"model", "content"}] — the training
                  signal consumed by the log sink (UMB-125)
    panel       : panel actually used
    judge       : judge model actually used
    cost        : total cost (sum of panel completions + judge) when available
    raw         : the untouched provider payload, for debugging
    """

    answer: str = ""
    analysis: dict[str, Any] | None = None
    responses: list[dict[str, Any]] = field(default_factory=list)
    panel: list[str] = field(default_factory=list)
    judge: str | None = None
    cost: float | None = None
    raw: dict[str, Any] | None = None


class FusionExecutor:
    def __init__(
        self,
        llm_data: dict[str, Any],
        judge: str | None = None,
        panel_preset: str = "Quality",
        cost_ceiling: float | None = None,
        api_endpoint: str | None = None,
        est_completion_tokens: int = DEFAULT_EST_COMPLETION_TOKENS,
    ):
        self.llm_data = llm_data
        self.judge = judge
        self.panel_preset = panel_preset
        self.cost_ceiling = cost_ceiling
        self.api_endpoint = api_endpoint or DEFAULT_ENDPOINT
        self.est_completion_tokens = int(est_completion_tokens)

    def run(
        self,
        query: str,
        panel: list[str],
        judge: str | None = None,
        api_keys: dict[str, str] | None = None,
        **gen_kwargs: Any,
    ) -> FusionResult:
        """Execute one fusion call against the OpenRouter `openrouter:fusion` tool.

        A SINGLE POST to ``{api_endpoint}/chat/completions`` carries the panel as
        the tool's ``analysis_models`` and the judge as the tool's ``model``, with
        ``tool_choice="required"`` so the gate's fuse decision is honored.

        Args:
            query: The user query to fuse over.
            panel: Panel model slugs (-> tool ``analysis_models``).
            judge: Judge model slug (-> tool ``model``); falls back to the
                executor's configured judge, then to the outer model when unset.
            api_keys: Optional ``{"OpenRouter": "<key>"}`` provider dict. When
                absent, the key is resolved from the ``OPENROUTER_API_KEY`` env
                var or an ``API_KEYS`` JSON env var.
            **gen_kwargs: Extra generation params merged into the request body
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            FusionResult with parsed ``responses``/``analysis``. On judge failure
            (status ``ok`` with ``analysis`` omitted) the answer is synthesized
            from ``responses`` and ``analysis`` is ``None``.

        Raises:
            CostCeilingExceeded: when the projected cost exceeds ``cost_ceiling``
                (raised before any network call).
            FusionExecutorError: on transport failure or an unparseable payload.
        """
        judge = judge or self.judge

        # Cost guard: abort BEFORE the HTTP call so a too-expensive fusion never
        # reaches the network.
        if self.cost_ceiling is not None:
            projected = self.project_cost(panel, judge, query=query)
            if projected > self.cost_ceiling:
                raise CostCeilingExceeded(projected, self.cost_ceiling)

        api_key = self._resolve_api_key(api_keys)

        body = self._build_request_body(query, panel, judge, gen_kwargs)
        payload = self._post_chat_completions(body, api_key)
        return self._parse_payload(payload, panel, judge)

    # ------------------------------------------------------- OpenRouter (BETA)
    # Everything below this line is OpenRouter-specific request/response handling
    # and MUST stay confined to this module (the beta server-tool blast point).

    def _resolve_api_key(self, api_keys: dict[str, str] | None) -> str:
        """Resolve the OpenRouter key without logging it.

        Resolution order:
          1. ``api_keys["OpenRouter"]`` (explicit provider dict),
          2. ``OPENROUTER_API_KEY`` env var,
          3. ``API_KEYS`` env var parsed as a JSON ``{"OpenRouter": "..."}`` dict.

        The key value is never logged or echoed.
        """
        if api_keys:
            key = api_keys.get(OPENROUTER_PROVIDER)
            if key:
                return key

        env_key = os.environ.get("OPENROUTER_API_KEY")
        if env_key:
            return env_key

        raw = os.environ.get("API_KEYS")
        if raw:
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise FusionExecutorError(
                    "API_KEYS env var is not valid JSON; cannot resolve the "
                    f"{OPENROUTER_PROVIDER} key."
                ) from exc
            key = parsed.get(OPENROUTER_PROVIDER) if isinstance(parsed, dict) else None
            if key:
                return key

        raise FusionExecutorError(
            f"No {OPENROUTER_PROVIDER} API key found. Provide api_keys="
            f'{{"{OPENROUTER_PROVIDER}": "..."}}, set OPENROUTER_API_KEY, or set '
            "API_KEYS as a JSON object."
        )

    def _build_request_body(
        self,
        query: str,
        panel: list[str],
        judge: str | None,
        gen_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the chat/completions body carrying the openrouter:fusion tool.

        The outer ``model`` defaults to the judge slug when one is configured,
        falling back to the panel head; the tool's ``model`` (judge) defaults to
        the outer model when unset, matching the scaffold contract.

        Raises:
            ValueError: when ``panel`` is empty. A fusion call has no meaning
                without at least one analysis model, and an empty ``model`` field
                would produce a nonsensical OpenRouter request.
        """
        if not panel:
            raise ValueError("panel must be non-empty for a fusion call")
        outer_model = judge or panel[0]
        parameters: dict[str, Any] = {"analysis_models": list(panel)}
        if judge:
            parameters["model"] = judge

        body: dict[str, Any] = {
            "model": outer_model,
            "messages": [{"role": "user", "content": query}],
            "tools": [{"type": FUSION_TOOL_TYPE, "parameters": parameters}],
            "tool_choice": "required",
        }
        # Allow callers to pass through generation params without overriding the
        # fusion-defining keys above.
        for key, value in gen_kwargs.items():
            if key not in body:
                body[key] = value
        return body

    def _post_chat_completions(
        self, body: dict[str, Any], api_key: str
    ) -> dict[str, Any]:
        """POST the request and return the decoded JSON payload.

        Prefers ``requests`` when importable; otherwise uses stdlib ``urllib``.
        The Authorization header carries the key but is never logged.
        """
        url = f"{self.api_endpoint}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            import requests  # type: ignore
        except ImportError:
            return self._post_urllib(url, headers, body)

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - normalize transport/HTTP errors
            # Surface the HTTP status (e.g. 429 / 503) when present so callers can
            # distinguish a retryable rate-limit/outage from a hard transport
            # failure. The status code carries no secret; the key/headers/body are
            # never included in the message.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            detail = f" (HTTP {status})" if status is not None else ""
            raise FusionExecutorError(
                f"OpenRouter fusion request failed: {type(exc).__name__}{detail}"
            ) from exc

    def _post_urllib(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        """stdlib fallback transport for the chat/completions POST."""
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except (urllib.error.URLError, ValueError) as exc:
            # Mirror the requests path: a urllib HTTPError carries ``.code`` (the
            # HTTP status); surface it so 429/503 are recoverable from the message.
            # No secret is included (only the status integer).
            status = getattr(exc, "code", None)
            detail = f" (HTTP {status})" if status is not None else ""
            raise FusionExecutorError(
                f"OpenRouter fusion request failed: {type(exc).__name__}{detail}"
            ) from exc

    def _parse_payload(
        self, payload: dict[str, Any], panel: list[str], judge: str | None
    ) -> FusionResult:
        """Parse the OpenRouter fusion tool payload into a FusionResult.

        The tool result is shaped ``{status, analysis?, responses: [...]}``. The
        ``responses[]`` entries are normalized to ``{"model", "content"}``. When
        the judge fails (status ``ok`` with ``analysis`` omitted) the answer is
        synthesized from the panel responses and ``analysis`` is ``None``.
        """
        tool = self._extract_tool_result(payload)

        responses: list[dict[str, Any]] = []
        for item in tool.get("responses", []) or []:
            if isinstance(item, dict):
                responses.append(
                    {"model": item.get("model"), "content": item.get("content", "")}
                )

        raw_analysis = tool.get("analysis")
        analysis: dict[str, Any] | None = None
        answer = tool.get("answer", "")
        if isinstance(raw_analysis, dict):
            analysis = {
                "consensus": raw_analysis.get("consensus"),
                "contradictions": raw_analysis.get("contradictions"),
                "blind_spots": raw_analysis.get("blind_spots"),
            }
            if not answer:
                answer = raw_analysis.get("consensus") or ""
        else:
            # Judge-failure mode: status "ok" but analysis omitted. Synthesize an
            # answer from the panel responses; do not crash.
            answer = self._synthesize_answer(responses)

        cost = tool.get("cost", payload.get("cost"))
        cost_value = float(cost) if isinstance(cost, (int, float)) else None

        return FusionResult(
            answer=answer or "",
            analysis=analysis,
            responses=responses,
            panel=list(panel),
            judge=judge,
            cost=cost_value,
            raw=payload,
        )

    def _extract_tool_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Locate the fusion tool result inside the chat/completions payload.

        Accepts either a top-level tool result (``{status, responses, ...}``) or
        the tool result nested in the first choice's message tool_calls.
        """
        if isinstance(payload, dict) and "responses" in payload:
            return payload

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if choices:
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            tool_calls = message.get("tool_calls") or []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                result = call.get("result")
                if isinstance(result, dict):
                    return result
                func = call.get("function", {})
                args = func.get("arguments") if isinstance(func, dict) else None
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except ValueError:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
                elif isinstance(args, dict):
                    return args

        raise FusionExecutorError(
            "OpenRouter fusion payload contained no parseable tool result."
        )

    @staticmethod
    def _synthesize_answer(responses: list[dict[str, Any]]) -> str:
        """Build a fallback answer from panel responses when the judge fails."""
        parts = [
            str(r.get("content", "")).strip()
            for r in responses
            if str(r.get("content", "")).strip()
        ]
        return "\n\n".join(parts)

    def project_cost(
        self,
        panel: list[str],
        judge: str | None,
        query: str | None = None,
        prompt_tokens: int | None = None,
    ) -> float:
        """Estimate the per-query DOLLAR cost of the panel + judge for the cost guard.

        DOLLARS: the returned value is an estimated per-query dollar cost, NOT a
        relative unit-price proxy. ``input_price`` / ``output_price`` in
        ``llm_data`` are per-million-token prices, so for each panel member plus
        the judge::

            dollars += (input_price * prompt_tokens
                        + output_price * completion_tokens) / 1e6

        ``prompt_tokens`` is taken from the explicit argument when given, else
        estimated from ``query`` as ``max(1, len(query) // 4)`` (~4 chars/token),
        else falls back to ``est_completion_tokens`` when neither is available.
        ``completion_tokens`` is the config-driven ``est_completion_tokens``
        default. The ``cost_ceiling`` comparison in both ``route_single`` and
        ``run`` is made against this dollar projection, so operators set
        ``cost_ceiling`` in dollars per query.
        """
        if prompt_tokens is not None:
            prompt_toks = max(1, int(prompt_tokens))
        elif query is not None:
            prompt_toks = max(1, len(query) // _CHARS_PER_TOKEN)
        else:
            prompt_toks = self.est_completion_tokens
        completion_toks = self.est_completion_tokens

        members = list(panel) + ([judge] if judge else [])
        total = 0.0
        for name in members:
            info = self.llm_data.get(name, {})
            input_price = float(info.get("input_price", 0.0))
            output_price = float(info.get("output_price", 0.0))
            total += (input_price * prompt_toks + output_price * completion_toks) / 1e6
        return total
