# Add FusionGateRouter — a route-vs-fuse meta-router

## Summary

Adds `FusionGateRouter`, a self-contained custom router plugin under
`custom_routers/fusion_gate/` that gates each query between the cheap
single-model path and a multi-model **fusion** path, with fusion delegated to
OpenRouter's `openrouter:fusion` server tool. **Zero edits to core `llmrouter/`
code** — the plugin is auto-discovered via the existing `custom_routers/`
mechanism, exactly like `randomrouter` and `thresholdrouter`.

## Motivation

LLMRouter today picks *which single model* answers a query. The interesting
lever for hard queries is a different one: **route vs. fuse** — decide whether a
query is worth running a panel of models and synthesizing their answers. This PR
makes route-vs-fuse the **primary per-query dial**, expressed as a three-tier
escalation driven by estimated difficulty:

```
single  ->  budget_fusion (cheap panel)  ->  fusion (full Quality panel)
```

Cheap queries stay cheap; only the hard ones escalate, and the middle tier lets
mid-difficulty queries fuse on a budget panel instead of jumping straight to the
full Quality panel.

## What's included

**In scope:**
- `FusionGateRouter` — the route-vs-fuse gate (difficulty + confidence) plus capability-scored panel selection with a Quality/Budget preset fallback.
- An `openrouter:fusion` adapter (`executor.py`) — the single, isolated blast point for the beta server-tool API.
- A configurable surface (`threshold`, `k`, `judge`, `provider`/`base_url`, `panel_preset`, `cost_ceiling`, `est_completion_tokens`) and a `--route-only` spend-free preview that returns the decision + intended panel/judge without any API call.
- A per-query **dollar** cost guard (`cost_ceiling`) that downgrades fusion → single when the projected spend exceeds the cap.
- Secret-scrubbed fusion-call logging (`fusion_log.py`) producing FusionFactory-style `(query, model, response, performance)` training rows.
- A three-arm offline eval harness + bundled fixtures (`eval/`) and an offline retrain step.
- Self-contained: **ONE optional provider** (OpenRouter), **ZERO core edits**.

**Out of scope (follow-ups):**
- **Local fan-out fallback is OUT of this PR.** Without an OpenRouter key only `--route-only` is exercisable. The executor interface is the seam a provider-agnostic local fan-out path would slot behind later — happy to add it if maintainers want it.
- A learned gate (the gate currently uses a duck-typed difficulty estimator with a deterministic lexical fallback so it runs with no trained model).

## Eval results

> **All committed numbers are from MOCK fixtures** (deterministic stub executor,
> zero spend, no network). They validate harness wiring and metric math, **not**
> real model quality. **Real numbers require a keyed live run**
> (`OPENROUTER_API_KEY` / `API_KEYS` set) against a real benchmark slice — that
> path is documented but intentionally not wired into the offline harness so a
> stray run cannot spend. See `eval/RESULTS.md`.

Dataset: 16 held-out queries (6 easy + 10 hard; GSM8K / MATH / GPQA / MBPP).
Quality / blended cost / escalation `p` are over the full 16-query dataset; **gate
precision is computed over the same fixed 10-query hard slice for every arm** so the
arms are comparable (`always_route` makes no escalation decision → N/A). Slice
definitions are documented in `eval/RESULTS.md`. Blended cost is an estimated
**per-query dollar** amount.

| Arm | n | Quality | Blended cost ($/query) | Escalation p | Gate-precision (hard slice) |
|-----|---|---------|------------------------|--------------|------------------------------|
| always_route | 16 | 0.3750 | 0.000650 | 0.0000 | n/a |
| always_fuse | 16 | 1.0000 | 0.001137 | 1.0000 | 1.0000 |
| fusion_gate | 16 | 1.0000 | 0.000767 | 0.6250 | 1.0000 |

- **Quality target** — gate ≥ 95% of always-fuse quality: 1.0000 vs target 0.9500 → **PASS** (mock).
- **Cost target** — blended cost ≤ 1.6× always-route: ratio 1.18 → **PASS** (mock).
- **Gate precision** — escalated answers beating best single, over the hard slice: fusion_gate 10/10, always_fuse 10/10 → **measured** (mock).
- **Retrain delta** — offline log→retrain holds gate-precision at 1.0000 (threshold refit 0.400 → 0.520, budget_threshold 0.100 → 0.180). **Real delta pending a keyed live run.**

## FusionFactory & continual learning

Each fusion call yields a panel of per-model responses plus a judge synthesis —
exactly the `(query, model, response, performance)` observations FusionFactory
needs. `fusion_log.to_training_rows` decomposes them into rows shaped for
`llmrouter/data/api_calling_evaluation.py`, and the retrain step replays the
logged sink to refit the gate thresholds offline. This directly serves the
repo's **continual-learning TODO**: the router's own fusion traffic becomes the
training signal that sharpens the route-vs-fuse gate over time, with no separate
labeling pass required.

## Beta server-tool caveat

`openrouter:fusion` is an OpenRouter **BETA** server tool; its request/response
shape may change. All OpenRouter HTTP specifics are confined to `executor.py`
(request body, tool type, key resolution, transport, payload parsing), so an
upstream beta change touches one file. The executor degrades gracefully on judge
failure (synthesizes from panel responses). No API keys, auth headers, or raw
provider payloads are ever logged.

## Testing

Torch-free, fully offline (HTTP mocked):

```bash
pytest custom_routers/fusion_gate/tests/
python -m custom_routers.fusion_gate.eval.eval_harness --mock --with-retrain \
  --out custom_routers/fusion_gate/eval/out
```
