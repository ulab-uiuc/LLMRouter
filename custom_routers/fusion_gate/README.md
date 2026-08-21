# FusionGateRouter

**Type:** Meta-router (route-vs-fuse gate). No training required to run; an optional offline retrain step refits the gate from logged fusion calls.

**Description:** A per-query gate that decides between the cheap **single-model**
path (classic LLMRouter routing) and a **fusion** path that runs a panel of
models and synthesizes their answers. Fusion is delegated to the OpenRouter
`openrouter:fusion` server tool (BETA — see the caveat below). Routing is
spend-free: the decision is computed locally and only `fuse()` ever calls the
provider.

The primary per-query dial is **route vs. fuse**, expressed as three tiers:

```
difficulty < budget_threshold          ->  single         (cheapest single model)
budget_threshold <= difficulty < threshold  ->  budget_fusion  (cheap Budget panel)
difficulty >= threshold                ->  fusion         (full Quality panel)
```

Set `budget_threshold: null` (or `>= threshold`) to disable the middle tier and
collapse to plain single/fusion. A `high_stakes: true` flag on a query forces
the full Quality `fusion` tier regardless of difficulty.

## Usage

```bash
# Inference (routes, then fuses via openrouter:fusion if the gate escalates)
llmrouter infer --router fusion_gate \
  --config custom_routers/fusion_gate/config.yaml \
  --query "Prove that the square root of 2 is irrational."

# Route-only — compute the decision with ZERO spend / no network call
llmrouter infer --router fusion_gate \
  --config custom_routers/fusion_gate/config.yaml \
  --query "What is the capital of France?" \
  --route-only
```

`--route-only` returns the decision dict (tier, panel, judge, projected cost)
without ever calling OpenRouter. Spend happens only when `fuse()` is invoked.

## Decision contract

`route_single` returns one of two shapes (both carry `strategy`, `tier`, and
`model_name` for drop-in CLI compatibility):

- **single:** `{query, strategy="single", tier="single", model_name, predicted_llm, difficulty, confidence}`
- **fusion:** `{query, strategy="fusion", tier="budget_fusion"|"fusion", panel[], judge, model_name, predicted_llm, difficulty, confidence, projected_cost}`

When the cost guard fires, a fusion decision is **downgraded** to single and the
result carries `downgraded_from`, `projected_cost`, and `cost_ceiling`.

## Configuration

All keys live under `hparam:` in `config.yaml` unless noted.

| Key | Default | Purpose |
|-----|---------|---------|
| `threshold` | `0.5` | Difficulty cutoff to escalate to the full Quality `fusion` tier. |
| `budget_threshold` | `0.3` | Lower boundary of the middle `budget_fusion` tier. `null` (or `>= threshold`) disables it. |
| `k` | `3` | Panel size — maps to the tool's `analysis_models`. |
| `judge` | `null` | Judge model slug — maps to the tool's `model`. `null` = use the outer model. |
| `panel_preset` | `Quality` | Fallback preset (`Quality` / `Budget`) when capability data is unavailable for a query. |
| `cost_ceiling` | `null` | Hard per-query **dollar** cap on the projected `Σ(panel)+judge` cost. `null` = off. See the cost-unit note. |
| `est_completion_tokens` | `512` | Per-completion output-token estimate feeding the dollar cost projection. |
| `provider` | `OpenRouter` | Informational; drives credential resolution. |
| `base_url` | `https://openrouter.ai/api/v1` | OpenRouter endpoint hosting the beta server tool. Overrides the top-level `api_endpoint`. |
| `log_sink_path` | `null` | JSONL sink for fusion-call logging. `null` = `fusion_log` default (`~/.llmrouter/openclaw_memory.jsonl`). |

Top-level `data_path` / `metric` keys mirror the other custom routers
(`randomrouter`, `thresholdrouter`); see `config.yaml` for the loaded candidate
and routing-data paths.

### Cost-unit note (important)

`cost_ceiling` is compared against `project_cost`, which estimates the **per-query
dollar cost** of the panel + judge. For each member,
`(input_price · prompt_tokens + output_price · completion_tokens) / 1e6`, where
`input_price` / `output_price` are the per-million-token prices from `llm_data`,
`prompt_tokens ≈ len(query) // 4`, and `completion_tokens = est_completion_tokens`
(default `512`). Set `cost_ceiling` in **dollars per query** (e.g. `0.05` ≈ five
cents per query).

## Panel selection

Panels are chosen by `CapabilityScorer`, which scores candidates per **query
category** (code / math / reasoning / general) from the LLMRouter routing-data
tables, lightly cost-penalized. When no usable capability data exists for a
query's category, selection falls back to a preset panel resolved by tier:
`budget_fusion` -> `Budget`, anything else -> the configured `panel_preset`
(`Quality` by default). The tier->preset mapping (`gate.resolve_preset`) is the
single source of truth shared with the eval harness.

## OpenRouter `openrouter:fusion` — BETA caveat

The fusion path depends on OpenRouter's `openrouter:fusion` **server tool, which
is BETA**: its request/response shape may change without notice. To contain that
risk, **every OpenRouter HTTP specific lives in `executor.py` and nowhere else**
— request body construction, the `openrouter:fusion` tool type, key resolution,
transport, and payload parsing. An upstream beta change should touch that one
file only. The executor also tolerates judge failure (status `ok` with
`analysis` omitted): it synthesizes the answer from the panel responses rather
than crashing.

OpenRouter is the **one optional provider**. There is no local fan-out fallback
(deferred to a follow-up); without a key, only `--route-only` is exercisable.

## Logging

Every `fuse()` call is appended (best-effort, append-only) to the JSONL sink via
`fusion_log.log_fusion`. The sink is **secret-scrubbed**: API keys, auth
headers, cookies, and the untouched provider payload are never written; only an
enumerated set of fields (query, panel, judge, normalized responses, analysis,
token/cost) is emitted. These rows are the FusionFactory-style training signal
consumed by the offline retrain step.

## Offline evaluation (`--mock`, zero spend)

The three-arm harness compares `always_route`, `always_fuse`, and `fusion_gate`
over a bundled hard-query slice (GSM8K / MATH / GPQA / MBPP). It is **offline by
default** — a deterministic stub executor reads canned answers from fixtures; no
network call is made and nothing is spent.

```bash
# Run the offline harness (mock is the default)
python -m custom_routers.fusion_gate.eval.eval_harness --mock \
  --out custom_routers/fusion_gate/eval/out

# Include the mock retrain (M3 before/after) delta in results.md
python -m custom_routers.fusion_gate.eval.eval_harness --mock --with-retrain \
  --out custom_routers/fusion_gate/eval/out
```

Tunable flags: `--threshold` (0.5), `--budget-threshold` (0.3), `--k` (2 in the
harness — kept cost-bounded for the M2 target; the plugin config uses `k=3`),
`--judge`, `--panel-preset`, `--dataset`, `--llm`, `--routing`, `--out`.
Outputs: `<out>/results.csv` and `<out>/results.md` (the `--out` dir defaults to
`eval/out/`, which is **gitignored** — runtime output, not source). The committed,
intentional report lives at [`eval/RESULTS.md`](eval/RESULTS.md), which also documents
the full-dataset vs hard-slice definitions used by the metrics.

`--live` is intentionally **not** wired into this harness, so a stray run cannot
spend; passing it errors out with a pointer to the keyed live-run path.

Run the unit tests (torch-free, fully offline, HTTP mocked):

```bash
pytest custom_routers/fusion_gate/tests/
```

## Live run (keyed, real spend)

The committed eval numbers are from MOCK fixtures. To produce real M1–M4 numbers
you must run keyed against real models:

```bash
# Provide an OpenRouter key (never commit it):
export OPENROUTER_API_KEY=sk-...           # or: export API_KEYS='{"OpenRouter": "sk-..."}'

# Then build the real FusionGateRouter from config.yaml and route+fuse a real
# benchmark slice; the executor makes the openrouter:fusion calls. The offline
# eval harness does NOT make live calls by design — see eval/RESULTS.md.
```

Keys are resolved (in order) from an explicit `api_keys={"OpenRouter": "..."}`
dict, `OPENROUTER_API_KEY`, or an `API_KEYS` JSON env var. Keys are never logged.

## Files

- `router.py` — `FusionGateRouter` entry point (MetaRouter contract).
- `gate.py` — `RouteGate`, `GateDecision`, the three-tier dial, `resolve_preset`.
- `capability.py` — `CapabilityScorer` panel selection.
- `executor.py` — **the only** OpenRouter `openrouter:fusion` blast point.
- `fusion_log.py` — secret-scrubbed JSONL logging + training-row decomposition.
- `eval/` — three-arm offline harness, fixtures, retrain, and `RESULTS.md` (the committed report; `eval/out/` is gitignored runtime output).
- `tests/` — torch-free offline unit tests.
