# FusionGateRouter — eval harness results

> **These numbers are from MOCK fixtures (deterministic stub executor, zero spend).**
> They validate the harness wiring and metric math, NOT real model quality.
> **Real M1–M4 numbers require a keyed live run** (`OPENROUTER_API_KEY` / `API_KEYS`
> set) against a real benchmark slice — see the *Live run* section below.

This file is the **committed, intentional** eval report. The harness also writes a
fresh `results.csv` / `results.md` into the gitignored `eval/out/` directory on every
run; those are runtime output and are never tracked. Regenerate the numbers below
with:

```bash
python -m custom_routers.fusion_gate.eval.eval_harness --mock --with-retrain \
  --out custom_routers/fusion_gate/eval/out
```

- Source: MOCK fixtures (zero spend)
- Dataset: `eval/fixtures/hard_slice.jsonl` (16 held-out queries; GSM8K / MATH / GPQA / MBPP)

## Slice definitions

The dataset mixes EASY and HARD queries (6 easy, 10 hard). Two distinct slices are
used so the metrics are comparable across arms:

- **Full dataset (16 queries)** — drives Quality, Blended cost, and Escalation `p`.
  Every arm is scored over all 16 records.
- **Hard slice (10 queries)** — the fixed, arm-independent set used for the **M3
  gate-precision** metric. A record is *hard* when its `id` carries the `-hard-`
  marker (e.g. `gsm8k-hard-01`); an explicit `difficulty: "hard"` field overrides
  the id heuristic when present. See `eval_harness.is_hard_record`.

**Why the hard slice matters for M3 (apples-to-apples):** M3 asks "among escalated
queries, how often does fusion beat the best single answer?" The `always_fuse` arm
escalates *every* query (easy + hard) while the `fusion_gate` arm escalates *only the
hard ones*. Scoring M3 over each arm's own escalation set would give the two arms
different denominators (16 vs 10) and the numbers would not be comparable. M3 is
therefore computed over the **same hard slice for every arm**. `always_route` makes
no escalation decision, so its M3 is **N/A** (undefined).

## Per-arm metrics

Quality / Blended cost / Escalation `p` are over the full 16-query dataset; M3 is over
the 10-query hard slice.

| Arm | n | Quality | Blended cost ($/query) | Escalation p | Gate-precision (M3, hard slice) |
|-----|---|---------|------------------------|--------------|---------------------------------|
| always_route | 16 | 0.3750 | 0.000650 | 0.0000 | n/a |
| always_fuse  | 16 | 1.0000 | 0.001137 | 1.0000 | 1.0000 |
| fusion_gate  | 16 | 1.0000 | 0.000767 | 0.6250 | 1.0000 |

Blended cost is an estimated **per-query dollar** amount: for each panel member + judge,
`(input_price · prompt_tokens + output_price · completion_tokens) / 1e6`, with
`input_price` / `output_price` the per-million-token prices from `llm_data`,
`prompt_tokens ≈ len(query) // 4`, and `completion_tokens = est_completion_tokens`
(default 512). This is the same projection the `cost_ceiling` guard compares against,
so `cost_ceiling` is set in dollars per query.

## Metric targets

- **M1** — gate quality ≥ 95% of always-fuse quality (hard slice): gate quality 1.0000
  vs target 0.9500 (95% of always-fuse 1.0000); ratio 1.0000 → **PASS**.
- **M2** — blended cost ≤ 1.6× always-route: gate cost 0.000767 vs target 0.001039
  (1.6× always-route 0.000650); ratio 1.1802 → **PASS**.
- **M3** — gate-precision over the hard slice (escalated answers that beat best single):
  fusion_gate 1.0000 (10/10), always_fuse 1.0000 (10/10) → **measured** (same slice for
  both arms; always_route N/A).

## Retrain: gate-precision before vs after

- Source: MOCK fixtures (synthesized fusion log, zero spend)
- Replayed 16 fusion-log entries → 32 graded training rows.
- Routing table augmented: 28 → 60 rows.
- Gate threshold refit: 0.400 → 0.520.
- Gate budget_threshold refit: 0.100 → 0.180 (raised so wasted low-difficulty
  escalations route single).

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| M3 gate-precision (hard slice) | 1.0000 | 1.0000 | +0.0000 |
| Escalated (hard slice) | 10 | 10 | +0 |
| Escalated-and-improved | 10 | 10 | +0 |

> With M3 scored over the fixed hard slice, the mock retrain holds gate-precision at
> 1.0000 (it no longer benefits from the prior easy/hard denominator mismatch). The
> real M3 delta (M4) requires a keyed live run replaying a real fusion-log sink.

## Live run (keyed, real spend)

The committed numbers above are from MOCK fixtures and a deterministic stub executor —
**zero spend, no network**. To produce real M1–M4 numbers you must run keyed against
real models:

```bash
# 1. Provide an OpenRouter key (never commit it):
export OPENROUTER_API_KEY=sk-...        # or: export API_KEYS='{"OpenRouter": "sk-..."}'

# 2. Build the real router from the plugin config and route+fuse a
#    real benchmark slice (GSM8K/MATH/GPQA/MBPP), scoring answers with
#    llmrouter/data/api_calling_evaluation.eval_perf. The real
#    FusionGateRouter + FusionExecutor make the openrouter:fusion calls;
#    all OpenRouter HTTP specifics stay inside executor.py.
#    (This offline harness does NOT make live calls by design.)
```

M4 (the offline log→retrain quality delta) is produced by `retrain.py`; its mock delta
is reported above when `--with-retrain` is passed.
