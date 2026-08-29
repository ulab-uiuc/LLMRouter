# Benchmark Pipeline — train & evaluate every router on xRouteBench

One-command pipeline that trains and evaluates LLMRouter's routers on the
[xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench) benchmark
(8 datasets), including cost-aware training with a GraphRouter-style composite
reward.

## Quick start

```bash
# 1. Get the data (the dataset is public, no token needed)
python download_data.py

# 2. Generate query embeddings (Qwen3-Embedding-0.6B, needs one GPU)
python generate_embeddings.py

# 3. Run the sweep — all local routers x all 8 datasets, pure-performance
python run_pipeline.py --datasets all --routers local

# 4. Aggregate into tables
python aggregate_results.py --csv
```

Results land in `results/<dataset>_<router>_a<alpha>_b<beta>.json`.
Completed pairs are skipped on re-run, so the sweep is resumable.

## Datasets (8)

| Dataset | Domain | Test queries |
|---|---|---|
| `llmrouter_generic` | 13 classic NLP benchmarks (MMLU, GSM8K, MATH, MBPP, …) | 3,729 |
| `memory_locomo` | Long-conversation memory QA (RAG top-k=5) | 314 |
| `memory_longmemeval` | Long-term memory eval (RAG top-k=5) | 101 |
| `timeseries` | Time-series understanding (7 sub-tasks) | 127 |
| `video` | Egocentric video QA (Charades-Ego) | 27 |
| `multimodal_geometry3k` | Geometry math | 61 |
| `multimodal_mathvista` | Visual math reasoning | 100 |
| `personalized` | Personalized preference (chat-format queries) | 303 |

Every query was pre-executed against all 18 candidate LLMs, so **evaluation
replays recorded outcomes — no API calls or cost** for the local router
category.

## Routers (17)

| Category | Routers | Notes |
|---|---|---|
| Embedding-based | `knnrouter` `svmrouter` `mlprouter` `mfrouter` `elorouter` `graphrouter` `routerdc` `hybrid_llm` | trained on query embeddings |
| Baselines | `largest_llm` `smallest_llm` | no training |
| Generic trainable | `gmtrouter` `personalizedrouter` | trained via the repo registry |
| Heavy | `causallm_router` | LoRA-finetunes Llama-2-7b + vLLM inference; runs train/infer as separate processes (a single process OOMs one 48GB GPU); needs `HF_TOKEN` with Llama-2 access |
| API-calling | `knnmultiroundrouter` `llmmultiroundrouter` `router_r1` `automix` | make live LLM calls at inference; enable with `--include-api-routers`, set `LLM_API_KEY` (+ `ROUTER_R1_API_BASE` for router_r1) |

`--routers local` = the first four categories (13 routers, zero API cost).

## Cost-aware (Pareto) training

```bash
# reward = alpha * norm(performance) - beta * norm(price_cost)
for a in 1.0 0.8 0.6 0.4 0.2; do
  b=$(python -c "print(round(1-$a,1))")
  python run_pipeline.py --datasets all --routers local --alpha $a --beta $b
done
python aggregate_results.py
```

Training data's `performance` column is replaced by the composite reward
(per-column min-max normalization), so any router that "picks the best model
per query" automatically becomes cost-aware. Price cost uses the per-model
prices in `data/llm_candidates/default_llm.json`:
`cost = input_tokens x input_price/1e6 + output_tokens x output_price/1e6`.

## Output format

```json
{
  "router": "graphrouter", "dataset": "timeseries", "alpha": 0.6, "beta": 0.4,
  "avg_performance": 0.4724, "avg_token_cost": 183.9,
  "avg_price_cost": 2.9e-05, "avg_reward": 0.17,
  "num_queries": 127, "routing_distribution": {"gpt-oss-20b": 127}
}
```

## Requirements

- Core: `llmrouter` package (this repo) + `datasets` (HF download)
- `generate_embeddings.py` / `graphrouter` / `routerdc` / `mlprouter` / `mfrouter`: GPU recommended
- `causallm_router`: GPU (>=40GB), `vllm`, `peft`, gated `meta-llama/Llama-2-7b-hf` access
- API routers: provider API key(s); these calls spend real money
