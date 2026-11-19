# Multi-Head Latent Control (MHLC) router

MHLC generates a candidate answer with a frozen target model. Its Capability Head reads the hidden states of the completed answer and estimates an adequacy score in `[0, 1]`. LLMRouter returns the candidate when `score >= threshold`. A lower score routes the original question to the configured fallback. The target generates the answer once and shares its resident vLLM engine with the scoring pass. [MHLC paper](https://arxiv.org/abs/2607.14277) · [Original implementation](https://github.com/Amirhosein-gh98/Multi-Head-Latent-Control)

The Qwen3.5-4B preset covers mixed-source generation, correctness labeling, head training, routing inference, and a two-model TriviaQA benchmark. The registered router is `gnosis_latent_router`, with the short alias `mhlc`. The examples below run from an LLMRouter checkout.

## Installation

Use a Linux CUDA environment with a vLLM-compatible GPU. Install the GPU extra in its own environment because the RouterR1 extra pins a different vLLM version. Python 3.12 is used in the example.

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -e ".[gnosis]"
```

The full preset loads Qwen3.5-4B for generation and training, a 30B-A3B FP8 judge for labels requiring judgment, and Qwen3.5-9B as the benchmark fallback. Select hardware that supports the judge's dtype or set `--judge-model-id` and `--judge-revision` to an appropriate instruction model and record that choice with your results. Use `HF_HOME` to place model and dataset downloads on a disk with sufficient space. All pipeline commands use the active Python interpreter unless `pipeline.python` is set.

## Run a two-question GPU smoke test

This example builds a small Qwen3.5 model whose answer is always `4`. It generates completions for `2 + 2` and `2 + 3`, grades them, trains the head, and routes a third question. The expected labels are 1 and 0. Only the tokenizer and model configuration are downloaded. Use held-out labeled data to measure routing quality.

```bash
work=$(mktemp -d)
python examples/mhlc/prepare_smoke.py "$work"
llmrouter mhlc run --config "$work/router.yaml" --query "What is 3 + 1?" --route-only
```

For an offline run, pass `--source /path/to/Qwen3.5-4B-snapshot` to the preparation script. Set `--gpu-memory-utilization` there to adjust the fraction of GPU memory available to each smoke stage. The final output includes a selected model and a capability score. Inspect `$work/run/verified/` for the labels and `$work/head/aux_head_final.pt` for the trained head.

## Run the mixed-source pipeline

Use a new workspace for each model, thinking mode, or source selection. Set the same workspace for training and inference. The workspace defaults to the checkout for the supplied preset.

```bash
export GNOSIS_WORKSPACE="$PWD/mhlc-workspace"
mkdir -p "$GNOSIS_WORKSPACE"
config=configs/model_config_train/gnosis_qwen35_4b.yaml

# Show the resolved commands and paths.
llmrouter mhlc run --config "$config" --dry-run

# Run all four stages in order.
llmrouter mhlc run --config "$config"
```

| Stage | Input and operation | Output in the workspace |
| --- | --- | --- |
| `generate` | Selects 40,000 questions across FineVision, MM-OpenR1, DAPO-Math, TriviaQA, AGUVIS, and APIGen. Generates candidates with the pinned Qwen3.5-4B revision and thinking off. | `{run_root}/raw/*.parquet`, selection manifest, generation statistics |
| `label` | Applies source-specific rules and uses a local Qwen judge when the rules require judgment. Preserves the prompt, images, completion, source identifiers, and continuous correctness score. | `{run_root}/verified/*.parquet`, verification statistics |
| `train` | Replays the completed chat, aligns processor token IDs to vLLM hidden states, and optimizes the Capability Head with weighted mean squared error. | Periodic checkpoints, `aux_head_final.pt`, and the effective training config in the recipe's `output_dir` |
| `benchmark` | Compares the target alone, fallback alone, and threshold-based handoff on the TriviaQA validation split. | Per-strategy generations, scores, summaries, and run manifest under `eval_outputs/` |

Run any stage individually, for example `llmrouter mhlc label --config "$config"`. Standard training is also available through `llmrouter train --router mhlc --config "$config"`. Each subprocess streams its output to the terminal and appends it to `logs/gnosis/<stage>.log` in the workspace.

Generation and labeling resume from completed Parquet shards. Keep the source selection, model revision, and judge consistent across retries. If a judge request exceeds its context or produces an unusable score, inspect the stage log and adjust the judge settings before retrying. For legacy raw shards without generation provenance, use them as labeling input or start generation in a fresh workspace.

## Route a question

```bash
llmrouter infer --router mhlc --config "$config" --route-only \
  --query "Why do eclipses happen?"
```

Route-only still runs local generation and head scoring. The result includes `model_name`, `candidate_response`, `gnosis_score`, `threshold`, and `accepted`. To receive the final answer, omit `--route-only`. Accepted candidates are returned immediately. On escalation, LLMRouter sends the question to the fallback endpoint.

The example catalog points at an OpenAI-compatible local service at `http://localhost:8000/v1` serving `Qwen/Qwen3.5-9B`. Start that service on hardware with room for the fallback, or edit `data_path.llm_data` and its catalog entry to point to your deployment. For a local service without authentication, set `export API_KEYS='{"local":[""]}'`. [Example model catalog](../../../data/gnosis_latent_router_llm_data.json)

Choose the threshold on held-out questions from the intended deployment. A higher threshold escalates more requests. Report the accepted fraction and final answer quality alongside fallback usage. The example threshold of 0.5 is a starting value.

## Data, recipes, and checkpoints

- **Custom raw inputs.** Add `--source <name>` and `--source-file /path/to/file.parquet` to the generation command in a copied config. Use the selected source's schema. The smoke example demonstrates DAPO's `prompt` and `solution` fields. TriviaQA uses `question` and `answer.value`. Without a local file, the generator loads the selected source from Hugging Face.
- **Already labeled inputs.** Set the training recipe's `dataset_path` to a verified Parquet directory, Hugging Face dataset saved to disk, or a JSON/JSONL file. A text record can contain `{"question": "What is 2 + 2?", "completion": "4", "correctness_score": 1.0}`. Multimodal rows also carry `images`. Preserve source-specific fields such as `system_prompt`, `human_prompt`, `tools`, and `context_text` for records that use them.
- **Training recipe.** The default uses the last hidden layer, completion text masking, two epochs, microbatch size 1, gradient accumulation 16, and a learning rate of `1e-4`. To customize, copy the recipe, update `pipeline.variables.training_recipe`, and keep `dataset_path`, `output_dir`, and `pipeline.variables.checkpoint` aligned. The trainer logs optimization-stream metrics. Reserve separate data for evaluation. [Thinking-off recipe](backend/recipes/training/qwen3_5_4b_think_off.yaml)
- **Model changes.** Match the model ID, revision, family, and thinking mode across generation, training, benchmarking, and inference. The packaged thinking-on recipe uses its own 120k run path. Give a new model or mode a new workspace and choose a matching judge and fallback. [Thinking-on recipe](backend/recipes/training/qwen3_5_4b_think_on.yaml)
- **Checkpoint compatibility.** Use a checkpoint produced by this vLLM trainer. The public paper heads use Transformers/Unsloth hidden states, which differ from vLLM's extracted features. Use the original implementation for those weights. The router reads the trained layer selection and token mask from the checkpoint. Version 2 scores answer tokens with the training template and image settings. Existing vLLM version 1 heads retain their original masking behavior. [Source and adaptations](backend/VENDORED_SOURCE.md)

The vLLM connector writes temporary tensors to shared memory and the reader removes them after scoring. Set `GNOSIS_HIDDEN_STATES_DIR` to a fast writable filesystem when shared memory is limited. Set `inference.sampling.max_new_tokens` to cap target generation and `--max-tokens` on the inference command to cap the fallback response. [Full pipeline config](../../../configs/model_config_train/gnosis_qwen35_4b.yaml)
