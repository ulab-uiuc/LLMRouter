# LLM Multi-Round Router

## Overview

The **LLM Multi-Round Router** uses LLM-based reasoning for both query decomposition and routing decisions. Unlike KNN Multi-Round Router, it doesn't require training data - it uses LLM prompts to intelligently decompose queries and select the best model for each sub-query.

## Paper Reference

This router implements multi-round routing as described in:

- **[Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](https://arxiv.org/abs/2506.09033)**
  - Zhang, H., Feng, T., & You, J. (2025). arXiv:2506.09033.
  - Proposes multi-round routing with decomposition and aggregation.

Zero-shot LLM-based routing with decomposition:
- **LLM Reasoning**: Uses language models to make routing decisions
- **Zero-Shot**: No training required, works with prompts only
- **Multi-Agent**: Decomposes and delegates to specialized models

## How It Works

### Architecture

```
Query → LLM Decomposition+Routing → [(Sub-Query 1, Model A), (Sub-Query 2, Model B), ...]
                                            ↓                          ↓
                                     Execute via API            Execute via API
                                            ↓                          ↓
                                    LLM Aggregation ← [Response 1, Response 2, ...]
                                            ↓
                                     Final Answer
```

### Pipeline

1. **Decomposition + Routing (Single LLM Call)**:
   - LLM decomposes query into sub-queries
   - Simultaneously selects best model for each sub-query
   - Based on model descriptions provided in prompt

2. **Execution**: Sub-queries executed with routed models via API

3. **Aggregation**: Base LLM combines responses into final answer

## Key Difference from KNN Multi-Round

- **KNN**: Uses K-nearest neighbors (requires training data)
- **LLM**: Uses LLM reasoning (zero-shot, no training)

## Configuration Parameters

### Hyperparameters (`hparam`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | str | `"Qwen/Qwen2.5-3B-Instruct"` | Base model for decomposition/aggregation/routing |
| `use_local_llm` | bool | `false` | Use local vLLM (true) or API (false) |
| `api_endpoint` | str | - | API endpoint for execution |
| `decomposition_max_tokens` | int | `2048` | Max tokens used by the decomposition + routing step |

### LLM Data

Requires `llm_data` with model descriptions for routing prompts.

## CLI Usage

The LLM Multi-Round Router can be used via the `llmrouter` command-line interface:

### Inference

> **Note**: This router does not require training - it uses zero-shot LLM prompts.

```bash
# Route a single query with decomposition
llmrouter infer --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml \
    --query "Compare neural networks and decision trees"

# Route queries from a file
llmrouter infer --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml \
    --input queries.jsonl --output results.json

# Route only (without calling LLM API)
llmrouter infer --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml \
    --query "Explain machine learning concepts" --route-only
```

### Interactive Chat

```bash
# Launch chat interface
llmrouter chat --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml

# Launch with custom port
llmrouter chat --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml --port 8080

# Create a public shareable link
llmrouter chat --router llmmultiroundrouter --config configs/model_config_test/llmmultiroundrouter.yaml --share
```

---

## Usage Examples

### Inference

```python
from llmrouter.models import LLMMultiRoundRouter

router = LLMMultiRoundRouter(yaml_path="configs/model_config_test/llmmultiroundrouter.yaml")

# Chat mode
response = router.route_single("Compare neural networks and decision trees")
print(response)

# Evaluation mode
query = {"query": "...", "ground_truth": "...", "task_name": "general"}
result = router.route_single(query)
print(f"Response: {result['response']}")
print(f"Performance: {result.get('task_performance', 'N/A')}")
```

## Advantages

- ✅ **No Training Required**: Zero-shot using LLM prompts
- ✅ **Intelligent Routing**: LLM understands model capabilities
- ✅ **Decomposition**: Handles complex multi-faceted queries
- ✅ **Flexible**: Works with any models in llm_data

## Limitations

- ❌ **High Cost**: Multiple LLM calls (decomposition + aggregation)
- ❌ **High Latency**: Sequential LLM generations
- ❌ **Prompt Sensitivity**: Routing quality depends on prompt engineering
- ❌ **No Learning**: Cannot improve from historical data

## When to Use

**Good For:**
- No training data available
- Need zero-shot routing solution
- Complex queries requiring decomposition
- Model capabilities well-described in metadata

**Alternatives:**
- Have training data → KNN Multi-Round Router
- Simple queries → Single-round routers
- Cost-sensitive → Avoid multi-round approaches

## Related Routers

- **KNN Multi-Round Router**: Trained alternative
- **Router-R1**: Agentic reasoning with external routing pool
- **Causal LM Router**: Finetuned LLM for routing (single-round)

---

For questions or issues, please refer to the main LLMRouter documentation or open an issue on GitHub.
