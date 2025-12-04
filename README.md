<div style="width: 100%;">
    <img src="assets/llmrouter_banner.png" style="width: 100%;"></img>
</div>

<h1 align="center">LLMRouter: An Open-Source Library for LLM Routing</h1>

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/downloads/release/python-3109/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-orange)](https://github.com/ulab-uiuc/LLMRouter/pulls)
[![Slack](https://img.shields.io/badge/Slack-Join%20Us-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/llmrouteropen-ri04588/shared_invite/zt-3jz3cc6d1-ncwKEHvvWe0OczHx7K5c0g)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Introduction

**LLMRouter** is an intelligent routing system designed to optimize LLM inference by dynamically selecting the most suitable model for each query. To achieve intelligent routing, it defines:

1. 🚀 *Smart Routing*: Automatically routes queries to the optimal LLM based on task complexity, cost, and performance requirements.
2. 📊 *Multiple Router Models*: Support for KNN, SVM, MLP, Matrix Factorization, Elo Rating, Graph-based, and more advanced routing strategies.
3. 🔄 *Multi-Round Routing*: Specialized routers for handling complex multi-turn conversations with context awareness.
4. 🛠️ *Unified CLI*: Complete command-line interface for training, inference, and interactive chat with Gradio-based UI.

## Get Started

### Install from pip

You can install LLMRouter from PyPI to use it as a package:
```bash
pip install llmrouter
```

### Install from scratch

Use a virtual environment, e.g. with anaconda3:
```bash
conda create -n llmrouter python=3.10
conda activate llmrouter
pip install -e .
```
### Training a Router

Train various router models with your configuration:
```bash
# Train KNN router
llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml

# Train MLP router with GPU
llmrouter train --router mlprouter --config configs/model_config_train/mlprouter.yaml --device cuda

# Train MF router quietly
llmrouter train --router mfrouter --config configs/model_config_train/mfrouter.yaml --quiet
```

Supported routers for training:
- `knnrouter`: K-Nearest Neighbors Router
- `svmrouter`: Support Vector Machine Router
- `mlprouter`: Multi-Layer Perceptron Router
- `mfrouter`: Matrix Factorization Router
- `elorouter`: Elo Rating Router
- `dcrouter`: Divide-and-Conquer Router
- `automix`: Automix Router
- `hybrid_llm`: Hybrid LLM Router
- `graphrouter`: Graph Router
- `causallm_router`: Causal Language Model Router
- `knnmultiroundrouter`: KNN Multi-Round Router

### Running Inference

Perform inference with trained routers:
```bash
# Single query inference
llmrouter infer --router knnrouter --config config.yaml --query "What is machine learning?"

# Batch inference from file
llmrouter infer --router knnrouter --config config.yaml --input queries.txt --output results.json

# Route only (without calling LLM API)
llmrouter infer --router knnrouter --config config.yaml --query "Hello" --route-only

# Custom generation parameters
llmrouter infer --router knnrouter --config config.yaml --query "Explain AI" --temp 0.7 --max-tokens 2048 --verbose
```

Input file formats supported: `.txt` (one query per line), `.json` (list of strings or objects with `"query"` field), `.jsonl` (one JSON object per line).

### Interactive Chat Interface

Launch a Gradio-based chat interface:
```bash
# Basic chat interface
llmrouter chat --router knnrouter --config config.yaml

# Custom host and port
llmrouter chat --router knnrouter --config config.yaml --host 0.0.0.0 --port 7860

# With public sharing link
llmrouter chat --router knnrouter --config config.yaml --share

# Specify query mode
llmrouter chat --router knnrouter --config config.yaml --mode full_context --top_k 5
```

Query Modes:
- `current_only`: Routes based on current query only (default)
- `full_context`: Combines all chat history with current query
- `retrieval`: Retrieves top-k similar historical queries for context

### Direct Script Execution

You can also run the CLI scripts directly:
```bash
# Training
python -m llmrouter.cli.router_train --router knnrouter --config config.yaml

# Inference
python -m llmrouter.cli.router_inference --router knnrouter --config config.yaml --query "Hello"

# Chat
python -m llmrouter.cli.router_chat --router knnrouter --config config.yaml
```

## Supported Routers

### Standard Routers
| Router | Training | Inference | Description |
|--------|:--------:|:---------:|-------------|
| `knnrouter` | ✅ | ✅ | K-Nearest Neighbors based routing |
| `svmrouter` | ✅ | ✅ | Support Vector Machine based routing |
| `mlprouter` | ✅ | ✅ | Multi-Layer Perceptron based routing |
| `mfrouter` | ✅ | ✅ | Matrix Factorization based routing |
| `elorouter` | ✅ | ✅ | Elo Rating based routing |
| `dcrouter` | ✅ | ✅ | Divide-and-Conquer routing |

### Advanced Routers
| Router | Training | Inference | Description |
|--------|:--------:|:---------:|-------------|
| `automix` | ✅ | ✅ | Automatic model mixing |
| `hybrid_llm` | ✅ | ✅ | Hybrid LLM routing strategy |
| `graphrouter` | ✅ | ❌ | Graph-based routing |
| `causallm_router` | ✅ | ✅ | Causal Language Model router |
| `router_r1` | ❌ | ✅ | Pre-trained Router-R1 model |

### Multi-Round Routers
| Router | Training | Inference | Description |
|--------|:--------:|:---------:|-------------|
| `knnmultiroundrouter` | ✅ | ✅ | KNN router for multi-turn conversations |
| `llmmultiroundrouter` | ❌ | ✅ | LLM-based multi-turn router |

### Baseline Routers
| Router | Training | Inference | Description |
|--------|:--------:|:---------:|-------------|
| `smallest_llm` | ❌ | ✅ | Always routes to smallest model |
| `largest_llm` | ❌ | ✅ | Always routes to largest model |


<p align="center">
    <a href="https://star-history.com/#ulab-uiuc/LLMRouter">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ulab-uiuc/LLMRouter&type=Date&theme=dark" />
            <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ulab-uiuc/LLMRouter&type=Date" />
            <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ulab-uiuc/LLMRouter&type=Date" />
        </picture>
    </a>
</p>
