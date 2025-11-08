# Query-Level Router Benchmark

This folder contains scripts and implementations for running query-level routing benchmarks, including baseline methods and the GraphRouter approach for LLM routing.

## Overview

The query-level routing task involves selecting the most appropriate LLM (Large Language Model) for a given query based on various factors such as performance, cost, and quality. This benchmark provides implementations of several baseline routing methods and an advanced GraphRouter approach.

## Directory Structure

```
query_level/
├── router_bench_baseline/    # Baseline routing methods
│   ├── router_mlp.py        # MLP-based router
│   ├── router_bert.py       # BERT-based router
│   ├── router_svm.py        # SVM-based router
│   ├── router_knn.py        # k-NN-based router
│   └── util.py              # Utility functions for metrics
├── GraphRouter/             # Graph-based router implementation
│   ├── configs/             # Configuration files and embeddings
│   ├── model/               # Graph router model implementation
│   ├── data_processing/     # Data preprocessing utilities
│   ├── run_exp.py          # Main experiment runner
│   └── evaluate.py         # Evaluation script
├── requirements.txt         # Dependencies
└── readme.md               # This file
```

## Data Format

The routing benchmarks expect JSON data with the following structure:

```json
[
  {
    "query": "Your input query here",
    "candidates": [
      {
        "candidate_name": "model_name_1",
        "cost": 0.01,
        "performance": 0.85,
        "score": 0.80
      },
      {
        "candidate_name": "model_name_2",
        "cost": 0.02,
        "performance": 0.90,
        "score": 0.85
      }
    ],
    "ground_truth": "model_name_1"
  }
]
```

## Baseline Methods

### 1. MLP Router (`router_mlp.py`)

A Multi-Layer Perceptron based router that uses sentence embeddings to predict the best LLM for each query.

**Usage:**
```bash
cd router_bench_baseline
python router_mlp.py
```

**Configuration:** Edit the `main()` function to set:
- `train_path`: Path to training data JSON file
- `test_path`: Path to test data JSON file
- `embedding_model`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `hidden_layer_sizes`: MLP architecture
- `learning_rate`: Training learning rate

### 2. BERT Router (`router_bert.py`)

Uses a pre-trained BERT-Router model for query-LLM matching based on cosine similarity of embeddings.

**Usage:**
```bash
cd router_bench_baseline
python router_bert.py
```

**Features:**
- Fine-tuning capability on custom data
- Uses BERT-Router model from Hugging Face
- Cosine similarity-based ranking

### 3. SVM Router (`router_svm.py`)

Support Vector Machine based router using query embeddings for classification.

**Usage:**
```bash
cd router_bench_baseline
python router_svm.py
```

### 4. k-NN Router (`router_knn.py`)

k-Nearest Neighbors based router that finds similar queries from training data.

**Usage:**
```bash
cd router_bench_baseline
python router_knn.py
```

## GraphRouter

An advanced graph neural network based approach that models the relationships between queries, LLMs, and tasks.

### Configuration

Edit `GraphRouter/configs/config.yaml` to configure:

```yaml
# Data paths
train_data_path: 'path/to/train.csv'
test_data_path: 'path/to/test.csv'
llm_description_path: 'configs/new_LLM_Descriptions_with_think.json'
llm_embedding_path: 'configs/new_llm_description_embedding_with_think.pkl'

# Training parameters
train_epoch: 70
learning_rate: 0.0001
batch_size: 640
scenario: "Performance First"  # "Performance First", "Balance", "Cost First"

# Model parameters
llm_num: 40
embedding_dim: 8
edge_dim: 3
```

### Usage

Run GraphRouter experiments:

```bash
cd GraphRouter
python run_exp.py --config_file configs/config.yaml
```

**Prerequisites:**
- Set up Weights & Biases (wandb) account and API key
- Prepare CSV data files with appropriate format
- Configure LLM descriptions and embeddings

## Evaluation Metrics

All baseline methods report the following metrics:
- **Average Performance**: Mean performance score of selected LLMs
- **Average Cost**: Mean cost of selected LLMs  
- **Average Score**: Mean overall score of selected LLMs

Additional metrics available in `util.py`:
- NDCG@k (Normalized Discounted Cumulative Gain)
- Hit@k (Hit ratio at k)
- MRR (Mean Reciprocal Rank)

## Customization

### Adding New Baseline Methods

1. Create a new router class following the pattern in existing scripts
2. Implement `train()`, `evaluate()`, and `predict_rankings()` methods
3. Use the utility functions from `util.py` for consistent evaluation

### Modifying GraphRouter

1. Edit configuration files in `GraphRouter/configs/`
2. Modify model architecture in `GraphRouter/model/`
3. Adjust data processing in `GraphRouter/data_processing/`

## Data Requirements

- **Training data**: JSON format with queries, candidates, and ground truth
- **Test data**: Same format as training data
- **LLM descriptions**: JSON file describing each LLM's capabilities
- **LLM embeddings**: Pre-computed embeddings for LLM descriptions

## Notes

- CUDA support is available for GPU acceleration
- Models cache embeddings for efficiency
- Set appropriate environment variables for GPU selection
- Some methods support incremental training and fine-tuning

## Troubleshooting

1. **CUDA errors**: Check `CUDA_VISIBLE_DEVICES` environment variable
2. **Memory issues**: Reduce batch size or embedding dimensions
3. **Missing data**: Ensure all data paths in config files are correct
4. **Wandb issues**: Verify API key and project settings for GraphRouter
