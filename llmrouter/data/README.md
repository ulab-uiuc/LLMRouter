# LLMRouter Data Generation Pipeline

This directory contains scripts for generating training and evaluation data for LLMRouter. The pipeline consists of three main steps that transform raw benchmark datasets into formatted routing data with embeddings.

**🚀 Quick Start**: Begin with [`sample_config.yaml`](sample_config.yaml) - a ready-to-use configuration file that references the example data directory. See [Step 1: Configuration Setup](#step-1-configuration-setup) for details.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Step-by-Step Pipeline](#step-by-step-pipeline)
- [Input File Formats](#input-file-formats)
- [Output File Formats](#output-file-formats)
- [Embedding Mapping System](#embedding-mapping-system)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)

---

## Pipeline Overview

The data generation pipeline follows this flow:

```
Step 1: Configuration (YAML)
    ↓
Step 2a: Generate Query Data → query_data_train.jsonl + query_data_test.jsonl
    ↓
Step 2b: Generate LLM Embeddings → default_llm_embeddings.json
    ↓
Step 3: API Calling & Evaluation → 
    - query_embeddings_longformer.pt (unified embeddings)
    - default_routing_train_data.jsonl
    - default_routing_test_data.jsonl
```

### Key Features

- **Unified Embeddings**: One `.pt` file contains embeddings for all queries (train + test)
- **Embedding ID Mapping**: Sequential IDs (0, 1, 2, ...) map directly to line numbers in the `.pt` file
- **Config-Driven**: All paths and parameters controlled via YAML configuration
- **Format Consistency**: Output formats match sample files exactly

---

## Step-by-Step Pipeline

### Step 1: Configuration Setup

**Start with the sample configuration file**: `llmrouter/data/sample_config.yaml`

This file contains all the necessary paths and parameters. You can use it as-is or copy and modify it for your own setup.

```bash
# Copy the sample config to your working directory
cp llmrouter/data/sample_config.yaml my_config.yaml

# Edit paths as needed
# Then use it with any of the pipeline scripts
```

**Sample Configuration Structure**:

```yaml
data_path:
  query_data_train: 'data/example_data/query_data/default_query_train.jsonl'
  query_data_test: 'data/example_data/query_data/default_query_test.jsonl'
  query_embedding_data: 'data/example_data/routing_data/query_embeddings_longformer.pt'
  routing_data_train: 'data/example_data/routing_data/default_routing_train_data.jsonl'
  routing_data_test: 'data/example_data/routing_data/default_routing_test_data.jsonl'
  llm_data: 'data/example_data/llm_candidates/default_llm.json'
  llm_embedding_data: 'data/example_data/llm_candidates/default_llm_embeddings.json'

data_generation:
  sample_size: 500  # Number of samples per task
  train_ratio: 0.8  # Train/test split ratio
  random_seed: 42
```

**Quick Start**: Use the sample config directly:
```bash
python llmrouter/data/data_generation.py --config llmrouter/data/sample_config.yaml
python llmrouter/data/generate_llm_embeddings.py --config llmrouter/data/sample_config.yaml
python llmrouter/data/api_calling_evaluation.py --config llmrouter/data/sample_config.yaml
```

### Step 2a: Generate Query Data (`data_generation.py`)

**Purpose**: Extract queries from benchmark datasets and create train/test split JSONL files.

**Input**: None (loads datasets directly from HuggingFace/local paths)

**Output**: 
- `query_data_train.jsonl` - Training query data
- `query_data_test.jsonl` - Test query data

**Usage**:
```bash
# Using config file (recommended)
python llmrouter/data/data_generation.py --config llmrouter/data/sample_config.yaml

# OR using command-line arguments
python llmrouter/data/data_generation.py --sample 100 \
    --output_train data/query_train.jsonl \
    --output_test data/query_test.jsonl
```

**What it does**:
1. Loads samples from 11 benchmark datasets (Natural QA, Trivia QA, MMLU, GPQA, MBPP, HumanEval, GSM8K, CommonsenseQA, MATH, OpenbookQA, ARC-Challenge)
2. Normalizes data format across different dataset structures
3. Splits data into train/test sets (default 80/20)
4. Saves as JSONL files matching `StandardQueryData` format

### Step 2b: Generate LLM Embeddings (`generate_llm_embeddings.py`)

**Purpose**: Generate embeddings for LLM candidates from their metadata.

**Input**: `default_llm.json` - LLM metadata file

**Output**: `default_llm_embeddings.json` - LLM metadata with embeddings

**Usage**:
```bash
# Using config file (recommended)
python llmrouter/data/generate_llm_embeddings.py --config llmrouter/data/sample_config.yaml

# OR using command-line arguments
python llmrouter/data/generate_llm_embeddings.py \
    --input data/example_data/llm_candidates/default_llm.json \
    --output data/example_data/llm_candidates/default_llm_embeddings.json
```

**What it does**:
1. Reads LLM metadata from JSON file
2. Generates embeddings for each LLM using the `feature` field description
3. Adds `embedding` field to each LLM entry
4. Saves updated JSON with embeddings

### Step 3: API Calling & Evaluation (`api_calling_evaluation.py`)

**Purpose**: Call LLM APIs, evaluate responses, and generate unified embeddings + routing data.

**Input**: 
- `query_data_train.jsonl` and `query_data_test.jsonl` (from Step 2a)
- `default_llm.json` (for model configuration)

**Output**:
- `query_embeddings_longformer.pt` - Unified embeddings for all queries
- `default_routing_train_data.jsonl` - Training routing data with model responses
- `default_routing_test_data.jsonl` - Test routing data with model responses

**Usage**:
```bash
# Set API keys as environment variable
# Service-specific dict format (recommended for multiple providers):
export API_KEYS='{"NVIDIA": "key1,key2", "OpenAI": ["key3", "key4"]}'
# OR legacy formats:
export API_KEYS='["key1", "key2", ...]'  # JSON array format
export API_KEYS='key1,key2,...'  # Comma-separated

# Run with sample config
python llmrouter/data/api_calling_evaluation.py --config llmrouter/data/sample_config.yaml --workers 100
```

**What it does**:
1. Loads query data from train and test JSONL files
2. For each query, calls all LLM candidates via LiteLLM Router (load balancing)
3. Evaluates responses using task-specific metrics
4. Generates embeddings for all unique queries (train + test together)
5. Creates unified `.pt` file with sequential embedding IDs
6. Maps `embedding_id` to routing data records
7. Saves routing data JSONL files with all fields

---

## Input File Formats

### Query Data JSONL (`query_data_train.jsonl` / `query_data_test.jsonl`)

**Format**: JSON Lines (one JSON object per line)

**Required Fields**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `task_name` | string | Task/dataset identifier | `"gsm8k"`, `"mmlu"`, `"mbpp"` |
| `query` | string | The query/question text | `"What is 2+2?"` |
| `ground_truth` | string | Correct answer/expected output | `"4"` or `"A"` |
| `metric` | string | Evaluation metric to use | `"GSM8K"`, `"em_mc"`, `"code_eval"` |
| `choices` | string \| null | JSON string of choices (for multiple choice) | `'{"text": ["A", "B"], "labels": ["A", "B"]}'` or `null` |
| `task_id` | string \| null | Task identifier (for code tasks) | `"HumanEval/0"` or `null` |

**Example**:
```json
{
  "task_name": "gsm8k",
  "query": "Janet has 4 apples. She gives 2 to Bob. How many does she have left?",
  "ground_truth": "2",
  "metric": "GSM8K",
  "choices": null,
  "task_id": null
}
```

**Multiple Choice Example**:
```json
{
  "task_name": "mmlu",
  "query": "What is the capital of France?",
  "ground_truth": "A",
  "metric": "em_mc",
  "choices": "{\"text\": [\"Paris\", \"London\", \"Berlin\"], \"labels\": [\"A\", \"B\", \"C\"]}",
  "task_id": null
}
```

**Note**: The `choices` field is stored as a JSON string (not a JSON object) to match the sample format.

### LLM Data JSON (`default_llm.json`)

**Format**: JSON object with LLM names as keys

**Required Fields** (per LLM):

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `size` | string | Model size | `"7B"`, `"70B"` |
| `feature` | string | Human-readable description | `"Qwen2.5-7B-Instruct represents..."` |
| `input_price` | float | Cost per million input tokens | `0.20` |
| `output_price` | float | Cost per million output tokens | `0.20` |
| `model` | string | API model identifier | `"qwen/qwen2.5-7b-instruct"` |
| `service` | string | Service provider | `"NVIDIA"` |
| `api_endpoint` | string | API endpoint URL for this model | `"https://integrate.api.nvidia.com/v1"` |

**Note on `api_endpoint`**: Required field specifying the base URL for API calls. If not specified here, routers fall back to `api_endpoint` in their YAML config. If neither exists, an error is raised. This allows different models to use different API providers. See [main README](../README.md#configuring-api-endpoints-) for details.

**Note on OpenAI-compatible gateways**: Gateways are configured with the same fields - set `service` to the name used in `API_KEYS`, `api_endpoint` to the gateway base URL (e.g. `https://api.edenai.run/v3` for Eden AI), and `model` to the gateway's own model identifier (e.g. `<provider>/<model>`). See [main README](../README.md#-using-eden-ai-openai-compatible-gateway) for a full Eden AI example.

**Example**:
```json
{
  "qwen2.5-7b-instruct": {
    "size": "7B",
    "feature": "Qwen2.5-7B-Instruct represents an upgraded version...",
    "input_price": 0.20,
    "output_price": 0.20,
    "model": "qwen/qwen2.5-7b-instruct",
    "service": "NVIDIA",
    "api_endpoint": "https://integrate.api.nvidia.com/v1"
  }
}
```

**API Endpoint Resolution**: Per-model `api_endpoint` (this field) → router YAML `api_endpoint` → error if missing. This allows different models to use different providers. See [main README](../README.md#configuring-api-endpoints-) for details.

---

## Output File Formats

### Routing Data JSONL (`default_routing_train_data.jsonl` / `default_routing_test_data.jsonl`)

**Format**: JSON Lines (one JSON object per line)

**Fields**: All fields from query data PLUS the following:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `model_name` | string | LLM model that generated the response | `"llama3-chatqa-1.5-8b"` |
| `response` | string | Model's response text | `"The answer is 4."` |
| `token_num` | int | Total tokens used (input + output) | `453` |
| `input_tokens` | int | Number of input tokens | `449` |
| `output_tokens` | int | Number of output tokens | `4` |
| `response_time` | float | API response time in seconds | `1.7864494324` |
| `api_key_used` | string | API key identifier (if available) | `"rivTkKeBPm"` or `""` |
| `performance` | float | Evaluation score (0.0 to 1.0) | `0.95` |
| `embedding_id` | int | ID mapping to embeddings .pt file | `61` |
| `user_id` | null | Reserved for future use | `null` |
| `fig_id` | null | Reserved for future use | `null` |

**Example**:
```json
{
  "task_name": "gsm8k",
  "query": "Janet has 4 apples. She gives 2 to Bob. How many does she have left?",
  "ground_truth": "2",
  "metric": "GSM8K",
  "choices": null,
  "task_id": null,
  "model_name": "llama3-chatqa-1.5-8b",
  "response": "Janet has 4 apples and gives 2 to Bob, so she has 4 - 2 = 2 apples left.",
  "token_num": 453,
  "input_tokens": 449,
  "output_tokens": 4,
  "response_time": 1.7864494324,
  "api_key_used": "",
  "performance": 1.0,
  "embedding_id": 42,
  "user_id": null,
  "fig_id": null
}
```

**Key Points**:
- Each query appears multiple times (once per LLM candidate)
- `embedding_id` is consistent across all model responses for the same query
- `performance` is computed using task-specific evaluation metrics
- `choices` remains as JSON string format

### Query Embeddings PyTorch File (`query_embeddings_longformer.pt`)

**Format**: PyTorch dictionary (saved via `torch.save()`)

**Structure**: Dictionary mapping `embedding_id` (int) → embedding tensor (torch.Tensor)

**Key Properties**:
- **Sequential IDs**: Embedding IDs start from 0 and increment sequentially (0, 1, 2, 3, ...)
- **Line Number Mapping**: `embedding_id` corresponds to the position in the dictionary
- **Unified Storage**: Contains embeddings for ALL unique queries (both train and test)
- **Tensor Format**: Each embedding is a `torch.FloatTensor` with shape `[embedding_dim]`

**Loading Example**:
```python
import torch

# Load embeddings
embeddings = torch.load("query_embeddings_longformer.pt")

# Access embedding by ID
embedding_id = 42
query_embedding = embeddings[embedding_id]  # Returns torch.Tensor

# Get embedding dimension
embedding_dim = embeddings[0].shape[0]  # e.g., 768
```

**Important**: The same query in train and test data will have the **same** `embedding_id` because embeddings are generated for unique queries only.

### LLM Embeddings JSON (`default_llm_embeddings.json`)

**Format**: Same structure as `default_llm.json` with added `embedding` field

**Additional Field**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `embedding` | array | Embedding vector (list of floats) | `[0.042, 0.090, -0.018, ...]` |

**Note**: This file contains all fields from `default_llm.json` including `api_endpoint`, plus the `embedding` field. The `api_endpoint` field works the same way as in `default_llm.json` - it specifies the API endpoint URL for each model and follows the same resolution priority (per-model endpoint → router config endpoint → error).

**Example**:
```json
{
  "qwen2.5-7b-instruct": {
    "feature": "Qwen2.5-7B-Instruct represents...",
    "input_price": 0.2,
    "output_price": 0.2,
    "model": "qwen/qwen2.5-7b-instruct",
    "api_endpoint": "https://integrate.api.nvidia.com/v1",
    "embedding": [0.04236221686005592, 0.09024723619222641, ...]
  }
}
```

---

## Embedding Mapping System

### How Embedding IDs Work

The embedding mapping system ensures efficient storage and retrieval of query embeddings:

1. **Unique Query Identification**: Queries are identified by the tuple `(task_name, query, ground_truth, metric)`

2. **Sequential ID Assignment**: 
   - All unique queries (from both train and test) are collected
   - Embeddings are generated for each unique query
   - Sequential IDs are assigned starting from 0: `0, 1, 2, 3, ...`

3. **Unified Storage**: 
   - One `.pt` file contains all embeddings
   - `embedding_id` maps directly to dictionary key in the `.pt` file
   - Same query = same `embedding_id` (whether in train or test)

4. **Mapping in Routing Data**:
   - Each routing data record has an `embedding_id` field
   - This ID points to the corresponding embedding in the `.pt` file
   - Multiple routing records (different models) can share the same `embedding_id` if they're for the same query

### Example Mapping

```
Query: "What is 2+2?" (task_name="gsm8k", ground_truth="4", metric="GSM8K")
  ↓
Gets assigned embedding_id = 42
  ↓
Stored in query_embeddings_longformer.pt as: embeddings[42] = tensor([...])
  ↓
All routing records for this query have embedding_id = 42:
  - {query: "What is 2+2?", model_name: "llama3-8b", embedding_id: 42, ...}
  - {query: "What is 2+2?", model_name: "gpt-4", embedding_id: 42, ...}
  - {query: "What is 2+2?", model_name: "qwen-7b", embedding_id: 42, ...}
```

### Retrieving Embeddings

```python
import torch
import json

# Load embeddings
embeddings = torch.load("query_embeddings_longformer.pt")

# Load routing data
with open("default_routing_train_data.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)
        embedding_id = record["embedding_id"]
        query_embedding = embeddings[embedding_id]
        
        # Now you have the embedding for this query
        print(f"Query: {record['query']}")
        print(f"Embedding shape: {query_embedding.shape}")
```

---

## Usage Examples

### Complete Pipeline Run

```bash
# Step 1: Generate query data
python llmrouter/data/data_generation.py --config llmrouter/data/sample_config.yaml

# Step 2: Generate LLM embeddings
python llmrouter/data/generate_llm_embeddings.py --config llmrouter/data/sample_config.yaml

# Step 3: API calling and evaluation (requires API_KEYS env var)
# Service-specific dict format (recommended for multiple providers):
export API_KEYS='{"NVIDIA": "nvidia-key-1,nvidia-key-2", "OpenAI": ["openai-key-1", "openai-key-2"]}'
# OR legacy format:
export API_KEYS='["your-key-1", "your-key-2"]'
# Service-specific dict format (recommended):
export API_KEYS='{"NVIDIA": "nvidia-key-1,nvidia-key-2", "OpenAI": ["openai-key-1", "openai-key-2"]}'
# OR legacy format:
export API_KEYS='["your-key-1", "your-key-2"]'
python llmrouter/data/api_calling_evaluation.py --config llmrouter/data/sample_config.yaml --workers 100
```

### Quick Test Run

```bash
# Generate small dataset for testing
python llmrouter/data/data_generation.py --config config.yaml --test

# Generate LLM embeddings
python llmrouter/data/generate_llm_embeddings.py --config config.yaml

# Test API calling with limited samples
python llmrouter/data/api_calling_evaluation.py --config config.yaml --test --workers 10
```

### Custom Configuration

```yaml
# config.yaml
data_path:
  query_data_train: 'my_data/train_queries.jsonl'
  query_data_test: 'my_data/test_queries.jsonl'
  query_embedding_data: 'my_data/embeddings.pt'
  routing_data_train: 'my_data/train_routing.jsonl'
  routing_data_test: 'my_data/test_routing.jsonl'
  llm_data: 'my_data/llms.json'
  llm_embedding_data: 'my_data/llm_embeddings.json'

data_generation:
  sample_size: 1000  # More samples per task
  train_ratio: 0.9   # 90% train, 10% test
  random_seed: 123
```

---

## Configuration

### Required Environment Variables

- `API_KEYS`: Service-specific dict, JSON array, or comma-separated list of API keys for LiteLLM Router
  ```bash
  # Service-specific dict format (recommended for multiple providers):
  export API_KEYS='{"NVIDIA": "key1,key2", "OpenAI": ["key3", "key4"]}'
  # OR legacy formats:
  export API_KEYS='["key1", "key2"]'  # JSON format
  export API_KEYS='key1,key2'  # Comma-separated
  ```
  **Note**: When using dict format, ensure the `service` field in your LLM candidate JSON matches the keys in `API_KEYS`.

### Configuration File Structure

```yaml
data_path:
  # Query data (input for Step 3, output from Step 2a)
  query_data_train: 'path/to/query_data_train.jsonl'
  query_data_test: 'path/to/query_data_test.jsonl'
  
  # Embeddings (output from Step 3)
  query_embedding_data: 'path/to/query_embeddings_longformer.pt'
  
  # Routing data (output from Step 3)
  routing_data_train: 'path/to/default_routing_train_data.jsonl'
  routing_data_test: 'path/to/default_routing_test_data.jsonl'
  
  # LLM data (input for Step 2b and Step 3)
  llm_data: 'path/to/default_llm.json'
  llm_embedding_data: 'path/to/default_llm_embeddings.json'  # Output from Step 2b

data_generation:
  sample_size: 500      # Samples per task (default: 500)
  train_ratio: 0.8      # Train/test split (default: 0.8)
  random_seed: 42       # Random seed for reproducibility
```

### Path Resolution

- **Relative paths**: Resolved relative to project root
- **Absolute paths**: Used as-is
- **Path resolution**: Handled by `DataLoader.to_abs()` method

---

## Evaluation Metrics

The pipeline supports various evaluation metrics based on task type:

| Metric | Description | Task Types |
|--------|-------------|------------|
| `GSM8K` | Math word problem evaluation | `gsm8k` |
| `MATH` | Advanced math problem evaluation | `math` |
| `em_mc` | Exact match for multiple choice | `mmlu`, `gpqa`, `commonsense_qa`, etc. |
| `f1_score` | F1 score for text matching | `natural_qa`, `trivia_qa` |
| `code_eval` | Code execution evaluation | `mbpp`, `human_eval` |
| `cem` | Close exact match | `natural_qa`, `trivia_qa` (auto-converted) |

Performance scores range from `0.0` (incorrect) to `1.0` (correct).

---

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure `API_KEYS` environment variable is set before running Step 3
2. **File Not Found**: Check that all paths in config file are correct
3. **Embedding ID Mismatch**: Ensure same config is used for all steps
4. **Memory Issues**: Reduce `--workers` count if running out of memory

### Validation

To verify output format matches samples:

```python
import json
import torch

# Check routing data format
with open("default_routing_train_data.jsonl", "r") as f:
    sample = json.loads(f.readline())
    print("Required fields:", set(sample.keys()))

# Check embeddings format
embeddings = torch.load("query_embeddings_longformer.pt")
print(f"Embedding count: {len(embeddings)}")
print(f"Embedding dimension: {embeddings[0].shape}")
print(f"ID range: 0 to {len(embeddings)-1}")
```

---

## File Structure

```
llmrouter/data/
├── README.md                    # This file
├── sample_config.yaml           # Sample configuration file (START HERE!)
├── __init__.py                  # Package initialization
├── data.py                      # Data format definitions and validators
├── data_loader.py               # Data loading utilities
├── data_generation.py           # Step 2a: Generate query data
├── generate_llm_embeddings.py   # Step 2b: Generate LLM embeddings
└── api_calling_evaluation.py    # Step 3: API calling and evaluation
```

---

## Additional Notes

- **Embedding Model**: Currently uses Longformer-based embeddings (via `get_longformer_embedding()`)
- **Load Balancing**: LiteLLM Router distributes API calls across multiple API keys
- **Parallel Processing**: API calls are parallelized using ThreadPoolExecutor
- **Error Handling**: Failed API calls are recorded with error messages in the response field
- **Format Consistency**: All outputs are designed to match sample files exactly for compatibility

For questions or issues, please refer to the main LLMRouter documentation or open an issue.

