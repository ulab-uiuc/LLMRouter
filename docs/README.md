# Router Planner: Intelligent LLM Routing System

This repository contains a comprehensive pipeline for training and evaluating intelligent router systems that can automatically select the most appropriate language model for different types of tasks.

## Overview

The Router Planner system learns to route queries to specialized language models based on task type and performance characteristics. It consists of three main components:

1. **Data Generation Pipeline**: Creates training data by having multiple models answer diverse tasks
2. **Router Training**: Trains various router models (KNN, MLP, SVM, Graph-based) to predict optimal model selection
3. **Evaluation Framework**: Comprehensive evaluation of router performance across different scenarios

## Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Datasets  │───▶│  Data Generation │───▶│ Router Training │
│  (11 tasks)     │    │   Pipeline       │    │   & Evaluation  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Performance     │
                       │  Evaluation      │
                       └──────────────────┘
```

## Directory Structure

```
router_planner/
├── embedding_based_router/           # Main router implementation
│   ├── train_data_gen.py            # Generate training data from 11 tasks
│   └── zijie_baseline/              # Baseline router implementations
│       ├── response_gen.py          # Generate model responses
│       └── evaluate_responses.py    # Evaluate model performance
├── config/                          # Configuration files
├── dataset/                         # Generated datasets
└── checkpoints_all/                 # Trained model checkpoints
```

## Data Generation Pipeline

### 1. Task Data Generation (`train_data_gen.py`)

**Purpose**: Creates base training data from 11 diverse benchmark datasets.

**Input**: Raw datasets from various sources
**Output**: `/dataset/router_data_train.csv`

**Tasks Included**:
- **World Knowledge**: Natural QA, Trivia QA
- **Academic Knowledge**: MMLU, GPQA  
- **Code Generation**: MBPP, HumanEval
- **Mathematical Reasoning**: GSM8K, MATH
- **Commonsense Reasoning**: CommonsenseQA, OpenbookQA, ARC-Challenge

**Key Features**:
- Samples 500 examples per task (configurable)
- Generates BERT embeddings for each query
- Standardizes data format across all tasks
- Preserves task-specific metadata (choices, task_id, etc.)

**Usage**:
```bash
cd embedding_based_router
python train_data_gen.py
```

### 2. Model Response Generation (`response_gen.py`)

**Purpose**: Generates responses from multiple language models for each query.

**Input**: Base training data from `train_data_gen.py`
**Output**: `/zijie_baseline/data/processed_train_14_task.csv`

**Key Features**:
- **Multi-model evaluation**: Tests each query against multiple LLMs
- **Load balancing**: Uses 5 NVIDIA API keys for parallel processing
- **Task-specific prompting**: Applies appropriate prompts for each task type
- **Comprehensive tracking**: Records response time, token usage, API costs
- **Error handling**: Graceful handling of API failures and timeouts

**Usage**:
```bash
cd embedding_based_router/zijie_baseline
python response_gen.py --workers 100
python response_gen.py --test  # Quick test with 10 rows
```

### 3. Performance Evaluation (`evaluate_responses.py`)

**Purpose**: Evaluates model responses and adds performance scores.

**Input**: Model responses from `response_gen.py`
**Output**: `*_evaluated.csv` files with performance scores

**Evaluation Metrics**:
- **Math tasks**: Numerical answer extraction and comparison
- **Code tasks**: Functional correctness via test case execution
- **Multiple choice**: Exact match for answer letters
- **QA tasks**: F1 score and exact match
- **Commonsense tasks**: Various semantic matching strategies

**Usage**:
```bash
python evaluate_responses.py --input processed_train_14_task.csv
python evaluate_responses.py --test  # Quick test with 100 rows
```

## Usage Examples

### Complete Pipeline

```bash
# 1. Generate base training data
cd embedding_based_router
python train_data_gen.py

# 2. Generate model responses
cd zijie_baseline
python response_gen.py --workers 100

# 3. Evaluate responses
python evaluate_responses.py --input data/processed_train_14_task.csv
```

### Quick Testing

```bash
# Test data generation with small sample
python response_gen.py --test

# Test evaluation with sample data
python evaluate_responses.py --test
```

## Output Files

### Generated Datasets
- `router_data_train.csv`: Base training data with embeddings
- `processed_train_14_task.csv`: Model responses for all queries
- `processed_train_14_task_evaluated.csv`: Responses with performance scores

## Task Categories

The system handles 7 main task categories:

1. **Math** (2 tasks): GSM8K, MATH
2. **Code** (2 tasks): MBPP, HumanEval  
3. **Commonsense** (3 tasks): CommonsenseQA, OpenbookQA, ARC-Challenge
4. **World Knowledge** (2 tasks): Natural QA, Trivia QA
5. **Reading** (3 tasks): SQuAD, QuAC, BoolQ
6. **Popular** (2 tasks): MMLU, GPQA
7. **AgentVerse** (2 tasks): LogicGrid, MGSM
