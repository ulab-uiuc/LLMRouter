"""
RouterDC Data Pipeline
----------------------
Data preprocessing pipeline for RouterDC training and evaluation.

This module provides functions to:
1. Convert training data format and add cluster IDs
2. Generate test scores using NVIDIA NIM API

Original source: RouterDC/convert_data_to_router_format.py and generate_test_scores_with_nim.py
Adapted for LLMRouter framework.
"""

import json
import os
import re
import time
from collections import Counter
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from tqdm import tqdm


# NIM API model names corresponding to 6 LLMs
LLM_MODELS = {
    "meta/llama-3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3": "mistralai/mistral-7b-instruct-v0.3",
    "qwen/qwen2.5-7b-instruct": "qwen/qwen2.5-7b-instruct",
    "google/gemma-2-27b-it": "google/gemma-2-27b-it",
    "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai/mixtral-8x22b-instruct-v0.1"
}


def convert_format(input_data: List[Dict]) -> List[Dict]:
    """
    Convert data/ format to RouterDC training format.

    Input format:
        {
            "query": "...",
            "llm_name_list": [...],
            "perf_list": [...]
        }

    Output format:
        {
            "question": "...",
            "scores": {"model1": 0.5, ...}
        }

    Args:
        input_data (list[dict]): List of original data

    Returns:
        list[dict]: List of converted data
    """
    converted_data = []

    for item in input_data:
        # Build scores dictionary
        scores = {}
        for llm_name, perf in zip(item['llm_name_list'], item['perf_list']):
            scores[llm_name] = perf

        # Build new format
        new_item = {
            "question": item['query'],
            "scores": scores
        }

        converted_data.append(new_item)

    return converted_data


def add_clusters(data: List[Dict], n_clusters: int = 5, random_state: int = 42) -> List[Dict]:
    """
    Add cluster_id to data using K-means clustering.

    Args:
        data (list[dict]): List of data, each item contains "scores" field
        n_clusters (int): Number of clusters (default: 5)
        random_state (int): Random seed (default: 42)

    Returns:
        list[dict]: List of data with cluster_id added
    """
    print(f"Assigning clusters to {len(data)} samples...")

    # Extract features: performance scores of each model
    features = []
    for item in data:
        perf_vector = list(item['scores'].values())
        features.append(perf_vector)

    features = np.array(features)

    # Use K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(features)

    # Add cluster_id
    for i, item in enumerate(data):
        item['cluster_id'] = int(cluster_labels[i])

    # Print cluster distribution
    cluster_dist = Counter(cluster_labels)
    print(f"Cluster distribution: {dict(cluster_dist)}")

    return data


def convert_train_data(
    input_path: str,
    output_path: str,
    add_cluster_id: bool = True,
    n_clusters: int = 5
) -> List[Dict]:
    """
    Convert training data to RouterDC format and add clustering.

    Args:
        input_path (str): Input JSON file path (e.g., data/router_train_nq.json)
        output_path (str): Output JSON file path (e.g., datasets/qa_cluster/nq_train.json)
        add_cluster_id (bool): Whether to add cluster_id (default: True)
        n_clusters (int): Number of clusters (default: 5)

    Returns:
        list[dict]: List of converted data
    """
    print(f"[Data Preprocessing] Converting training data")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    # Read data
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    print(f"  Original sample count: {len(input_data)}")
    print(f"  LLM list: {input_data[0]['llm_name_list']}")

    # Convert format
    converted_data = convert_format(input_data)

    # Add clustering
    if add_cluster_id:
        converted_data = add_clusters(converted_data, n_clusters=n_clusters)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Conversion completed! Sample count: {len(converted_data)}")

    return converted_data


def extract_answer(response_text: str) -> str:
    """
    Extract answer from model response.

    Args:
        response_text (str): Complete model response

    Returns:
        str: Extracted answer
    """
    # Try to extract <answer>...</answer> format
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no tags, return full response (strip leading/trailing whitespace)
    return response_text.strip()


def evaluate_answer(predicted_answer: str, golden_answers: List[str]) -> float:
    """
    Evaluate whether predicted answer matches golden answers.

    Args:
        predicted_answer (str): Predicted answer
        golden_answers (list[str]): List of golden answers

    Returns:
        float: 1.0 (exact match) or 0.0 (no match)
    """
    predicted_lower = predicted_answer.lower().strip()

    for golden in golden_answers:
        golden_lower = str(golden).lower().strip()

        # Exact match
        if predicted_lower == golden_lower:
            return 1.0

        # Containment relationship
        if golden_lower in predicted_lower or predicted_lower in golden_lower:
            return 1.0

    return 0.0


def call_nim_api(
    client: OpenAI,
    model_name: str,
    question: str,
    max_retries: int = 3
) -> str:
    """
    Call NVIDIA NIM API.

    Args:
        client (OpenAI): OpenAI client instance
        model_name (str): Model name
        question (str): Question text
        max_retries (int): Maximum number of retries (default: 3)

    Returns:
        str: Model response text
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"Answer the following question directly and concisely. Provide only the answer without explanation.\n\nQuestion: {question}\n\nAnswer:"
                    }
                ],
                temperature=0.2,
                max_tokens=100,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  API call failed ({model_name}), retrying {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  ❌ API call failed ({model_name}): {str(e)}")
                return ""
    return ""


def generate_test_scores(
    input_path: str,
    output_path: str,
    max_samples: Optional[int] = None,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Generate test scores using NVIDIA NIM API.

    Args:
        input_path (str): Input json file path (e.g., data/final_test_nq.json)
        output_path (str): Output JSON file path (e.g., datasets/qa_6model/nq_test.json)
        max_samples (int | None): Maximum number of samples to process (None means all)
        api_key (str | None): NVIDIA API Key (None means use default or environment variable)

    Returns:
        list[dict]: List of generated test data
    """
    print(f"[Data Preprocessing] Generating test set scores")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    # Read json file
    df = pd.read_json(input_path, lines=True)

    if max_samples:
        df = df.head(max_samples)

    print(f"  Sample count: {len(df)}")

    # Initialize OpenAI client (NIM API is compatible with OpenAI format)
    default_api_key = "your_api_key"
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key or os.environ.get("NVIDIA_API_KEY") or default_api_key
    )

    # Process each sample
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating scores"):
        question = row['question']
        golden_answers = row['golden_answers']

        # Call 6 LLMs
        scores = {}

        print(f"\n[{idx+1}/{len(df)}] Question: {question[:60]}...")

        for llm_name, model_name in LLM_MODELS.items():
            # Call API
            response = call_nim_api(client, model_name, question)

            # Extract answer
            answer = extract_answer(response)

            # Evaluate answer
            score = evaluate_answer(answer, golden_answers)

            scores[llm_name] = score

            print(f"  {llm_name}: {score} (Answer: {answer[:40]}...)")

            # Avoid requests too fast
            time.sleep(0.5)

        # Build output format
        result = {
            "question": question,
            "scores": scores
        }

        results.append(result)

        # Periodic save (prevent failure midway)
        if (idx + 1) % 10 == 0:
            temp_path = output_path + ".tmp"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  💾 Temporary file saved: {temp_path}")

    # Save final results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Completed! Sample count: {len(results)}")

    # Display statistics
    avg_scores = {llm: 0.0 for llm in LLM_MODELS.keys()}
    for result in results:
        for llm, score in result['scores'].items():
            avg_scores[llm] += score

    print(f"\n  📊 Average accuracy:")
    for llm, total_score in avg_scores.items():
        avg = total_score / len(results)
        print(f"    {llm}: {avg:.2%}")

    return results


def prepare_routerdc_data(
    train_input_path: str,
    train_output_path: str,
    test_input_path: Optional[str] = None,
    test_output_path: Optional[str] = None,
    n_clusters: int = 5,
    max_test_samples: Optional[int] = None,
    api_key: Optional[str] = None,
    skip_existing: bool = True
) -> Dict[str, List[Dict]]:
    """
    Complete RouterDC data preparation pipeline.

    Args:
        train_input_path (str): Training data input path (JSON)
        train_output_path (str): Training data output path (JSON with cluster_id)
        test_input_path (str | None): Test data input path (parquet)
        test_output_path (str | None): Test data output path (JSON)
        n_clusters (int): Number of clusters for training data (default: 5)
        max_test_samples (int | None): Maximum number of test samples (default: None)
        api_key (str | None): NVIDIA API Key (default: None)
        skip_existing (bool): Skip existing files (default: True)

    Returns:
        dict: Dictionary with "train" and "test" keys, values are data lists
    """
    print("=" * 70)
    print("RouterDC Data Preparation Pipeline")
    print("=" * 70)

    results = {}

    # Step 1: Convert training data
    if skip_existing and os.path.exists(train_output_path):
        print(f"\n[Skipped] Training data already exists: {train_output_path}")
        with open(train_output_path, 'r') as f:
            results['train'] = json.load(f)
    else:
        results['train'] = convert_train_data(
            input_path=train_input_path,
            output_path=train_output_path,
            add_cluster_id=True,
            n_clusters=n_clusters
        )

    # Step 2: Generate test set scores (optional)
    if test_input_path and test_output_path:
        if skip_existing and os.path.exists(test_output_path):
            print(f"\n[Skipped] Test data already exists: {test_output_path}")
            with open(test_output_path, 'r') as f:
                results['test'] = json.load(f)
        else:
            results['test'] = generate_test_scores(
                input_path=test_input_path,
                output_path=test_output_path,
                max_samples=max_test_samples,
                api_key=api_key
            )
    else:
        print("\n[Skipped] Test data path not specified")
        results['test'] = []

    print("\n" + "=" * 70)
    print("Data preparation completed!")
    print("=" * 70)
    print(f"Training samples: {len(results['train'])}")
    if results['test']:
        print(f"Test samples: {len(results['test'])}")

    return results
