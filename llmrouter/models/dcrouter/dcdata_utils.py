"""
DCRouter Data Utilities
-----------------------
Data preprocessing utilities for DCRouter.

This module provides functions to convert data to DCRouter format.
"""

import json
import os
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from typing import List, Dict


def aggregate_by_query(records: List[Dict]) -> List[Dict]:
    """
    Aggregate records by query, combining responses from different LLMs.

    Args:
        records (list[dict]): List of individual LLM response records

    Returns:
        list[dict]: List of aggregated records with llm_name_list and perf_list
    """
    # Group by query
    grouped = defaultdict(list)
    for record in records:
        query = record.get("query", "")
        if query:
            grouped[query].append(record)

    # Convert each group
    aggregated = []
    for query, group_records in grouped.items():
        llm_name_list = []
        perf_list = []

        for record in group_records:
            model_name = record.get("model_name", "")
            performance = record.get("performance", 0.0)

            if model_name:
                llm_name_list.append(model_name)
                perf_list.append(float(performance))

        if llm_name_list:  # Only add if we have at least one LLM response
            aggregated.append({
                "query": query,
                "llm_name_list": llm_name_list,
                "perf_list": perf_list
            })

    return aggregated


def convert_format(input_data: List[Dict]) -> List[Dict]:
    """
    Convert default data format to DCRouter training format.

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


def add_clusters(data: List[Dict], n_clusters: int = 3, random_state: int = 42) -> List[Dict]:
    """
    Add cluster_id to data using K-means clustering.

    Args:
        data (list[dict]): List of data, each item contains "scores" field
        n_clusters (int): Number of clusters (default: 3)
        random_state (int): Random seed (default: 42)

    Returns:
        list[dict]: List of data with cluster_id added
    """
    print(f"  Assigning clusters to {len(data)} samples...")

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
    print(f"  Cluster distribution: {dict(cluster_dist)}")

    return data


def preprocess_data(
    input_path: str,
    output_path: str,
    add_cluster_id: bool = True,
    n_clusters: int = 3,
    max_samples: int = None
):
    """
    Preprocess data from JSONL format to DCRouter JSON format.

    Args:
        input_path (str): Input JSONL file path
        output_path (str): Output JSON file path
        add_cluster_id (bool): Whether to add cluster_id (default: True)
        n_clusters (int): Number of clusters (default: 3)
        max_samples (int): Maximum number of samples (default: None for all)
    """
    print(f"  Reading data from: {input_path}")

    # Read JSONL data
    input_data = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                input_data.append(json.loads(line.strip()))

    print(f"  Loaded {len(input_data)} records")

    # Aggregate by query
    print(f"  Aggregating records by query...")
    aggregated_data = aggregate_by_query(input_data)
    print(f"  Aggregated to {len(aggregated_data)} unique queries")

    # Limit samples if specified
    if max_samples and len(aggregated_data) > max_samples:
        aggregated_data = aggregated_data[:max_samples]
        print(f"  Limited to {len(aggregated_data)} samples")

    # Convert format
    converted_data = convert_format(aggregated_data)

    # Add clustering
    if add_cluster_id:
        converted_data = add_clusters(converted_data, n_clusters=n_clusters)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(converted_data)} samples to: {output_path}")

    return converted_data
