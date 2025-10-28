#!/usr/bin/env python3
"""
Convert Chatbot Arena Conversations Dataset to Router Planner Format

This script converts the lmsys/chatbot_arena_conversations dataset to match
the output format from process.py for use in the router planner pipeline.

Evaluation approach:
- Aggregates all pairwise preferences for each unique query
- For each query, counts wins for each model across all comparisons
- Models with any wins get evaluation score 1.0; models with no wins get 0.0

Input: lmsys/chatbot_arena_conversations dataset
Output: Same 3 files as process.py:
        - default_routing_train_data.jsonl
        - default_routing_test_data.jsonl
        - query_embeddings.pt

Usage:
    python chatbot_arena_to_jsonl.py [--sample N] [--test]

Examples:
    python chatbot_arena_to_jsonl.py                           # Convert all data
    python chatbot_arena_to_jsonl.py --sample 1000             # Convert 1000 samples
    python chatbot_arena_to_jsonl.py --test                    # Quick test with 100 samples
"""

import time
import argparse
import pandas as pd
from datasets import load_dataset

# Import utils
from utils import (
    setup_environment, HF_TOKEN,
    process_final_data, generate_embeddings_for_data,
    aggregate_arena_preferences_by_query, calculate_arena_model_scores,
)

# Setup environment (loads env vars, tokens, etc.)
setup_environment()


def convert_chatbot_arena(sample_size=None):
    """
    Convert Chatbot Arena dataset to router planner format.

    Returns:
        (df_train_indexed, df_test_indexed)
    """
    print("=== CONVERTING CHATBOT ARENA DATASET ===")

    # Load the dataset
    print("Loading Chatbot Arena dataset...")
    try:
        dataset = load_dataset("lmsys/chatbot_arena_conversations", token=HF_TOKEN)
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Loaded {len(data)} conversations from {split_name} split")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Optional sampling for faster runs
    if sample_size and sample_size < len(data):
        data = data.select(range(sample_size))
        print(f"Using {len(data)} samples")

    # Aggregate preferences by query using utils
    print("Aggregating preferences by query...")
    query_groups = aggregate_arena_preferences_by_query(data)
    if len(query_groups) == 0:
        print("No valid queries found. Exiting.")
        return None, None

    # Calculate model scores per query
    print("Calculating model scores...")
    converted_data = calculate_arena_model_scores(query_groups)

    print(f"Converted {len(converted_data)} samples from {len(query_groups)} unique queries")

    # Generate embeddings for queries
    ret_1 = generate_embeddings_for_data(converted_data, "Generating embeddings")

    # Build final rows matching process.py format
    rows = []
    for rid, row in enumerate(converted_data):
        query_embedding = ret_1[rid][1]
        row_data = {
            'task_name': row['task_name'],
            'query': row['query'],
            'gt': row['gt'],
            'metric': row['metric'],
            'choices': row['choices'],
            'query_embedding': query_embedding,
        }
        if 'question_id' in row:
            row_data['task_id'] = row['question_id']
        rows.append(row_data)

    # Create DataFrame
    columns = ['task_name', 'query', 'gt', 'metric', 'choices', 'query_embedding']
    if any('task_id' in row for row in rows):
        columns.append('task_id')
    df_all = pd.DataFrame(rows, columns=columns)

    # Final processing saves jsonl and embeddings
    df_train_indexed, df_test_indexed, _ = process_final_data(df_all)
    return df_train_indexed, df_test_indexed


def main():
    parser = argparse.ArgumentParser(description="Convert Chatbot Arena Conversations to JSONL for router planner")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples to convert (default: all)")
    parser.add_argument("--test", action="store_true", help="Run with 100 samples for quick testing")

    args = parser.parse_args()
    if args.test:
        args.sample = 100
        print("Running in test mode with 100 samples...")

    start_time = time.time()
    try:
        train_df, test_df = convert_chatbot_arena(sample_size=args.sample)
        if train_df is None or test_df is None:
            print("Conversion failed.")
            return

        total_time = time.time() - start_time
        print(f"\nConversion completed successfully in {total_time:.1f} seconds!")
        print("Stats:")
        print(f"  - Train samples: {len(train_df)}")
        print(f"  - Test samples: {len(test_df)}")
        print(f"  - Total samples: {len(train_df) + len(test_df)}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

dataset = load_dataset("lmsys/chatbot_arena_conversations", token=HF_TOKEN)