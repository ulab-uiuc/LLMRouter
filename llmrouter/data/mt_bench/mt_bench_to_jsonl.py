#!/usr/bin/env python3
"""
Convert MT Bench Human Judgments Dataset to Router Planner Format (per-turn)

This script converts the lmsys/mt_bench_human_judgments dataset to match
the output format from process.py for use in the router planner pipeline.

Turn handling:
- The dataset contains multi-turn conversations in fields `conversation_a` and `conversation_b`
- Each record has a `turn` value of 1 or 2
- We split by turn so that each turn is treated as an independent row in the output

Evaluation approach:
- Aggregates all pairwise preferences for each unique query (per turn)
- For each query, counts wins for each model across comparisons
- Models with any wins get evaluation score 1.0; models with no wins get 0.0

Input: lmsys/mt_bench_human_judgments dataset
Output: Same 3 files as process.py:
        - default_routing_train_data.jsonl
        - default_routing_test_data.jsonl
        - query_embeddings.pt

Usage:
    python mt_bench_to_jsonl.py [--sample N] [--test] [--turn TURN]

Examples:
    python mt_bench_to_jsonl.py                           # Convert all data (both turns)
    python mt_bench_to_jsonl.py --sample 1000             # Convert 1000 samples
    python mt_bench_to_jsonl.py --test                    # Quick test with 100 samples
    python mt_bench_to_jsonl.py --turn 1                  # Only use turn 1
    python mt_bench_to_jsonl.py --turn 2                  # Only use turn 2
"""

import time
import argparse
import pandas as pd
from datasets import load_dataset

# Import utils
from llmrouter.utils import (
    setup_environment, HF_TOKEN,
    process_final_data, generate_embeddings_for_data,
    aggregate_preferences_by_query, calculate_model_scores,
)

# Setup environment (loads env vars, tokens, etc.)
setup_environment()


def convert_mt_bench(sample_size=None, turn_filter=None):
    """
    Convert MT Bench dataset to router planner format, splitting by turn.

    Args:
        sample_size: Optional cap on number of rows from the dataset
        turn_filter: Optional int 1 or 2 to filter to a single turn

    Returns:
        (df_train_indexed, df_test_indexed)
    """
    print("=== CONVERTING MT BENCH DATASET (per-turn) ===")

    # Load the dataset
    print("Loading MT Bench dataset...")
    try:
        dataset = load_dataset("lmsys/mt_bench_human_judgments", token=HF_TOKEN)
        split_name = 'human' if 'human' in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Loaded {len(data)} judgments from {split_name} split")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Optional sampling for faster runs
    if sample_size and sample_size < len(data):
        data = data.select(range(sample_size))
        print(f"Using {len(data)} samples")

    # Show turn distribution
    turn_counts = {}
    for sample in data:
        t = sample.get('turn', 1)
        turn_counts[t] = turn_counts.get(t, 0) + 1
    print(f"Turn distribution: {dict(turn_counts)}")
    if turn_filter is not None:
        print(f"Filtering to turn {turn_filter} only")

    # Aggregate preferences by query (per turn)
    print("Aggregating preferences by query (per turn)...")
    query_groups = aggregate_preferences_by_query(data, turn_filter)
    if len(query_groups) == 0:
        print("No valid queries found. Exiting.")
        return None, None

    # Calculate model scores per query
    print("Calculating model scores...")
    converted_data = calculate_model_scores(query_groups)

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
    parser = argparse.ArgumentParser(description="Convert MT Bench Human Judgments to JSONL for router planner (per-turn)")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples to convert (default: all)")
    parser.add_argument("--test", action="store_true", help="Run with 100 samples for quick testing")
    parser.add_argument("--turn", type=int, choices=[1, 2], default=None, help="Filter to specific turn (1 or 2)")

    args = parser.parse_args()
    if args.test:
        args.sample = 100
        print("Running in test mode with 100 samples...")

    start_time = time.time()
    try:
        train_df, test_df = convert_mt_bench(sample_size=args.sample, turn_filter=args.turn)
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
