#!/usr/bin/env python3
"""
API Calling and Evaluation Script using LiteLLM Router

This script performs API calling and response evaluation using LiteLLM Router
for load balancing across multiple API keys, then applies final processing
to match the exact output format of process.py and preprocessing_all_in_one.py.

Input: Pre-generated base data CSV file
Output: Same format as process.py (3 files: .pt + 2 .jsonl)

Usage:
    python api_calling_evaluation.py --input base_data.csv [--workers N] [--test]
    
Examples:
    python api_calling_evaluation.py --input dataset/14_task_train.csv
    python api_calling_evaluation.py --input dataset/14_task_train.csv --workers 50
    python api_calling_evaluation.py --input dataset/14_task_train.csv --test
"""

import os
import sys
import time
import json
import ast
import re
import argparse
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from litellm import Router

# Import utils
from utils import (
    setup_environment, API_KEYS,
    format_mc_prompt, format_gsm8k_prompt, format_math_prompt,
    format_commonsense_qa_prompt, format_mbpp_prompt, format_humaneval_prompt,
    generate_task_query, ProgressTracker, to_tensor, clean_df,
    process_final_data
)

# Import evaluation functions
from utils import f1_score, exact_match_score, get_bert_score, evaluate_code, cem_score
from human_eval.evaluate_functional_correctness import entry_point_item
from mbpp.mbpp_eval import entry_point_item_mbpp
from math_eval import last_boxed_only_string, remove_boxed, is_equiv

# Setup environment
setup_environment()

class LiteLLMRouterManager:
    """Manages LiteLLM Router instances for different models"""
    
    def __init__(self, config_path="llm_descriptions.json"):
        self.config_path = config_path
        self.routers = {}
        self._load_model_config()
        self._create_routers()
    
    def _load_model_config(self):
        """Load model configuration from JSON file"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Filter to non-_think variants only
        all_models = list(self.config.keys())
        self.allowed_models = [model for model in all_models if not model.endswith('_think')]
        
        print(f"Total models in config: {len(all_models)}")
        print(f"Using {len(self.allowed_models)} non-think models: {self.allowed_models}")
    
    def _create_routers(self):
        """Create Router instances for each model"""
        for model_name in self.allowed_models:
            api_model_name = self.config[model_name]["model"]
            
            # Create model list with all API keys for load balancing
            model_list = []
            for i, api_key in enumerate(API_KEYS):
                model_list.append({
                    "model_name": model_name,
                    "litellm_params": {
                        "model": api_model_name,
                        "api_key": api_key,
                        "api_base": "https://integrate.api.nvidia.com/v1",
                        "timeout": self._get_timeout_for_model(model_name),
                        "max_retries": 3
                    }
                })
            
            # Create router for this model
            self.routers[model_name] = Router(
                model_list=model_list,
                routing_strategy="round_robin"  # Distribute load evenly
            )
            
            print(f"Created router for {model_name} with {len(API_KEYS)} API keys")
    
    def _get_timeout_for_model(self, model_name):
        """Get timeout setting for specific model"""
        timeout_settings = {
            'llama-3.3-nemotron-super-49b-v1': 120,
            'llama-3.1-nemotron-51b-instruct': 90,
            'llama3-chatqa-1.5-70b': 90,
        }
        return timeout_settings.get(model_name, 30)
    
    def get_router(self, model_name):
        """Get router for specific model"""
        return self.routers.get(model_name)


# ============================================================================
# API CALLING WITH LITELLM ROUTER
# ============================================================================

def process_single_query_model(args):
    """Process a single query with a single model using LiteLLM Router"""
    base_row, model_name, router_manager, tracker = args
    
    try:
        # Generate task-specific prompt
        formatted_query = generate_task_query(base_row['task_name'], base_row.to_dict())
        
        # Get router for this model
        router = router_manager.get_router(model_name)
        if not router:
            raise ValueError(f"No router found for model: {model_name}")
        
        # Call the API using LiteLLM Router
        start_time = time.time()
        
        try:
            response = router.completion(
                model=model_name,
                messages=[{"role": "user", "content": formatted_query}],
                max_tokens=512,
                temperature=0.01,
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content
            usage = response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None
                    
        except Exception as api_error:
            error_msg = str(api_error)
            if "timeout" in error_msg.lower():
                response_text = f"API Error: Request timed out for model {model_name}"
            else:
                response_text = f"API Error: {error_msg[:100]}"
            usage = None
            
        end_time = time.time()
        
        # Extract token count
        if usage:
            token_num = usage.get("total_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = len(formatted_query.split()) if formatted_query else 0
            completion_tokens = len(response_text.split()) if isinstance(response_text, str) else 0
            token_num = prompt_tokens + completion_tokens
        
        # Create result row
        result_row = base_row.copy()
        result_row['model_name'] = model_name
        result_row['formatted_query'] = formatted_query
        result_row['response'] = response_text
        result_row['token_num'] = token_num
        result_row['prompt_tokens'] = prompt_tokens
        result_row['completion_tokens'] = completion_tokens
        result_row['response_time'] = end_time - start_time
        
        tracker.update(success=True, model_name=model_name)
        return result_row, True
        
    except Exception as e:
        print(f"Error processing {base_row.get('task_name', 'unknown')} with {model_name}: {str(e)}")
        
        # Create error row
        result_row = base_row.copy()
        result_row['model_name'] = model_name
        result_row['formatted_query'] = "ERROR"
        result_row['response'] = f"ERROR: {str(e)}"
        result_row['token_num'] = 0
        result_row['prompt_tokens'] = 0
        result_row['completion_tokens'] = 0
        result_row['response_time'] = 0
        
        tracker.update(success=False, model_name=model_name)
        return result_row, False

def generate_responses(base_df, max_workers=100):
    """Generate responses from multiple models using LiteLLM Router"""
    print("=== API CALLING WITH LITELLM ROUTER ===")
    
    # Initialize router manager
    router_manager = LiteLLMRouterManager()
    
    # Create all query-model combinations
    print(f"Creating query-model combinations...")
    all_tasks = []
    
    for _, base_row in base_df.iterrows():
        for model_name in router_manager.allowed_models:
            all_tasks.append((base_row, model_name, router_manager))
    
    print(f"Total tasks to process: {len(all_tasks)} ({len(base_df)} queries × {len(router_manager.allowed_models)} models)")
    
    # Optimize worker count
    optimal_workers = min(max_workers, len(all_tasks) // 10, 100)
    if optimal_workers < max_workers:
        print(f"Optimizing workers: {max_workers} → {optimal_workers}")
        max_workers = optimal_workers
    
    print(f"Processing with {max_workers} parallel workers...")
    
    tracker = ProgressTracker(len(all_tasks), "Generating responses")
    results = []
    
    # Add tracker to each task
    tasks_with_tracker = [(task[0], task[1], task[2], tracker) for task in all_tasks]
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_single_query_model, task): task
                for task in tasks_with_tracker
            }
            
            for future in as_completed(future_to_task):
                result_row, success = future.result()
                results.append(result_row)
    
    finally:
        tracker.close()
    
    # Convert results to DataFrame
    print(f"Converting {len(results)} results to DataFrame...")
    result_df = pd.DataFrame(results)
    
    # Display summary
    print(f"\n=== Processing Summary ===")
    print(f"Total rows generated: {len(result_df)}")
    print(f"Unique queries: {len(result_df['query'].unique())}")
    print(f"Unique models: {len(result_df['model_name'].unique())}")
    
    # Error analysis
    error_count = len(result_df[result_df['response'].str.startswith('ERROR')])
    success_rate = ((len(result_df) - error_count) / len(result_df)) * 100
    print(f"Success rate: {success_rate:.2f}% ({len(result_df) - error_count}/{len(result_df)})")
    
    return result_df

# ============================================================================
# PERFORMANCE EVALUATION (from preprocessing_all_in_one.py)
# ============================================================================

def eval_perf(metric, prediction, ground_truth, task_name, task_id=None):
    """Evaluate performance of a prediction against ground truth"""
    if task_name in ["natural_qa", "trivia_qa", "squad", "boolq"]:
        metric = "cem"
    
    # Exact match evaluation
    if metric == 'em':
        result = exact_match_score(prediction, ground_truth)
        return float(result)
    elif metric == 'cem':
        result = cem_score(prediction, ground_truth)
        return float(result)
    # Multiple choice exact match
    elif metric == 'em_mc':
        result = exact_match_score(prediction, ground_truth, normal_method="mc")
        return float(result)

    # BERT-based semantic similarity score
    elif metric == 'bert_score':
        result = get_bert_score([prediction], [ground_truth])
        return result

    # GSM8K math problem evaluation
    elif metric == 'GSM8K':
        ground_truth = ground_truth.split("####")[-1].replace(',', '').replace('$', '').replace('.', '').strip()
        answer = re.findall("(\\-?[0-9\\.\\,]+)", prediction)
        final_answer = None
        if len(answer) == 0:
            return 0
        else:
            invalid_str = ['', '.']
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
        final_answer = final_answer.replace(',', '').replace('$', '').replace('.', '').strip()
        if final_answer == ground_truth:
            return 1
        else:
            return 0
            
    elif metric == 'MATH':
        # Handle ground truth - it might be in \boxed{} format or plain text
        gt_boxed = last_boxed_only_string(ground_truth)
        if gt_boxed is not None:
            ground_truth_processed = remove_boxed(gt_boxed)
        else:
            ground_truth_processed = ground_truth
        
        try:
            # Extract answer from prediction (should be in \boxed{} format)
            string_in_last_boxed = last_boxed_only_string(prediction)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
                if is_equiv(answer, ground_truth_processed):
                    return 1
        except Exception as e:
            return 0
        return 0
    
    # F1 score for partial matching
    elif metric == 'f1_score' or task_name in ['quac']:
        f1, prec, recall = f1_score(prediction, ground_truth)
        return f1

    elif metric == 'code_eval':
        if task_id is None:
            raise ValueError("task_id is required for code_eval metric")

        # Check if this is MBPP or HumanEval based on task_id format
        is_mbpp = not str(task_id).startswith("HumanEval")

        if is_mbpp:
            # Case-insensitive pattern to match between [BEGIN] and [DONE]/[Done]
            code_match = re.search(r'\[BEGIN\](.*?)(?:\[DONE\]|\[Done\]|$)', prediction, re.DOTALL | re.IGNORECASE)

            if code_match:
                code = code_match.group(1).strip()
            else:
                code = prediction.strip()

            mbpp_sample = {"task_id": int(task_id), "completion": code}
            pass_1 = entry_point_item_mbpp(mbpp_sample, '/data/taofeng2/Router_bench/dataset/Code/mbpp.jsonl')
            return pass_1['pass@1']

        else:
            # Extract code between [BEGIN] and optional [DONE]
            code_match = re.search(r'\[BEGIN\](.*?)(?:\[DONE\]|$)', prediction, re.DOTALL | re.IGNORECASE)
            if code_match:
                raw_code = code_match.group(1).strip()
                if raw_code.lstrip().startswith("def "):
                    code = raw_code
                else:
                    code = "    " + raw_code.replace("\n", "\n    ")
            else:
                code = prediction.strip()
            
            dict = {"task_id": task_id, "completion": code}
            pass_1 = entry_point_item(dict, '/data/taofeng2/Router_bench/dataset/Code/HumanEval.jsonl')
            return pass_1['pass@1']

    # Default case for unrecognized metrics
    else:
        return 0

def evaluate_responses(df):
    """Evaluate responses and add performance scores"""
    print("=== PERFORMANCE EVALUATION ===")
    
    print(f"Evaluating {len(df)} responses...")
    
    # Check required columns
    required_columns = ['response', 'gt', 'metric', 'task_name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Optional columns
    has_task_id = 'task_id' in df.columns
    
    print(f"Tasks in data: {sorted(df['task_name'].unique())}")
    print(f"Metrics in data: {sorted(df['metric'].unique())}")
    print(f"Models in data: {sorted(df['model_name'].unique())}")
    
    # Define task categories for analysis
    MATH_TASK = ['gsm8k', 'math']
    CODE_TASK = ["mbpp", "human_eval"]
    COMMONSENSE_TASK = ['commonsense_qa', 'openbook_qa', 'arc_challenge']
    WORLD_KNOWLEDGE_TASK = ["natural_qa", "trivia_qa"]
    POPULAR_TASK = ["mmlu", "gpqa"]
    
    # Initialize results storage
    performance_scores = []
    task_results = defaultdict(list)
    model_results = defaultdict(list)
    category_results = {
        'math': [],
        'code': [],
        'commonsense': [],
        'world_knowledge': [],
        'popular': []
    }
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating responses", ncols=100):
        try:
            # Get evaluation parameters
            prediction = row["response"] if not pd.isna(row["response"]) else ""
            ground_truth = row["gt"]
            task_name = row["task_name"]
            metric = row["metric"]
            model_name = row["model_name"]
            task_id = row["task_id"] if has_task_id and not pd.isna(row["task_id"]) else None
            
            # Skip error responses
            if isinstance(prediction, str) and prediction.startswith("ERROR"):
                performance = 0.0
            else:
                # Handle task_id formatting for code evaluation
                if task_name in CODE_TASK:
                    if task_id is not None and not pd.isna(task_id) and not str(task_id).startswith("HumanEval"):
                        task_id = int(str(task_id).strip())
                
                # Evaluate performance
                performance = eval_perf(
                    metric=metric,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    task_name=task_name,
                    task_id=task_id
                )
                
                # Handle dict results (from code evaluation)
                if isinstance(performance, dict):
                    performance = performance.get('pass@1', 0.0)
            
            performance_scores.append(performance)
            
            # Store results for analysis
            task_results[task_name].append(performance)
            model_results[model_name].append(performance)
            
            # Store by category
            if task_name in MATH_TASK:
                category_results['math'].append(performance)
            elif task_name in CODE_TASK:
                category_results['code'].append(performance)
            elif task_name in COMMONSENSE_TASK:
                category_results['commonsense'].append(performance)
            elif task_name in WORLD_KNOWLEDGE_TASK:
                category_results['world_knowledge'].append(performance)
            elif task_name in POPULAR_TASK:
                category_results['popular'].append(performance)
                
        except Exception as e:
            print(f"\nError evaluating row {idx} (task: {row.get('task_name', 'unknown')}, model: {row.get('model_name', 'unknown')}): {e}")
            performance_scores.append(0.0)
            continue
    
    # Add performance column to dataframe
    df['performance'] = performance_scores
    
    # Print evaluation summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Overall average performance: {np.mean(performance_scores):.4f}")
    
    print(f"\n=== Performance by Task Category ===")
    for category, results in category_results.items():
        if results:
            print(f"{category.title()}: {np.mean(results):.4f} ({len(results)} samples)")
    
    print(f"\n=== Performance by Task ===")
    for task_name, results in sorted(task_results.items()):
        if results:
            print(f"{task_name}: {np.mean(results):.4f} ({len(results)} samples)")
    
    print(f"\n=== Performance by Model ===")
    for model_name, results in sorted(model_results.items()):
        if results:
            print(f"{model_name}: {np.mean(results):.4f} ({len(results)} samples)")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="API Calling and Evaluation with LiteLLM Router")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input CSV file with base data")
    parser.add_argument("--workers", type=int, default=100,
                       help="Number of parallel workers for API calls")
    parser.add_argument("--test", action="store_true",
                       help="Run with first 100 rows for quick testing")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load base data
    print(f"Loading base data from: {args.input}")
    base_df = pd.read_csv(args.input)
    
    if args.test:
        base_df = base_df.head(100)
        print(f"Running in test mode with {len(base_df)} rows...")
    
    print(f"Base data shape: {base_df.shape}")
    print(f"Tasks in data: {sorted(base_df['task_name'].unique())}")
    
    start_time = time.time()
    
    try:
        # Generate responses using LiteLLM Router
        response_df = generate_responses(base_df, max_workers=args.workers)
        
        # Evaluate responses
        evaluated_df = evaluate_responses(response_df)
        
        # Final processing to match process.py format
        train_df, test_df, embedding_dict = process_final_data(evaluated_df)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Processing completed successfully in {total_time:.1f} seconds!")
        print(f"📊 Final statistics:")
        print(f"  - Total samples processed: {len(evaluated_df)}")
        print(f"  - Train samples: {len(train_df)}")
        print(f"  - Test samples: {len(test_df)}")
        print(f"  - Average performance: {evaluated_df['performance'].mean():.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
