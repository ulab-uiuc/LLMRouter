#!/usr/bin/env python3
"""
API Calling and Evaluation Script - Step 3: Generate Routing Data and Unified Embeddings

This script performs API calling and response evaluation using LiteLLM Router,
then generates unified embeddings .pt file and routing data JSONL files.

Input: Query data JSONL files (train/test) from config
Output: Unified embeddings .pt + routing data JSONL files (train/test)

Usage:
    python api_calling_evaluation.py --config config.yaml [--workers N] [--test]
    
Examples:
    python api_calling_evaluation.py --config llmrouter/data/sample_config.yaml
    python api_calling_evaluation.py --config config.yaml --workers 50
    python api_calling_evaluation.py --config config.yaml --test
"""

import os
import sys
import time
import json
import re
import argparse
import yaml
from typing import Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Allow importing local helper packages under repo `data/` (e.g., `human_eval`, `mbpp`)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_DATA_DIR = _PROJECT_ROOT / "data"
if _LOCAL_DATA_DIR.exists() and str(_LOCAL_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DATA_DIR))

# Import utils
from llmrouter.utils import (
    setup_environment,
    generate_task_query, ProgressTracker, call_api
)
from llmrouter.utils.data_processing import process_unified_embeddings_and_routing
from llmrouter.data.data_loader import DataLoader

# Import evaluation functions
from llmrouter.utils import f1_score, exact_match_score, get_bert_score, cem_score
from llmrouter.utils.evaluation import last_boxed_only_string, remove_boxed, is_equiv
try:
    from human_eval.evaluate_functional_correctness import entry_point_item
    from human_eval.data import HUMAN_EVAL as DEFAULT_HUMAN_EVAL_PATH
except ImportError:  # pragma: no cover
    entry_point_item = None
    DEFAULT_HUMAN_EVAL_PATH = None

try:
    from mbpp.mbpp_eval import entry_point_item_mbpp
except ImportError:  # pragma: no cover
    entry_point_item_mbpp = None

# Setup environment
setup_environment()


def _parse_api_keys_env() -> List[str]:
    api_keys_env = (os.environ.get("API_KEYS") or "").strip()
    if not api_keys_env:
        return []

    try:
        parsed = json.loads(api_keys_env)
        if isinstance(parsed, list):
            return [str(k).strip() for k in parsed if str(k).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except Exception:
        pass

    # Fallback: comma-separated list
    return [k.strip() for k in api_keys_env.split(",") if k.strip()]


API_KEYS = _parse_api_keys_env()


class LiteLLMRouterManager:
    """Manages LiteLLM Router instances for different models"""
    
    def __init__(self, llm_data_path=None, llm_data_dict=None):
        """
        Initialize router manager.
        
        Args:
            llm_data_path: Path to llm_descriptions.json file (legacy)
            llm_data_dict: Dictionary with LLM data (preferred, from config)
        """
        if llm_data_dict:
            self.config = llm_data_dict
        elif llm_data_path:
            with open(llm_data_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Try default path
            default_path = "llm_descriptions.json"
            if os.path.exists(default_path):
                with open(default_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError("No LLM data provided. Use llm_data_path or llm_data_dict")
        
        self._load_model_config()
    
    def _load_model_config(self):
        """Load model configuration"""
        # Filter to non-_think variants only
        all_models = list(self.config.keys())
        self.allowed_models = [model for model in all_models if not model.endswith('_think')]
        
        print(f"Total models in config: {len(all_models)}")
        print(f"Using {len(self.allowed_models)} non-think models: {self.allowed_models}")
    
    def _get_timeout_for_model(self, model_name):
        """Get timeout setting for specific model"""
        timeout_settings = {
            'llama-3.3-nemotron-super-49b-v1': 120,
            'llama-3.1-nemotron-51b-instruct': 90,
            'llama3-chatqa-1.5-70b': 90,
        }
        return timeout_settings.get(model_name, 30)

# ============================================================================
# API CALLING WITH LITELLM ROUTER
# ============================================================================

def process_single_query_model(args):
    """Process a single query with a single model using call_api"""
    base_row, model_name, router_manager, tracker = args
    
    try:
        # Generate task-specific prompt
        formatted_query = generate_task_query(base_row['task_name'], base_row.to_dict())
        
        model_config = router_manager.config[model_name]
        
        # Call the API
        start_time = time.time()
        request = {
            "api_endpoint": model_config["api_endpoint"],
            "query": formatted_query['user'],
            "model_name": model_name,
            "api_name": model_config["model"],
            "service": model_config["service"],
        }
        
        if formatted_query.get('system'):
            request["system_prompt"] = formatted_query['system']
        
        timeout = router_manager._get_timeout_for_model(model_name)
        result = call_api(request, max_tokens=512, temperature=0.01, top_p=0.9, timeout=timeout)
        
        result_row = base_row.copy()
        result_row['model_name'] = model_name
        result_row['formatted_query'] = formatted_query
        result_row['response'] = result['response']
        result_row['token_num'] = result['token_num']
        result_row['input_tokens'] = result['prompt_tokens']
        result_row['output_tokens'] = result['completion_tokens']
        result_row['response_time'] = result['response_time']
        result_row['response_time'] = time.time() - start_time
        result_row['api_key_used'] = ""
        
        success = 'error' not in result
        tracker.update(success=success, model_name=model_name)
        return result_row, success
        
    except Exception as e:
        print(f"Error processing {base_row.get('task_name', 'unknown')} with {model_name}: {str(e)}")
        
        # Create error row
        result_row = base_row.copy()
        result_row['model_name'] = model_name
        result_row['formatted_query'] = "ERROR"
        result_row['response'] = f"ERROR: {str(e)}"
        result_row['token_num'] = 0
        result_row['input_tokens'] = 0
        result_row['output_tokens'] = 0
        result_row['response_time'] = 0
        result_row['api_key_used'] = ""
        
        tracker.update(success=False, model_name=model_name)
        return result_row, False

def generate_responses(base_df, router_manager, max_workers=100):
    """Generate responses from multiple models using LiteLLM Router"""
    print("=== API CALLING WITH LITELLM ROUTER ===")
    
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
    error_count = len(result_df[
        result_df['response'].str.startswith('ERROR') |
        result_df['response'].str.startswith('API Error')
    ])
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
        except Exception:
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
            if entry_point_item_mbpp is None:
                raise ImportError(
                    "MBPP evaluation helpers not available. Ensure repo `data/mbpp` is importable (e.g., run from repo root)."
                )
            # Use relative path based on project root
            mbpp_dataset_path = _PROJECT_ROOT / "data" / "mbpp.jsonl"
            if not mbpp_dataset_path.exists():
                # Try alternative location
                mbpp_dataset_path = _PROJECT_ROOT / "data" / "mbpp" / "mbpp.jsonl"
            if not mbpp_dataset_path.exists():
                raise FileNotFoundError(
                    f"MBPP dataset file not found. Expected at: {_PROJECT_ROOT / 'data' / 'mbpp.jsonl'} "
                    f"or {_PROJECT_ROOT / 'data' / 'mbpp' / 'mbpp.jsonl'}. "
                    f"Please download the MBPP dataset and place it in the data directory."
                )
            pass_1 = entry_point_item_mbpp(mbpp_sample, str(mbpp_dataset_path))
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
             
            code_dict = {"task_id": task_id, "completion": code}
            if entry_point_item is None:
                raise ImportError(
                    "HumanEval evaluation helpers not available. Ensure repo `data/human_eval` is importable (e.g., run from repo root)."
                )
            # Use default path from human_eval.data, or construct relative path
            if DEFAULT_HUMAN_EVAL_PATH and os.path.exists(DEFAULT_HUMAN_EVAL_PATH):
                human_eval_path = DEFAULT_HUMAN_EVAL_PATH
            else:
                # Fallback: try relative path based on project root
                human_eval_path = _PROJECT_ROOT / "data" / "HumanEval.jsonl.gz"
                if not human_eval_path.exists():
                    # Try uncompressed version
                    human_eval_path = _PROJECT_ROOT / "data" / "HumanEval.jsonl"
                if not human_eval_path.exists():
                    raise FileNotFoundError(
                        f"HumanEval dataset file not found. Expected at: {DEFAULT_HUMAN_EVAL_PATH if DEFAULT_HUMAN_EVAL_PATH else 'N/A'} "
                        f"or {_PROJECT_ROOT / 'data' / 'HumanEval.jsonl.gz'}. "
                        f"Please download the HumanEval dataset and place it in the data directory."
                    )
            pass_1 = entry_point_item(code_dict, str(human_eval_path))
            return pass_1['pass@1']

    # Default case for unrecognized metrics
    else:
        return 0

def evaluate_responses(df):
    """Evaluate responses and add performance scores"""
    print("=== PERFORMANCE EVALUATION ===")
    
    print(f"Evaluating {len(df)} responses...")
    
    # Check required columns
    required_columns = ['response', 'ground_truth', 'metric', 'task_name']
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
    MULTIMODAL_TASK = ["geometry3k", "mathvista", "charades_ego_activity", "charades_ego_verb", "charades_ego_object"]
    
    # Initialize results storage
    performance_scores = []
    task_results = defaultdict(list)
    model_results = defaultdict(list)
    category_results = {
        'math': [],
        'code': [],
        'commonsense': [],
        'world_knowledge': [],
        'popular': [],
        'multimodal': []
    }
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating responses", ncols=100):
        try:
            # Get evaluation parameters
            prediction = row["response"] if not pd.isna(row["response"]) else ""
            # Handle both 'gt' and 'ground_truth' field names
            ground_truth = row.get("ground_truth") if "ground_truth" in row else row.get("gt")
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
            elif task_name in MULTIMODAL_TASK:
                category_results['multimodal'].append(performance)
                
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

def load_query_data_jsonl(file_path: str) -> List[Dict]:
    """Load query data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                # Parse choices if it's a string
                if 'choices' in record and isinstance(record['choices'], str):
                    try:
                        record['choices'] = json.loads(record['choices'])
                    except (json.JSONDecodeError, ValueError, TypeError):
                        record['choices'] = None
                data.append(record)
    return data


def query_data_to_dataframe(query_data: List[Dict]) -> pd.DataFrame:
    """Convert query data list to DataFrame for processing"""
    # Convert to DataFrame format expected by API calling
    # Keep ground_truth (not gt) to match sample format
    rows = []
    for item in query_data:
        row = {
            'task_name': item['task_name'],
            'query': item['query'],
            'ground_truth': item['ground_truth'],  # Keep as ground_truth to match sample
            'metric': item['metric'],
            'choices': item.get('choices'),
            'task_id': item.get('task_id')
        }
        if 'question_type' in item:
            qt = item['question_type']
            if qt is not None and not (isinstance(qt, float) and np.isnan(qt)):
                row['question_type'] = qt
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="API Calling and Evaluation with LiteLLM Router")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    parser.add_argument("--workers", type=int, default=100,
                       help="Number of parallel workers for API calls")
    parser.add_argument("--test", action="store_true",
                       help="Run with limited samples for quick testing")
    
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    loader = DataLoader(project_root)
    data_path = config.get("data_path", {})
    
    # Get paths from config
    query_data_train_path = loader.to_abs(data_path.get("query_data_train", ""))
    query_data_test_path = loader.to_abs(data_path.get("query_data_test", ""))
    llm_data_path = loader.to_abs(data_path.get("llm_data", ""))
    embedding_output_path = loader.to_abs(data_path.get("query_embedding_data", ""))
    routing_train_output_path = loader.to_abs(data_path.get("routing_data_train", ""))
    routing_test_output_path = loader.to_abs(data_path.get("routing_data_test", ""))
    
    # Check if input files exist
    if not os.path.exists(query_data_train_path):
        print(f"Error: Train query data file not found: {query_data_train_path}")
        sys.exit(1)
    if not os.path.exists(query_data_test_path):
        print(f"Error: Test query data file not found: {query_data_test_path}")
        sys.exit(1)
    if not os.path.exists(llm_data_path):
        print(f"Error: LLM data file not found: {llm_data_path}")
        sys.exit(1)
    
    # Load query data
    print(f"Loading train query data from: {query_data_train_path}")
    query_data_train = load_query_data_jsonl(query_data_train_path)
    
    print(f"Loading test query data from: {query_data_test_path}")
    query_data_test = load_query_data_jsonl(query_data_test_path)
    
    if args.test:
        query_data_train = query_data_train[:50]  # Limit for testing
        query_data_test = query_data_test[:20]
        print(f"Running in test mode with {len(query_data_train)} train and {len(query_data_test)} test samples...")
    
    # Load LLM data
    print(f"Loading LLM data from: {llm_data_path}")
    with open(llm_data_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    # Convert query data to DataFrames
    train_df = query_data_to_dataframe(query_data_train)
    test_df = query_data_to_dataframe(query_data_test)
    
    print(f"Train data: {len(train_df)} queries")
    print(f"Test data: {len(test_df)} queries")
    print(f"Tasks: {sorted(set(train_df['task_name'].unique()) | set(test_df['task_name'].unique()))}")
    
    start_time = time.time()
    
    try:
        # Initialize router manager
        router_manager = LiteLLMRouterManager(llm_data_dict=llm_data)
        
        # Generate responses for train data
        print("\n=== Processing Train Data ===")
        train_response_df = generate_responses(train_df, router_manager, max_workers=args.workers)
        train_evaluated_df = evaluate_responses(train_response_df)
        
        # Generate responses for test data
        print("\n=== Processing Test Data ===")
        test_response_df = generate_responses(test_df, router_manager, max_workers=args.workers)
        test_evaluated_df = evaluate_responses(test_response_df)
        
        # Process unified embeddings and routing data
        print("\n=== Processing Unified Embeddings and Routing Data ===")
        embedding_dict, train_final, test_final = process_unified_embeddings_and_routing(
            df_train=train_evaluated_df,
            df_test=test_evaluated_df,
            query_data_train=query_data_train,
            query_data_test=query_data_test,
            embedding_output_path=embedding_output_path,
            routing_train_output_path=routing_train_output_path,
            routing_test_output_path=routing_test_output_path
        )
        
        total_time = time.time() - start_time
        print(f"\n🎉 Processing completed successfully in {total_time:.1f} seconds!")
        print(f"📊 Final statistics:")
        print(f"  - Train samples: {len(train_final)}")
        print(f"  - Test samples: {len(test_final)}")
        print(f"  - Unified embeddings: {len(embedding_dict)}")
        print(f"  - Average train performance: {train_evaluated_df['performance'].mean():.4f}")
        print(f"  - Average test performance: {test_evaluated_df['performance'].mean():.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
