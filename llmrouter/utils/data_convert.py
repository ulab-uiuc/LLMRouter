#!/usr/bin/env python3
"""
Data format conversion script
Convert default_routing_test_data.jsonl format to router_test_data_nq.json format
Convert default_routing_train_data.jsonl format to router_train_data_nq.json format
"""

import os
import sys
import json
import ast
import argparse
from typing import Dict, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from litellm import completion

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


API_KEYS = 'your_api_key'

# Model name mapping (map input format to output format)
MODEL_NAME_MAPPING = {
    "qwen2.5-7b-instruct": "qwen/qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct": "meta/llama-3.1-8b-instruct",
    "llama-3.1-70b-instruct": "meta/llama-3.1-70b-instruct",
    "mistral-7b-instruct-v0.3": "mistralai/mistral-7b-instruct-v0.3",
    "mixtral-8x22b-instruct-v0.1": "mistralai/mixtral-8x22b-instruct-v0.1",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama3-chatqa-1.5-8b": "meta/llama-3.1-8b-instruct",  # Approximate mapping
    "llama3-chatqa-1.5-70b": "meta/llama-3.1-70b-instruct",  # Approximate mapping
    "llama-3.1-nemotron-51b-instruct": "meta/llama-3.1-70b-instruct",  # Approximate mapping
}

# Import prompt template from centralized prompts module
from functools import lru_cache

from llmrouter.prompts import load_prompt_template


@lru_cache(maxsize=1)
def _get_prompt_template() -> str:
    """Load the data-conversion prompt template on first use.

    Loaded lazily (and cached) rather than at import time so that importing this
    module never performs disk I/O — importing it previously raised
    FileNotFoundError wherever the template path didn't resolve (e.g. a
    non-editable install). It is only needed when actually converting data.
    """
    return load_prompt_template("data_conversion")


def normalize_model_name(model_name: str) -> str:
    """Normalize model name"""
    if not model_name:
        return ""
    model_name_lower = model_name.lower()
    # Direct match
    if model_name_lower in MODEL_NAME_MAPPING:
        return MODEL_NAME_MAPPING[model_name_lower]
    # Try partial match
    for key, value in MODEL_NAME_MAPPING.items():
        if key in model_name_lower:
            return value
    # If no match, try to construct standard format
    if "qwen" in model_name_lower:
        return f"qwen/{model_name}"
    elif "llama" in model_name_lower:
        return f"meta/{model_name}"
    elif "mistral" in model_name_lower:
        return f"mistralai/{model_name}"
    elif "gemma" in model_name_lower:
        return f"google/{model_name}"
    else:
        return model_name


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate cost based on token count and model name
    This uses a simplified cost calculation, actual pricing may need adjustment based on specific model pricing
    """
    # Simplified cost calculation (price per 1000 tokens, unit: USD)
    # These are example prices and may need adjustment based on actual pricing
    pricing = {
        "qwen/qwen2.5-7b-instruct": {"input": 0.1, "output": 0.1},
        "meta/llama-3.1-8b-instruct": {"input": 0.1, "output": 0.1},
        "meta/llama-3.1-70b-instruct": {"input": 0.7, "output": 0.7},
        "mistralai/mistral-7b-instruct-v0.3": {"input": 0.2, "output": 0.2},
        "mistralai/mixtral-8x22b-instruct-v0.1": {"input": 0.7, "output": 0.7},
        "google/gemma-2-27b-it": {"input": 0.6, "output": 0.6},
        "google/gemma-2-9b-it": {"input": 0.2, "output": 0.2},
    }
    
    model_name_lower = model_name.lower()
    # Find matching pricing
    for key, prices in pricing.items():
        if key.lower() in model_name_lower or model_name_lower in key.lower():
            input_cost = (input_tokens / 1000) * prices["input"]
            output_cost = (output_tokens / 1000) * prices["output"]
            return input_cost + output_cost
    
    # Default pricing
    default_input_price = 0.1
    default_output_price = 0.1
    return (input_tokens / 1000) * default_input_price + (output_tokens / 1000) * default_output_price


def parse_choices(choices_str: str) -> Optional[Dict]:
    """Parse choices string"""
    if not choices_str or choices_str == "null" or choices_str == "None":
        return None
    try:
        # Try direct parsing
        if isinstance(choices_str, str):
            choices = ast.literal_eval(choices_str)
            return choices
        return choices_str
    except (ValueError, SyntaxError, TypeError):
        return None


def generate_id(index: int, task_name: str = "default") -> str:
    """Generate unique ID"""
    return f"{task_name}_{index}"


def generate_prompt(question: str) -> List[Dict]:
    """Generate prompt format"""
    content = _get_prompt_template().format(question=question)
    return [{"content": content, "role": "user"}]


def determine_ability(task_name: str, query: str) -> str:
    """Determine ability based on task name and query"""
    # Infer ability from task name
    ability_mapping = {
        "natural_qa": "fact-reasoning",
        "trivia_qa": "fact-reasoning",
        "mmlu": "fact-reasoning",
        "gpqa": "fact-reasoning",
        "gsm8k": "math-reasoning",
        "math": "math-reasoning",
        "commonsense_qa": "commonsense-reasoning",
        "openbook_qa": "commonsense-reasoning",
        "arc_challenge": "commonsense-reasoning",
        "mbpp": "code-generation",
        "human_eval": "code-generation",
        "agentverse-logicgrid": "logical-reasoning",
    }
    
    # First try to get from task name
    if task_name in ability_mapping:
        return ability_mapping[task_name]
    
    # If cannot determine, return default value
    return "fact-reasoning"


def call_llm_for_ability(query: str, task_name: str, api_key: str, api_base: str = None) -> str:
    """Use LLM API to generate ability"""
    # Get api_base from environment or parameter
    if not api_base:
        import os
        api_base = (
            os.environ.get("OPENAI_API_BASE")
            or os.environ.get("NVIDIA_API_BASE")
        )
    
    if not api_base:
        raise ValueError(
            "API endpoint (api_base) not found. Please specify 'api_base' parameter or set "
            "OPENAI_API_BASE or NVIDIA_API_BASE environment variable."
        )
    
    try:
        prompt = f"""Based on the following task and query, determine which ability type this task belongs to. Please choose one from the following options: fact-reasoning, math-reasoning, commonsense-reasoning, code-generation, logical-reasoning.

Task name: {task_name}
Query: {query[:200]}

Please only return the ability type, do not return any other content."""
        
        response = completion(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base,
            max_tokens=50,
            temperature=0.1
        )
        
        ability = response.choices[0].message.content.strip()
        # Clean response, only keep ability type
        for ab in ["fact-reasoning", "math-reasoning", "commonsense-reasoning", "code-generation", "logical-reasoning"]:
            if ab in ability.lower():
                return ab
        return "fact-reasoning"
    except Exception as e:
        print(f"Error generating ability: {e}")
        return determine_ability(task_name, query)


def convert_single_record(old_record: Dict, index: int, use_llm: bool = False, api_key: str = None) -> Dict:
    """Convert single record"""
    # Basic field mapping
    question = old_record.get("query", "")
    ground_truth = old_record.get("ground_truth", "")
    
    # Convert golden_answers (from ground_truth to list)
    if isinstance(ground_truth, str):
        golden_answers = [ground_truth]
    elif isinstance(ground_truth, list):
        golden_answers = ground_truth
    else:
        golden_answers = [str(ground_truth)]
    
    # Generate new record
    new_record = {
        "id": generate_id(index, old_record.get("task_name", "default")),
        "question": question,
        "golden_answers": golden_answers,
        "data_source": old_record.get("task_name", "default"),  # Use task_name as data_source
        "prompt": generate_prompt(question),
        "ability": determine_ability(old_record.get("task_name", ""), question),
        "reward_model": {
            "ground_truth": {
                "target": golden_answers
            },
            "style": "rule"
        },
        "extra_info": {
            "index": index,
            "split": "test"
        }
    }
    
    # If need to use LLM to generate ability
    if use_llm and api_key:
        new_record["ability"] = call_llm_for_ability(
            question, 
            old_record.get("task_name", ""), 
            api_key
        )
    
    return new_record


def process_batch(records: List[Dict], start_index: int, use_llm: bool = False, api_key: str = None) -> List[Dict]:
    """Process records in batch"""
    results = []
    for i, record in enumerate(records):
        try:
            new_record = convert_single_record(record, start_index + i, use_llm, api_key)
            results.append(new_record)
        except Exception as e:
            print(f"Error processing record {start_index + i}: {e}")
            continue
    return results


def convert_data(
    input_file: str,
    output_file: str,
    use_llm: bool = False,
    max_workers: int = 10,
    start_line: int = None,
    end_line: int = None
):
    """Convert data format"""
    print(f"Reading input file: {input_file}")
    
    # Read JSONL file
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if start_line and i < start_line - 1:
                continue
            if end_line and i >= end_line:
                break
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line {i+1}: {e}")
                continue
    
    print(f"Successfully read {len(records)} records")
    
    # If line range is specified, adjust index
    start_index = (start_line - 1) if start_line else 0
    
    # Convert records
    converted_records = []
    
    if use_llm:
        print(f"Using LLM API to generate missing fields (using {max_workers} parallel workers)...")
        # Use API key
        api_key = API_KEYS if API_KEYS else None
        
        if not api_key:
            print("Warning: API key not found, will not use LLM to generate fields")
            use_llm = False
        
        if use_llm:
            # Process in batches
            batch_size = max(1, len(records) // max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    future = executor.submit(process_batch, batch, start_index + i, use_llm, api_key)
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Converting records"):
                    batch_results = future.result()
                    converted_records.extend(batch_results)
    else:
        print("Converting records directly (without using LLM)...")
        for i, record in enumerate(tqdm(records, desc="Converting records")):
            try:
                new_record = convert_single_record(record, start_index + i, use_llm, None)
                converted_records.append(new_record)
            except Exception as e:
                print(f"Error processing record {start_index + i}: {e}")
                continue
    
    # Save as JSON file (one JSON object per line)
    print(f"Saving to output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in converted_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Conversion completed! Converted {len(converted_records)} records")
    print(f"Output file: {output_file}")


def convert_train_data(
    input_file: str,
    output_file: str,
    start_line: int = None,
    end_line: int = None
):
    """Convert training data format (group by query)"""
    print(f"Reading input file: {input_file}")
    
    # Read JSONL file
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if start_line and i < start_line - 1:
                continue
            if end_line and i >= end_line:
                break
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line {i+1}: {e}")
                continue
    
    print(f"Successfully read {len(records)} records")
    
    # Group by query
    print("Grouping by query...")
    grouped_records = defaultdict(list)
    for record in records:
        query = record.get("query", "")
        if query:
            grouped_records[query].append(record)
    
    print(f"Total {len(grouped_records)} unique queries")
    
    # Convert each group
    converted_records = []
    for q_id, (query, group_records) in enumerate(tqdm(grouped_records.items(), desc="Converting records")):
        try:
            # Get ground_truth (from first record, as ground_truth for same query should be the same)
            ground_truth = group_records[0].get("ground_truth", "")
            if isinstance(ground_truth, str):
                gt = [ground_truth]
            elif isinstance(ground_truth, list):
                gt = ground_truth
            else:
                gt = [str(ground_truth)]
            
            # Get data_source (infer from task_name)
            task_name = group_records[0].get("task_name", "")
            source = task_name if task_name else "default"
            
            # Collect all model responses
            llm_name_list = []
            response_list = []
            cost_list = []
            perf_list = []
            
            for record in group_records:
                model_name = record.get("model_name", "")
                response = record.get("response", "")
                input_tokens = record.get("input_tokens", 0)
                output_tokens = record.get("output_tokens", 0)
                performance = record.get("performance", 0.0)
                
                if model_name and response:
                    normalized_model = normalize_model_name(model_name)
                    cost = calculate_cost(input_tokens, output_tokens, normalized_model)
                    
                    llm_name_list.append(normalized_model)
                    response_list.append(response)
                    cost_list.append(cost)
                    perf_list.append(float(performance))
            
            # Find index with best performance
            if perf_list:
                best_perf_idx = max(range(len(perf_list)), key=lambda i: perf_list[i])
                best_llm_name = llm_name_list[best_perf_idx] if best_perf_idx < len(llm_name_list) else ""
            else:
                best_perf_idx = 0
                best_llm_name = ""
            
            # Build new record
            new_record = {
                "q_id": q_id,
                "query": query,
                "gt": gt,
                "source": source,
                "llm_name_list": llm_name_list,
                "response_list": response_list,
                "cost_list": cost_list,
                "perf_list": perf_list,
                "best_perf_idx": best_perf_idx,
                "best_llm_name": best_llm_name
            }
            
            converted_records.append(new_record)
        except Exception as e:
            print(f"Error processing query {q_id} ({query[:50]}...): {e}")
            continue
    
    # Save as JSON array format
    print(f"Saving to output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_records, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Conversion completed! Converted {len(converted_records)} records")
    print(f"Output file: {output_file}")


def merge_train_test(
    test_file: str,
    train_file: str,
    output_file: str
):
    """Merge train and test data into unified format"""
    print(f"Reading test file: {test_file}")
    print(f"Reading train file: {train_file}")
    
    # Read test data (JSONL format)
    test_records = []
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # Ensure split field exists
                    if "extra_info" in record:
                        record["extra_info"]["split"] = "test"
                    else:
                        record["extra_info"] = {"split": "test"}
                    # Add split field to top level (for easy access)
                    record["split"] = "test"
                    test_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid test JSON line: {e}")
                    continue
        print(f"Successfully read {len(test_records)} test records")
    else:
        print(f"Warning: test file does not exist: {test_file}")
    
    # Read train data (JSON array format)
    train_records = []
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            try:
                train_data = json.load(f)
                if isinstance(train_data, list):
                    for record in train_data:
                        # Add split field
                        record["split"] = "train"
                        train_records.append(record)
                else:
                    print("Warning: train file format is incorrect, should be JSON array")
            except json.JSONDecodeError as e:
                print(f"Error: unable to parse train file: {e}")
        print(f"Successfully read {len(train_records)} train records")
    else:
        print(f"Warning: train file does not exist: {train_file}")
    
    # Merge data
    print(f"Merging data...")
    merged_records = test_records + train_records
    print(f"Total {len(merged_records)} merged records (test: {len(test_records)}, train: {len(train_records)})")
    
    # Save as JSONL format
    print(f"Saving to output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Merge completed! Merged {len(merged_records)} records")
    print(f"Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Data format conversion tool")
    parser.add_argument("--input", type=str, default=None,
                       help="Input file path (required for conversion mode)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output file path")
    parser.add_argument("--mode", type=str, choices=["test", "train", "merge"], default="test",
                       help="Conversion mode: test, train, or merge (default: test)")
    parser.add_argument("--test-file", type=str, default=None,
                       help="Test file path (required for merge mode)")
    parser.add_argument("--train-file", type=str, default=None,
                       help="Train file path (required for merge mode)")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM API to generate missing fields (e.g., ability, only for test mode)")
    parser.add_argument("--workers", type=int, default=10,
                       help="Number of parallel workers (default: 10, only for test mode)")
    parser.add_argument("--start-line", type=int, default=None,
                       help="Start line number (starting from 1)")
    parser.add_argument("--end-line", type=int, default=None,
                       help="End line number (exclusive)")
    
    args = parser.parse_args()
    
    if args.mode == "merge":
        # Merge mode
        if not args.test_file or not args.train_file:
            print("Error: merge mode requires --test-file and --train-file to be specified")
            sys.exit(1)
        merge_train_test(
            test_file=args.test_file,
            train_file=args.train_file,
            output_file=args.output
        )
    elif args.mode == "train":
        # Train conversion mode
        if not args.input:
            print("Error: train mode requires --input to be specified")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"Error: input file does not exist: {args.input}")
            sys.exit(1)
        convert_train_data(
            input_file=args.input,
            output_file=args.output,
            start_line=args.start_line,
            end_line=args.end_line
        )
    else:
        # Test conversion mode
        if not args.input:
            print("Error: test mode requires --input to be specified")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"Error: input file does not exist: {args.input}")
            sys.exit(1)
        convert_data(
            input_file=args.input,
            output_file=args.output,
            use_llm=args.use_llm,
            max_workers=args.workers,
            start_line=args.start_line,
            end_line=args.end_line
        )


if __name__ == "__main__":
    main()

# # Basic conversion (test data, without using LLM)
# python data_convert.py --input default_routing_test_data.jsonl --output router_test_data_nq.json --mode test

# # Use LLM to generate missing fields (test data)
# python data_convert.py --input default_routing_test_data.jsonl --output router_test_data_nq.json --mode test --use-llm --workers 10 

# # Convert train data
# python data_convert.py --input default_routing_train_data.jsonl --output router_train_data_nq.json --mode train

# # Merge train and test data
# python data_convert.py --test-file router_test_data_nq.json --train-file router_train_data_nq.json --output train_test_nq.jsonl --mode merge