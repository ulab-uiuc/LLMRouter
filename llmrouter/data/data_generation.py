#!/usr/bin/env python3
"""
Data Generation Script - Stage 1 from preprocessing_all_in_one.py

This script extracts the data generation phase from preprocessing_all_in_one.py,
creating base training data with embeddings from 11 diverse benchmark datasets.

Input: None (loads datasets directly)
Output: Base data CSV file with embeddings (same format as preprocessing_all_in_one.py Stage 1)

Usage:
    python data_generation.py [--sample N] [--output OUTPUT_FILE]
    
Examples:
    python data_generation.py                           # Generate 500 samples per task
    python data_generation.py --sample 100              # Generate 100 samples per task
    python data_generation.py --output my_data.csv      # Custom output file
    python data_generation.py --sample 10 --test        # Quick test with 10 samples
"""

import os
import sys
import time
import random
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# Import utils
from utils import (
    setup_environment, TASK_DESCRIPTIONS, CASE_NUM,
    get_bert_representation, parallel_embedding_task,
    generate_embeddings_for_data
)

# Setup environment
setup_environment()


def get_n_samples(N=10, random_seed=42):
    """Extract samples from all datasets"""
    random.seed(random_seed)
    
    # Initialize empty lists for each dataset
    natural_qa_samples = []
    trivia_qa_samples = []
    mmlu_samples = []
    gpqa_samples = []
    mbpp_samples = []
    humaneval_samples = []
    gsm8k_samples = []
    commonsense_qa_samples = []
    math_samples = []
    openbook_qa_samples = []
    arc_challenge_samples = []

    # 1. Natural QA dataset
    try:
        natural_qa = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq',
                                  cache_dir='/data/taofeng2/Router_bench/dataset/World Knowledge')
        split_name = 'train' if 'train' in natural_qa else list(natural_qa.keys())[0]
        dataset_size = len(natural_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            natural_qa_samples = [natural_qa[split_name][i] for i in indices]
        else:
            natural_qa_samples = [natural_qa[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(natural_qa_samples)} samples from Natural QA")
    except Exception as e:
        print(f"Error extracting from Natural QA: {e}")

    # 2. Trivia QA dataset
    try:
        trivia_qa = load_dataset("trivia_qa", "rc.nocontext",
                                 cache_dir='/data/taofeng2/Router_bench/dataset/World Knowledge')
        split_name = 'train' if 'train' in trivia_qa else list(trivia_qa.keys())[0]
        dataset_size = len(trivia_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            trivia_qa_samples = [trivia_qa[split_name][i] for i in indices]
        else:
            trivia_qa_samples = [trivia_qa[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(trivia_qa_samples)} samples from Trivia QA")
    except Exception as e:
        print(f"Error extracting from Trivia QA: {e}")

    # 3. MMLU dataset
    try:
        mmlu = load_dataset("cais/mmlu", "all", cache_dir='/data/taofeng2/Router_bench/dataset/Popular')
        split_name = 'auxiliary_train' if 'auxiliary_train' in mmlu else list(mmlu.keys())[0]
        dataset_size = len(mmlu[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            mmlu_samples = [mmlu[split_name][i] for i in indices]
        else:
            mmlu_samples = [mmlu[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(mmlu_samples)} samples from MMLU")
    except Exception as e:
        print(f"Error extracting from MMLU: {e}")

    # 4. GPQA dataset
    try:
        gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main",
                            cache_dir='/data/taofeng2/Router_bench/dataset/Popular/gpqa')
        split_name = 'train' if 'train' in gpqa else list(gpqa.keys())[0]
        dataset_size = len(gpqa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            gpqa_samples = [gpqa[split_name][i] for i in indices]
        else:
            gpqa_samples = [gpqa[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(gpqa_samples)} samples from GPQA")
    except Exception as e:
        print(f"Error extracting from GPQA: {e}")

    # 5. MBPP dataset
    try:
        mdpp_path = '/data/taofeng2/Router_bench/dataset/Code/mbpp.jsonl'
        with open(mdpp_path, 'r') as f:
            lines = f.readlines()
        mbpp_samples_all = [json.loads(line) for line in lines]
        dataset_size = len(mbpp_samples_all)
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            mbpp_samples = [mbpp_samples_all[i] for i in indices]
        else:
            mbpp_samples = mbpp_samples_all
        print(f"Successfully extracted {len(mbpp_samples)} samples from MBPP")
    except Exception as e:
        print(f"Error extracting from MBPP: {e}")

    # 6. HumanEval dataset
    try:
        humaneval_path = '/data/taofeng2/Router_bench/dataset/Code/HumanEval.jsonl'
        with open(humaneval_path, 'r') as f:
            lines = f.readlines()
        humaneval_samples_all = [json.loads(line) for line in lines]
        dataset_size = len(humaneval_samples_all)
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            humaneval_samples = [humaneval_samples_all[i] for i in indices]
        else:
            humaneval_samples = humaneval_samples_all
        print(f"Successfully extracted {len(humaneval_samples)} samples from HumanEval")
    except Exception as e:
        print(f"Error extracting from HumanEval: {e}")

    # 7. GSM8K dataset
    try:
        gsm8k = load_dataset('gsm8k', 'main',
                             cache_dir='/data/taofeng2/Router_bench/dataset/Math')
        split_name = 'train' if 'train' in gsm8k else list(gsm8k.keys())[0]
        dataset_size = len(gsm8k[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            gsm8k_samples = [gsm8k[split_name][i] for i in indices]
        else:
            gsm8k_samples = [gsm8k[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(gsm8k_samples)} samples from GSM8K")
    except Exception as e:
        print(f"Error extracting from GSM8K: {e}")

    # 8. CommonsenseQA dataset
    try:
        commonsense_qa = load_dataset('commonsense_qa',
                                      cache_dir='/data/taofeng2/Router_bench/dataset/Commonsense Reasoning')
        split_name = 'train' if 'train' in commonsense_qa else list(commonsense_qa.keys())[0]
        dataset_size = len(commonsense_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            commonsense_qa_samples = [commonsense_qa[split_name][i] for i in indices]
        else:
            commonsense_qa_samples = [commonsense_qa[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(commonsense_qa_samples)} samples from CommonsenseQA")
    except Exception as e:
        print(f"Error extracting from CommonsenseQA: {e}")

    # 9. ARC-Challenge dataset
    try:
        arc_challenge = load_dataset('allenai/ai2_arc', 'ARC-Challenge',
                                     cache_dir='/data/taofeng2/Router_bench/dataset/Commonsense Reasoning')
        split_name = 'train' if 'train' in arc_challenge else list(arc_challenge.keys())[0]
        dataset_size = len(arc_challenge[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            arc_challenge_samples = [arc_challenge[split_name][i] for i in indices]
        else:
            arc_challenge_samples = [arc_challenge[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(arc_challenge_samples)} samples from ARC-Challenge")
    except Exception as e:
        print(f"Error extracting from ARC-Challenge: {e}")

    # 10. OpenbookQA dataset
    try:
        openbook_qa = load_dataset('allenai/openbookqa', 'main',
                                   cache_dir='/data/taofeng2/Router_bench/dataset/Commonsense Reasoning')
        split_name = 'train' if 'train' in openbook_qa else list(openbook_qa.keys())[0]
        dataset_size = len(openbook_qa[split_name])
        if dataset_size >= N:
            indices = random.sample(range(dataset_size), N)
            openbook_qa_samples = [openbook_qa[split_name][i] for i in indices]
        else:
            openbook_qa_samples = [openbook_qa[split_name][i] for i in range(dataset_size)]
        print(f"Successfully extracted {len(openbook_qa_samples)} samples from OpenbookQA")
    except Exception as e:
        print(f"Error extracting from OpenbookQA: {e}")

    # 11. MATH dataset
    try:
        CATEGORY = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                    'prealgebra', 'precalculus']
        for cate in CATEGORY:
            math = load_dataset('EleutherAI/hendrycks_math', cate,
                                cache_dir='/data/taofeng2/Router_bench/dataset/Math')
            split_name = 'train' if 'train' in math else list(math.keys())[0]
            dataset_size = len(math[split_name])
            target_samples = N // len(CATEGORY) + 1
            if dataset_size >= target_samples:
                indices = random.sample(range(dataset_size), target_samples)
                math_samples.extend([math[split_name][i] for i in indices])
            else:
                math_samples.extend([math[split_name][i] for i in range(dataset_size)])
        print(f"Successfully extracted {len(math_samples)} samples from MATH")
    except Exception as e:
        print(f"Error extracting from MATH: {e}")

    return {
        "natural_qa": natural_qa_samples,
        "trivia_qa": trivia_qa_samples,
        "mmlu": mmlu_samples,
        'gpqa': gpqa_samples,
        'mbpp': mbpp_samples,
        'human_eval': humaneval_samples,
        'gsm8k': gsm8k_samples,
        'commonsense_qa': commonsense_qa_samples,
        'math': math_samples,
        'openbook_qa': openbook_qa_samples,
        'arc_challenge': arc_challenge_samples,
    }

def generate_base_data(sample_size=None):
    """Generate base training data with embeddings"""
    print("=== DATA GENERATION ===")
    
    # Use sample_size if provided, otherwise use CASE_NUM
    n_samples = sample_size if sample_size else CASE_NUM
    
    print(f"Extracting {n_samples} samples per task...")
    samples = get_n_samples(N=n_samples)
    
    # Use task descriptions from utils
    task_description = TASK_DESCRIPTIONS
    
    data_all = []
    
    # Process each task type
    for task_name, task_samples in samples.items():
        if task_name == "natural_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['question'],
                    'choices': None,
                    'gt': sample['golden_answers'][0],
                    'metric': 'f1_score'
                }
                data_all.append(case)
                
        elif task_name == "trivia_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['question'],
                    'choices': None,
                    'gt': sample['answer']['normalized_aliases'][0],
                    'metric': 'f1_score'
                }
                data_all.append(case)
                
        elif task_name == "mmlu":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['question'],
                    'choices': sample['choices'],
                    'gt': chr(65 + sample['answer']),  # Convert index to A, B, C, D
                    'metric': 'em_mc'
                }
                data_all.append(case)
                
        elif task_name == "gpqa":
            for sample in task_samples:
                options = [
                    sample['Correct Answer'], sample['Incorrect Answer 1'], 
                    sample['Incorrect Answer 2'], sample['Incorrect Answer 3']
                ]
                correct_index = 0
                mapping = list(range(len(options)))
                random.shuffle(mapping)
                new_correct_index = mapping.index(correct_index)
                shuffled_options = [options[mapping.index(i)] for i in range(len(options))]
                
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['Question'],
                    'choices': shuffled_options,
                    'gt': chr(65 + new_correct_index),
                    'metric': 'em_mc'
                }
                data_all.append(case)
                
        elif task_name == "mbpp":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'task_id': sample['task_id'],
                    'query': sample['text'],
                    'choices': sample['test_list'],
                    'gt': sample['test_list'],
                    'metric': 'code_eval'
                }
                data_all.append(case)
                
        elif task_name == "human_eval":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['prompt'],
                    'choices': None,
                    'gt': sample['test'],
                    'task_id': sample['task_id'],
                    'metric': 'code_eval'
                }
                data_all.append(case)
                
        elif task_name == "gsm8k":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['question'],
                    'choices': None,
                    'gt': sample['answer'],
                    'metric': 'GSM8K'
                }
                data_all.append(case)
                
        elif task_name == "commonsense_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['question'],
                    'choices': sample['choices'],
                    'gt': sample['answerKey'],
                    'metric': 'em_mc'
                }
                data_all.append(case)
                
        elif task_name == "math":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['problem'],
                    'choices': None,
                    'gt': sample['solution'],
                    'metric': 'MATH'
                }
                data_all.append(case)
                
        elif task_name == "openbook_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['question_stem'],
                    'choices': sample['choices'],
                    'gt': sample['answerKey'],
                    'metric': 'em_mc'
                }
                data_all.append(case)
                
        elif task_name == "arc_challenge":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'task_description': task_description[task_name],
                    'query': sample['question'],
                    'choices': sample['choices'],
                    'gt': sample['answerKey'],
                    'metric': 'em_mc'
                }
                data_all.append(case)
    
    print(f"Generated {len(data_all)} base samples")
    
    # Generate embeddings using utils
    ret_1 = generate_embeddings_for_data(data_all, "Generating embeddings")
    
    # Build final data with embeddings
    rows = []
    for rid, row in enumerate(data_all):
        query_embedding = ret_1[rid][1]
        row_data = {
            'task_name': row['task_name'],
            'query': row['query'],
            'gt': row['gt'],
            'metric': row['metric'],
            'choices': row.get('choices'),
            'query_embedding': query_embedding,
        }
        if 'task_id' in row:
            row_data['task_id'] = row['task_id']
        rows.append(row_data)
    
    # Create DataFrame
    columns = ['task_name', 'query', 'gt', 'metric', 'choices', 'query_embedding']
    if any('task_id' in row for row in rows):
        columns.append('task_id')
    
    df = pd.DataFrame(rows, columns=columns)
    
    return df

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Data Generation - Stage 1 from preprocessing_all_in_one.py")
    parser.add_argument("--sample", type=int, default=None, 
                       help="Number of samples per task (default: 500)")
    parser.add_argument("--output", type=str, default="dataset/14_task_train.csv",
                       help="Output CSV file path")
    parser.add_argument("--test", action="store_true", 
                       help="Run with 10 samples for quick testing")
    
    args = parser.parse_args()
    
    if args.test:
        args.sample = 10
        print("Running in test mode with 10 samples per task...")
    
    start_time = time.time()
    
    try:
        # Generate base data
        base_df = generate_base_data(sample_size=args.sample)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save base data
        base_df.to_csv(args.output, index=False)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Data generation completed successfully in {total_time:.1f} seconds!")
        print(f"📊 Generated data statistics:")
        print(f"  - Total samples: {len(base_df)}")
        print(f"  - Tasks: {sorted(base_df['task_name'].unique())}")
        print(f"  - Output file: {args.output}")
        
        # Show sample counts by task
        print(f"\n📈 Samples per task:")
        task_counts = base_df['task_name'].value_counts().sort_index()
        for task, count in task_counts.items():
            print(f"  - {task}: {count}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Data generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during data generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
