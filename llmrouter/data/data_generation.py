#!/usr/bin/env python3
"""
Data Generation Script - Step 2a: Generate Query Data JSONL Files

This script generates query data JSONL files (train/test) from 14 diverse benchmark datasets.
The output format matches StandardQueryData format without embeddings.

Supported Datasets:
- Text: Natural QA, Trivia QA, MMLU, GPQA, CommonsenseQA, OpenbookQA, ARC-Challenge
- Math: GSM8K, MATH
- Code: MBPP, HumanEval
- Multimodal: Geometry3K, MathVista, Charades-Ego (Activity/Object/Verb)

Input: Config YAML file (optional, can use command-line args)
Output: query_data_train.jsonl and query_data_test.jsonl files

Usage:
    python data_generation.py --config config.yaml
    python data_generation.py --sample N --output_train PATH --output_test PATH
    
Examples:
    python data_generation.py --config llmrouter/data/sample_config.yaml
    python data_generation.py --sample 100 --output_train data/query_train.jsonl --output_test data/query_test.jsonl
    python data_generation.py --sample 10 --test  # Quick test with 10 samples
"""

import os
import sys
import time
import random
import json
import argparse
import yaml

import numpy as np
from datasets import load_dataset

# Import utils
from llmrouter.utils import (
    setup_environment, CASE_NUM
)
from llmrouter.data.data_loader import DataLoader
from llmrouter.data import batch_vlm_describe_images

# Setup environment
setup_environment()


def get_n_samples(N=10, random_seed=42, cache_dir=None, charades_ego_path=None, datasets=None):
    """Extract samples from all datasets

    Args:
        N: Number of samples per dataset
        random_seed: Random seed for reproducibility
        cache_dir: Optional cache directory for datasets. If None, uses default HuggingFace cache.
        charades_ego_path: Path to Charades-Ego dataset root.
        datasets: Optional list of dataset names to include. If None, includes all.
    """
    random.seed(random_seed)

    # all supported datasets
    all_datasets = [
        "natural_qa", "trivia_qa", "mmlu", "gpqa", "mbpp", "human_eval", 
        "gsm8k", "commonsense_qa", "math", "openbook_qa", "arc_challenge", 
        "geometry3k", "mathvista", 
        "charades_ego_activity", "charades_ego_object", "charades_ego_verb"
    ]

    if datasets:
         # Validate and filter
         datasets = [d for d in datasets if d in all_datasets]
         if not datasets:
             print("Warning: No valid datasets specified. Generating nothing.")
             return {}
    else:
        datasets = all_datasets

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
    geometry3k_samples = []
    mathvista_samples = []
    charades_ego_activity_samples = []
    charades_ego_object_samples = []
    charades_ego_verb_samples = []

    # 1. Natural QA dataset
    if "natural_qa" in datasets:
        try:
            natural_qa = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq',
                                      cache_dir=cache_dir)
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
    if "trivia_qa" in datasets:
        try:
            trivia_qa = load_dataset("trivia_qa", "rc.nocontext",
                                     cache_dir=cache_dir)
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
    if "mmlu" in datasets:
        try:
            mmlu = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
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
    if "gpqa" in datasets:
        try:
            gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main",
                                cache_dir=cache_dir)
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

    # 5. MBPP dataset (loaded from HuggingFace)
    if "mbpp" in datasets:
        try:
            mbpp_dataset = load_dataset("mbpp", "full", cache_dir=cache_dir)
            split_name = 'train' if 'train' in mbpp_dataset else list(mbpp_dataset.keys())[0]
            mbpp_samples_all = list(mbpp_dataset[split_name])
            dataset_size = len(mbpp_samples_all)
            if dataset_size >= N:
                indices = random.sample(range(dataset_size), N)
                mbpp_samples = [mbpp_samples_all[i] for i in indices]
            else:
                mbpp_samples = mbpp_samples_all
            print(f"Successfully extracted {len(mbpp_samples)} samples from MBPP")
        except Exception as e:
            print(f"Error extracting from MBPP: {e}")

    # 6. HumanEval dataset (loaded from HuggingFace)
    if "human_eval" in datasets:
        try:
            humaneval_dataset = load_dataset("openai/openai_humaneval", cache_dir=cache_dir)
            split_name = 'test' if 'test' in humaneval_dataset else list(humaneval_dataset.keys())[0]
            humaneval_samples_all = list(humaneval_dataset[split_name])
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
    if "gsm8k" in datasets:
        try:
            gsm8k = load_dataset('gsm8k', 'main',
                                 cache_dir=cache_dir)
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
    if "commonsense_qa" in datasets:
        try:
            commonsense_qa = load_dataset('commonsense_qa',
                                          cache_dir=cache_dir)
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
    if "arc_challenge" in datasets:
        try:
            arc_challenge = load_dataset('allenai/ai2_arc', 'ARC-Challenge',
                                         cache_dir=cache_dir)
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
    if "openbook_qa" in datasets:
        try:
            openbook_qa = load_dataset('allenai/openbookqa', 'main',
                                       cache_dir=cache_dir)
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
    if "math" in datasets:
        try:
            CATEGORY = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                        'prealgebra', 'precalculus']
            for cate in CATEGORY:
                math = load_dataset('EleutherAI/hendrycks_math', cate,
                                    cache_dir=cache_dir)
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

    # 12. Geometry3K (multimodal geometry QA)
    if "geometry3k" in datasets:
        try:
            geometry3k = load_dataset("hiyouga/geometry3k", cache_dir=cache_dir)
            split_name = 'train' if 'train' in geometry3k else list(geometry3k.keys())[0]
            dataset_size = len(geometry3k[split_name])
            if dataset_size >= N:
                indices = random.sample(range(dataset_size), N)
                geometry3k_samples = [geometry3k[split_name][i] for i in indices]
            else:
                geometry3k_samples = [geometry3k[split_name][i] for i in range(dataset_size)]
            print(f"Successfully extracted {len(geometry3k_samples)} samples from Geometry3K")
        except Exception as e:
            print(f"Error extracting from Geometry3K: {e}")

    # 13. MathVista (multimodal visual math QA)
    if "mathvista" in datasets:
        try:
            mathvista = load_dataset("AI4Math/MathVista", cache_dir=cache_dir)
            split_name = 'train' if 'train' in mathvista else list(mathvista.keys())[0]
            dataset_size = len(mathvista[split_name])
            if dataset_size >= N:
                indices = random.sample(range(dataset_size), N)
                mathvista_samples = [mathvista[split_name][i] for i in indices]
            else:
                mathvista_samples = [mathvista[split_name][i] for i in range(dataset_size)]
            print(f"Successfully extracted {len(mathvista_samples)} samples from MathVista")
        except Exception as e:
            print(f"Error extracting from MathVista: {e}")

    # 14. Charades-Ego Activity (multimodal video activity recognition)
    if "charades_ego_activity" in datasets:
        try:
            # Need to provide data_root from argument
            if charades_ego_path and os.path.exists(charades_ego_path):
                sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "charades_ego"))
                from charades_ego_to_json import load_charades_ego_samples
                charades_ego_activity_samples = load_charades_ego_samples(
                    N=N, task_type="activity", data_root=charades_ego_path, 
                    random_seed=random_seed, cache_dir=cache_dir
                )
                print(f"Successfully extracted {len(charades_ego_activity_samples)} samples from Charades-Ego Activity")
            else:
                print("Skipping Charades-Ego Activity: charades_ego_path not set or invalid")
        except Exception as e:
            print(f"Error extracting from Charades-Ego Activity: {e}")
    
    # 15. Charades-Ego Object (multimodal video object recognition)
    charades_ego_object_samples = []
    if "charades_ego_object" in datasets:
        try:
            if charades_ego_path and os.path.exists(charades_ego_path):
                sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "charades_ego"))
                from charades_ego_to_json import load_charades_ego_samples
                charades_ego_object_samples = load_charades_ego_samples(
                    N=N, task_type="object", data_root=charades_ego_path,
                    random_seed=random_seed, cache_dir=cache_dir
                )
                print(f"Successfully extracted {len(charades_ego_object_samples)} samples from Charades-Ego Object")
            else:
                print("Skipping Charades-Ego Object: charades_ego_path not set or invalid")
        except Exception as e:
            print(f"Error extracting from Charades-Ego Object: {e}")
    
    # 16. Charades-Ego Verb (multimodal video verb recognition)
    charades_ego_verb_samples = []
    if "charades_ego_verb" in datasets:
        try:
            if charades_ego_path and os.path.exists(charades_ego_path):
                sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "charades_ego"))
                from charades_ego_to_json import load_charades_ego_samples
                charades_ego_verb_samples = load_charades_ego_samples(
                    N=N, task_type="verb", data_root=charades_ego_path,
                    random_seed=random_seed, cache_dir=cache_dir
                )
                print(f"Successfully extracted {len(charades_ego_verb_samples)} samples from Charades-Ego Verb")
            else:
                print("Skipping Charades-Ego Verb: charades_ego_path not set or invalid")
        except Exception as e:
            print(f"Error extracting from Charades-Ego Verb: {e}")

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
        'geometry3k': geometry3k_samples,
        'mathvista': mathvista_samples,
        'charades_ego_activity': charades_ego_activity_samples,
        'charades_ego_object': charades_ego_object_samples,
        'charades_ego_verb': charades_ego_verb_samples,
    }

def generate_query_data(sample_size=None, train_ratio=0.8, random_seed=42, charades_ego_path=None, datasets=None):
    """
    Generate query data from benchmark datasets.
    
    Args:
        sample_size: Number of samples per task
        train_ratio: Ratio for train/test split (default 0.8)
        random_seed: Random seed for reproducibility
        charades_ego_path: Path to Charades-Ego dataset root
        datasets: Optional list of dataset names to include
        
    Returns:
        tuple: (train_data, test_data) - Lists of dictionaries matching query_data format
    """
    print("=== QUERY DATA GENERATION ===")
    
    # Use sample_size if provided, otherwise use CASE_NUM
    n_samples = sample_size if sample_size else CASE_NUM
    
    print(f"Extracting {n_samples} samples per task...")
    if datasets:
        print(f"Datasets: {', '.join(datasets)}")
    samples = get_n_samples(N=n_samples, charades_ego_path=charades_ego_path, datasets=datasets)
    
    data_all = []
    
    # Process each task type
    for task_name, task_samples in samples.items():
        if task_name == "natural_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['question'],
                    'ground_truth': sample['golden_answers'][0],
                    'metric': 'f1_score',
                    'choices': None,
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "trivia_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['question'],
                    'ground_truth': sample['answer']['normalized_aliases'][0],
                    'metric': 'f1_score',
                    'choices': None,
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "mmlu":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['question'],
                    'ground_truth': chr(65 + sample['answer']),  # Convert index to A, B, C, D
                    'metric': 'em_mc',
                    'choices': sample['choices'],
                    'task_id': None
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
                    'query': sample['Question'],
                    'ground_truth': chr(65 + new_correct_index),
                    'metric': 'em_mc',
                    'choices': shuffled_options,
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "mbpp":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['text'],
                    'ground_truth': sample['test_list'],
                    'metric': 'code_eval',
                    'choices': sample['test_list'],
                    'task_id': sample['task_id']
                }
                data_all.append(case)
                
        elif task_name == "human_eval":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['prompt'],
                    'ground_truth': sample['test'],
                    'metric': 'code_eval',
                    'choices': None,
                    'task_id': sample['task_id']
                }
                data_all.append(case)
                
        elif task_name == "gsm8k":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['question'],
                    'ground_truth': sample['answer'],
                    'metric': 'GSM8K',
                    'choices': None,
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "commonsense_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['question'],
                    'ground_truth': sample['answerKey'],
                    'metric': 'em_mc',
                    'choices': sample['choices'],
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "math":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['problem'],
                    'ground_truth': sample['solution'],
                    'metric': 'MATH',
                    'choices': None,
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "openbook_qa":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['question_stem'],
                    'ground_truth': sample['answerKey'],
                    'metric': 'em_mc',
                    'choices': sample['choices'],
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "arc_challenge":
            for sample in task_samples:
                case = {
                    'task_name': task_name,
                    'query': sample['question'],
                    'ground_truth': sample['answerKey'],
                    'metric': 'em_mc',
                    'choices': sample['choices'],
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "geometry3k":
            vlm_prompt = (
                "Describe the geometry diagram for solving the problem.\n"
                "Include all visible text, numbers, symbols, angles, lengths, and geometric relationships.\n"
                "Be concise and factual. Do NOT solve the problem."
            )
            images_list = [sample["images"] for sample in task_samples]
            image_descriptions = batch_vlm_describe_images(vlm_prompt, images_list)
            
            for sample, image_desc in zip(task_samples, image_descriptions):
                # Remove <image> tags from problem text
                problem_text = sample['problem'].replace('<image>', '').strip()
                query = f"{problem_text}\n\n[Diagram description]\n{image_desc}"
                
                case = {
                    'task_name': task_name,
                    'query': query,
                    'ground_truth': sample["answer"],
                    'metric': 'MATH',
                    'choices': None,
                    'task_id': None
                }
                data_all.append(case)
                
        elif task_name == "mathvista":
            vlm_prompt = (
                "Describe the image for solving the question.\n"
                "Include all visible text, numbers, symbols, tables/charts/axes, and geometry relationships.\n"
                "Be concise and factual. Do NOT solve the problem."
            )
            # Batch process all images in parallel
            images_list = [[sample["decoded_image"]] for sample in task_samples]
            image_descriptions = batch_vlm_describe_images(vlm_prompt, images_list)
            
            # Build cases with descriptions
            for sample, image_desc in zip(task_samples, image_descriptions):
                question_text = sample['question']
                query = f"{question_text}\n\n[Image description]\n{image_desc}"
                
                case = {
                    'task_name': task_name,
                    'query': query,
                    'ground_truth': sample["answer"],
                    'metric': 'MATH',
                    'question_type': sample["question_type"],
                    'choices': sample["choices"],
                    'task_id': sample["pid"]
                }
                data_all.append(case)
                
        elif task_name in ["charades_ego_activity", "charades_ego_object", "charades_ego_verb"]:
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "charades_ego"))
            from charades_ego_to_json import process_charades_ego_jsonl_samples
            
            charades_cases = process_charades_ego_jsonl_samples(task_name, task_samples, charades_ego_path)
            data_all.extend(charades_cases)
    
    print(f"Generated {len(data_all)} base samples")
    
    # Convert choices to string format if needed (matching sample format)
    for case in data_all:
        if case['choices'] is not None:
            if isinstance(case['choices'], dict):
                # Already in correct format
                pass
            elif isinstance(case['choices'], list):
                # Convert to dict format like sample: {'text': [...], 'labels': [...]}
                case['choices'] = {
                    'text': case['choices'],
                    'labels': [chr(65 + i) for i in range(len(case['choices']))]
                }
    
    # Split into train/test
    random.seed(random_seed)
    random.shuffle(data_all)
    train_size = int(len(data_all) * train_ratio)
    train_data = data_all[:train_size]
    test_data = data_all[train_size:]
    
    print(f"Split into {len(train_data)} train and {len(test_data)} test samples")
    
    return train_data, test_data

def save_query_data_jsonl(data_list, output_path):
    """Save query data to JSONL file matching sample format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            # Format matches sample: task_name, query, ground_truth, metric, choices, task_id
            record = {
                'task_name': item['task_name'],
                'query': item['query'],
                'ground_truth': item['ground_truth'],
                'metric': item['metric'],
                'choices': json.dumps(item['choices']) if item['choices'] is not None else None,
                'task_id': item['task_id']
            }
            if 'question_type' in item:
                qt = item['question_type']
                if qt is not None and not (isinstance(qt, float) and np.isnan(qt)):
                    record['question_type'] = qt
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(data_list)} records to {output_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate Query Data JSONL Files")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--sample", type=int, default=None, 
                       help="Number of samples per task (default: 500)")
    parser.add_argument("--output_train", type=str, default=None,
                       help="Output path for train JSONL file")
    parser.add_argument("--output_test", type=str, default=None,
                       help="Output path for test JSONL file")
    parser.add_argument("--charades_ego_path", type=str, default=None,
                       help="Path to Charades-Ego dataset root")
    parser.add_argument("--test", action="store_true", 
                       help="Run with 10 samples for quick testing")
    parser.add_argument("--datasets", type=str, default=None,
                       help="Comma-separated list of datasets to include. If not specified, includes all.")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        loader = DataLoader(project_root)
        data_path = config.get("data_path", {})
        
        # Get paths from config
        output_train = loader.to_abs(data_path.get("query_data_train", ""))
        output_test = loader.to_abs(data_path.get("query_data_test", ""))
        
        # Get sample size from config if not provided
        if args.sample is None:
            args.sample = config.get("data_generation", {}).get("sample_size", CASE_NUM)
    else:
        # Use command-line args
        if args.output_train is None or args.output_test is None:
            parser.error("Either --config or both --output_train and --output_test must be provided")
        output_train = args.output_train
        output_test = args.output_test
    
    if args.test:
        args.sample = 10
        print("Running in test mode with 10 samples per task...")
    
    start_time = time.time()
    
    try:
        # Generate query data
        train_data, test_data = generate_query_data(
            sample_size=args.sample,
            train_ratio=config.get("data_generation", {}).get("train_ratio", 0.8),
            random_seed=config.get("data_generation", {}).get("random_seed", 42),
            charades_ego_path=args.charades_ego_path,
            datasets=args.datasets.split(',') if args.datasets else None
        )
        
        # Save to JSONL files
        save_query_data_jsonl(train_data, output_train)
        save_query_data_jsonl(test_data, output_test)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Query data generation completed successfully in {total_time:.1f} seconds!")
        print(f"📊 Generated data statistics:")
        print(f"  - Train samples: {len(train_data)}")
        print(f"  - Test samples: {len(test_data)}")
        print(f"  - Total samples: {len(train_data) + len(test_data)}")
        
        # Show sample counts by task
        all_tasks = set(item['task_name'] for item in train_data + test_data)
        print(f"  - Tasks: {sorted(all_tasks)}")
        
        print(f"\n📁 Output files:")
        print(f"  - Train: {output_train}")
        print(f"  - Test: {output_test}")
        
    except Exception as e:
        print(f"\n❌ Error during data generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
