"""
Utils package for LLMRouter scripts
"""

from .data_loader import (
    load_csv,
    load_jsonl,
    jsonl_to_csv,
    load_pt,
)

from .model_loader import (
    save_model,
    load_model,
)

from .embeddings import get_longformer_embedding, parallel_embedding_task
from .tensor_utils import to_tensor
from .dataframe_utils import clean_df
from .prompting import (
    format_mc_prompt, format_gsm8k_prompt, format_math_prompt,
    format_commonsense_qa_prompt, format_mbpp_prompt, format_humaneval_prompt,
    generate_task_query
)
from .progress import ProgressTracker
from .conversation import (
    extract_user_prompt, extract_model_response,
    aggregate_preferences_by_query, calculate_model_scores
)
from .arena_conversation import (
    extract_user_prompt as extract_arena_user_prompt,
    extract_model_response as extract_arena_model_response,
    aggregate_preferences_by_query as aggregate_arena_preferences_by_query,
    calculate_model_scores as calculate_arena_model_scores
)
from .data_processing import process_final_data, generate_embeddings_for_data
from .constants import TASK_DESCRIPTIONS, TASK_CATEGORIES, API_KEYS, HF_TOKEN, CASE_NUM
from .setup import setup_environment
from .api_calling import call_api

# Import evaluation functions from evaluation.py
from .evaluation import f1_score, exact_match_score, get_bert_score, evaluate_code, cem_score

__all__ = ["load_csv",
    "load_jsonl",
    "jsonl_to_csv",
    "load_pt",
    'get_bert_representation', 'parallel_embedding_task',
    'to_tensor', 'clean_df',
    'format_mc_prompt', 'format_gsm8k_prompt', 'format_math_prompt',
    'format_commonsense_qa_prompt', 'format_mbpp_prompt', 'format_humaneval_prompt',
    'generate_task_query', 'ProgressTracker',
    'extract_user_prompt', 'extract_model_response',
    'aggregate_preferences_by_query', 'calculate_model_scores',
    'extract_arena_user_prompt', 'extract_arena_model_response',
    'aggregate_arena_preferences_by_query', 'calculate_arena_model_scores',
    'process_final_data', 'generate_embeddings_for_data',
    'TASK_DESCRIPTIONS', 'TASK_CATEGORIES', 'API_KEYS', 'HF_TOKEN', 'CASE_NUM',
    'setup_environment',
    'f1_score', 'exact_match_score', 'get_bert_score', 'evaluate_code', 'cem_score',
    'call_api'
]
