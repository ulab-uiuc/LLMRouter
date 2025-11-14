"""
Automix Data Pipeline
---------------------
This module contains data preparation functions for the Automix router.

Combines functionality from:
- Step1_SolveQueries.py: Get predictions from small and large models
- Step2_SelfVerify.py: Perform self-verification and categorization

Original source: automix/colabs/
Adapted for LLMRouter framework.
"""

import os
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# ============================================================================
# Environment and API Configuration
# ============================================================================

def _env_or(default_value: str, *env_keys: str) -> str:
    """
    Get environment variable or return default value.

    Args:
        default_value: Default value if no env var is found
        *env_keys: Environment variable keys to check

    Returns:
        Environment variable value or default
    """
    for k in env_keys:
        v = os.environ.get(k)
        if v and len(v.strip()) > 0:
            return v.strip()
    return default_value


def init_providers() -> None:
    """
    Initialize API providers (HuggingFace, OpenAI, etc.).

    This function should be called before using any API-based functions.
    """
    try:
        from huggingface_hub import login
        import openai

        os.environ.pop("HF_ENDPOINT", None)
        hf_token = _env_or(
            "your_hf_token", "HF_TOKEN", "HUGGINGFACE_TOKEN"
        )
        login(token=hf_token)

        openai.api_key = _env_or(
            (
                "your_api_key"
            ),
            "OPENAI_API_KEY",
            "NVIDIA_API_KEY",
            "NVAPI_KEY",
        )
        openai.api_base = _env_or(
            "https://integrate.api.nvidia.com/v1", "OPENAI_API_BASE", "NVIDIA_API_BASE"
        )
    except ImportError:
        print("Warning: Some API providers could not be initialized")


# ============================================================================
# Tokenizer Management
# ============================================================================

_tokenizer_singleton = None


def get_tokenizer():
    """
    Get singleton tokenizer instance.

    Returns:
        AutoTokenizer instance
    """
    global _tokenizer_singleton
    if _tokenizer_singleton is None:
        _tokenizer_singleton = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b")
    return _tokenizer_singleton


# ============================================================================
# Prompt Templates
# ============================================================================

dataset_prompts_and_instructions = {
    "router": {
        "instruction": (
            "You are given  a question. Answer the question as "
            "concisely as you can, using a single phrase if possible."
        ),
        "prompt": """
{instruction}

Question: {question}

Answer: The answer is'""",
    }
}

verifier_prompt = """

Question: Whose lost work was discovered in a dusty attic in 1980?

AI Generated Answer: Shakespeare

Instruction: Your task is to evaluate if the AI Generated Answer is correct,
based on the provided question. Provide the judgement and reasoning for each
case. Choose between Correct or Incorrect.

Evaluation: The lost work of Shakespeare was discovered in 1980 in a dusty attic.

Verification Decision: The AI generated answer is Correct.

---


Question: In which month does the celestial event, the Pink Moon, occur?

AI Generated Answer: July

Instruction: Your task is to evaluate if the AI Generated Answer is correct,
based on the provided question. Provide the judgement and reasoning for each
case. Choose between Correct or Incorrect.

Evaluation: The Pink Moon is unique to the month of April.

Verification Decision: The AI generated answer is Incorrect.

---


Question: Who is believed to have painted the Mona Lisa in the early 16th century?

AI Generated Answer: Vincent van Gogh

Instruction: Your task is to evaluate if the AI Generated Answer is correct,
based on the provided question. Provide the judgement and reasoning for each
case. Choose between Correct or Incorrect.

Evaluation: The  Mona Lisa was painted by Leonardo da Vinci in the early 16th century.

Verification Decision: The AI generated answer is Incorrect.

---


Question: How far away is the planet Kepler-442b?

AI Generated Answer: 1,100 light-years

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The Kepler-442b is located 1,100 light-years away.

Verification Decision: The AI generated answer is Correct.

---

Question: {question}

AI Generated Answer: {generated_answer}

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation:"""


# ============================================================================
# API Call Functions
# ============================================================================

def call_openai_api(
    prompt: str,
    engine_name: str,
    temperature: float = 0.0,
    n: int = 1,
    stop: str = None,
    max_tokens: int = 100,
    batch_size: int = 32,
):
    """
    Call OpenAI API to get model predictions.

    Args:
        prompt: Input prompt
        engine_name: Model engine name
        temperature: Sampling temperature
        n: Number of completions
        stop: Stop sequence
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for API calls

    Returns:
        Single response string or list of responses
    """
    try:
        import openai
    except ImportError:
        print("Error: openai package not installed")
        return None

    all_responses = []
    orig_n = n

    try:
        while n > 0:
            current_batch_size = min(n, batch_size)
            response = openai.ChatCompletion.create(
                model=engine_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=current_batch_size,
                max_tokens=max_tokens,
            )
            all_responses.extend(
                [choice["message"]["content"] for choice in response["choices"]]
            )
            n -= current_batch_size
        return all_responses if orig_n > 1 else all_responses[0]
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


# ============================================================================
# Text Processing Functions
# ============================================================================

def normalize_answer(s: str) -> str:
    """
    Normalize answer string for comparison.

    Args:
        s: Input string

    Returns:
        Normalized string
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)

    # Clean quotes and trailing periods
    s = s.strip().strip("'").strip('"')
    if s.endswith("."):
        s = s[:-1]

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def clean_answer(ans: str) -> str:
    """
    Clean answer by removing quotes.

    Args:
        ans: Input answer string

    Returns:
        Cleaned answer or NA
    """
    return ans.replace("'", "") if ans else pd.NA


# ============================================================================
# Evaluation Metrics
# ============================================================================

def f1_score_single(prediction: str, ground_truth: str) -> float:
    """
    Calculate F1 score between prediction and ground truth.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_f1(
    prediction: Union[str, None], ground_truth: Union[str, List[str], None]
) -> float:
    """
    Compute F1 score, handling multiple ground truths.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer(s)

    Returns:
        F1 score
    """
    if prediction is None or (isinstance(prediction, float) and pd.isna(prediction)):
        return 0.0
    if isinstance(ground_truth, list):
        if len(ground_truth) == 0:
            return 0.0
        return max(f1_score_single(prediction, gt) for gt in ground_truth)
    return f1_score_single(
        prediction, ground_truth if ground_truth is not None else ""
    )


def calculate_f1_for_models(
    df: pd.DataFrame, model_sizes: List[str], ground_truth_col: str = "gt"
) -> pd.DataFrame:
    """
    Calculate F1 scores for multiple models.

    Args:
        df: Input dataframe
        model_sizes: List of model size identifiers
        ground_truth_col: Column name for ground truth

    Returns:
        DataFrame with F1 scores added
    """
    for size in model_sizes:
        pred_col = f"llama{size}_pred_ans"
        f1_col = f"llama{size}_f1"
        df[f1_col] = df.apply(
            lambda r: compute_f1(r.get(pred_col, None), r.get(ground_truth_col, None)),
            axis=1,
        )
    return df


def calculate_f1_for_multi_choice(
    df: pd.DataFrame, model_sizes: List[str], datasets: List[str] = ["quality"]
) -> pd.DataFrame:
    """
    Calculate F1 scores for multiple choice questions.

    Args:
        df: Input dataframe
        model_sizes: List of model size identifiers
        datasets: List of dataset names with multiple choice format

    Returns:
        DataFrame with F1 scores updated for multiple choice
    """

    def extract_option(row: pd.Series) -> str:
        options = re.findall(
            r"\((\w)\)\s+([\w\W]*?)(?=\s*\(\w\)\s+|$)", row["question"]
        )
        for option, value in options:
            if value.strip() == str(row["output"]).strip():
                return option
        return None

    def extract_option_from_prediction(pred: str) -> str:
        if (
            pred is None
            or (isinstance(pred, float) and pd.isna(pred))
            or len(str(pred).strip()) == 0
        ):
            return None
        option_token = str(pred).split()[0]
        for ch in option_token:
            if ch in ["A", "B", "C", "D"]:
                return ch
        return None

    if "question" in df.columns and "output" in df.columns:
        df["correct_option"] = df.apply(extract_option, axis=1)
    else:
        df["correct_option"] = None

    for size in model_sizes:
        pred_ans_col = f"llama{size}_pred_ans"
        pred_option_col = f"llama{size}_pred_option"
        f1_col = f"llama{size}_f1"

        def clean_pred(r):
            val = r.get(pred_ans_col, None)
            if isinstance(val, str) and r.get("dataset") in datasets:
                return val.replace("'", "")
            return val

        df[pred_ans_col] = df.apply(clean_pred, axis=1)
        df[pred_option_col] = df[pred_ans_col].apply(extract_option_from_prediction)

        def maybe_override_f1(r):
            if r.get("dataset") in datasets:
                return (
                    1.0
                    if (
                        r.get(pred_option_col) is not None
                        and r.get(pred_option_col) == r.get("correct_option")
                    )
                    else 0.0
                )
            return r.get(f1_col, 0.0)

        df[f1_col] = df.apply(maybe_override_f1, axis=1)

    return df


# ============================================================================
# Verification Functions
# ============================================================================

def make_verifier_input(question: str, generated_answer: str) -> str:
    """
    Create verifier prompt from question and answer.

    Args:
        question: Question text
        generated_answer: Generated answer to verify

    Returns:
        Formatted verifier prompt
    """
    prompt_text = verifier_prompt.format(
        question=question, generated_answer=generated_answer
    )
    tokens = get_tokenizer().tokenize(prompt_text)
    if len(tokens) > 3950:
        tokens = tokens[-3950:]
        truncated_prompt = get_tokenizer().convert_tokens_to_string(tokens)
    else:
        truncated_prompt = prompt_text

    return truncated_prompt


def run_verification(
    df: pd.DataFrame,
    ans_col: str,
    engine_name: str,
    temperature: float = 1.0,
    n: int = 8,
    stop: str = "---",
    max_tokens: int = 250,
    max_workers: int = 15,
) -> List:
    """
    Run verification on predictions.

    Args:
        df: Input dataframe
        ans_col: Column name containing answers to verify
        engine_name: Model engine name for verification
        temperature: Sampling temperature
        n: Number of verification samples
        stop: Stop sequence
        max_tokens: Maximum tokens to generate
        max_workers: Number of parallel workers

    Returns:
        List of verification results
    """
    verifier_inputs = df.apply(
        lambda row: make_verifier_input(row["question"], row[ans_col]),
        axis=1,
    )
    verifier_call = partial(
        call_openai_api,
        engine_name=engine_name,
        temperature=temperature,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
    )

    print("Inputs prepared, starting verification now.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(executor.map(verifier_call, verifier_inputs), total=df.shape[0])
        )
    return results


def compute_fraction_correct(lst: List[str]) -> float:
    """
    Compute fraction of verifications marked as correct.

    Args:
        lst: List of verification responses

    Returns:
        Fraction of correct verifications
    """
    total_valid = sum(
        [1 for item in lst if "the ai generated answer is" in item.lower()]
    )
    if total_valid == 0:
        return 0
    correct_count = sum(
        [1 for item in lst if "the ai generated answer is correct" in item.lower()]
    )
    return correct_count / total_valid


def categorize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize rows based on model performance.

    Categories:
    - NEEDY: Small model worse than large model
    - GOOD: Both models perform equally well
    - HOPELESS: Both models perform poorly

    Args:
        df: Input dataframe

    Returns:
        DataFrame with 'category' column added
    """
    # Calculate 10th percentile values
    p_10_13b = df["llama13b_f1"].quantile(0.10)
    p_10_70b = df["llama70b_f1"].quantile(0.10)

    # Define conditions for each category
    conditions = [
        (df["llama13b_f1"] <= df["llama70b_f1"])
        & (df["llama13b_f1"] != df["llama70b_f1"]),
        (df["llama13b_f1"] == df["llama70b_f1"]) & (df["llama13b_f1"] != 0),
        (df["llama13b_f1"] <= p_10_13b) & (df["llama70b_f1"] <= p_10_70b),
    ]

    categories = ["NEEDY", "GOOD", "HOPELESS"]
    df["category"] = np.select(conditions, categories, default="UNDEFINED")

    return df


# ============================================================================
# High-Level Pipeline Functions
# ============================================================================

def prepare_row(row: pd.Series, dataset: str = "router") -> str:
    """
    Prepare a single row for model inference.

    Args:
        row: DataFrame row
        dataset: Dataset name

    Returns:
        Formatted prompt
    """
    prompt = dataset_prompts_and_instructions[dataset]["prompt"]
    instruction = dataset_prompts_and_instructions[dataset]["instruction"]
    question = row["query"]
    full_text = prompt.format(instruction=instruction, question=question)
    tokens = get_tokenizer().encode(full_text)
    if len(tokens) > 3096:
        tokens = tokens[-3096:]
    return get_tokenizer().decode(tokens)


def run_solver_job(
    df: pd.DataFrame,
    prepare_row_func,
    engine_name: str,
    max_workers: int = 1,
    temperature: float = 0.0,
    n: int = 1,
    stop: str = "\n",
    max_tokens: int = 100,
    batch_size: int = 32,
) -> List:
    """
    Run solver job on dataframe.

    Args:
        df: Input dataframe
        prepare_row_func: Function to prepare each row
        engine_name: Model engine name
        max_workers: Number of parallel workers
        temperature: Sampling temperature
        n: Number of completions
        stop: Stop sequence
        max_tokens: Maximum tokens
        batch_size: Batch size

    Returns:
        List of model predictions
    """
    solver_call = partial(
        call_openai_api,
        engine_name=engine_name,
        temperature=temperature,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(solver_call, df.apply(prepare_row_func, axis=1)),
                total=df.shape[0],
            )
        )
    return results


def solve_queries(
    data_path: str = "./data/train_test_nq.jsonl",
    save_dir: str = "./data",
    engine_small: str = "meta/llama-3.1-8b-instruct",
    engine_large: str = "meta/llama-3.1-70b-instruct",
    max_workers: int = 1,
    temperature: float = 0.0,
    n: int = 1,
    stop: str = "\n",
    max_tokens: int = 100,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Step 1: Solve queries using small and large models.

    Args:
        data_path: Path to input data
        save_dir: Directory to save results
        engine_small: Small model engine name
        engine_large: Large model engine name
        max_workers: Number of parallel workers
        temperature: Sampling temperature
        n: Number of completions
        stop: Stop sequence
        max_tokens: Maximum tokens
        batch_size: Batch size

    Returns:
        DataFrame with predictions
    """
    init_providers()

    inputs = pd.read_json(data_path, lines=True, orient="records")

    results_13b = run_solver_job(
        inputs,
        prepare_row,
        engine_small,
        max_workers=max_workers,
        temperature=temperature,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )
    results_70b = run_solver_job(
        inputs,
        prepare_row,
        engine_large,
        max_workers=max_workers,
        temperature=temperature,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )

    inputs["llama13b_pred_ans"] = [clean_answer(ans) for ans in results_13b]
    inputs["llama70b_pred_ans"] = [clean_answer(ans) for ans in results_70b]

    inputs_with_predictions = inputs
    print(f"{len(inputs_with_predictions)}/{len(inputs)} inputs have predictions")

    model_sizes = ["13b", "70b"]
    inputs_with_predictions = calculate_f1_for_models(
        inputs_with_predictions, model_sizes
    )
    inputs_with_predictions = calculate_f1_for_multi_choice(
        inputs_with_predictions, model_sizes
    )

    print(
        "result f1 scores: llama13b",
        inputs_with_predictions[["llama13b_f1"]].mean(),
    )
    print(
        "result f1 scores: llama70b",
        inputs_with_predictions[["llama70b_f1"]].mean(),
    )

    # Prepare save dataframe
    def _first_or_str(x):
        if isinstance(x, list):
            return x[0] if len(x) > 0 else None
        return x

    cols = {}
    cols["base_ctx"] = (
        inputs_with_predictions["base_ctx"]
        if "base_ctx" in inputs_with_predictions.columns
        else ""
    )
    cols["question"] = (
        inputs_with_predictions["question"]
        if "question" in inputs_with_predictions.columns
        else inputs_with_predictions.get("query", "")
    )
    cols["output"] = (
        inputs_with_predictions["output"]
        if "output" in inputs_with_predictions.columns
        else inputs_with_predictions.get("gt", None).apply(_first_or_str)
    )
    cols["dataset"] = (
        inputs_with_predictions["dataset"]
        if "dataset" in inputs_with_predictions.columns
        else "router"
    )
    cols["split"] = (
        inputs_with_predictions["split"]
        if "split" in inputs_with_predictions.columns
        else None
    )
    cols["llama13b_pred_ans"] = inputs_with_predictions.get("llama13b_pred_ans")
    cols["llama70b_pred_ans"] = inputs_with_predictions.get("llama70b_pred_ans")

    save_df = pd.DataFrame(cols)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "router_automix_llamapair_outputs.jsonl")
    save_df.to_json(save_path, lines=True, orient="records", force_ascii=False)
    print(f"Saved Step1 outputs for Step2: {save_path} | rows={len(save_df)}")

    return save_df


def self_verify(
    input_path: str = "./data/router_automix_llamapair_outputs.jsonl",
    save_path: str = "./data/router_automix_llamapair_ver_outputs.jsonl",
    engine_name: str = "meta/llama-3.1-8b-instruct",
    temperature: float = 1.0,
    n: int = 2,
    stop: str = "---",
    max_tokens: int = 250,
    max_workers: int = 1,
    verifier_on_column: str = "llama13b_pred_ans",
) -> pd.DataFrame:
    """
    Step 2: Perform self-verification on predictions.

    Args:
        input_path: Path to input data from Step 1
        save_path: Path to save results
        engine_name: Model engine name for verification
        temperature: Sampling temperature
        n: Number of verification samples
        stop: Stop sequence
        max_tokens: Maximum tokens
        max_workers: Number of parallel workers
        verifier_on_column: Column to verify

    Returns:
        DataFrame with verification scores and categories
    """
    init_providers()
    df = pd.read_json(input_path, lines=True, orient="records")

    ver_results = run_verification(
        df,
        ans_col=verifier_on_column,
        engine_name=engine_name,
        temperature=temperature,
        n=n,
        stop=stop,
        max_tokens=max_tokens,
        max_workers=max_workers,
    )
    print(ver_results)
    df["llama13b_ver"] = ver_results
    df["p_ver_13b"] = df["llama13b_ver"].apply(compute_fraction_correct)

    # Compute F1 scores
    df["llama13b_f1"] = df.apply(
        lambda r: f1_score_single(r.get("llama13b_pred_ans"), r.get("output")),
        axis=1,
    )
    df["llama70b_f1"] = df.apply(
        lambda r: f1_score_single(r.get("llama70b_pred_ans"), r.get("output")),
        axis=1,
    )

    df = categorize_rows(df)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_json(save_path, lines=True, orient="records")
    print(f"Saved Step2 outputs: {save_path} | rows={len(df)}")
    return df


def prepare_automix_data(
    input_data_path: str,
    output_dir: str = "./data",
    engine_small: str = "meta/llama-3.1-8b-instruct",
    engine_large: str = "meta/llama-3.1-70b-instruct",
    skip_step1: bool = False,
    skip_step2: bool = False,
) -> pd.DataFrame:
    """
    Complete data preparation pipeline for Automix.

    Runs both Step 1 (solve queries) and Step 2 (self verify).

    Args:
        input_data_path: Path to input data
        output_dir: Directory to save intermediate and final results
        engine_small: Small model engine name
        engine_large: Large model engine name
        skip_step1: Skip Step 1 if already done
        skip_step2: Skip Step 2 if already done

    Returns:
        Final prepared dataframe with predictions, verification, and categories
    """
    step1_output = os.path.join(output_dir, "router_automix_llamapair_outputs.jsonl")
    step2_output = os.path.join(
        output_dir, "router_automix_llamapair_ver_outputs.jsonl"
    )

    if not skip_step1:
        print("=" * 60)
        print("Step 1: Solving queries with small and large models")
        print("=" * 60)
        solve_queries(
            data_path=input_data_path,
            save_dir=output_dir,
            engine_small=engine_small,
            engine_large=engine_large,
        )

    if not skip_step2:
        print("\n" + "=" * 60)
        print("Step 2: Self-verification and categorization")
        print("=" * 60)
        df_final = self_verify(
            input_path=step1_output,
            save_path=step2_output,
            engine_name=engine_small,
        )
    else:
        df_final = pd.read_json(step2_output, lines=True, orient="records")

    return df_final
