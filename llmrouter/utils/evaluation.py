import json
import os
import re
import string
import pickle
from collections import Counter
from typing import Any, List, Optional, Tuple

import numpy as np


# File I/O functions
def loadjson(filename: str) -> dict:
    """
    Load data from a JSON file.

    Args:
        filename: Path to the JSON file

    Returns:
        Dictionary containing the loaded JSON data
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def savejson(data: dict, filename: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Dictionary to save
        filename: Path where the JSON file will be saved
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def loadpkl(filename: str) -> Any:
    """
    Load data from a pickle file.

    Args:
        filename: Path to the pickle file

    Returns:
        The unpickled object
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def savepkl(data: Any, filename: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data: Object to save
        filename: Path where the pickle file will be saved
    """
    with open(filename, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


# Text normalization and evaluation functions
def normalize_answer(s: str, normal_method: str = "") -> str:
    """
    Normalize text for evaluation.

    Args:
        s: String to normalize
        normal_method: Method for normalization ("mc" for multiple choice, "" for standard)

    Returns:
        Normalized string
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def mc_remove(text):
        a1 = re.findall(r'\(\s*[a-zA-Z]\s*\)', text)
        if len(a1) == 0:
            return ""
        return a1[-1]

    if normal_method == "mc":
        return mc_remove(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Calculate F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str, normal_method: str = "") -> bool:
    """
    Check if prediction exactly matches ground truth after normalization.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text
        normal_method: Method for normalization

    Returns:
        True if exact match, False otherwise
    """
    if normal_method == "mc":
        return ground_truth.strip().lower() in normalize_answer(prediction, normal_method=normal_method).strip().lower()
    return (normalize_answer(prediction, normal_method=normal_method) ==
            normalize_answer(ground_truth, normal_method=normal_method))


def cemf1_score(prediction: str, ground_truth: str):
    norm_prediction = normalize_answer(prediction, normal_method="")
    norm_gt = normalize_answer(ground_truth, normal_method="")
    if norm_prediction == norm_gt or norm_gt in norm_prediction:
        return 1.0
    else:
        return f1_score(prediction=prediction, ground_truth=ground_truth)[0]


def cem_score(prediction: str, ground_truth: str):
    norm_prediction = normalize_answer(prediction, normal_method="")
    norm_gt = normalize_answer(ground_truth, normal_method="")
    if norm_prediction == norm_gt or norm_gt in norm_prediction:
        return 1.0
    else:
        return 0.0


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    """Remove \\boxed{} wrapper from LaTeX string."""
    if "\\boxed " in s:
        left = "\\boxed "
        if s[:len(left)] == left:
            return s[len(left):]
        # If doesn't start with \boxed but contains it, try to extract
        idx = s.find(left)
        if idx >= 0:
            return s[idx + len(left):]

    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]

    # Try to find and extract \boxed{...} anywhere in string
    idx = s.find(left)
    if idx >= 0:
        # Find matching closing brace
        depth = 0
        start = idx + len(left)
        for i in range(start, len(s)):
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                if depth == 0:
                    return s[start:i]
                depth -= 1

    return s  # Return original if no valid boxed content found


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    """Remove units from LaTeX string (e.g., \\text{ meters})."""
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) >= 2:
            return splits[0]
        # If only one part, return as-is
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def calculate_task_performance(
    prediction: str, 
    ground_truth: Optional[str], 
    task_name: Optional[str] = None,
    metric: Optional[str] = None
) -> Optional[float]:
    """
    Calculate task performance score for a prediction against ground truth.
    
    Args:
        prediction: The model's response/prediction
        ground_truth: Ground truth answer (optional)
        task_name: Task name to determine metric if not provided
        metric: Evaluation metric to use (optional, will be inferred from task_name if not provided)
        
    Returns:
        Performance score (0.0 to 1.0) or None if ground_truth is not available
    """
    if not ground_truth:
        return None
    
    # Determine metric based on task_name if not provided
    if metric is None and task_name:
        # First check custom task-to-metric registry (for user-defined tasks)
        try:
            from llmrouter.utils.prompting import TASK_METRIC_REGISTRY
            if task_name in TASK_METRIC_REGISTRY:
                metric = TASK_METRIC_REGISTRY[task_name]
        except ImportError:
            pass
        
        # If not found in registry, check built-in task-to-metric mappings
        if metric is None:
            if task_name in ["natural_qa", "trivia_qa", "squad", "boolq"]:
                metric = "cem"
            elif task_name in ["mmlu", "gpqa", "commonsense_qa", "openbook_qa", "arc_challenge"]:
                metric = "em_mc"
            elif task_name == "gsm8k":
                metric = "gsm8k"
            elif task_name == "math":
                metric = "math"
            else:
                metric = "cem"  # Default to CEM
    
    # Evaluate based on metric
    try:
        # First check if metric is registered in EVALUATION_METRICS (for custom metrics)
        try:
            from llmrouter.evaluation.batch_evaluator import EVALUATION_METRICS
            if metric in EVALUATION_METRICS:
                eval_func = EVALUATION_METRICS[metric]
                return float(eval_func(prediction, ground_truth))
        except (ImportError, KeyError):
            pass
        
        # Fall back to built-in metric implementations
        if metric == "em":
            return float(exact_match_score(prediction, ground_truth))
        elif metric == "em_mc":
            return float(exact_match_score(prediction, ground_truth, normal_method="mc"))
        elif metric == "cem":
            return float(cem_score(prediction, ground_truth))
        elif metric == "gsm8k":
            # GSM8K evaluation: extract number from ground truth and prediction
            ground_truth_processed = ground_truth.split("####")[-1].replace(',', '').replace('$', '').replace('.', '').strip()
            answer = re.findall(r"(\-?[0-9\.\,]+)", prediction)
            if len(answer) == 0:
                return 0.0
            invalid_str = ['', '.']
            final_answer = None
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
            if final_answer is None:
                return 0.0
            final_answer = final_answer.replace(',', '').replace('$', '').replace('.', '').strip()
            return 1.0 if final_answer == ground_truth_processed else 0.0
        elif metric == "math":
            # MATH evaluation: extract from \boxed{} format using robust parsing
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
                        return 1.0
            except Exception:
                pass
            return 0.0
        elif metric == "f1":
            f1, _, _ = f1_score(prediction, ground_truth)
            return float(f1)
        else:
            # Default to CEM
            return float(cem_score(prediction, ground_truth))
    except Exception as e:
        print(f"Warning: Error calculating task_performance: {e}")
        return None


def hellaswag_preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def get_bert_score(generate_response: List[str], ground_truth: List[str]) -> float:
    """
    Calculate BERT score between generated responses and ground truths.

    Args:
        generate_response: List of generated responses
        ground_truth: List of ground truth texts

    Returns:
        Average BERT score (F1)
    """
    try:
        from bert_score import score as bert_score
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "BERTScore metric requires extra dependency `bert-score`. "
            "Install it with: `pip install bert-score`."
        ) from e

    F_l = []
    for inter in range(len(generate_response)):
        generation = generate_response[inter]
        gt = ground_truth[inter]
        P, R, F = bert_score([generation], [gt], lang="en", verbose=False)
        F_l.append(F.mean().numpy().reshape(1)[0])
    return np.array(F_l).mean()



# this is the code used to evluates generated code against test case 
def evaluate_code(generated_code, test_cases, timeout=5):
    """
    Evaluates generated code against test cases
    
    Args:
        generated_code (str): The code generated by the model
        test_cases (list): List of test case strings (assertions)
        timeout (int): Maximum execution time in seconds
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    import signal
    
    # Create a safe execution environment
    local_vars = {}
    
    # Define timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    try:
        # Set timeout (SIGALRM is not available on Windows)
        alarm_supported = hasattr(signal, "SIGALRM")
        if alarm_supported:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        # Execute the generated code
        exec(generated_code, {}, local_vars)
        
        # Run test cases
        for test in test_cases:
            exec(test, local_vars)
            
        # If we get here, all tests passed
        return True
        
    except AssertionError:
        # Test failed
        return False
    except Exception as e:
        # Code execution error
        print(f"Error during execution: {str(e)}")
        return False
    finally:
        # Disable the alarm
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

# LLM prompting
# def model_prompting(
#         llm_model: str,
#         prompt: str,
#         return_num: Optional[int] = 1,
#         max_token_num: Optional[int] = 512,
#         temperature: Optional[float] = 0.0,
#         top_p: Optional[float] = None,
#         stream: Optional[bool] = None,
# ) -> str:
#     """
#     Get a response from an LLM model using LiteLLM.
#
#     Args:
#         llm_model: Name of the model to use
#         prompt: Input prompt text
#         return_num: Number of completions to generate
#         max_token_num: Maximum number of tokens to generate
#         temperature: Sampling temperature
#         top_p: Top-p sampling parameter
#         stream: Whether to stream the response
#
#     Returns:
#         Generated text response
#     """
#     completion = litellm.completion(
#         model=llm_model,
#         messages=[{'role': 'user', 'content': prompt}],
#         max_tokens=max_token_num,
#         api_key=os.environ.get("NVAPI_KEY", ""),
#         api_base="https://integrate.api.nvidia.com/v1",
#         n=return_num,
#         top_p=top_p,
#         temperature=temperature,
#         stream=stream,
#     )
#     content = completion.choices[0].message.content
#     return content

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

def model_prompting(
    llm_model: str,
    prompt: str,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.2,
    top_p: Optional[float] = 0.7,
    stream: Optional[bool] = True,
) -> str:
    """
    Get a response from an LLM model using the OpenAI-compatible NVIDIA API.

    Args:
        llm_model: Name of the model to use (e.g., "nvdev/nvidia/llama-3.1-nemotron-70b-instruct")
        prompt: Input prompt text
        return_num: Number of completions to generate
        max_token_num: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stream: Whether to stream the response

    Returns:
        Generated text response
    """
    if OpenAI is None:  # pragma: no cover
        raise ImportError(
            "Optional dependency `openai` is required for model_prompting(). "
            "Install it with: `pip install openai`."
        )

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVAPI_KEY")
        or ""
    ).strip()
    if not api_key:
        raise ValueError(
            "Missing API key for model_prompting(); set OPENAI_API_KEY/NVIDIA_API_KEY/NVAPI_KEY."
        )

    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("NVIDIA_API_BASE")
    )
    
    if not base_url:
        raise ValueError(
            "API endpoint (base_url) not found. Please set OPENAI_API_BASE or "
            "NVIDIA_API_BASE environment variable."
        )
    
    base_url = base_url.strip()

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=300,
        max_retries=2,
    )

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_token_num,
        temperature=temperature,
        top_p=top_p,
        stream=stream
    )

    response_text = ""
    if stream:
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        return response_text

    return completion.choices[0].message.content or ""
