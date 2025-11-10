"""
Prompt formatting utilities for LLMRouter scripts
"""

from typing import List, Dict, Any

def format_mc_prompt(question, choices):
    """Format prompt for multiple choice tasks"""
    formatted_choices = ""
    options = ["A", "B", "C", "D"]
    
    for i, choice in enumerate(choices):
        formatted_choices += f"{options[i]}. {choice}\n"
    
    return f"""Answer the following multiple-choice question by selecting the correct option (A, B, C, or D). You Must put your final answer letter in a parenthesis.

## Question:
{question}

## Options:
{formatted_choices}
"""

def format_gsm8k_prompt(query):
    """Format prompt for GSM8K math tasks"""
    return f"""
    Answer the following question.

    Question: {query}
    """

def format_math_prompt(query):
    """Format prompt for MATH tasks"""
    return f"""
    Answer the following question. Make sure to put the answer ( and only answer ) inside \\boxed{{}}.

    Question: {query}
    """

def format_commonsense_qa_prompt(query, choices):
    """Format prompt for commonsense QA tasks"""
    label = choices["label"]
    text = choices["text"]
    choice_text = ""
    for i, j in zip(label, text):
        choice_text += "\n" + "(" + i + ")" + " " + j
    return f"""
    Answer the following multiple-choice question by selecting the correct option. You Must put your final answer letter in a parenthesis.

    Question: {query} \n
    """ + choice_text

def format_mbpp_prompt(text, tests):
    """Format prompt for MBPP code generation tasks"""
    tests_str = "\n".join(tests)
    return f"You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{tests_str}\n Implement the function (no irrelevant words or comments) in this format: [BEGIN] <Your Code> [Done]\n"

def format_humaneval_prompt(prompt):
    """Format prompt for HumanEval code generation tasks"""
    return f"""You are an expert Python programmer. Complete the following function:

            {prompt}

            Implement the function body only. Do not repeat the function signature or docstring.
            Put the code in this format: [BEGIN] <Your Code - function body only> [Done]
            """


# ============================================================================
# Decorator-based prompt registration system
# ============================================================================

# Registry for prompt formatters
PROMPT_REGISTRY = {}


def register_prompt(task_name: str):
    """
    Decorator to register a custom prompt formatter for a task.
    
    The decorated function should take one argument:
    - sample_data: dict - Dictionary containing task data
    
    Args:
        task_name: Name of the task to register (e.g., 'my_custom_task')
    
    Returns:
        The decorated function (unchanged)
    
    Usage:
        # custom_prompts/my_task.py
        from llmrouter.utils.prompting import register_prompt
        
        @register_prompt('my_custom_task')
        def format_my_task(sample_data):
            query = sample_data['query']
            return f"Custom format: {query}"
        
        Then import the module to register it:
        
        # main.py
        from llmrouter.utils import generate_task_query
        import custom_prompts.my_task  # Import to trigger decorator registration
        
        result = generate_task_query('my_custom_task', {'query': 'test'})
    """
    def decorator(func):
        PROMPT_REGISTRY[task_name] = func
        return func
    return decorator


def generate_task_query(task_name, sample_data):
    """
    Generate query prompt based on task name and sample data.
    
    First checks if the task is registered via @register_prompt decorator.
    If not found, falls back to built-in task formatters.
    
    Args:
        task_name: Name of the task
        sample_data: Dictionary containing task-specific data (query, choices, etc.)
    
    Returns:
        Formatted prompt string
    
    Raises:
        ValueError: If task_name is not registered and not in built-in tasks
    """
    # Check if task is registered via decorator
    if task_name in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[task_name](sample_data)
    
    # Fall back to built-in formatters
    if task_name in ["natural_qa", "trivia_qa"]:
        return sample_data['query']
    elif task_name in ["mmlu"]:
        return format_mc_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "gpqa":
        return format_mc_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "mbpp":
        return format_mbpp_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "human_eval":
        return format_humaneval_prompt(sample_data['query'])
    elif task_name == "gsm8k":
        return format_gsm8k_prompt(sample_data['query'])
    elif task_name == "commonsense_qa":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "math":
        return format_math_prompt(sample_data['query'])
    elif task_name == "openbook_qa":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    elif task_name == "arc_challenge":
        return format_commonsense_qa_prompt(sample_data['query'], sample_data['choices'])
    else:
        raise ValueError(f"Unknown task name: {task_name}")


def format_batch_queries(data: List[Dict[str, Any]], task_name_key: str = 'task_name') -> List[str]:
    """
    Format queries for a batch of data (list of dictionaries, e.g., JSONL elements).
    
    Each dictionary in the input list should contain:
    - A task_name field (or specify custom key via task_name_key)
    - Task-specific data fields (query, choices, etc.)
    
    Args:
        data: List of dictionaries, each representing a task sample
        task_name_key: Key name in each dictionary that contains the task name (default: 'task_name')
    
    Returns:
        List of formatted prompt strings, one for each input dictionary
    
    Example:
        jsonl_data = [
            {'task_name': 'gsm8k', 'query': 'What is 2+2?'},
            {'task_name': 'em', 'query': 'hello', 'ground_truth': 'hello'}
        ]
        formatted = format_batch_queries(jsonl_data)
        # Returns: [formatted_gsm8k_prompt, formatted_em_prompt]
    
    Raises:
        ValueError: If a task_name is not found and not registered
        KeyError: If task_name_key is missing from a dictionary
    """
    formatted_queries = []
    
    for item in data:
        # Get task name from the dictionary
        if task_name_key not in item:
            raise KeyError(f"Task name key '{task_name_key}' not found in item: {item}")
        
        task_name = item[task_name_key]
        
        # Use the item itself as sample_data (it contains all the fields)
        formatted_query = generate_task_query(task_name, item)
        formatted_queries.append(formatted_query)
    
    return formatted_queries


def get_available_tasks() -> List[str]:
    """
    Get list of all available task names (both registered and built-in).
    
    Returns:
        List of task names
    """
    built_in_tasks = [
        "natural_qa", "trivia_qa", "mmlu", "gpqa", "mbpp", "human_eval",
        "gsm8k", "commonsense_qa", "math", "openbook_qa", "arc_challenge"
    ]
    registered_tasks = list(PROMPT_REGISTRY.keys())
    return sorted(set(built_in_tasks + registered_tasks))




