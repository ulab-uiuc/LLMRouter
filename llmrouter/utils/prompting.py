"""
Prompt formatting utilities for LLMRouter scripts
"""

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

def generate_task_query(task_name, sample_data):
    """Generate query prompt based on task name and sample data"""
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








