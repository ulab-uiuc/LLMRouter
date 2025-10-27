"""
Constants and configurations for LLMRouter scripts
"""

# Task descriptions
TASK_DESCRIPTIONS = {
    "natural_qa": 'Natural Questions consists of real Google search queries paired with full Wikipedia articles.',
    "trivia_qa": 'TriviaQA features complex trivia-style questions with evidence from multiple web sources.',
    "gsm8k": 'GSM8K is a benchmark of grade school math word problems.',
    "commonsense_qa": 'CommonsenseQA is a multiple-choice question dataset that requires commonsense knowledge.',
    "mmlu": 'MMLU covers 57 subjects ranging from STEM to humanities.',
    "gpqa": 'GPQA evaluates graduate-level multiple-choice questions in physics, chemistry, biology.',
    "mbpp": 'MBPP features Python programming tasks with test cases.',
    "human_eval": 'HumanEval is a challenging programming benchmark.',
    "math": 'MATH is a dataset of high school and competition-level mathematics problems.',
    "arc_challenge": 'ARC-Challenge is a benchmark of difficult grade-school science questions.',
    "openbook_qa": 'OpenbookQA consists of elementary science questions.',
    "mt_bench": 'MT Bench is a multi-turn conversation evaluation dataset.',
    "chatbot_arena": 'Chatbot Arena is a human preference evaluation dataset.',
}

# Task categories for analysis
TASK_CATEGORIES = {
    'MATH_TASK': ['gsm8k', 'math'],
    'CODE_TASK': ["mbpp", "human_eval"],
    'COMMONSENSE_TASK': ['commonsense_qa', 'openbook_qa', 'arc_challenge'],
    'WORLD_KNOWLEDGE_TASK': ["natural_qa", "trivia_qa"],
    'POPULAR_TASK': ["mmlu", "gpqa"],
    'PREFERENCE_TASK': ["mt_bench", "chatbot_arena"]
}



# Hugging Face token
HF_TOKEN = ""

# Default configuration
CASE_NUM = 500  # Number of samples per task
