"""
Example custom prompt formatter.

This file demonstrates how to create a custom prompt formatter for a new task
without modifying the original prompting.py file.
"""

from llmrouter.utils.prompting import register_prompt


@register_prompt('code_refine')
def format_code_refine_prompt(sample_data):
    """
    Format prompt for code refinement task.
    
    This is a custom task formatter that users can add without modifying
    the original prompting.py file.
    
    Args:
        sample_data: Dictionary containing task data with keys:
            - 'code': str - The code to refine
            - 'instruction': str (optional) - Custom instruction
    
    Returns:
        Formatted prompt string
    """
    code = sample_data.get("code", "")
    instruction = sample_data.get("instruction", "Refine the following code.")
    
    return f"""
{instruction}

Code:
{code}

Please output the refined version below:
"""


@register_prompt('summarization')
def format_summarization_prompt(sample_data):
    """
    Format prompt for text summarization task.
    
    Another example of a custom task formatter.
    
    Args:
        sample_data: Dictionary containing:
            - 'text': str - Text to summarize
            - 'max_length': int (optional) - Maximum summary length
    
    Returns:
        Formatted prompt string
    """
    text = sample_data.get("text", "")
    max_length = sample_data.get("max_length", 100)
    
    return f"""Please summarize the following text in no more than {max_length} words:

{text}

Summary:"""

