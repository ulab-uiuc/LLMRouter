"""
Custom prompt formatters for user-defined tasks.

This folder contains custom prompt formatting functions that extend
the built-in task formatters in llmrouter.utils.prompting.

To add a new custom task formatter:
1. Create a new Python file in this folder
2. Import register_prompt from llmrouter.utils.prompting
3. Use @register_prompt('task_name') decorator
4. Import this module in your main script to register it
"""

