# Custom Prompts

This folder contains custom prompt formatters for user-defined tasks.

## How to Add a Custom Task Formatter

1. **Create a new Python file** in this folder (e.g., `my_custom_task.py`)

2. **Import the decorator**:
   ```python
   from llmrouter.utils.prompting import register_prompt
   ```

3. **Define your formatter function** with the `@register_prompt` decorator:
   ```python
   @register_prompt('my_task_name')
   def format_my_task(sample_data):
       # Your formatting logic here
       query = sample_data['query']
       return f"Formatted: {query}"
   ```

4. **Import the module** in your main script to trigger registration:
   ```python
   from llmrouter.utils import generate_task_query
   import custom_prompts.my_custom_task  # Import triggers decorator
   
   # Now you can use it
   result = generate_task_query('my_task_name', {'query': 'test'})
   ```

## Example Files

- `example_custom_task.py` - Contains example custom formatters (`code_refine`, `summarization`)

## Notes

- The decorator executes when the module is imported
- You can register multiple tasks in one file
- Custom tasks take precedence over built-in tasks with the same name
- See `llmrouter/utils/prompting.py` for built-in task formatters

