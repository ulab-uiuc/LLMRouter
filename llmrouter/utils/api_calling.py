"""
API calling utilities using LiteLLM for API calls with manual load balancing

This module provides functions for making API calls to LLM services with
manual round-robin load balancing across multiple API keys.
"""

import os
import json
import time
from typing import Dict, List, Union, Optional, Any
from litellm import completion

# Global counter for round-robin API key selection
# Key: (api_endpoint, api_name) -> counter value
_api_key_counters: Dict[tuple, int] = {}


def _parse_api_keys(api_keys_env: Optional[str] = None) -> List[str]:
    """
    Parse API keys from environment variable.
    
    Supports both single string and JSON list format:
    - Single key: "your-api-key"
    - Multiple keys: '["key1", "key2", "key3"]'
    
    Args:
        api_keys_env: Environment variable value for API_KEYS.
                     If None, reads from os.environ['API_KEYS']
    
    Returns:
        List of API key strings
    
    Raises:
        ValueError: If API_KEYS is not set or invalid
    """
    if api_keys_env is None:
        api_keys_env = os.environ.get('API_KEYS', '')
    
    if not api_keys_env:
        raise ValueError("API_KEYS environment variable is not set")
    
    # Try to parse as JSON (list format)
    try:
        parsed = json.loads(api_keys_env)
        if isinstance(parsed, list):
            return [str(key) for key in parsed if key]
        elif isinstance(parsed, str):
            return [parsed] if parsed else []
    except (json.JSONDecodeError, TypeError):
        pass
    
    # If not JSON, treat as single string
    if isinstance(api_keys_env, str) and api_keys_env.strip():
        return [api_keys_env.strip()]
    
    raise ValueError(f"Invalid API_KEYS format: {api_keys_env}")


def _get_api_key(
    api_endpoint: str,
    api_name: str,
    api_keys: List[str],
    is_batch: bool = False,
    request_index: int = 0
) -> str:
    """
    Get an API key using round-robin selection.
    
    For single requests, uses a counter that increments per call.
    For batch requests, distributes requests across keys based on request_index.
    
    Args:
        api_endpoint: API endpoint URL (for counter key)
        api_name: API model name (for counter key)
        api_keys: List of available API keys
        is_batch: Whether this is part of a batch request
        request_index: Index of the request in a batch (only used for batch requests)
    
    Returns:
        Selected API key string
    """
    if not api_keys:
        raise ValueError("No API keys provided")
    
    cache_key = (api_endpoint, api_name)
    
    if is_batch:
        # Batch request: distribute based on request_index
        selected_index = request_index % len(api_keys)
    else:
        # Single request: use counter and increment
        if cache_key not in _api_key_counters:
            _api_key_counters[cache_key] = 0
        
        selected_index = _api_key_counters[cache_key] % len(api_keys)
        _api_key_counters[cache_key] = (_api_key_counters[cache_key] + 1) % len(api_keys)
    
    return api_keys[selected_index]


def call_api(
    request: Union[Dict[str, Any], List[Dict[str, Any]]],
    api_keys_env: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.01,
    top_p: float = 0.9,
    timeout: int = 30,
    max_retries: int = 3
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Call LLM API using LiteLLM completion with manual round-robin load balancing.
    
    This function distributes API calls evenly across multiple API keys using
    manual round-robin selection. For batch requests, requests are distributed
    across API keys based on their index in the batch.
    
    Args:
        request: Single dict or list of dicts, each containing:
            - api_endpoint (str): API endpoint URL
            - query (str): The query/prompt to send
            - model_name (str): Model identifier name (not used for API call)
            - api_name (str): Actual API model name/path (e.g., "qwen/qwen2.5-7b-instruct")
        api_keys_env: Optional override for API_KEYS env var (for testing)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        timeout: Request timeout in seconds
        max_retries: Maximum retries for failed requests
    
    Returns:
        Single dict or list of dicts (matching input format) with added fields:
            - response (str): API response text
            - token_num (int): Total tokens used
            - prompt_tokens (int): Input tokens
            - completion_tokens (int): Output tokens
            - response_time (float): Time taken in seconds
            - error (str, optional): Error message if request failed
    
    Example:
        Single request:
        >>> request = {
        ...     "api_endpoint": "https://integrate.api.nvidia.com/v1",
        ...     "query": "What is 2+2?",
        ...     "model_name": "qwen2.5-7b-instruct",
        ...     "api_name": "qwen/qwen2.5-7b-instruct"
        ... }
        >>> result = call_api(request)
        >>> print(result["response"])
        
        Batch requests (distributed across API keys):
        >>> requests = [request1, request2, request3]
        >>> results = call_api(requests)
    """
    # Parse API keys from environment
    api_keys = _parse_api_keys(api_keys_env)
    
    # Handle single request vs batch
    is_single = isinstance(request, dict)
    requests = [request] if is_single else request
    
    # Validate request format
    required_keys = {'api_endpoint', 'query', 'model_name', 'api_name'}
    for req in requests:
        missing = required_keys - set(req.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
    
    results = []
    
    # Process each request
    for idx, req in enumerate(requests):
        result = req.copy()
        start_time = time.time()
        
        try:
            # Select API key using round-robin
            # For batch requests, use request index; for single requests, use counter
            selected_api_key = _get_api_key(
                api_endpoint=req['api_endpoint'],
                api_name=req['api_name'],
                api_keys=api_keys,
                is_batch=not is_single,
                request_index=idx
            )
            
            # Make API call using LiteLLM completion directly
            # Format: openai/{api_name} tells LiteLLM to use OpenAI-compatible client
            model_for_litellm = f"openai/{req['api_name']}"
            
            response = completion(
                model=model_for_litellm,
                messages=[{"role": "user", "content": req['query']}],
                api_key=selected_api_key,
                api_base=req['api_endpoint'],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            usage = response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None
            
            # Extract token counts
            if usage:
                token_num = usage.get("total_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                # Fallback estimation
                prompt_tokens = len(req['query'].split()) if req['query'] else 0
                completion_tokens = len(response_text.split()) if isinstance(response_text, str) else 0
                token_num = prompt_tokens + completion_tokens
            
            end_time = time.time()
            
            # Add results to response
            result['response'] = response_text
            result['token_num'] = token_num
            result['prompt_tokens'] = prompt_tokens
            result['completion_tokens'] = completion_tokens
            result['response_time'] = end_time - start_time
            
        except Exception as e:
            error_msg = str(e)
            end_time = time.time()
            
            # Add error information
            result['response'] = f"API Error: {error_msg[:200]}"
            result['token_num'] = 0
            result['prompt_tokens'] = 0
            result['completion_tokens'] = 0
            result['response_time'] = end_time - start_time
            result['error'] = error_msg
        
        results.append(result)
    
    # Return single result or list based on input
    return results[0] if is_single else results

