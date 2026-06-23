from typing import Any, Dict, List, Optional

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import call_api, generate_task_query, calculate_task_performance
import copy

def parse_size(size_str: str) -> float:
    """
    Parse a model size string (e.g., '7B', '13B', '512M') into
    a numeric value in billions.

    Supported suffixes:
        - K: thousands
        - M: millions
        - B: billions
        - T: trillions

    If parsing fails, this function returns 0.0.
    """
    size_str = str(size_str).strip().upper()
    try:
        if size_str.endswith("K"):
            return float(size_str[:-1]) / 1e6
        elif size_str.endswith("M"):
            return float(size_str[:-1]) / 1e3
        elif size_str.endswith("B"):
            return float(size_str[:-1])
        elif size_str.endswith("T"):
            return float(size_str[:-1]) * 1e3
        else:
            # Treat raw numeric string as billions directly
            return float(size_str)
    except Exception:
        return 0.0


class LargestLLM(MetaRouter):
    """
    LargestLLM Router
    -----------------
    A heuristic router that always selects the largest LLM based on the
    'size' field in `self.llm_data`, restricted to models whose size
    string ends with 'B'.

    This router does not perform any learning and does not depend on
    the input batch. It only uses metadata loaded from the YAML config.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the LargestLLM router.

        Args:
            yaml_path (str):
                Path to the YAML configuration file. The corresponding
                DataLoader is expected to populate `self.llm_data` based
                on this configuration.
        """
        # Use a dummy model because this router does not train or forward
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)
        print("✅ LargestLLM initialized successfully")

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query to the largest LLM (by size ending with 'B').

        This method ignores the content of the input query and purely relies on
        `self.llm_data`, which should be populated during MetaRouter initialization.

        Args:
            query (dict):
                A single query dictionary. The content is unused here but required
                for interface compatibility with multi-query routing methods.

        Returns:
            dict:
                A dictionary containing:
                    - "model_name": name of the selected model
                    - "model_size": size string of the selected model
                    - "model_info": full metadata entry from `self.llm_data`
        """
        # --- Validate LLM metadata ---
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError(
                "LLM data not loaded or missing in YAML configuration. "
                "Expected `self.llm_data` to be populated by DataLoader."
            )

        # --- Filter models whose size ends with 'B' ---
        filtered_names = [
            name
            for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str)
               and info["size"].upper().endswith("B")
        ]

        if not filtered_names:
            raise ValueError(
                "No models with size ending in 'B' found in `llm_data`."
            )

        # --- Select the largest model among candidates ---
        largest_model_name = max(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B")),
        )

        query_output = copy.copy(query)
        query_output["model_name"]=largest_model_name

        return query_output


    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Select the largest LLM and execute queries with it.

        This method performs end-to-end processing:
        1. Selects the largest model based on size
        2. Applies task-specific prompt formatting if task_name is provided
        3. Calls the model via API to get response
        4. Calculates performance metrics if ground truth is available

        Args:
            batch (Any, optional):
                If provided, routes the provided batch. If None, uses self.query_data_test from loaded data.
            task_name (str, optional):
                Task name for prompt formatting (e.g., "mmlu", "gsm8k", "commonsense_qa").

        Returns:
            list of dict:
                A list of query dictionaries with response, tokens, and performance metrics.
        """
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError(
                "LLM data not loaded or missing in YAML configuration. "
                "Expected `self.llm_data` to be populated by DataLoader."
            )

        # Filter only models whose size ends with 'B'
        filtered_names = [
            name
            for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str)
            and info["size"].upper().endswith("B")
        ]

        if not filtered_names:
            raise ValueError(
                "No models with size ending in 'B' found in `llm_data`."
            )

        # Find the largest model among the filtered candidates
        largest_model_name = max(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B")),
        )

        query_data = self._resolve_query_data(batch)
        if query_data is None:
            return []

        query_data_output = []
        for row in query_data:
            # Handle both dict and non-dict inputs
            row_copy, original_query, row_task_name = self._normalize_row(row, task_name)

            # Step 1: Route - always use largest model
            model_name = largest_model_name
            row_copy["model_name"] = model_name

            # Step 2: Format query if task_name is provided
            if row_task_name:
                try:
                    sample_data = {
                        "query": original_query,
                        "choices": row_copy.get("choices", None) if isinstance(row_copy, dict) else None
                    }
                    formatted_query = generate_task_query(row_task_name, sample_data)
                    row_copy["formatted_query"] = formatted_query
                    query_text_for_execution = formatted_query
                except (ValueError, KeyError) as e:
                    print(f"Warning: Failed to format query with task '{row_task_name}': {e}. Using original query.")
                    query_text_for_execution = original_query
            else:
                query_text_for_execution = original_query

            # Step 3: Call API to get response
            # Get API endpoint and model name from llm_data if available
            api_model_name = model_name
            api_endpoint = None
            service = None
            if hasattr(self, 'llm_data') and self.llm_data and model_name in self.llm_data:
                api_model_name = self.llm_data[model_name].get("model", model_name)
                # Get API endpoint from llm_data, fallback to router config
                api_endpoint = self.llm_data[model_name].get(
                    "api_endpoint",
                    self.cfg.get("api_endpoint")
                )
                # Get service field for service-specific API key selection
                service = self.llm_data[model_name].get("service")
            
            # If still no endpoint found, try router config
            if api_endpoint is None:
                api_endpoint = self.cfg.get("api_endpoint")
            
            # Validate that we have an endpoint
            if not api_endpoint:
                raise ValueError(
                    f"API endpoint not found for model '{model_name}'. "
                    f"Please specify 'api_endpoint' in llm_data JSON for this model or in router YAML config."
                )

            request = {
                "api_endpoint": api_endpoint,
                "query": query_text_for_execution,
                "model_name": model_name,
                "api_name": api_model_name
            }
            # Add service field if available (for service-specific API key selection)
            if service:
                request["service"] = service

            try:
                result = call_api(request, max_tokens=1024, temperature=0.7)
                response = result.get("response", "")
                prompt_tokens = result.get("prompt_tokens", 0)
                completion_tokens = result.get("completion_tokens", 0)
                success = "error" not in result
            except Exception as e:
                print(f"Error calling API for query: {e}")
                response = ""
                prompt_tokens = 0
                completion_tokens = 0
                success = False

            row_copy["response"] = response
            row_copy["prompt_tokens"] = prompt_tokens
            row_copy["completion_tokens"] = completion_tokens
            row_copy["input_token"] = prompt_tokens
            row_copy["output_token"] = completion_tokens
            row_copy["success"] = success

            # Step 4: Calculate task performance if ground truth is available
            ground_truth = row_copy.get("ground_truth") or row_copy.get("gt") or row_copy.get("answer")
            metric = row_copy.get("metric")
            if ground_truth:
                task_performance = calculate_task_performance(
                    prediction=response,
                    ground_truth=ground_truth,
                    task_name=row_task_name,
                    metric=metric
                )
                if task_performance is not None:
                    row_copy["task_performance"] = task_performance

            query_data_output.append(row_copy)

        return query_data_output