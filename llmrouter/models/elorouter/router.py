from typing import Any, Dict, List, Optional
import os
import copy

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, call_api, generate_task_query, calculate_task_performance


class EloRouter(MetaRouter):
    """
    EloRouter
    ----------
    A routing module that selects the model with the highest Elo score.
    Elo scores are precomputed by EloRouterTrainer and saved to disk.

    IMPORTANT:
    - Does NOT load model during __init__()
    - Only loads during route_single() / route_batch() (lazy-load)
    - Matches behavior of MFRouter, MLPRouter, SVMRouter
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the EloRouter with configuration only.
        No loading of Elo scores here.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        routing = self.routing_data_train.reset_index(drop=True)
        self.model_names = sorted(routing["model_name"].unique().tolist())

        self.elo_scores = None
        self.model_loaded = False  # lazy load flag

        print(f"[EloRouter] Initialized with {len(self.model_names)} models.")

    # ---------------------------------------------------------
    # Lazy load Elo scores
    # ---------------------------------------------------------
    def _load_elo_if_needed(self):
        """Load Elo scores once at inference time."""
        if self.model_loaded:
            return

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_path = os.path.join(project_root, self.cfg["model_path"]["save_model_path"])

        elo_obj = load_model(load_path)

        # Normalize to dict
        if hasattr(elo_obj, "to_dict"):
            self.elo_scores = elo_obj.to_dict()
        else:
            self.elo_scores = dict(elo_obj)

        self.model_loaded = True
        print(f"[EloRouter] Loaded Elo scores from {load_path}")

    # ---------------------------------------------------------
    # Select best model
    # ---------------------------------------------------------
    def _select_best_model(self) -> str:
        """Pick the model with the highest Elo rating."""
        self._load_elo_if_needed()
        return max(self.elo_scores.items(), key=lambda kv: kv[1])[0]

    # ---------------------------------------------------------
    # Route single
    # ---------------------------------------------------------
    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        best_model = self._select_best_model()
        query_out = copy.copy(query)
        query_out["model_name"] = best_model
        return query_out

    # ---------------------------------------------------------
    # Route batch and execute
    # ---------------------------------------------------------
    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using Elo scores and execute them.

        This method performs end-to-end processing for each query:
        1. Selects the model with the highest Elo score
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
        # Select best model once (same for all queries)
        best_model = self._select_best_model()

        # Determine which data to use
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
        else:
            if hasattr(self, "query_data_test") and self.query_data_test is not None:
                query_data = copy.copy(self.query_data_test)
            else:
                print("Warning: No batch provided and no test data available for batch routing.")
                return []

        query_data_output = []
        for row in query_data:
            # Handle both dict and non-dict inputs
            if isinstance(row, dict):
                row_copy = copy.copy(row)
                original_query = row_copy.get("query", "")
                row_task_name = row_copy.get("task_name", task_name)
            else:
                row_copy = {"query": str(row)}
                original_query = str(row)
                row_task_name = task_name

            # Step 1: Route - always use best model
            model_name = best_model
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

