from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import copy

from sklearn.neural_network import MLPRegressor
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding, call_api, generate_task_query, calculate_task_performance
from llmrouter.models.largest_llm import parse_size


class HybridLLMRouter(MetaRouter):
    """
    HybridLLMRouter
    ----------------
    Implements the routing logic from "Hybrid LLM (Ding et al., 2024)":

    Modes (YAML configurable):
      - deterministic :     y = 1[q(S) >= q(L)]
      - probabilistic :     y = sigmoid((q(S)-q(L)) / tau)
      - transformed  :      y = 1[q(S) >= q(L) - t*],  t* chosen to maximize label separation

    Additional configs:
      router_tau: float          # Only used in 'probabilistic'
      router_threshold: float    # Routing decision threshold
    """

    def __init__(self, yaml_path: str):
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # -------------------------------
        # Load router hyperparameters
        # -------------------------------
        self.router_mode: str = self.cfg.get("router_mode", "deterministic")
        assert self.router_mode in ["deterministic", "probabilistic", "transformed"], \
            f"Router mode '{self.router_mode}' must be one of: deterministic, probabilistic, transformed."

        self.router_tau: float = float(self.cfg.get("router_tau", 0.1))
        self.router_threshold: float = float(self.cfg.get("router_threshold", 0.5))

        # -------------------------------
        # Build MLP Regressor
        # -------------------------------
        mlp_params = self.cfg["hparam"]
        self.mlp_model = MLPRegressor(**mlp_params)

        # -------------------------------
        # Determine smallest / largest LLM
        # -------------------------------
        self.small_model_name, self.large_model_name = self._resolve_small_large(self.routing_data_train)
        print(
            f"[HybridLLMRouter] Mode={self.router_mode}, "
            f"Small='{self.small_model_name}', Large='{self.large_model_name}'"
        )

        # -------------------------------
        # Build training dataset
        # -------------------------------
        (
            self.query_embedding_list,
            self.router_label_list
        ) = self._create_training_dataset()

    # ==============================================================
    # Compute smallest and largest models from llm_data
    # ==============================================================
    def _resolve_small_large(self, routing_data=None):
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError("[HybridLLMRouter] llm_data missing.")

        # Only models with sizes ending in 'B'
        available = [
            name
            for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str)
            and info["size"].upper().endswith("B")
        ]

        if len(available) < 2:
            raise ValueError("[HybridLLMRouter] Need at least 2 models (size ends with B).")

        # Filter models to only those that exist in routing_data (if provided)
        if routing_data is not None and not routing_data.empty:
            available_in_data = routing_data["model_name"].unique().tolist()
            available = [m for m in available if m in available_in_data]
            if len(available) < 2:
                raise ValueError(
                    f"[HybridLLMRouter] Only found {len(available)} models with routing data. "
                    f"Need at least 2 models that exist in routing_data."
                )
            print(f"[HybridLLMRouter] Filtered to {len(available)} models with routing data.")

        sorted_models = sorted(
            available,
            key=lambda m: parse_size(self.llm_data[m].get("size", "0B"))
        )

        return sorted_models[0], sorted_models[-1]

    # ==============================================================
    # Build training X, y from routing_data_train
    # ==============================================================
    def _create_training_dataset(self):
        """
        Return:
            X: List[np.ndarray] - query embeddings
            y: List[float]      - labels in [0,1]
        """

        df = self.routing_data_train
        small = self.small_model_name
        large = self.large_model_name

        # Filter for small & large
        pair_df = df[df["model_name"].isin([small, large])]
        if pair_df.empty:
            raise ValueError("[HybridLLMRouter] No routing data found for small/large pair.")

        # Pivot: one row/query with small & large performances
        pivot = pair_df.pivot_table(
            index=["query", "embedding_id"],
            columns="model_name",
            values="performance",
            aggfunc="mean"
        )

        pivot = pivot.dropna(subset=[small, large])
        if pivot.empty:
            raise ValueError("[HybridLLMRouter] No complete rows with both small & large scores.")

        q_s = pivot[small].values
        q_l = pivot[large].values
        gaps = q_s - q_l   # quality gap

        embedding_ids = pivot.index.get_level_values("embedding_id").tolist()

        # -------------------------------------------------
        # Produce labels depending on router_mode
        # -------------------------------------------------
        if self.router_mode == "deterministic":
            labels = (gaps >= 0).astype(float)

        elif self.router_mode == "probabilistic":
            labels = 1 / (1 + np.exp(-gaps / self.router_tau))

        elif self.router_mode == "transformed":
            labels = self._compute_transformed_labels(gaps)

        else:
            raise ValueError("[HybridLLMRouter] Invalid router_mode.")

        # Embeddings
        X = [self.query_embedding_data[i].numpy() for i in embedding_ids]
        return X, labels.tolist()

    # ==============================================================
    # Transformed router: find best t*
    # ==============================================================
    def _compute_transformed_labels(self, gaps: np.ndarray) -> np.ndarray:
        if gaps.size == 0:
            raise ValueError("[HybridLLMRouter] gaps array is empty.")

        # Search t on a grid
        t_values = np.linspace(0, np.max(np.abs(gaps)) + 1e-8, 50)

        best_t = 0.0
        best_score = -1.0

        for t in t_values:
            y_t = (gaps >= -t).astype(float)
            p = y_t.mean()
            score = 2 * p * (1 - p)  # maximize balance

            if score > best_score:
                best_score = score
                best_t = t

        print(f"[HybridLLMRouter] transformed mode: best t = {best_t:.4f}")

        return (gaps >= -best_t).astype(float)

    # ==============================================================
    # Route a single query
    # ==============================================================
    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_path)

        emb = [get_longformer_embedding(query["query"]).numpy()]
        score = float(self.mlp_model.predict(emb)[0])
        score = max(0.0, min(1.0, score))

        chosen = self.small_model_name if score >= self.router_threshold else self.large_model_name

        out = copy.copy(query)
        out["model_name"] = chosen
        out["router_score"] = score
        return out

    # ==============================================================
    # Route a batch
    # ==============================================================
    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using HybridLLMRouter and execute them.

        This method performs end-to-end processing for each query:
        1. Routes the query to get the best model (small or large)
        2. Applies task-specific prompt formatting if task_name is provided
        3. Calls the routed model via API to get response
        4. Calculates performance metrics if ground truth is available

        Args:
            batch (Any, optional):
                If provided, routes the provided batch. If None, uses self.query_data_test from loaded data.
            task_name (str, optional):
                Task name for prompt formatting (e.g., "mmlu", "gsm8k", "commonsense_qa").
                If provided, queries will be formatted using task-specific prompts before execution.
                If None, queries are executed as-is. Can also be extracted from each row's 'task_name' field.

        Returns:
            list of dict:
                A list of query dictionaries, each updated with:
                    - "query": original query text (preserved)
                    - "formatted_query": formatted query if task_name was provided (optional)
                    - "model_name": predicted model name (small or large)
                    - "router_score": routing score from MLP
                    - "response": final answer from the routed model
                    - "prompt_tokens": total prompt tokens used
                    - "completion_tokens": total completion tokens used
                    - "input_token": total input tokens (alias for prompt_tokens)
                    - "output_token": total output tokens (alias for completion_tokens)
                    - "task_performance": evaluation score (0.0-1.0) if ground truth available
                    - "success": whether the API call succeeded
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_path)

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
                # Use task_name from row if available, otherwise use parameter
                row_task_name = row_copy.get("task_name", task_name)
            else:
                row_copy = {"query": str(row)}
                original_query = str(row)
                row_task_name = task_name

            # Step 1: Route the query to get model_name
            emb = [get_longformer_embedding(original_query).numpy()]
            score = float(self.mlp_model.predict(emb)[0])
            score = max(0.0, min(1.0, score))

            model_name = (
                self.small_model_name if score >= self.router_threshold else self.large_model_name
            )
            row_copy["model_name"] = model_name
            row_copy["router_score"] = score

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

