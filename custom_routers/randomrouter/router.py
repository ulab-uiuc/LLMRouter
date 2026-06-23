"""
Random Router - Example Custom Router Implementation
=====================================================

This is a simple example router that randomly selects an LLM.
It demonstrates the minimal interface required for custom routers.

Usage:
    llmrouter infer --router randomrouter --config custom_routers/randomrouter/config.yaml --query "Hello"
"""

import random
from typing import Any, Dict, List
import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter


class RandomRouter(MetaRouter):
    """
    Random Router - Baseline router that randomly selects an LLM.

    This router serves as both a baseline and an example of how to
    implement custom routers. It randomly selects from available LLMs
    without considering the query content.

    Required Methods (from MetaRouter):
        - route_single(batch): Route a single query
        - route_batch(batch): Route multiple queries

    YAML Configuration Example:
    ---------------------------
    data_path:
      llm_data: 'path/to/llm_candidates.json'

    hparam:
      seed: 42  # Optional random seed for reproducibility
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the RandomRouter.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        The YAML should contain:
            - data_path.llm_data: Path to LLM candidates JSON file
            - hparam.seed (optional): Random seed
        """
        # Create a dummy model (required by MetaRouter, but not used)
        dummy_model = nn.Identity()

        # Initialize parent class - this will load config and data
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Extract hyperparameters
        hparam = self.cfg.get("hparam", {})
        seed = hparam.get("seed", None)

        if seed is not None:
            random.seed(seed)

        # Get list of available LLM names
        if hasattr(self, 'llm_data') and self.llm_data:
            self.llm_names = list(self.llm_data.keys())
        else:
            raise ValueError(
                "No LLM data found. Please specify 'llm_data' in YAML config."
            )

        if not self.llm_names:
            raise ValueError("LLM data is empty. At least one LLM is required.")

        print(f"✅ RandomRouter initialized with {len(self.llm_names)} LLMs")
        print(f"   Available LLMs: {', '.join(self.llm_names)}")

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query by randomly selecting an LLM.

        Args:
            query_input (dict): Input dictionary containing:
                - query (str): The query text
                - ... (other optional fields)

        Returns:
            dict: Routing result containing:
                - query (str): Original query
                - model_name (str): Selected LLM name
                - predicted_llm (str): Same as model_name (for compatibility)
                - confidence (float): Always 1.0 for random selection
                - method (str): "random"
        """
        # Randomly select an LLM
        selected_llm = random.choice(self.llm_names)

        result = {
            "query": query_input.get("query", ""),
            "model_name": selected_llm,
            "predicted_llm": selected_llm,  # Alternative field name for compatibility
            "predicted_llm_name": selected_llm,  # Another alternative
            "confidence": 1.0,  # Random selection, no confidence measure
            "method": "random",
        }

        return result

    def route_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Route a batch of queries.

        Args:
            batch (list): List of query input dictionaries

        Returns:
            list: List of routing results
        """
        results = []
        for query_input in batch:
            result = self.route_single(query_input)
            results.append(result)

        return results

    def forward(self, batch):
        """
        PyTorch-compatible forward method.

        This allows the router to be used in training loops if needed.
        """
        if isinstance(batch, list):
            return self.route_batch(batch)
        else:
            return self.route_single(batch)
