"""
Automix Router
--------------
Router implementation for the Automix routing strategy.

This module provides the AutomixRouter class that integrates with the
LLMRouter framework.

Original source: automix/colabs/Step3_MetaVerify.py
Adapted for LLMRouter framework.
"""

import torch.nn as nn
from llmrouter.models.meta_router import MetaRouter


class AutomixRouter(MetaRouter):
    """
    AutomixRouter
    -------------
    Router that uses Automix strategy for LLM routing decisions.

    Automix uses self-verification and various strategies (Threshold, POMDP,
    Self-Consistency) to decide when to route queries to a larger model.

    This class:
        - Inherits MetaRouter to reuse configuration and utilities
        - Implements the `route()` method using the underlying AutomixModel
        - Provides compute_metrics() for evaluation
    """

    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        """
        Initialize AutomixRouter.

        Args:
            model (nn.Module):
                Underlying AutomixModel instance containing routing logic.
            yaml_path (str | None):
                Optional path to YAML config for this router.
            resources (Any, optional):
                Additional shared resources (e.g., tokenizer, API clients).
        """
        super().__init__(model=model, yaml_path=yaml_path, resources=resources)

    def route(self, batch):
        """
        Perform routing on a batch of data.

        Args:
            batch (dict):
                A batch containing:
                    - "data": pandas DataFrame with columns:
                        - "p_ver_13b": verification scores
                        - "llama13b_f1": small model F1 scores
                        - "llama70b_f1": large model F1 scores
                        - other relevant fields
                    - "mode": "train" or "infer" (optional, defaults to "infer")

        Returns:
            dict:
                A dictionary with routing outputs:
                    - "decisions": Boolean tensor of routing decisions
                        (True = route to large model, False = use small model)
                    - "performance": Average performance score
                    - "cost": Average cost
        """
        return self.model(batch)

    def compute_metrics(self, outputs, batch) -> dict:
        """
        Compute evaluation metrics for routing decisions.

        Args:
            outputs (dict):
                Dictionary containing:
                    - "decisions": Boolean tensor of routing decisions
                    - "performance": Average performance
                    - "cost": Average cost
            batch (dict):
                Original input batch with data

        Returns:
            dict:
                Dictionary of metrics:
                    - "avg_performance": Average task performance
                    - "avg_cost": Average routing cost
                    - "routing_percentage": Percentage routed to large model
                    - "num_routed": Number of queries routed to large model
                    - "total_samples": Total number of samples
        """
        decisions = outputs["decisions"]
        num_routed = int(decisions.sum().item())
        total_samples = len(decisions)

        metrics = {
            "avg_performance": outputs["performance"],
            "avg_cost": outputs["cost"],
            "routing_percentage": (num_routed / total_samples * 100.0)
            if total_samples > 0
            else 0.0,
            "num_routed": num_routed,
            "total_samples": total_samples,
        }

        return metrics
