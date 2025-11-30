"""
RouterDC Router
---------------
Router implementation for the RouterDC routing strategy.

This module provides the RouterDCRouter class that integrates with the
LLMRouter framework.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework.
"""

import torch
import torch.nn as nn
from llmrouter.models.meta_router import MetaRouter


class RouterDCRouter(MetaRouter):
    """
    RouterDCRouter
    --------------
    Router that uses dual-contrastive learning strategy for LLM routing decisions.

    RouterDC uses a pre-trained encoder (e.g., mDeBERTa) combined with learnable
    LLM embeddings to make routing decisions. The model is trained with three
    contrastive learning objectives:
    1. Sample-LLM contrastive loss
    2. Sample-Sample contrastive loss (task-level)
    3. Cluster contrastive loss

    This class:
        - Inherits MetaRouter to reuse configuration and utilities
        - Implements the `route()` method using the underlying RouterModule
        - Provides compute_metrics() for evaluation
    """

    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        """
        Initialize RouterDCRouter.

        Args:
            model (nn.Module):
                Underlying RouterModule instance containing routing logic.
                Should be an instance of RouterModule from model.py.
            yaml_path (str | None):
                Optional path to YAML config for this router.
            resources (Any, optional):
                Additional shared resources (e.g., tokenizer, API clients).
        """
        super().__init__(model=model, yaml_path=yaml_path, resources=resources)

    def route(self, batch):
        """
        Perform routing on a batch of data.

        This method:
        1. Extracts tokenized inputs from the batch
        2. Passes them through the RouterModule
        3. Returns similarity scores for each LLM

        Args:
            batch (dict):
                A batch containing tokenized inputs:
                    - "input_ids": Token IDs, shape (batch_size, seq_len)
                    - "attention_mask": Attention mask, shape (batch_size, seq_len)
                    - "temperature": Optional temperature for scaling (default: 1.0)

        Returns:
            dict:
                A dictionary with routing outputs:
                    - "scores": Similarity scores for each LLM,
                               shape (batch_size, num_llms)
                    - "hidden_state": Hidden states from the encoder,
                                     shape (batch_size, hidden_dim)
                    - "predictions": Predicted LLM indices (argmax of scores),
                                    shape (batch_size,)
        """
        # Extract temperature if provided, default to 1.0
        temperature = batch.get("temperature", 1.0)

        # Prepare inputs for the model
        input_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        # Forward pass through RouterModule
        scores, hidden_state = self.model(t=temperature, **input_kwargs)

        # Get predicted LLM indices (argmax)
        predictions = torch.argmax(scores, dim=1)

        return {
            "scores": scores,
            "hidden_state": hidden_state,
            "predictions": predictions,
        }

    def route_batch(self, batch):
        """Compatibility wrapper for MetaRouter expectations."""
        return self.route(batch)

    def route_single(self, sample):
        """
        Route a single tokenized sample.

        Args:
            sample (dict): Dict with "input_ids" and "attention_mask".

        Returns:
            dict: Routing outputs identical to `route`.
        """
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]

        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "temperature": sample.get("temperature", 1.0),
        }
        if "true_scores" in sample:
            batch["true_scores"] = sample["true_scores"]
        if "data_type" in sample:
            batch["data_type"] = sample["data_type"]
        return self.route(batch)

    def compute_metrics(self, outputs, batch) -> dict:
        """
        Compute evaluation metrics for routing decisions.

        This method computes:
        1. Routing accuracy: Percentage of times the router selects the best LLM
        2. Task accuracy: Percentage of times the selected LLM would answer correctly

        Args:
            outputs (dict):
                Dictionary containing:
                    - "scores": Similarity scores, shape (batch_size, num_llms)
                    - "predictions": Predicted LLM indices, shape (batch_size,)
            batch (dict):
                Original input batch with:
                    - "true_scores": Ground truth performance scores,
                                    shape (batch_size, num_llms)
                    - "data_type": Type of evaluation ("probability" or "multi_attempt")

        Returns:
            dict:
                Dictionary of metrics:
                    - "routing_accuracy": Percentage of correct routing decisions
                    - "task_accuracy": Percentage of correctly answered questions
                    - "avg_score": Average routing score
        """
        predictions = outputs["predictions"]
        scores = outputs["scores"]

        # Get ground truth scores
        true_scores = batch["true_scores"]

        # Compute routing accuracy (did we select the best LLM?)
        best_llm_indices = torch.argmax(true_scores, dim=1)
        routing_correct = (predictions == best_llm_indices).sum().item()
        routing_accuracy = routing_correct / len(predictions) * 100.0

        # Compute task accuracy (did the selected LLM answer correctly?)
        data_type = batch.get("data_type", "multi_attempt")
        batch_size = predictions.shape[0]
        num_llms = true_scores.shape[1]

        # Create one-hot mask for selected LLMs
        mask = torch.zeros_like(true_scores)
        mask.scatter_(1, predictions.unsqueeze(1), 1)

        if data_type == "probability":
            # For probability tasks, score > 0 means correct
            binary_scores = (true_scores > 0).float()
            correct_predictions = (binary_scores * mask).sum().item()
        else:  # multi_attempt
            # For multi-attempt tasks, use scores directly
            correct_predictions = (true_scores * mask).sum().item()

        task_accuracy = correct_predictions / batch_size * 100.0

        # Compute average routing score
        avg_score = scores.mean().item()

        metrics = {
            "routing_accuracy": routing_accuracy,
            "task_accuracy": task_accuracy,
            "avg_score": avg_score,
        }

        return metrics
