"""
Automix Router Trainer
----------------------
Training implementation for AutomixRouter.

This module provides the AutomixRouterTrainer class that handles
training of Automix routing strategies.

Original source: automix/colabs/Step3_MetaVerify.py
Adapted for LLMRouter framework.
"""

import torch
import pandas as pd
from typing import Any, Dict
from llmrouter.models.base_trainer import BaseTrainer


class AutomixRouterTrainer(BaseTrainer):
    """
    AutomixRouterTrainer
    -------------------
    Trainer implementation for AutomixRouter.

    Unlike typical neural network training with gradient descent,
    Automix training involves:
    1. Searching over candidate routing parameters
    2. Evaluating each on the training data
    3. Selecting the parameter with best IBC (Incremental Benefit over Cost) lift

    This is a discrete optimization problem rather than continuous optimization.
    """

    def __init__(
        self,
        router,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu",
        cost_constraint: tuple | None = None,
        **kwargs: Any,
    ):
        """
        Initialize AutomixRouterTrainer.

        Args:
            router:
                An AutomixRouter instance containing the routing model.
            optimizer (torch.optim.Optimizer | None):
                Not used for Automix (no gradient-based optimization).
                Included for API compatibility.
            device (str):
                Device for computation (default: "cpu").
                Automix primarily uses pandas operations, so GPU not required.
            cost_constraint (tuple | None):
                Optional (min_cost, max_cost) constraint for routing.
            **kwargs:
                Additional keyword arguments.
        """
        # Note: Automix doesn't use gradient-based optimization,
        # but BaseTrainer requires an optimizer. Create a dummy one if not provided.
        if optimizer is None:
            # Create a dummy parameter and optimizer for API compatibility
            dummy_param = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.Adam([dummy_param], lr=1e-4)

        super().__init__(router=router, optimizer=optimizer, device=device, **kwargs)
        self.cost_constraint = cost_constraint

    def loss_func(self, outputs, batch) -> torch.Tensor:
        """
        Compute loss (not used for Automix).

        Automix uses discrete parameter search rather than gradient descent,
        so traditional loss is not applicable. We return a dummy loss of 0
        for API compatibility.

        Args:
            outputs (dict):
                Dictionary containing routing outputs.
            batch (dict):
                Input batch.

        Returns:
            torch.Tensor:
                Dummy loss tensor (always 0).
        """
        # Automix doesn't use gradient-based loss
        # Return dummy loss for compatibility
        return torch.tensor(0.0, device=self.device)

    def train(self, dataloader):
        """
        Train the AutomixRouter.

        For Automix, "training" means:
        1. Iterate through training data
        2. For each batch, search over candidate parameters
        3. Select best parameter based on IBC lift
        4. Store best parameter in the model

        Args:
            dataloader:
                Iterable of batches. Each batch should be a dict with:
                    - "data": pandas DataFrame with required columns
                    - Other metadata as needed
        """
        self.router.eval()  # Automix doesn't have learnable parameters

        print("[AutomixRouterTrainer] Starting training (parameter search)")

        for step, batch in enumerate(dataloader):
            # Ensure data is pandas DataFrame (not moved to GPU)
            if not isinstance(batch["data"], pd.DataFrame):
                raise TypeError(
                    "Automix requires batch['data'] to be a pandas DataFrame"
                )

            # Set mode to train
            batch["mode"] = "train"

            # Run forward pass (this performs parameter search)
            with torch.no_grad():
                outputs = self.router(batch)

            # Log progress
            if step % 10 == 0:
                metrics = self.router.compute_metrics(outputs, batch)
                print(
                    f"[AutomixRouterTrainer] Step {step} | "
                    f"Performance: {metrics['avg_performance']:.4f} | "
                    f"Cost: {metrics['avg_cost']:.2f} | "
                    f"Routing %: {metrics['routing_percentage']:.1f}%"
                )

        # Training complete - best parameter is now stored in model
        print(
            f"[AutomixRouterTrainer] Training complete. "
            f"Best parameter: {self.router.model.best_param}"
        )

    def train_on_dataframe(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convenience method to train directly on a pandas DataFrame.

        This is the recommended way to train Automix, as it works
        directly with DataFrame data.

        Args:
            train_df (pd.DataFrame):
                Training dataframe with required columns:
                    - "p_ver_13b": verification scores
                    - "llama13b_f1": small model F1 scores
                    - "llama70b_f1": large model F1 scores
                    - "split": train/test split indicator (optional)

        Returns:
            dict:
                Training results including best parameter and metrics
        """
        # Filter to training split if available
        if "split" in train_df.columns:
            train_data = train_df[train_df["split"] == "train"].copy()
        else:
            train_data = train_df.copy()

        print(f"[AutomixRouterTrainer] Training on {len(train_data)} samples")

        # Perform parameter search
        best_param = self.router.model.train_routing(
            train_data, cost_constraint=self.cost_constraint
        )

        # Evaluate on training data
        metrics = self.router.model.evaluate(train_data, return_dict=True)

        print("[AutomixRouterTrainer] Training complete!")
        print(f"  Best parameter: {best_param}")
        print(f"  IBC Lift: {metrics['ibc_lift']:.4f}")
        print(f"  Avg Performance: {metrics['avg_performance']:.4f}")
        print(f"  Avg Cost: {metrics['avg_cost']:.2f}")

        return {
            "best_param": best_param,
            "metrics": metrics,
        }

    def evaluate_on_dataframe(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convenience method to evaluate directly on a pandas DataFrame.

        Args:
            test_df (pd.DataFrame):
                Test dataframe with same structure as training data.

        Returns:
            dict:
                Evaluation metrics
        """
        # Filter to test split if available
        if "split" in test_df.columns:
            test_data = test_df[test_df["split"] == "test"].copy()
        else:
            test_data = test_df.copy()

        print(f"[AutomixRouterTrainer] Evaluating on {len(test_data)} samples")

        # Ensure model is trained
        if self.router.model.best_param is None:
            raise ValueError(
                "Model must be trained before evaluation. "
                "Call train_on_dataframe() first."
            )

        # Evaluate
        metrics = self.router.model.evaluate(
            test_data, return_dict=True, return_decisions=True
        )

        # Compute additional metrics
        decisions = metrics["route_to_llm"]
        num_routed = int(decisions.sum())
        total = len(test_data)

        print("[AutomixRouterTrainer] Evaluation complete!")
        print(f"  IBC Lift: {metrics['ibc_lift']:.4f}")
        print(f"  Avg Performance: {metrics['avg_performance']:.4f}")
        print(f"  Avg Cost: {metrics['avg_cost']:.2f}")
        print(f"  Routed to LLM: {num_routed}/{total} ({num_routed/total*100:.1f}%)")

        return metrics

    def train_and_evaluate(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train on training data and evaluate on test data.

        Args:
            train_df (pd.DataFrame):
                Training dataframe
            test_df (pd.DataFrame):
                Test dataframe

        Returns:
            dict:
                Combined results with training and evaluation metrics
        """
        # Train
        train_results = self.train_on_dataframe(train_df)

        # Evaluate
        test_metrics = self.evaluate_on_dataframe(test_df)

        return {
            "train": train_results,
            "test": test_metrics,
        }
