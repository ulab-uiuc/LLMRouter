"""
Automix Router Trainer
----------------------
Training implementation for AutomixRouter.

Original source: automix/colabs/Step3_MetaVerify.py
Adapted for LLMRouter framework.
"""

import torch
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
    """

    def __init__(self, router, device: str = "cpu"):
        """
        Initialize AutomixRouterTrainer.

        Args:
            router: An AutomixRouter instance
            device (str): Device for computation (default: "cpu")
        """
        # Create a dummy optimizer for API compatibility
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([dummy_param], lr=1e-4)

        super().__init__(router=router, optimizer=optimizer, device=device)

        # Get config from router
        self.cfg = router.cfg
        self.train_df = router.train_df
        self.test_df = router.test_df

        # Get training parameters (support both train_param and hparam)
        hparam = self.cfg.get("hparam", {})

        # Try to get cost_constraint from train_param first, then hparam
        self.cost_constraint = hparam.get("cost_constraint", None)

        # Try to get verbose from train_param first, then hparam
        self.verbose = hparam.get("verbose", False)

    def loss_func(self, outputs, batch) -> torch.Tensor:
        """
        Compute loss (not used for Automix).

        Automix uses discrete parameter search rather than gradient descent.

        Returns:
            torch.Tensor: Dummy loss tensor (always 0)
        """
        return torch.tensor(0.0, device=self.device)

    def train(self):
        """
        Train the AutomixRouter.

        For Automix, "training" means:
        1. Search over candidate parameters on training data
        2. Select best parameter based on IBC lift
        3. Evaluate on test data
        """
        # Perform parameter search on training data
        best_param = self.router.model.train_routing(
            self.train_df,
            cost_constraint=self.cost_constraint
        )

        # Evaluate on training data
        train_metrics = self.router.model.evaluate(self.train_df, return_dict=True)

        # Evaluate on test data
        test_metrics = self.router.model.evaluate(
            self.test_df,
            return_dict=True,
            return_decisions=True
        )

        return {
            "train": {
                "best_param": best_param,
                "metrics": train_metrics,
            },
            "test": test_metrics,
        }
