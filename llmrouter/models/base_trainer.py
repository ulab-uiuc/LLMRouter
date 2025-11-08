from abc import ABC, abstractmethod
from typing import Any

import torch
import joblib

class BaseTrainer(ABC):
    """
    BaseTrainer (Abstract Class)
    ----------------------------
    Defines a unified training interface for all routers.

    Each specific router baseline (e.g., GraphRouter, RouterDC) should:
        - Inherit from this class
        - Implement `loss_func()` and `train()` for its own training logic
    """

    def __init__(
        self,
        router,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ):
        """
        Args:
            router (nn.Module):
                A MetaRouter subclass instance containing the routing model.
            optimizer (torch.optim.Optimizer | None):
                Optional optimizer. If None, a default Adam optimizer is created.
            device (str):
                Device to place the router and batches on (e.g., "cuda", "cpu").
            **kwargs:
                Extra keyword arguments for future extensions (e.g., schedulers).
        """
        self.router = router.to(device)
        self.device = device
        self.optimizer = optimizer or torch.optim.Adam(
            self.router.parameters(), lr=1e-4
        )
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # Abstract methods to be customized by each baseline
    # ------------------------------------------------------------------

    def loss_func(self, outputs, batch) -> torch.Tensor:
        """
        Compute task-specific loss.

        Args:
            outputs (Any):
                Model outputs (e.g., logits, scores) from `router(batch)`.
            batch (Any):
                Input batch, typically containing labels and additional info.

        Returns:
            torch.Tensor:
                A scalar loss tensor used for backpropagation.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, dataloader):
        """
        Define the full training loop.

        This method should handle:
            - iterating over `dataloader`
            - calling `self.router(batch)`
            - computing loss via `self.loss_func()`
            - backward pass and optimizer step
            - optional logging / progress reporting
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional shared utilities
    # ------------------------------------------------------------------

    def evaluate(self, dataloader):
        """
        Optional evaluation loop.

        Iterates over a dataloader, collects metrics from the router, and
        returns a list or aggregated metrics.

        Args:
            dataloader:
                Iterable of batches used for evaluation.

        Returns:
            list[dict]:
                A list of metric dictionaries, one per batch.
        """
        self.router.eval()
        all_metrics = []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                outputs = self.router(batch)
                metrics = self.router.compute_metrics(outputs, batch)
                all_metrics.append(metrics)
        return all_metrics

    def save_checkpoint(self, path: str):
        """
        Save model and optimizer states to a checkpoint file.

        Args:
            path (str):
                Target file path for the checkpoint.
        """
        torch.save(
            {
                "model": self.router.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"✅ Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        """
        Load model and optimizer states from a checkpoint file.

        Args:
            path (str):
                Path of an existing checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.router.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"📂 Checkpoint loaded from: {path}")


    def save_model_lib(self, save_path: str):
        joblib.dump(self.router, save_path)

    def load_model_lib(self, load_path: str):
        self.router= joblib.load(load_path)


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_batch_to_device(self, batch):
        """
        Move a batch to the target device.

        This helper assumes the batch is either:
            - a tensor, or
            - a dict of tensors.

        Args:
            batch (Any):
                Original batch.

        Returns:
            Any:
                Batch placed on the trainer device.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        # Extend here if you have more complex structures
        return batch
