# models/graphrouter/trainer.py

import torch
from llmrouter.models.base_trainer import BaseTrainer


class KNNRouterTrainer(BaseTrainer):
    """
    GraphRouterTrainer
    ------------------
    Example trainer implementation for GraphRouter.

    Uses a simple supervised learning objective with cross-entropy loss.
    """

    def train(self, dataloader):
        """
        Train the GraphRouter for one or multiple epochs.

        Args:
            dataloader:
                Iterable of batches for training.
        """
        self.router.train()
        for step, batch in enumerate(dataloader):
            batch = self._move_batch_to_device(batch)

            outputs = self.router(batch)
            loss = self.loss_func(outputs, batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                print(f"[GraphRouterTrainer] Step {step} | loss = {loss.item():.4f}")
