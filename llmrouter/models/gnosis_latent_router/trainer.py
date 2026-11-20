"""LLMRouter trainer interface for the integrated Gnosis trainer."""

from __future__ import annotations

from typing import Any

from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.models.gnosis_latent_router.adapter import run_gnosis_stage


class GnosisLatentRouterTrainer(BaseTrainer):
    """Expose Gnosis training through LLMRouter's standard trainer registry."""

    def __init__(self, router, optimizer=None, device: str = "auto", **kwargs: Any):
        super().__init__(
            router=router,
            optimizer=optimizer,
            device=device,
            **kwargs,
        )

    def train(self, dataloader=None):
        del dataloader
        stage = run_gnosis_stage(
            self.router.gnosis_config,
            "train",
            workspace=self.router.workspace_root,
        )
        return {
            "backend": "gnosis_latent_router",
            "stage": stage,
            "checkpoint_path": str(self.router._checkpoint_path()),
        }
