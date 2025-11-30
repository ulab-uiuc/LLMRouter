import os
import numpy as np
import torch
import torch.nn as nn

from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.utils import save_model, get_longformer_embedding

from .mfrouter import BilinearMF


class MFRouterTrainer(BaseTrainer):
    """
    MFRouterTrainer
    Trains BilinearMF using pairwise logistic loss:
        L = BCE( δ(win,q) − δ(loss,q), 1 )
    """

    def __init__(self, router, optimizer=None, device="cpu"):
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.router = router
        self.pairs = router.pairs
        self.dim = router.dim
        self.text_dim = router.text_dim

        # -----------------------
        # MATCH mlptrainer.py: save path only
        # -----------------------
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.save_model_path = os.path.join(
            project_root, router.cfg["model_path"]["save_model_path"]
        )

        # Build bilinear MF model
        self.model = BilinearMF(
            dim=self.dim,
            num_models=len(router.model_to_idx),
            text_dim=self.text_dim,
        ).to(device)

        hparam = router.cfg["hparam"]
        self.lr = hparam.get("lr", 1e-3)
        self.epochs = hparam.get("epochs", 3)
        self.noise_alpha = hparam.get("noise_alpha", 0.0)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print("[MFRouterTrainer] Initialized.")

    # ---------------------------------------------------------
    # Longformer query embedding
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        emb = get_longformer_embedding(text).numpy()
        return torch.tensor(emb, dtype=torch.float32).to(self.model.device)

    # ---------------------------------------------------------
    # Full training loop
    # ---------------------------------------------------------
    def train(self):
        model = self.model
        optimizer = self.optimizer
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            np.random.shuffle(self.pairs)
            epoch_losses = []

            for sample in self.pairs:
                win_id = torch.tensor([sample["winner"]], dtype=torch.long, device=model.device)
                loss_id = torch.tensor([sample["loser"]], dtype=torch.long, device=model.device)

                q_emb = self.embed_query(sample["query"])
                q_emb_proj = model.project_text(q_emb)

                if self.noise_alpha > 0:
                    q_emb_proj = q_emb_proj + torch.randn_like(q_emb_proj) * self.noise_alpha

                logit = model(win_id, loss_id, q_emb_proj)
                target = torch.ones_like(logit)

                optimizer.zero_grad()
                loss = loss_fn(logit, target)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            print(
                f"[MFRouterTrainer] Epoch {epoch+1}/{self.epochs} "
                f"- Loss={np.mean(epoch_losses):.6f}"
            )

        # ---------------------------------------------------------
        # Save model (MATCH MLPRouter format)
        # ---------------------------------------------------------
        save_model(model.state_dict(), self.save_model_path)
        print(f"[MFRouterTrainer] Model saved to {self.save_model_path}")
