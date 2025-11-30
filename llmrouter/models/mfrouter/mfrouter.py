from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding


class BilinearMF(nn.Module):
    """
    Bilinear Matrix Factorization model used by MFRouter.
    Implements:
        δ(M, q) = w2^T ( v_m ⊙ (W1 * v_q) )
    """

    def __init__(self, dim: int, num_models: int, text_dim: int):
        super().__init__()

        # Latent model embeddings
        self.P = nn.Embedding(num_models, dim)

        # Text projection (Longformer embedding → router latent space)
        self.text_proj = nn.Linear(text_dim, dim, bias=False)

        # Final scoring layer
        self.classifier = nn.Linear(dim, 1, bias=False)

    @property
    def device(self):
        return self.P.weight.device

    def project_text(self, q_emb: torch.Tensor) -> torch.Tensor:
        """Project raw Longformer embedding into latent routing space."""
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        proj = self.text_proj(q_emb)
        return proj.squeeze(0)

    def forward(self, model_win, model_loss, q_emb):
        """Pairwise scoring: δ(win, q) − δ(loss, q)."""
        v_win = F.normalize(self.P(model_win), p=2, dim=-1)
        v_loss = F.normalize(self.P(model_loss), p=2, dim=-1)
        h = v_win - v_loss

        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)

        interaction = h * q_emb
        logit = self.classifier(interaction).squeeze(-1)
        return logit

    def score_all(self, q_emb: torch.Tensor):
        """Return δ(M, q) for all models."""
        P_all = F.normalize(self.P.weight, p=2, dim=-1)
        interaction = P_all * q_emb
        logits = self.classifier(interaction).squeeze(-1)
        return logits


class MFRouter(MetaRouter):
    """
    MFRouter (RouteLLM-style Bilinear Matrix Factorization Router)
    Predicts best model for each query:
        best = argmax_M δ(M, q)
    """

    def __init__(self, yaml_path: str):
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Hyperparameters
        hparam = self.cfg["hparam"]
        self.dim = hparam.get("latent_dim", 128)
        self.text_dim = hparam.get("text_dim", 768)

        # Routing data
        routing = self.routing_data_train.reset_index(drop=True)

        # Build model index mappings
        models = routing["model_name"].unique().tolist()
        self.model_to_idx = {m: i for i, m in enumerate(models)}
        self.idx_to_model = {i: m for m, i in self.model_to_idx.items()}

        # Construct pairwise samples
        self.pairs = []
        grouped = routing.groupby("query")
        for q, df in grouped:
            best_row = df.loc[df["performance"].idxmax()]
            winner_id = self.model_to_idx[best_row["model_name"]]

            for _, row in df.iterrows():
                loser_id = self.model_to_idx[row["model_name"]]
                if loser_id != winner_id:
                    self.pairs.append({
                        "query": q,
                        "winner": winner_id,
                        "loser": loser_id
                    })
    # ---------------------------------------------------------
    # Query embedding using Longformer
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        emb = get_longformer_embedding(text).numpy()
        return torch.tensor(emb, dtype=torch.float32)

    # ---------------------------------------------------------
    # Load MF model from file (MATCH MLPRouter)
    # ---------------------------------------------------------
    def _load_mf_model(self):
        state_dict = load_model(self.load_model_path)
        model = BilinearMF(self.dim, len(self.model_to_idx), self.text_dim)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # ---------------------------------------------------------
    # Route a single query
    # ---------------------------------------------------------
    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        model = self._load_mf_model()

        q_emb = self.embed_query(query["query"]).to(model.device)
        q_emb_proj = model.project_text(q_emb)

        scores = model.score_all(q_emb_proj)
        best_id = torch.argmax(scores).item()

        out = copy.copy(query)
        out["model_name"] = self.idx_to_model[best_id]
        return out

    # ---------------------------------------------------------
    # Route a batch of queries
    # ---------------------------------------------------------
    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        model = self._load_mf_model()

        results = copy.copy(self.query_data_test)
        for row in results:
            q_emb = self.embed_query(row["query"]).to(model.device)
            q_emb_proj = model.project_text(q_emb)

            scores = model.score_all(q_emb_proj)
            best_id = torch.argmax(scores).item()

            row["model_name"] = self.idx_to_model[best_id]

        return results

