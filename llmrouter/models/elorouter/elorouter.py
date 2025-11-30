from typing import Any, Dict, List, Optional
import os
import copy

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model


class EloRouter(MetaRouter):
    """
    EloRouter
    ----------
    A routing module that selects the model with the highest Elo score.
    Elo scores are precomputed by EloRouterTrainer and saved to disk.

    IMPORTANT:
    - Does NOT load model during __init__()
    - Only loads during route_single() / route_batch() (lazy-load)
    - Matches behavior of MFRouter, MLPRouter, SVMRouter
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the EloRouter with configuration only.
        No loading of Elo scores here.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        routing = self.routing_data_train.reset_index(drop=True)
        self.model_names = sorted(routing["model_name"].unique().tolist())

        self.elo_scores = None
        self.model_loaded = False  # lazy load flag

        print(f"[EloRouter] Initialized with {len(self.model_names)} models.")

    # ---------------------------------------------------------
    # Lazy load Elo scores
    # ---------------------------------------------------------
    def _load_elo_if_needed(self):
        """Load Elo scores once at inference time."""
        if self.model_loaded:
            return

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])

        elo_obj = load_model(load_path)

        # Normalize to dict
        if hasattr(elo_obj, "to_dict"):
            self.elo_scores = elo_obj.to_dict()
        else:
            self.elo_scores = dict(elo_obj)

        self.model_loaded = True
        print(f"[EloRouter] Loaded Elo scores from {load_path}")

    # ---------------------------------------------------------
    # Select best model
    # ---------------------------------------------------------
    def _select_best_model(self) -> str:
        """Pick the model with the highest Elo rating."""
        self._load_elo_if_needed()
        return max(self.elo_scores.items(), key=lambda kv: kv[1])[0]

    # ---------------------------------------------------------
    # Route single
    # ---------------------------------------------------------
    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        best_model = self._select_best_model()
        query_out = copy.copy(query)
        query_out["model_name"] = best_model
        return query_out

    # ---------------------------------------------------------
    # Route batch
    # ---------------------------------------------------------
    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        best_model = self._select_best_model()
        output = copy.copy(self.query_data_test)
        for row in output:
            row["model_name"] = best_model
        return output

