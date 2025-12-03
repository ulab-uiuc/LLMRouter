from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import copy

from sklearn.neural_network import MLPRegressor
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding
from llmrouter.models.largest_llm import parse_size


class HybridLLMRouter(MetaRouter):
    """
    HybridLLMRouter
    ----------------
    Implements the routing logic from "Hybrid LLM (Ding et al., 2024)":

    Modes (YAML configurable):
      - deterministic :     y = 1[q(S) >= q(L)]
      - probabilistic :     y = sigmoid((q(S)-q(L)) / tau)
      - transformed  :      y = 1[q(S) >= q(L) - t*],  t* chosen to maximize label separation

    Additional configs:
      router_tau: float          # Only used in 'probabilistic'
      router_threshold: float    # Routing decision threshold
    """

    def __init__(self, yaml_path: str):
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # -------------------------------
        # Load router hyperparameters
        # -------------------------------
        self.router_mode: str = self.cfg.get("router_mode", "deterministic")
        assert self.router_mode in ["deterministic", "probabilistic", "transformed"], \
            f"Router mode '{self.router_mode}' must be one of: deterministic, probabilistic, transformed."

        self.router_tau: float = float(self.cfg.get("router_tau", 0.1))
        self.router_threshold: float = float(self.cfg.get("router_threshold", 0.5))

        # -------------------------------
        # Build MLP Regressor
        # -------------------------------
        mlp_params = self.cfg["hparam"]
        self.mlp_model = MLPRegressor(**mlp_params)

        # -------------------------------
        # Determine smallest / largest LLM
        # -------------------------------
        self.small_model_name, self.large_model_name = self._resolve_small_large()
        print(
            f"[HybridLLMRouter] Mode={self.router_mode}, "
            f"Small='{self.small_model_name}', Large='{self.large_model_name}'"
        )

        # -------------------------------
        # Build training dataset
        # -------------------------------
        (
            self.query_embedding_list,
            self.router_label_list
        ) = self._create_training_dataset()

    # ==============================================================
    # Compute smallest and largest models from llm_data
    # ==============================================================
    def _resolve_small_large(self):
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError("[HybridLLMRouter] llm_data missing.")

        # Only models with sizes ending in 'B'
        available = [
            name
            for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str)
            and info["size"].upper().endswith("B")
        ]

        if len(available) < 2:
            raise ValueError("[HybridLLMRouter] Need at least 2 models (size ends with B).")

        sorted_models = sorted(
            available,
            key=lambda m: parse_size(self.llm_data[m].get("size", "0B"))
        )

        return sorted_models[0], sorted_models[-1]

    # ==============================================================
    # Build training X, y from routing_data_train
    # ==============================================================
    def _create_training_dataset(self):
        """
        Return:
            X: List[np.ndarray] - query embeddings
            y: List[float]      - labels in [0,1]
        """

        df = self.routing_data_train
        small = self.small_model_name
        large = self.large_model_name

        # Filter for small & large
        pair_df = df[df["model_name"].isin([small, large])]
        if pair_df.empty:
            raise ValueError("[HybridLLMRouter] No routing data found for small/large pair.")

        # Pivot: one row/query with small & large performances
        pivot = pair_df.pivot_table(
            index=["query", "embedding_id"],
            columns="model_name",
            values="performance",
            aggfunc="mean"
        )

        pivot = pivot.dropna(subset=[small, large])
        if pivot.empty:
            raise ValueError("[HybridLLMRouter] No complete rows with both small & large scores.")

        q_s = pivot[small].values
        q_l = pivot[large].values
        gaps = q_s - q_l   # quality gap

        embedding_ids = pivot.index.get_level_values("embedding_id").tolist()

        # -------------------------------------------------
        # Produce labels depending on router_mode
        # -------------------------------------------------
        if self.router_mode == "deterministic":
            labels = (gaps >= 0).astype(float)

        elif self.router_mode == "probabilistic":
            labels = 1 / (1 + np.exp(-gaps / self.router_tau))

        elif self.router_mode == "transformed":
            labels = self._compute_transformed_labels(gaps)

        else:
            raise ValueError("[HybridLLMRouter] Invalid router_mode.")

        # Embeddings
        X = [self.query_embedding_data[i].numpy() for i in embedding_ids]
        return X, labels.tolist()

    # ==============================================================
    # Transformed router: find best t*
    # ==============================================================
    def _compute_transformed_labels(self, gaps: np.ndarray) -> np.ndarray:
        if gaps.size == 0:
            raise ValueError("[HybridLLMRouter] gaps array is empty.")

        # Search t on a grid
        t_values = np.linspace(0, np.max(np.abs(gaps)) + 1e-8, 50)

        best_t = 0.0
        best_score = -1.0

        for t in t_values:
            y_t = (gaps >= -t).astype(float)
            p = y_t.mean()
            score = 2 * p * (1 - p)  # maximize balance

            if score > best_score:
                best_score = score
                best_t = t

        print(f"[HybridLLMRouter] transformed mode: best t = {best_t:.4f}")

        return (gaps >= -best_t).astype(float)

    # ==============================================================
    # Route a single query
    # ==============================================================
    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_path)

        emb = [get_longformer_embedding(query["query"]).numpy()]
        score = float(self.mlp_model.predict(emb)[0])
        score = max(0.0, min(1.0, score))

        chosen = self.small_model_name if score >= self.router_threshold else self.large_model_name

        out = copy.copy(query)
        out["model_name"] = chosen
        out["router_score"] = score
        return out

    # ==============================================================
    # Route a batch
    # ==============================================================
    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_path)

        data = copy.copy(batch if batch is not None else self.query_data_test)

        for row in data:
            emb = [get_longformer_embedding(row["query"]).numpy()]
            score = float(self.mlp_model.predict(emb)[0])
            score = max(0.0, min(1.0, score))

            row["model_name"] = (
                self.small_model_name if score >= self.router_threshold else self.large_model_name
            )
            row["router_score"] = score

        return data

