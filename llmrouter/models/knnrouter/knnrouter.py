from typing import Any, Dict, List, Optional, Union
import os
import pickle
import random
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from llmrouter.models.meta_router import MetaRouter


class KNNRouter(MetaRouter):
    """
    KNNRouter
    ----------
    A router that uses a pre-trained or pre-saved KNN model to select
    the most similar LLM based on embedding proximity. If no KNN model
    file is found, it falls back to random selection.

    The router inherits from MetaRouter (which includes an nn.Module)
    for consistent interface, but it does not perform training.

    YAML format:
    ------------
    llm_data:
      GPT4:
        size: "175B"
        embedding: [0.12, 0.33, 0.78, 0.44]
      Claude3:
        size: "52B"
        embedding: [0.10, 0.25, 0.70, 0.50]
    optional:
      knn_model_path: "configs/knn_model.pkl"
      n_neighbors: 2
      metric: "cosine"
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the KNNRouter.

        Args:
            yaml_path (str): Path to YAML config file.
        """
        # Keep nn.Module for MetaRouter interface
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Core params
        self.n_neighbors: int = self.config.get("n_neighbors", 1)
        self.metric: str = self.config.get("metric", "cosine")
        self.knn_model_path: Optional[str] = self.config.get("knn_model_path")

        # Prepare embeddings
        self.llm_names: List[str] = []
        self.llm_embs: List[np.ndarray] = []
        for name, info in self.llm_data.items():
            emb = info.get("embedding")
            if emb is not None:
                self.llm_names.append(name)
                self.llm_embs.append(np.array(emb, dtype=np.float32))

        if not self.llm_embs:
            raise ValueError("No valid embeddings found in YAML config.")
        self.llm_embs = np.vstack(self.llm_embs)

        # Try to load pre-saved KNN model
        self.model: Optional[NearestNeighbors] = None
        if self.knn_model_path and os.path.exists(self.knn_model_path):
            try:
                with open(self.knn_model_path, "rb") as f:
                    self.model = pickle.load(f)
                print(f"✅ Loaded KNN model from {self.knn_model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load KNN model: {e}")
                self.model = None
        else:
            print("⚠️ No KNN model file found — router will fallback to random routing.")

        print(f"✅ KNNRouter initialized ({len(self.llm_names)} LLMs loaded).")

    def route_single(self, batch: Optional[Union[np.ndarray, List[float]]] = None) -> Dict[str, Any]:
        """
        Route the given query embedding(s) to the best-matching LLM(s).

        Args:
            batch (np.ndarray | list[float], optional):
                Query embedding vector or batch (shape [N, dim]).

        Returns:
            dict:
                {
                    "query_shape": tuple,
                    "results": [
                        {
                            "model_name": str,
                            "distance": float,
                            "model_info": dict
                        },
                        ...
                    ]
                }
        """
        if batch is None:
            raise ValueError("Missing input query embedding for routing.")

        query_emb = np.array(batch, dtype=np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        results: List[Dict[str, Any]] = []

        # Case 1: KNN model available
        if self.model is not None:
            distances, indices = self.model.kneighbors(query_emb, n_neighbors=self.n_neighbors)
            for j in range(len(indices[0])):
                llm_idx = indices[0][j]
                llm_name = self.llm_names[llm_idx]
                dist = float(distances[0][j])
                info = self.llm_data[llm_name]
                results.append({
                    "model_name": llm_name,
                    "distance": dist,
                    "model_info": info,
                })
            print(f"🔍 Routed to {results[0]['model_name']} (distance={results[0]['distance']:.4f})")

        # Case 2: Fallback (no KNN model)
        else:
            random_name = random.choice(self.llm_names)
            results.append({
                "model_name": random_name,
                "distance": float("nan"),
                "model_info": self.llm_data[random_name],
            })
            print(f"🎲 Randomly selected LLM: {random_name}")

        return {
            "query_shape": query_emb.shape,
            "results": results,
        }

    def route_batch(self, batch: Optional[Union[np.ndarray, List[float]]] = None) -> Dict[str, Any]:
        """
        Route the given query embedding(s) to the best-matching LLM(s).

        Args:
            batch (np.ndarray | list[float], optional):
                Query embedding vector or batch (shape [N, dim]).

        Returns:
            dict:
                {
                    "query_shape": tuple,
                    "results": [
                        {
                            "model_name": str,
                            "distance": float,
                            "model_info": dict
                        },
                        ...
                    ]
                }
        """
        if batch is None:
            raise ValueError("Missing input query embedding for routing.")

        query_emb = np.array(batch, dtype=np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        results: List[Dict[str, Any]] = []

        # Case 1: KNN model available
        if self.model is not None:
            distances, indices = self.model.kneighbors(query_emb, n_neighbors=self.n_neighbors)
            for j in range(len(indices[0])):
                llm_idx = indices[0][j]
                llm_name = self.llm_names[llm_idx]
                dist = float(distances[0][j])
                info = self.llm_data[llm_name]
                results.append({
                    "model_name": llm_name,
                    "distance": dist,
                    "model_info": info,
                })
            print(f"🔍 Routed to {results[0]['model_name']} (distance={results[0]['distance']:.4f})")

        # Case 2: Fallback (no KNN model)
        else:
            random_name = random.choice(self.llm_names)
            results.append({
                "model_name": random_name,
                "distance": float("nan"),
                "model_info": self.llm_data[random_name],
            })
            print(f"🎲 Randomly selected LLM: {random_name}")

        return {
            "query_shape": query_emb.shape,
            "results": results,
        }

