"""
Threshold Router - Difficulty-based routing
"""

from typing import Any, Dict, List
import torch
import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter


class DifficultyEstimator(nn.Module):
    """Simple MLP to estimate query difficulty."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output difficulty score in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Query embeddings [batch_size, input_dim]

        Returns:
            Difficulty scores [batch_size, 1]
        """
        return self.network(x)


class ThresholdRouter(MetaRouter):
    """
    Threshold-based router that estimates query difficulty.

    Routes queries based on estimated difficulty:
    - difficulty < threshold -> use smaller/cheaper model
    - difficulty >= threshold -> use larger/more capable model

    Hyperparameters:
        - threshold: Difficulty threshold (default: 0.5)
        - small_model: Name of the small/cheap model
        - large_model: Name of the large/capable model
        - embedding_dim: Dimension of query embeddings
        - hidden_dim: Hidden dimension for difficulty estimator
    """

    def __init__(self, yaml_path: str):
        """Initialize ThresholdRouter."""
        # Get hyperparameters from config
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        hparam = cfg.get('hparam', {})
        embedding_dim = hparam.get('embedding_dim', 768)
        hidden_dim = hparam.get('hidden_dim', 128)

        # Create difficulty estimator model
        difficulty_model = DifficultyEstimator(embedding_dim, hidden_dim)

        # Initialize parent
        super().__init__(model=difficulty_model, yaml_path=yaml_path)

        # Extract hyperparameters
        self.threshold = hparam.get('threshold', 0.5)
        self.small_model = hparam.get('small_model', None)
        self.large_model = hparam.get('large_model', None)

        # Validate LLM data
        if not hasattr(self, 'llm_data') or not self.llm_data:
            raise ValueError("No LLM data found in config")

        self.llm_names = list(self.llm_data.keys())

        # Auto-detect small/large models if not specified
        if self.small_model is None or self.large_model is None:
            if len(self.llm_names) < 2:
                raise ValueError("At least 2 LLMs required for ThresholdRouter")
            # Assume first is small, last is large (or specify in config)
            self.small_model = self.small_model or self.llm_names[0]
            self.large_model = self.large_model or self.llm_names[-1]

        # Load trained model if available
        model_path = self.cfg.get('model_path', {}).get('load_model_path')
        if model_path:
            try:
                self.load_router(model_path)
                print(f"✅ Loaded trained model from {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model from {model_path}: {e}")
                print("   Using random initialization")

        print(f"✅ ThresholdRouter initialized:")
        print(f"   Small model (difficulty < {self.threshold}): {self.small_model}")
        print(f"   Large model (difficulty >= {self.threshold}): {self.large_model}")

    def _estimate_difficulty(self, query_embedding: torch.Tensor) -> float:
        """
        Estimate difficulty of a query.

        Args:
            query_embedding: Query embedding tensor [embedding_dim]

        Returns:
            Difficulty score in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)

            difficulty = self.model(query_embedding)
            return difficulty.item()

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        """Route a single query based on difficulty estimation."""
        # Get query embedding
        if 'embedding' in query_input:
            embedding = query_input['embedding']
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
        elif hasattr(self, 'query_embeddings') and 'query' in query_input:
            # Try to get from loaded embeddings (if available).
            # This is a simplified version - real implementation would hash or lookup
            # the query embedding here; for now we require an explicit 'embedding'.
            raise ValueError(
                "Query embedding not provided. "
                "Pass 'embedding' in query_input or implement embedding generation."
            )
        else:
            raise ValueError(
                "No embedding found for query. "
                "Please provide 'embedding' in query_input."
            )

        # Estimate difficulty
        difficulty = self._estimate_difficulty(embedding)

        # Select model based on threshold
        selected_model = self.small_model if difficulty < self.threshold else self.large_model

        return {
            "query": query_input.get("query", ""),
            "model_name": selected_model,
            "predicted_llm": selected_model,
            "predicted_llm_name": selected_model,
            "difficulty_score": difficulty,
            "threshold": self.threshold,
            "method": "threshold",
        }

    def route_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route a batch of queries."""
        results = []
        for query_input in batch:
            result = self.route_single(query_input)
            results.append(result)
        return results
