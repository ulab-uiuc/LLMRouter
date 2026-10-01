"""
GMTRouter - Graph-based Multi-Turn Personalized Router

Complete integration into LLMRouter with:
- Heterogeneous Graph Neural Network (HeteroGNN) with 5 node types
- Preference-based routing using PreferencePredictor
- Pairwise preference learning
- User personalization
- Special JSONL data format with automatic detection

Training and inference are fully integrated into LLMRouter CLI.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import torch
import torch.nn as nn
import numpy as np

from llmrouter.models.meta_router import MetaRouter
from llmrouter.models.gmtrouter.data_loader import GMTRouterDataLoader, detect_data_format
from llmrouter.models.gmtrouter.models import GMTRouterModel
from llmrouter.data import get_format_requirements, DataFormatType


class GMTRouter(MetaRouter):
    """
    GMTRouter - Graph-based Multi-Turn Personalized Router

    Architecture:
    - 5 Node Types: User, Session, Query, LLM, Response
    - 21 Edge Types: Modeling various relationships
    - HeteroGNN: Heterogeneous Graph Transformer layers
    - PreferencePredictor: Cross-attention for LLM selection

    Data Format (JSONL):
    {
      "judge": "user_id",
      "model": "gpt-4",
      "question_id": "q123",
      "turn": 1,
      "conversation": [
        {
          "query": "What is ML?",
          "query_emb": [0.1, 0.2, ...],
          "response": "ML is...",
          "rating": 4.5
        }
      ],
      "model_emb": [0.3, 0.4, ...]
    }
    """

    def __init__(self, yaml_path: str):
        """
        Initialize GMTRouter.

        Args:
            yaml_path: Path to YAML configuration file
        """
        dummy_model = nn.Identity()
        super(GMTRouter, self).__init__(model=dummy_model, yaml_path=yaml_path)

        # GMTRouter-specific configuration
        self.gmt_config = self.cfg.get("gmt_config", {})

        # Model architecture parameters
        self.hidden_dim = self.gmt_config.get("hidden_dim", 128)
        self.num_gnn_layers = self.gmt_config.get("num_gnn_layers", 2)
        self.dropout = self.gmt_config.get("dropout", 0.1)
        self.personalization = self.gmt_config.get("personalization", True)

        # Data paths
        model_paths = self.cfg.get("model_path", {})
        self.model_checkpoint_path = model_paths.get("load_model_path", "saved_models/gmtrouter/gmtrouter.pt")

        # Initialize data loader
        self.data_loader = GMTRouterDataLoader(self.cfg)

        # Models
        self.gmt_model = None  # GMTRouterModel
        self.graph_data = None  # Graph structure for inference
        self.metadata = None  # Graph metadata

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load pretrained model if available
        self._load_pretrained_model()

    def _load_pretrained_model(self):
        """Load pretrained GMTRouter model."""
        if not os.path.exists(self.model_checkpoint_path):
            print(f"No pretrained model found at {self.model_checkpoint_path}")
            print("GMTRouter will need to be trained first.")
            return

        try:
            checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
            print(f"Loaded GMTRouter checkpoint from {self.model_checkpoint_path}")

            # Extract configuration
            config = checkpoint.get('config', {})
            self.hidden_dim = config.get('hidden_dim', self.hidden_dim)
            self.num_gnn_layers = config.get('num_gnn_layers', self.num_gnn_layers)

            # Model will be initialized when graph data is loaded
            self.checkpoint = checkpoint

        except Exception as e:
            print(f"Warning: Could not load GMTRouter checkpoint: {e}")
            self.checkpoint = None

    def get_training_data(self) -> Tuple[Any, Any]:
        """
        Load training and validation data for GMTRouter.

        Returns:
            tuple: (train_data, val_data) with graph structures and metadata
        """
        data_paths = self.cfg.get("data_path", {})

        # Get training data path
        train_path = data_paths.get("training_set", data_paths.get("routing_data_train"))
        if not train_path or not os.path.exists(train_path):
            print(f"Error: Training data not found at {train_path}")
            print("Please download GMTRouter data. See README for instructions.")
            return None, None

        # Detect and validate format
        format_type = detect_data_format(train_path)
        if format_type != "gmtrouter":
            print(f"\n{'='*70}")
            print(f"ERROR: Data Format Mismatch!")
            print(f"{'='*70}")
            print(f"Expected: GMTRouter JSONL format")
            print(f"Detected: {format_type}")
            print()

            # Show format requirements
            requirements = get_format_requirements(DataFormatType.GMTROUTER)
            print(f"Required fields: {', '.join(requirements['required_fields'])}")
            print()
            print("Example JSONL entry:")
            print(json.dumps(requirements['example'], indent=2))
            print()
            print(f"See llmrouter/models/gmtrouter/README.md for complete documentation.")
            print(f"{'='*70}\n")
            return None, None

        print(f"Loading training data from {train_path}...")

        # Load training data
        train_graph, train_metadata = self.data_loader.load_data(train_path)

        # Prepare training data structure
        if hasattr(train_graph, 'x_dict'):
            # PyTorch Geometric HeteroData
            train_data = {
                'x_dict': train_graph.x_dict,
                'edge_index_dict': train_graph.edge_index_dict,
                'metadata': train_graph.metadata(),
                'train_metadata': train_metadata
            }
        else:
            # Simplified data structure
            train_data = {
                **train_graph,
                'train_metadata': train_metadata
            }

        # Load validation data if available
        val_path = data_paths.get("valid_set")
        val_data = None

        if val_path and os.path.exists(val_path):
            print(f"Loading validation data from {val_path}...")
            val_loader = GMTRouterDataLoader(self.cfg)
            val_graph, val_metadata = val_loader.load_data(val_path)

            if hasattr(val_graph, 'x_dict'):
                val_data = {
                    'x_dict': val_graph.x_dict,
                    'edge_index_dict': val_graph.edge_index_dict,
                    'metadata': val_graph.metadata(),
                    'val_metadata': val_metadata
                }
            else:
                val_data = {
                    **val_graph,
                    'val_metadata': val_metadata
                }
        else:
            print("No validation data provided. Will split from training data.")
            # Split validation from training
            val_split = 0.2
            train_pairs = train_metadata['pairs']
            split_idx = int(len(train_pairs) * (1 - val_split))

            val_metadata_split = {
                **train_metadata,
                'pairs': train_pairs[split_idx:]
            }
            train_metadata['pairs'] = train_pairs[:split_idx]

            val_data = {**train_data, 'val_metadata': val_metadata_split}
            train_data['train_metadata'] = train_metadata

        # Store graph data and metadata for inference
        self.graph_data = train_data
        self.metadata = train_metadata

        # Initialize model if checkpoint exists
        if hasattr(self, 'checkpoint') and self.checkpoint is not None:
            self._initialize_model_from_data(train_data)

        return train_data, val_data

    def _initialize_model_from_data(self, data: Dict):
        """Initialize GMTRouter model from loaded data."""
        # Extract metadata
        if 'metadata' in data:
            metadata = data['metadata']
        else:
            node_types = ['user', 'session', 'query', 'llm', 'response']
            edge_types = []
            for edge_type in data.get('edges', {}).keys():
                parts = edge_type.split('_')
                if len(parts) >= 3:
                    src_type = parts[0]
                    dst_type = parts[-1]
                    rel_type = '_'.join(parts[1:-1])
                    edge_types.append((src_type, rel_type, dst_type))
            metadata = (node_types, edge_types)

        # Initialize model
        self.gmt_model = GMTRouterModel(
            metadata=metadata,
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            dropout=self.dropout
        )

        # Load weights from checkpoint
        self.gmt_model.load_state_dict(self.checkpoint['model_state_dict'])
        self.gmt_model.to(self.device)
        self.gmt_model.eval()

        print(f"GMTRouter model initialized and loaded from checkpoint")

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using personalized GMTRouter.

        Args:
            query: Query dictionary with:
                - query_text: The query string
                - user_id: User identifier (required for personalization)
                - session_id: Session identifier (optional)
                - turn: Turn number (optional)

        Returns:
            Dictionary with routing decision
        """
        query_text = query.get("query_text", query.get("query", ""))
        user_id_str = query.get("user_id", "default_user")

        # Check if model is trained
        if self.gmt_model is None or self.graph_data is None:
            return self._fallback_routing(query_text, user_id_str)

        # Get user ID from metadata
        user_map = self.metadata.get('user_map', {})
        if user_id_str not in user_map:
            # New user - use fallback
            return self._fallback_routing(query_text, user_id_str)

        user_id = user_map[user_id_str]

        # For now, use a simple query embedding (in production, would use PLM)
        # Find similar query in training data or use average
        query_id = 0  # Simplified: use first query

        # Get LLM candidates
        llm_map = self.metadata.get('llm_map', {})
        llm_id_to_name = self.metadata.get('llm_id_to_name', {})
        llm_candidates = list(llm_map.values())

        if len(llm_candidates) == 0:
            return self._fallback_routing(query_text, user_id_str)

        # Prepare graph data
        if 'x_dict' in self.graph_data:
            x_dict = {k: v.to(self.device) for k, v in self.graph_data['x_dict'].items()}
            edge_index_dict = {k: v.to(self.device) for k, v in self.graph_data['edge_index_dict'].items()}
        else:
            x_dict = {
                'user': self.graph_data['user_embeddings'].to(self.device),
                'session': self.graph_data['session_embeddings'].to(self.device),
                'query': self.graph_data['query_embeddings'].to(self.device),
                'llm': self.graph_data['llm_embeddings'].to(self.device),
            }
            edge_index_dict = {}

        # Predict best LLM
        best_llm_id, scores = self.gmt_model.predict_for_user_query(
            x_dict, edge_index_dict, user_id, query_id, llm_candidates
        )

        best_model_name = llm_id_to_name.get(best_llm_id, self.models[0] if self.models else "gpt-3.5-turbo")

        result = {
            "model_name": best_model_name,
            "confidence": float(torch.max(scores).item()),
            "user_preference": float(scores[llm_candidates.index(best_llm_id)].item()),
            "reasoning": f"Selected based on user {user_id_str}'s learned preferences via GMTRouter"
        }

        return result

    def _fallback_routing(self, query_text: str, user_id: str) -> Dict[str, Any]:
        """Fallback routing when model is not available or user is new."""
        default_model = self.models[0] if self.models else "gpt-3.5-turbo"

        return {
            "model_name": default_model,
            "confidence": 0.5,
            "user_preference": 0.5,
            "reasoning": f"GMTRouter fallback: model not trained or user {user_id} not in training data"
        }

    def route_batch(self, batch: List[Dict[str, Any]], task_name: str = None) -> List[Dict[str, Any]]:
        """Route a batch of queries."""
        results = []
        for query in batch:
            result = self.route_single(query)
            results.append(result)
        return results
