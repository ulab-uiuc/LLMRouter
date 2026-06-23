"""
GMTRouter neural network models: HeteroGNN and PreferencePredictor.

Implements:
- HeteroGNN: Heterogeneous Graph Neural Network with HGT layers
- PreferencePredictor: Cross-attention based preference scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network using HGT (Heterogeneous Graph Transformer) layers.

    Processes a heterogeneous graph with multiple node types and edge types
    to produce aggregated node embeddings.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        """
        Initialize HeteroGNN.

        Args:
            metadata: Tuple of (node_types, edge_types)
            hidden_dim: Hidden dimension for embeddings
            num_layers: Number of HGT layers
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        try:
            from torch_geometric.nn import HGTConv, Linear
            self.has_pyg = True
        except ImportError:
            print("Warning: PyTorch Geometric not installed. Using simplified GNN.")
            self.has_pyg = False
            self._init_simplified_gnn(metadata)
            return

        node_types, edge_types = metadata

        # Linear projections for each node type to hidden_dim
        self.node_projections = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim)
            for node_type in node_types
        })

        # HGT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads
            )
            self.convs.append(conv)

            # Layer normalization for each node type
            norm_dict = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in node_types
            })
            self.norms.append(norm_dict)

    def _init_simplified_gnn(self, metadata):
        """Initialize simplified GNN when PyTorch Geometric is not available."""
        node_types, _ = metadata

        self.node_projections = nn.ModuleDict({
            node_type: nn.Linear(self.hidden_dim, self.hidden_dim)
            for node_type in node_types
        })

        # Simple GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(self.num_layers)
        ])

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HeteroGNN.

        Args:
            x_dict: Dictionary of node features {node_type: features}
            edge_index_dict: Dictionary of edge indices {(src, rel, dst): edge_index}

        Returns:
            Dictionary of aggregated node embeddings {node_type: embeddings}
        """
        if not self.has_pyg:
            return self._forward_simplified(x_dict)

        # Project node features to hidden_dim
        x_dict = {
            node_type: self.node_projections[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Apply HGT layers
        for i, conv in enumerate(self.convs):
            # HGT convolution
            x_dict_new = conv(x_dict, edge_index_dict)

            # Layer normalization and dropout for each node type
            x_dict = {
                node_type: self.norms[i][node_type](x)
                for node_type, x in x_dict_new.items()
            }

            x_dict = {
                node_type: F.dropout(x, p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }

        return x_dict

    def _forward_simplified(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Simplified forward pass without PyTorch Geometric."""
        # Project features
        x_dict = {
            node_type: self.node_projections[node_type](x)
            for node_type, x in x_dict.items()
        }

        # Apply simple transformations
        for i in range(self.num_layers):
            x_dict = {
                node_type: self.gnn_layers[i](x)
                for node_type, x in x_dict.items()
            }

            x_dict = {
                node_type: self.layer_norms[i](x)
                for node_type, x in x_dict.items()
            }

            x_dict = {
                node_type: F.relu(x)
                for node_type, x in x_dict.items()
            }

            x_dict = {
                node_type: F.dropout(x, p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }

        return x_dict


class PreferencePredictor(nn.Module):
    """
    Preference predictor using cross-attention mechanism.

    Scores LLM candidates based on user embeddings and query context.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize PreferencePredictor.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Cross-attention: query attends to user preferences and LLM features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP for final scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        user_emb: torch.Tensor,
        llm_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict preference score for a query-user-LLM triple.

        Args:
            query_emb: Query embeddings [batch_size, hidden_dim]
            user_emb: User embeddings [batch_size, hidden_dim]
            llm_emb: LLM embeddings [batch_size, hidden_dim]

        Returns:
            Preference scores [batch_size, 1]
        """
        # Combine user and LLM embeddings as context
        # Shape: [batch_size, 2, hidden_dim]
        context = torch.stack([user_emb, llm_emb], dim=1)

        # Query as query for attention
        # Shape: [batch_size, 1, hidden_dim]
        query = query_emb.unsqueeze(1)

        # Cross-attention: query attends to [user, llm]
        # attn_output: [batch_size, 1, hidden_dim]
        attn_output, _ = self.cross_attention(query, context, context)

        # Remove sequence dimension
        # Shape: [batch_size, hidden_dim]
        attn_output = attn_output.squeeze(1)

        # Score
        # Shape: [batch_size, 1]
        score = self.scorer(attn_output)

        return score


class GMTRouterModel(nn.Module):
    """
    Complete GMTRouter model combining HeteroGNN and PreferencePredictor.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize complete GMTRouter model.

        Args:
            metadata: Graph metadata (node_types, edge_types)
            hidden_dim: Hidden dimension
            num_gnn_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # HeteroGNN for graph embedding
        self.hetero_gnn = HeteroGNN(
            metadata=metadata,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            num_heads=num_heads
        )

        # Preference predictor
        self.preference_predictor = PreferencePredictor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        user_ids: torch.Tensor,
        query_ids: torch.Tensor,
        llm_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: embed graph and predict preferences.

        Args:
            x_dict: Node features dictionary
            edge_index_dict: Edge indices dictionary
            user_ids: User node IDs [batch_size]
            query_ids: Query node IDs [batch_size]
            llm_ids: LLM node IDs [batch_size]

        Returns:
            Preference scores [batch_size, 1]
        """
        # Get aggregated embeddings from GNN
        agg_emb = self.hetero_gnn(x_dict, edge_index_dict)

        # Extract embeddings for the specific nodes
        user_emb = agg_emb['user'][user_ids]  # [batch_size, hidden_dim]
        query_emb = agg_emb['query'][query_ids]  # [batch_size, hidden_dim]
        llm_emb = agg_emb['llm'][llm_ids]  # [batch_size, hidden_dim]

        # Predict preference scores
        scores = self.preference_predictor(query_emb, user_emb, llm_emb)

        return scores

    def predict_for_user_query(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        user_id: int,
        query_id: int,
        llm_candidates: List[int]
    ) -> Tuple[int, torch.Tensor]:
        """
        Predict best LLM for a user-query pair.

        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            user_id: User node ID
            query_id: Query node ID
            llm_candidates: List of candidate LLM IDs

        Returns:
            Tuple of (best_llm_id, scores)
        """
        self.eval()

        with torch.no_grad():
            # Get aggregated embeddings
            agg_emb = self.hetero_gnn(x_dict, edge_index_dict)

            user_emb = agg_emb['user'][user_id].unsqueeze(0)  # [1, hidden_dim]
            query_emb = agg_emb['query'][query_id].unsqueeze(0)  # [1, hidden_dim]

            # Score each LLM candidate
            scores = []
            for llm_id in llm_candidates:
                llm_emb = agg_emb['llm'][llm_id].unsqueeze(0)  # [1, hidden_dim]
                score = self.preference_predictor(query_emb, user_emb, llm_emb)
                scores.append(score.item())

            scores = torch.tensor(scores)
            best_idx = torch.argmax(scores).item()
            best_llm_id = llm_candidates[best_idx]

        return best_llm_id, scores
