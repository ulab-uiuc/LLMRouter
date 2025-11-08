"""
RouterDC Model
--------------
Core model implementation for RouterDC routing strategy.

This module provides the RouterModule class that implements the dual-contrastive
learning approach for LLM routing.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework while preserving all original logic.
"""

import random
import torch
import torch.nn as nn


class RouterModule(nn.Module):
    """
    RouterModule
    ------------
    Core model for RouterDC routing strategy.

    Architecture:
        - Backbone: Pre-trained encoder (e.g., mDeBERTa-v3-base)
        - Learnable LLM embeddings: nn.Embedding for each LLM node
        - Similarity computation: Cosine similarity or inner product

    Training strategy:
        - Dual-contrastive learning with three loss components:
          1. Sample-LLM contrastive loss
          2. Sample-Sample contrastive loss (task-level)
          3. Cluster contrastive loss
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_state_dim: int = 768,
        node_size: int = 3,
        similarity_function: str = "cos"
    ):
        """
        Initialize RouterModule.

        Args:
            backbone (nn.Module):
                Pre-trained encoder model (e.g., DebertaV2Model).
            hidden_state_dim (int):
                Dimension of hidden states from backbone (default: 768).
            node_size (int):
                Number of LLM nodes to route between (default: 3).
            similarity_function (str):
                Similarity function to use: "cos" for cosine similarity,
                or "inner" for inner product (default: "cos").
        """
        super(RouterModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        self.similarity_function = similarity_function

        # Initialize embeddings with normal distribution
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=std_dev)

    def compute_similarity(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two tensors.

        Args:
            input1 (torch.Tensor): First input tensor, shape (batch_size, hidden_dim)
            input2 (torch.Tensor): Second input tensor, shape (num_nodes, hidden_dim)

        Returns:
            torch.Tensor: Similarity matrix, shape (batch_size, num_nodes)
        """
        if self.similarity_function == "cos":
            # Cosine similarity
            return (input1 @ input2.T) / (
                torch.norm(input1, dim=1).unsqueeze(1) *
                torch.norm(input2, dim=1).unsqueeze(0)
            )
        else:
            # Inner product
            return input1 @ input2.T

    def forward(self, t: float = 1, **input_kwargs):
        """
        Forward pass through the router model.

        The forward function:
        1. Passes input through the backbone encoder
        2. Extracts the [CLS] token representation
        3. Computes similarity with LLM embeddings
        4. Applies temperature scaling

        Args:
            t (float): Temperature for scaling similarity scores (default: 1)
            **input_kwargs: Keyword arguments for the backbone model
                           (typically input_ids, attention_mask)

        Returns:
            tuple: (scores, hidden_state)
                - scores (torch.Tensor): Similarity scores for each LLM,
                                        shape (batch_size, node_size)
                - hidden_state (torch.Tensor): Hidden states from backbone,
                                               shape (batch_size, hidden_dim)
        """
        x = self.backbone(**input_kwargs)
        # Use the first token ([CLS]) as the sequence representation
        hidden_state = x['last_hidden_state'][:, 0, :]
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(
        self,
        x: torch.Tensor,
        index_true: torch.Tensor,
        top_k: int,
        last_k: int
    ) -> torch.Tensor:
        """
        Compute sample-LLM contrastive loss.

        This loss encourages the model to:
        - Assign higher scores to better-performing LLMs
        - Distinguish between high-performing and low-performing LLMs

        Args:
            x (torch.Tensor): Similarity scores, shape (batch_size, num_llms)
            index_true (torch.Tensor): True performance scores for each LLM,
                                      shape (batch_size, num_llms)
            top_k (int): Number of top-performing LLMs to consider as positives
            last_k (int): Number of low-performing LLMs to consider as negatives

        Returns:
            torch.Tensor: Scalar loss value
        """
        loss = 0
        # Sort by true performance to get top-k best LLMs
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        # Get last-k worst LLMs
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False, dim=-1)

        for i in range(top_k):
            positive_index = top_index[:, i].view(-1, 1)

            # Skip if positive model doesn't perform well (score <= 0)
            mask = torch.where(top_index_true[:, i].view(-1, 1) > 0, 1, 0)

            # Get scores for positive and negative LLMs
            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)

            # Mask out negatives that actually perform well (score > 0.5)
            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)

            # Concatenate positive and negative scores
            temp_x = torch.concat([top_x, last_x], dim=-1)

            # Compute softmax and log-likelihood
            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:, 0])
            log_x = log_x * mask
            loss += torch.mean(-log_x)

        return loss

    def compute_sample_sample_loss_with_task_tag(
        self,
        hidden_state: torch.Tensor,
        dataset_ids: torch.Tensor,
        t: float,
        H: int = 3
    ) -> torch.Tensor:
        """
        Compute sample-sample contrastive loss with task tags.

        This loss encourages similar hidden representations for samples
        from the same task/dataset, and dissimilar representations for
        samples from different tasks.

        Args:
            hidden_state (torch.Tensor): Hidden states, shape (batch_size, hidden_dim)
            dataset_ids (torch.Tensor): Dataset ID for each sample, shape (batch_size,)
            t (float): Temperature for scaling similarity scores
            H (int): Number of negative samples to consider (default: 3)

        Returns:
            torch.Tensor: Scalar loss value
        """
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H

        # Build positive and negative sample indices for each sample
        all_index = []
        for dataset_id in dataset_ids:
            # Find samples from the same dataset (positives)
            positive_indexs = torch.nonzero(dataset_ids == dataset_id)
            select_positive_index = random.choice(positive_indexs)

            # Find samples from different datasets (negatives)
            negtive_indexs = torch.nonzero(dataset_ids != dataset_id)
            if len(negtive_indexs) < last_k2:
                # Skip if not enough negative samples
                continue

            # Randomly sample H negative samples
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].squeeze()

            # Combine positive and negative indices
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)

        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        # Compute softmax and log-likelihood
        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:, 0])

        return loss

    def compute_cluster_loss(
        self,
        hidden_state: torch.Tensor,
        cluster_ids: torch.Tensor,
        t: float,
        H: int = 3
    ) -> torch.Tensor:
        """
        Compute cluster contrastive loss.

        Similar to sample-sample loss, but uses cluster IDs instead of
        dataset IDs. This encourages similar representations within
        clusters and dissimilar representations across clusters.

        Args:
            hidden_state (torch.Tensor): Hidden states, shape (batch_size, hidden_dim)
            cluster_ids (torch.Tensor): Cluster ID for each sample, shape (batch_size,)
            t (float): Temperature for scaling similarity scores
            H (int): Number of negative samples to consider (default: 3)

        Returns:
            torch.Tensor: Scalar loss value
        """
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H

        # Build positive and negative sample indices for each sample
        all_index = []
        for cluster_id in cluster_ids:
            # Find samples from the same cluster (positives)
            positive_indexs = torch.nonzero(cluster_ids == cluster_id)
            select_positive_index = random.choice(positive_indexs)

            # Find samples from different clusters (negatives)
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id)
            if len(negtive_indexs) < last_k2:
                # Skip if not enough negative samples
                continue

            # Randomly sample H negative samples
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].view(-1)

            # Combine positive and negative indices
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)

        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        # Compute softmax and log-likelihood
        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:, 0])

        return loss
