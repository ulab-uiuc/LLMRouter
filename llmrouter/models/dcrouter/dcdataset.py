"""
DCRouter Dataset
----------------
Dataset implementation for DCRouter training and evaluation.

This module provides the DCDataset class for loading and processing
DCRouter training data.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework while preserving all original logic.
"""

import json
import torch
from torch.utils.data import Dataset


class DCDataset(Dataset):
    """
    DCDataset
    ---------
    Dataset class for DCRouter training and evaluation.

    Data format:
        Each data point should be a dictionary with:
        - "question" (str): The input query/question
        - "scores" (dict): Performance scores for each LLM
                          e.g., {"llm1": 0.8, "llm2": 0.6, ...}
        - "cluster_id" (int, optional): Cluster ID for the sample

    The dataset handles tokenization and returns:
        - Tokenized input (input_ids, attention_mask)
        - Performance scores tensor
        - Dataset ID (for task-level contrastive learning)
        - Cluster ID (for cluster-level contrastive learning)
    """

    def __init__(
        self,
        data_path: str,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        size: int | None = None,
        data_type: str = "multi_attempt",
        dataset_id: int = 0,
    ):
        """
        Initialize DCDataset.

        Args:
            data_path (str):
                Path to JSON file containing training/test data.
            source_max_token_len (int):
                Maximum length for source tokens (default: 512).
            target_max_token_len (int):
                Maximum length for target tokens (default: 512).
            size (int | None):
                Optional fixed size for dataset. If provided and dataset
                is smaller, data will be repeated to reach this size.
            data_type (str):
                Type of data: "multi_attempt" or "probability" (default: "multi_attempt").
            dataset_id (int):
                ID for this dataset, used for task-level contrastive learning
                (default: 0).
        """
        # Load data from JSON file
        with open(data_path, 'r') as f:
            if data_path.endswith('.json'):
                self.data = json.load(f)
            elif data_path.endswith('.jsonl'):
                # Load JSONL format
                self.data = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                # Default to JSON
                self.data = json.load(f)

        # Repeat data if size is specified and larger than current dataset
        if size:
            while len(self.data) < size:
                self.data.extend(self.data)
            self.data = self.data[:size]

        # Extract LLM names from first data point
        self.router_node = list(self.data[0]['scores'].keys())

        # Tokenizer will be registered later via register_tokenizer()
        self.tokenizer = None
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.data_type = data_type
        self.dataset_id = dataset_id

    def __getitem__(self, index: int):
        """
        Get a single data point.

        Args:
            index (int): Index of the data point

        Returns:
            tuple: (question_tokens, scores, dataset_id, cluster_id)
                - question_tokens (dict): Tokenized input with input_ids and attention_mask
                - scores (torch.Tensor): Performance scores for each LLM
                - dataset_id (int): Dataset ID for this sample
                - cluster_id (int): Cluster ID for this sample (0 if not available)
        """
        data_point = self.data[index]

        # Extract performance scores as tensor
        scores = torch.tensor(list(data_point['scores'].values()))

        # Extract question text
        question = data_point['question']

        # Tokenize question
        question_id = self.tokenizer(
            question,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Flatten tensors (remove batch dimension)
        question_id['input_ids'] = question_id.input_ids.flatten()
        question_id['attention_mask'] = question_id.attention_mask.flatten()

        # Extract cluster_id if available, otherwise default to 0
        cluster_id = data_point.get('cluster_id', 0)

        return question_id, scores, self.dataset_id, cluster_id

    def __len__(self) -> int:
        """
        Get dataset size.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data)

    def register_tokenizer(self, tokenizer):
        """
        Register a tokenizer for this dataset.

        This method must be called before using the dataset, as tokenization
        happens in __getitem__.

        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer
