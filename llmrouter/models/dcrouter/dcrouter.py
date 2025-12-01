"""
DCRouter Router
---------------
Router implementation for the DCRouter routing strategy.

This module provides the DCRouter class that integrates with the
LLMRouter framework.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework.
"""

import os
import yaml
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DebertaV2Model
from llmrouter.models.meta_router import MetaRouter
from .dcmodel import RouterModule
from .dcdataset import DCDataset
from .dcdata_utils import preprocess_data


class DCRouter(MetaRouter):
    """
    DCRouter
    --------
    Router that uses dual-contrastive learning strategy for LLM routing decisions.

    DCRouter uses a pre-trained encoder (e.g., mDeBERTa) combined with learnable
    LLM embeddings to make routing decisions. The model is trained with three
    contrastive learning objectives:
    1. Sample-LLM contrastive loss
    2. Sample-Sample contrastive loss (task-level)
    3. Cluster contrastive loss
    """

    def __init__(self, yaml_path: str):
        """
        Initialize DCRouter.

        Args:
            yaml_path (str): Path to YAML config file
        """
        # Load configuration
        with open(yaml_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Resolve project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Prepare data
        self._prepare_data()

        # Initialize tokenizer and backbone
        backbone_model = self.cfg['model_path']['backbone_model']
        print(f"[DCRouter] Loading backbone model: {backbone_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone_model,
            truncation_side='left',
            padding=True
        )
        encoder_model = DebertaV2Model.from_pretrained(backbone_model)
        print("[DCRouter] Backbone model loaded successfully!")

        # Load datasets
        print(f"[DCRouter] Loading datasets...")
        self.train_dataset = DCDataset(
            data_path=self.train_data_processed,
            source_max_token_len=self.cfg.get('data_preprocessing', {}).get('source_max_token_len', 512),
            target_max_token_len=self.cfg.get('data_preprocessing', {}).get('target_max_token_len', 512),
            dataset_id=0
        )
        self.train_dataset.register_tokenizer(self.tokenizer)

        self.test_dataset = DCDataset(
            data_path=self.test_data_processed,
            source_max_token_len=self.cfg.get('data_preprocessing', {}).get('source_max_token_len', 512),
            target_max_token_len=self.cfg.get('data_preprocessing', {}).get('target_max_token_len', 512),
            dataset_id=1
        )
        self.test_dataset.register_tokenizer(self.tokenizer)

        num_llms = len(self.train_dataset.router_node)
        print(f"[DCRouter] Datasets loaded:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        print(f"  Number of LLMs: {num_llms}")
        print(f"  LLM names: {self.train_dataset.router_node}")

        # Create RouterModule
        model_config = self.cfg['model']
        model = RouterModule(
            backbone=encoder_model,
            hidden_state_dim=model_config['hidden_state_dim'],
            node_size=num_llms,
            similarity_function=model_config['similarity_function']
        )
        print("[DCRouter] RouterModule created successfully!")

        # Initialize parent class
        super().__init__(model=model, yaml_path=yaml_path)

    def _prepare_data(self):
        """Prepare and preprocess data if needed."""
        data_path_config = self.cfg['data_path']
        train_data_raw = os.path.join(self.project_root, data_path_config['routing_data_train'])
        test_data_raw = os.path.join(self.project_root, data_path_config['routing_data_test'])

        # Preprocessed data paths
        preprocessed_dir = os.path.join(
            self.project_root,
            data_path_config.get('preprocessed_dir', 'data/dcrouter_preprocessed')
        )
        self.train_data_processed = os.path.join(preprocessed_dir, "train.json")
        self.test_data_processed = os.path.join(preprocessed_dir, "test.json")

        # Check if preprocessing is needed
        if not os.path.exists(self.train_data_processed) or not os.path.exists(self.test_data_processed):
            print("\n[DCRouter] Preprocessed data not found. Starting data preprocessing...")

            preprocess_config = self.cfg.get('data_preprocessing', {})
            n_clusters = preprocess_config.get('n_clusters', 3)
            max_test_samples = preprocess_config.get('max_test_samples', 500)

            # Preprocess training data
            print("\n[DCRouter] Preprocessing training data...")
            preprocess_data(
                input_path=train_data_raw,
                output_path=self.train_data_processed,
                add_cluster_id=True,
                n_clusters=n_clusters,
                max_samples=None
            )

            # Preprocess test data
            print("\n[DCRouter] Preprocessing test data...")
            preprocess_data(
                input_path=test_data_raw,
                output_path=self.test_data_processed,
                add_cluster_id=False,
                n_clusters=n_clusters,
                max_samples=max_test_samples
            )

            print("[DCRouter] Data preprocessing completed!\n")

    def route(self, batch):
        """
        Perform routing on a batch of data.

        Args:
            batch (dict): A batch containing tokenized inputs

        Returns:
            dict: A dictionary with routing outputs
        """
        # Extract temperature if provided, default to 1.0
        temperature = batch.get("temperature", 1.0)

        # Prepare inputs for the model
        input_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        # Forward pass through RouterModule
        scores, hidden_state = self.model(t=temperature, **input_kwargs)

        # Get predicted LLM indices (argmax)
        predictions = torch.argmax(scores, dim=1)

        return {
            "scores": scores,
            "hidden_state": hidden_state,
            "predictions": predictions,
        }

    def route_batch(self):
        """
        Route a batch of data from the test dataset.

        Returns:
            dict: Routing results
        """
        from torch.utils.data import DataLoader

        # Load model if exists
        inference_config = self.cfg.get('inference', {})
        device = inference_config.get('device', 'cpu')

        # Try to load checkpoint
        save_dir = os.path.join(self.project_root, os.path.dirname(self.cfg['model_path']['load_model_path']))
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(save_dir, 'best_training_model.pth')

        if os.path.exists(checkpoint_path):
            print(f"[DCRouter] Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"[DCRouter] Warning: No checkpoint found. Using untrained model.")

        self.model = self.model.to(device)
        self.model.eval()

        # Run inference
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=inference_config.get('batch_size', 64),
            shuffle=False
        )

        all_predictions = []
        routing_correct = 0
        task_correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in test_dataloader:
                inputs, scores, _, _ = batch_data
                inputs = {k: v.to(device) for k, v in inputs.items()}
                scores = scores.to(device)

                batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "temperature": inference_config.get('temperature', 1.0),
                }

                outputs = self.route(batch)
                predictions = outputs["predictions"]
                best_llm = torch.argmax(scores, dim=1)

                routing_correct += (predictions == best_llm).sum().item()
                binary_scores = (scores > 0).float()
                mask = torch.zeros_like(scores)
                mask.scatter_(1, predictions.unsqueeze(1), 1)
                task_correct += (binary_scores * mask).sum().item()
                total += len(predictions)

                all_predictions.extend(predictions.cpu().tolist())

        return {
            "total": total,
            "routing_accuracy": routing_correct / total,
            "task_accuracy": task_correct / total,
            "predictions": all_predictions
        }

    def route_single(self, data):
        """
        Route a single query.

        Args:
            data (dict): Query data with 'query' key

        Returns:
            dict: Routing result
        """
        inference_config = self.cfg.get('inference', {})
        device = inference_config.get('device', 'cpu')

        # Load model if exists
        save_dir = os.path.join(self.project_root, os.path.dirname(self.cfg['model_path']['load_model_path']))
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(save_dir, 'best_training_model.pth')

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(device)
        self.model.eval()

        # Tokenize query
        query_text = data["query"]
        query_tokens = self.tokenizer(
            query_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        batch = {
            "input_ids": query_tokens["input_ids"],
            "attention_mask": query_tokens["attention_mask"],
            "temperature": inference_config.get('temperature', 1.0),
        }

        with torch.no_grad():
            outputs = self.route(batch)

        predicted_llm_idx = outputs["predictions"][0].item()
        predicted_llm = self.test_dataset.router_node[predicted_llm_idx]
        routing_scores = outputs["scores"][0].cpu().tolist()

        return {
            "query": query_text,
            "predicted_llm": predicted_llm,
            "predicted_llm_idx": predicted_llm_idx,
            "routing_scores": dict(zip(self.test_dataset.router_node, routing_scores))
        }
