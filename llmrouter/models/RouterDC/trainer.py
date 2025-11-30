"""
RouterDC Trainer
----------------
Training implementation for RouterDCRouter.

This module provides the RouterDCTrainer class that handles
training of RouterDC routing strategies with dual-contrastive learning.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework while preserving all original logic.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Any, List
from tqdm import tqdm

from llmrouter.models.base_trainer import BaseTrainer
from .utils import AverageMeter


class RouterDCTrainer(BaseTrainer):
    """
    RouterDCTrainer
    ---------------
    Trainer implementation for RouterDCRouter.

    This trainer implements dual-contrastive learning with three loss components:
    1. Sample-LLM contrastive loss (main routing objective)
    2. Sample-Sample contrastive loss (task-level similarity)
    3. Cluster contrastive loss (cluster-level similarity)

    Training process:
    1. Forward pass through the router
    2. Compute combined loss with three components
    3. Backpropagate and update parameters
    4. Periodic evaluation on validation and test sets
    """

    def __init__(
        self,
        router,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cuda",
        # Training hyperparameters
        top_k: int = 3,
        last_k: int = 3,
        temperature: float = 1.0,
        sample_loss_weight: float = 0.0,
        cluster_loss_weight: float = 0.0,
        H: int = 3,
        gradient_accumulation: int = 1,
        # Evaluation settings
        eval_datasets: List[tuple] | None = None,
        eval_steps: int = 50,
        save_path: str = "./logs/routerdc/",
        **kwargs: Any,
    ):
        """
        Initialize RouterDCTrainer.

        Args:
            router:
                A RouterDCRouter instance containing the routing model.
            optimizer (torch.optim.Optimizer | None):
                Optimizer for training. If None, AdamW with lr=5e-5 is used.
            device (str):
                Device for training (default: "cuda").
            top_k (int):
                Number of top-performing LLMs to consider as positives in
                sample-LLM loss (default: 3).
            last_k (int):
                Number of low-performing LLMs to consider as negatives in
                sample-LLM loss (default: 3).
            temperature (float):
                Temperature for scaling similarity scores (default: 1.0).
            sample_loss_weight (float):
                Weight for sample-sample contrastive loss (default: 0.0).
            cluster_loss_weight (float):
                Weight for cluster contrastive loss (default: 0.0).
            H (int):
                Number of negative samples for sample-sample and cluster loss
                (default: 3).
            gradient_accumulation (int):
                Number of steps to accumulate gradients before updating
                (default: 1).
            eval_datasets (list[tuple] | None):
                List of (dataset, data_type, name) tuples for evaluation.
                Each tuple contains:
                    - dataset: RouterDataset instance
                    - data_type: "probability" or "multi_attempt"
                    - name: Name for logging
            eval_steps (int):
                Evaluate every N steps (default: 50).
            save_path (str):
                Path to save model checkpoints and logs (default: "./logs/routerdc/").
            **kwargs:
                Additional keyword arguments.
        """
        # Use AdamW optimizer with lr=5e-5 if not provided
        if optimizer is None:
            optimizer = torch.optim.AdamW(router.parameters(), lr=5e-5)

        super().__init__(router=router, optimizer=optimizer, device=device, **kwargs)

        # Training hyperparameters
        self.top_k = top_k
        self.last_k = last_k
        self.temperature = temperature
        self.sample_loss_weight = sample_loss_weight
        self.cluster_loss_weight = cluster_loss_weight
        self.H = H
        self.gradient_accumulation = gradient_accumulation

        # Evaluation settings
        self.eval_datasets = eval_datasets or []
        self.eval_steps = eval_steps
        self.save_path = save_path

        # Training state
        self.training_log = []
        self.max_average = 0.0
        self.max_training_average = 0.0

        # Create save directory
        os.makedirs(self.save_path, exist_ok=True)

    def loss_func(self, outputs, batch) -> torch.Tensor:
        """
        Compute the combined dual-contrastive loss.

        This loss consists of three components:
        1. Sample-LLM loss: Main routing objective
        2. Sample-Sample loss (optional): Task-level contrastive learning
        3. Cluster loss (optional): Cluster-level contrastive learning

        Args:
            outputs (dict):
                Dictionary containing:
                    - "scores": Similarity scores, shape (batch_size, num_llms)
                    - "hidden_state": Hidden states, shape (batch_size, hidden_dim)
            batch (dict):
                Input batch with:
                    - "true_scores": Ground truth scores, shape (batch_size, num_llms)
                    - "dataset_ids": Dataset IDs, shape (batch_size,)
                    - "cluster_ids": Cluster IDs, shape (batch_size,)

        Returns:
            torch.Tensor:
                Scalar loss tensor for backpropagation
        """
        scores = outputs["scores"]
        hidden_state = outputs["hidden_state"]
        true_scores = batch["true_scores"]

        # 1. Sample-LLM contrastive loss (main objective)
        loss = self.router.model.compute_sample_llm_loss(
            x=scores,
            index_true=true_scores,
            top_k=self.top_k,
            last_k=self.last_k
        )

        # 2. Sample-Sample contrastive loss (optional)
        if self.sample_loss_weight > 0:
            dataset_ids = batch["dataset_ids"]
            sample_sample_loss = self.router.model.compute_sample_sample_loss_with_task_tag(
                hidden_state=hidden_state,
                dataset_ids=dataset_ids,
                t=self.temperature,
                H=self.H
            )
            loss = loss + self.sample_loss_weight * sample_sample_loss

        # 3. Cluster contrastive loss (optional)
        if self.cluster_loss_weight > 0:
            cluster_ids = batch["cluster_ids"]
            cluster_loss = self.router.model.compute_cluster_loss(
                hidden_state=hidden_state,
                cluster_ids=cluster_ids,
                t=self.temperature,
                H=self.H
            )
            loss = loss + self.cluster_loss_weight * cluster_loss

        return loss

    def train(self, dataloader, training_steps: int = 1000):
        """
        Train the RouterDCRouter.

        Training loop:
        1. Iterate through training batches
        2. Forward pass through router
        3. Compute combined loss
        4. Backpropagate and update parameters
        5. Periodic evaluation and checkpointing

        Args:
            dataloader:
                Training data loader. Each batch should be a tuple:
                (inputs, scores, dataset_ids, cluster_ids)
            training_steps (int):
                Total number of training steps (default: 1000)
        """
        self.router.train()
        print(f"[RouterDCTrainer] Starting training for {training_steps} steps")

        # Setup progress bar
        pbar = tqdm(range(training_steps), desc="Training")
        step = 0
        losses = AverageMeter('Loss', ':3.2f')

        while step < training_steps:
            for batch_data in dataloader:
                # Parse batch data
                inputs, scores, dataset_ids, cluster_ids = batch_data

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = scores.to(self.device)
                dataset_ids = dataset_ids.to(self.device)
                cluster_ids = cluster_ids.to(self.device)

                # Prepare batch dict
                batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "temperature": self.temperature,
                    "true_scores": scores,
                    "dataset_ids": dataset_ids,
                    "cluster_ids": cluster_ids,
                }

                # Forward pass
                outputs = self.router(batch)

                # Compute loss
                loss = self.loss_func(outputs, batch)

                # Backward pass
                loss.backward()

                # Update parameters (with gradient accumulation)
                if step % self.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update statistics
                losses.update(loss.item(), scores.size(0))
                pbar.set_postfix({"step": f"{step}", "loss": f"{loss.item():.4f}"})
                pbar.update(1)

                step += 1

                # Evaluation
                if (step + 1) % self.eval_steps == 0:
                    self._evaluate_and_save(step)
                    self.router.train()  # Back to training mode

                # Check if training is complete
                if step >= training_steps:
                    break

        pbar.close()

        # Save final logs
        self._save_logs()

        print(f"[RouterDCTrainer] Training complete!")
        print(f"  Best test average: {self.max_average:.4f}")
        print(f"  Best train average: {self.max_training_average:.4f}")

    def _evaluate_and_save(self, step: int):
        """
        Evaluate on all evaluation datasets and save checkpoints.

        Args:
            step (int): Current training step
        """
        print(f"\n[RouterDCTrainer] Evaluation at step {step + 1}")

        self.router.eval()

        # Separate validation and test datasets
        val_datasets = [d for d in self.eval_datasets if "train" in d[2].lower()]
        test_datasets = [d for d in self.eval_datasets if "test" in d[2].lower()]

        # Evaluate on validation sets
        val_results = {}
        if val_datasets:
            print("  Validation:")
            for dataset, data_type, name in val_datasets:
                result = self._evaluate_dataset(dataset, data_type)
                val_results[name] = result
                print(f"    {name}: routing_acc={result[0]:.4f}, task_acc={result[1]:.4f}")

        # Evaluate on test sets
        test_results = {}
        if test_datasets:
            print("  Test:")
            for dataset, data_type, name in test_datasets:
                result = self._evaluate_dataset(dataset, data_type)
                test_results[name] = result
                print(f"    {name}: routing_acc={result[0]:.4f}, task_acc={result[1]:.4f}")

        # Compute averages
        if test_results:
            test_average = sum([r[1] for r in test_results.values()]) / len(test_results)
            print(f"  Average test task_acc: {test_average:.4f}")

            # Save best test model
            if test_average > self.max_average:
                self.save_checkpoint(os.path.join(self.save_path, "best_model.pth"))
                self.max_average = test_average
                print(f"  ✅ New best test model saved!")

        if val_results:
            train_average = sum([r[1] for r in val_results.values()]) / len(val_results)
            print(f"  Average train task_acc: {train_average:.4f}")

            # Save best training model
            if train_average > self.max_training_average:
                self.save_checkpoint(os.path.join(self.save_path, "best_training_model.pth"))
                self.max_training_average = train_average
                print(f"  ✅ New best training model saved!")

        # Log results
        all_results = {**val_results, **test_results}
        self.training_log.append({"step": step + 1, "results": all_results})

    def _evaluate_dataset(self, dataset, data_type: str):
        """
        Evaluate on a single dataset.

        Args:
            dataset: RouterDataset instance
            data_type (str): "probability" or "multi_attempt"

        Returns:
            tuple: (routing_accuracy, task_accuracy)
        """
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        correct_routing = 0
        correct_task = 0
        total = 0

        with torch.no_grad():
            for batch_data in dataloader:
                inputs, scores, _, _ = batch_data

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = scores.to(self.device)

                # Prepare batch
                batch = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "temperature": self.temperature,
                    "true_scores": scores,
                    "data_type": data_type,
                }

                # Forward pass
                outputs = self.router(batch)

                # Compute metrics
                predictions = outputs["predictions"]
                best_llm = torch.argmax(scores, dim=1)

                # Routing accuracy
                correct_routing += (predictions == best_llm).sum().item()

                # Task accuracy
                mask = torch.zeros_like(scores)
                mask.scatter_(1, predictions.unsqueeze(1), 1)

                if data_type == "probability":
                    binary_scores = (scores > 0).float()
                    correct_task += (binary_scores * mask).sum().item()
                else:  # multi_attempt
                    correct_task += (scores * mask).sum().item()

                total += len(predictions)

        routing_acc = correct_routing / total
        task_acc = correct_task / total

        return (routing_acc, task_acc)

    def _save_logs(self):
        """Save training logs and configuration."""
        # Save training log
        log_path = os.path.join(self.save_path, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)

        # Save configuration
        config = {
            "top_k": self.top_k,
            "last_k": self.last_k,
            "temperature": self.temperature,
            "sample_loss_weight": self.sample_loss_weight,
            "cluster_loss_weight": self.cluster_loss_weight,
            "H": self.H,
            "gradient_accumulation": self.gradient_accumulation,
            "eval_steps": self.eval_steps,
            "max_average": self.max_average,
            "max_training_average": self.max_training_average,
        }

        config_path = os.path.join(self.save_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[RouterDCTrainer] Logs saved to {self.save_path}")
