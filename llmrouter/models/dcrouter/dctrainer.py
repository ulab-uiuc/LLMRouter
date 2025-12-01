"""
DCRouter Trainer
----------------
Training implementation for DCRouter.

This module provides the DCTrainer class that handles
training of DCRouter routing strategies with dual-contrastive learning.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework while preserving all original logic.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from llmrouter.models.base_trainer import BaseTrainer
from .dcutils import AverageMeter


class DCTrainer(BaseTrainer):
    """
    DCTrainer
    ---------
    Trainer implementation for DCRouter.

    This trainer implements dual-contrastive learning with three loss components:
    1. Sample-LLM contrastive loss (main routing objective)
    2. Sample-Sample contrastive loss (task-level similarity)
    3. Cluster contrastive loss (cluster-level similarity)
    """

    def __init__(self, router, device=None):
        """
        Initialize DCTrainer.

        Args:
            router: A DCRouter instance containing the routing model
            device (str, optional): Device for training (default: from config)
        """
        # Get config from router
        self.cfg = router.cfg
        training_config = self.cfg['training']

        # Set device
        if device is None:
            device = training_config.get('device', 'cpu')

        # Create optimizer
        optimizer = torch.optim.AdamW(router.parameters(), lr=training_config['learning_rate'])

        super().__init__(router=router, optimizer=optimizer, device=device)

        # Training hyperparameters
        self.top_k = training_config['top_k']
        self.last_k = training_config['last_k']
        self.temperature = training_config['temperature']
        self.sample_loss_weight = training_config['sample_loss_weight']
        self.cluster_loss_weight = training_config['cluster_loss_weight']
        self.H = training_config['H']
        self.gradient_accumulation = training_config['gradient_accumulation']
        self.batch_size = training_config['batch_size']
        self.training_steps = training_config['training_steps']

        # Evaluation settings
        eval_config = self.cfg['evaluation']
        self.eval_steps = eval_config['eval_steps']

        # Save paths
        save_model_path = self.cfg['model_path']['save_model_path']
        self.save_dir = os.path.join(router.project_root, os.path.dirname(save_model_path))
        self.final_model_path = os.path.join(router.project_root, save_model_path)
        os.makedirs(self.save_dir, exist_ok=True)

        # Datasets
        self.train_dataset = router.train_dataset
        self.test_dataset = router.test_dataset

        # Prepare evaluation datasets
        self.eval_datasets = [
            (self.test_dataset, "probability", "test_data"),
            (self.train_dataset, "probability", "train_data")
        ]

        # Training state
        self.training_log = []
        self.max_average = 0.0
        self.max_training_average = 0.0

        print("[DCTrainer] Initialized successfully!")
        print(f"  Top-k: {self.top_k}")
        print(f"  Last-k: {self.last_k}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Sample loss weight: {self.sample_loss_weight}")
        print(f"  Cluster loss weight: {self.cluster_loss_weight}")
        print(f"  Save directory: {self.save_dir}")

    def loss_func(self, outputs, batch) -> torch.Tensor:
        """
        Compute the combined dual-contrastive loss.

        Args:
            outputs (dict): Dictionary containing scores and hidden_state
            batch (dict): Input batch with true_scores, dataset_ids, cluster_ids

        Returns:
            torch.Tensor: Scalar loss tensor for backpropagation
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

    def train(self):
        """
        Train the DCRouter.

        This method contains the complete training loop including:
        - Data loading
        - Forward/backward passes
        - Periodic evaluation
        - Model checkpointing
        """
        self.router.train()
        print(f"\n[DCTrainer] Starting training for {self.training_steps} steps")
        print("=" * 70)

        # Create dataloader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Setup progress bar
        pbar = tqdm(range(self.training_steps), desc="Training")
        step = 0
        losses = AverageMeter('Loss', ':3.2f')

        while step < self.training_steps:
            for batch_data in train_dataloader:
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
                if step >= self.training_steps:
                    break

        pbar.close()

        # Save final model
        torch.save(self.router.model.state_dict(), self.final_model_path)
        print(f"\n[DCTrainer] Final model saved to: {self.final_model_path}")

        # Save final logs
        self._save_logs()

        print("\n" + "=" * 70)
        print("[DCTrainer] Training complete!")
        print(f"  Best test average: {self.max_average:.4f}")
        print(f"  Best train average: {self.max_training_average:.4f}")
        print("=" * 70)

    def _evaluate_and_save(self, step: int):
        """
        Evaluate on all evaluation datasets and save checkpoints.

        Args:
            step (int): Current training step
        """
        print(f"\n[DCTrainer] Evaluation at step {step + 1}")

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
                checkpoint_path = os.path.join(self.save_dir, "best_model.pth")
                torch.save(self.router.model.state_dict(), checkpoint_path)
                self.max_average = test_average
                print(f"  New best test model saved!")

        if val_results:
            train_average = sum([r[1] for r in val_results.values()]) / len(val_results)
            print(f"  Average train task_acc: {train_average:.4f}")

            # Save best training model
            if train_average > self.max_training_average:
                checkpoint_path = os.path.join(self.save_dir, "best_training_model.pth")
                torch.save(self.router.model.state_dict(), checkpoint_path)
                self.max_training_average = train_average
                print(f"  New best training model saved!")

        # Log results
        all_results = {**val_results, **test_results}
        self.training_log.append({"step": step + 1, "results": all_results})

    def _evaluate_dataset(self, dataset, data_type: str):
        """
        Evaluate on a single dataset.

        Args:
            dataset: DCDataset instance
            data_type (str): "probability" or "multi_attempt"

        Returns:
            tuple: (routing_accuracy, task_accuracy)
        """
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
        log_path = os.path.join(self.save_dir, "training_log.json")
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

        config_path = os.path.join(self.save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[DCTrainer] Logs saved to {self.save_dir}")
