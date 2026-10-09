import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json

from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.utils import save_model, load_model
from .router import MLPClassifierNN


class MLPTrainer(BaseTrainer):
    """
    MLPTrainer
    ------------------
    A trainer class for MLPRouter using PyTorch.
    Supports CUDA acceleration.
    """

    def __init__(self, router, optimizer=None, device="cpu"):
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.device = device
        self.router = router

        # Training data from router
        self.query_embeddings = router.query_embedding_list  # Already a tensor
        self.label_indices = router.label_indices

        # Model configuration
        self.input_dim = router.input_dim
        self.num_classes = router.num_classes
        self.hparam = router.hparam

        # Paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.ini_model_path = os.path.join(project_root, router.cfg["model_path"]["ini_model_path"])
        self.save_model_path = os.path.join(project_root, router.cfg["model_path"]["save_model_path"])

        # Build model
        hidden_layer_sizes = self.hparam.get("hidden_layer_sizes", [128, 64])
        activation = self.hparam.get("activation", "relu")
        self.model = MLPClassifierNN(
            input_dim=self.input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            num_classes=self.num_classes,
            activation=activation
        ).to(device)

        # Training hyperparameters
        self.lr = self.hparam.get("lr", self.hparam.get("learning_rate_init", 0.001))
        self.epochs = self.hparam.get("epochs", self.hparam.get("max_iter", 100))
        self.batch_size = self.hparam.get("batch_size", 32)
        self.weight_decay = self.hparam.get("alpha", 0.0001)  # L2 regularization

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        print(f"[MLPTrainer] Initialized on device: {device}")
        print(f"[MLPTrainer] Model: {hidden_layer_sizes}, lr={self.lr}, epochs={self.epochs}, batch_size={self.batch_size}")

    def train(self):
        """
        Train the MLP model using PyTorch.
        """
        # Load initial model if exists
        if self.ini_model_path and os.path.exists(self.ini_model_path) and self.ini_model_path.endswith(".pkl"):
            try:
                state_dict = load_model(self.ini_model_path)
                if isinstance(state_dict, dict) and any(k.startswith("layers") for k in state_dict.keys()):
                    self.model.load_state_dict(state_dict)
                    print(f"[MLPTrainer] Loaded initial model from {self.ini_model_path}")
            except Exception as e:
                print(f"[MLPTrainer] Could not load initial model: {e}")
        self.loss_history = []
        # Prepare data
        X = self.query_embeddings.float().to(self.device)
        y = self.label_indices.to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_losses = []

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)

            # Print progress every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                # Calculate accuracy
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model.predict(X)
                    accuracy = (predictions == y).float().mean().item()
                self.model.train()

                print(f"[MLPTrainer] Epoch {epoch+1}/{self.epochs} - Loss={avg_loss:.6f}, Accuracy={accuracy:.4f}")

        # Save model
        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
        save_model(self.model.state_dict(), self.save_model_path)
        print(f"[MLPTrainer] Model saved to {self.save_model_path}")
        loss_path = self.save_model_path.replace('.pth', '_loss.json').replace('.pkl', '_loss.json')
        with open(loss_path, 'w') as f:
            json.dump(self.loss_history, f)
        print(f"[MLPTrainer] Loss history saved to {loss_path}")

    def evaluate(self):
        """
        Evaluate the model on training data.
        """
        self.model.eval()
        X = self.query_embeddings.float().to(self.device)
        y = self.label_indices.to(self.device)

        with torch.no_grad():
            predictions = self.model.predict(X)
            accuracy = (predictions == y).float().mean().item()

        print(f"[MLPTrainer] Training Accuracy: {accuracy:.4f}")
        return accuracy
