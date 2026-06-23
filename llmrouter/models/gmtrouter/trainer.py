"""
GMTRouter trainer for heterogeneous graph-based personalized routing.

Trains:
- HeteroGNN: Graph neural network for node embeddings
- PreferencePredictor: Preference scoring model
"""

import torch
import torch.nn.functional as F
import os
from typing import Dict, Tuple
import numpy as np

from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.models.gmtrouter.data_loader import GMTRouterDataLoader
from llmrouter.models.gmtrouter.models import GMTRouterModel


class GMTRouterTrainer(BaseTrainer):
    """
    Trainer for GMTRouter using heterogeneous graph neural networks.

    Training workflow:
    1. Load GMTRouter JSONL data with special format detection
    2. Build heterogeneous graph with 5 node types and 21 edge types
    3. Train HeteroGNN + PreferencePredictor with pairwise preference learning
    4. Save model checkpoint and user embeddings
    """

    def __init__(self, router, optimizer=None, device=None):
        """
        Initialize GMTRouterTrainer.

        Args:
            router: GMTRouter instance
            optimizer: Optional optimizer (if None, use Adam)
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.router = router
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Get configuration
        self.gmt_config = router.cfg.get("gmt_config", {})
        training_config = router.cfg.get("train", {})

        # Model hyperparameters
        self.hidden_dim = self.gmt_config.get("hidden_dim", 128)
        self.num_gnn_layers = self.gmt_config.get("num_gnn_layers", 2)
        self.dropout = self.gmt_config.get("dropout", 0.1)
        self.num_heads = 4

        # Training hyperparameters
        self.epochs = training_config.get("epochs", 350)
        self.learning_rate = training_config.get("lr", 5e-4)
        self.prediction_count = training_config.get("prediction_count", 256)
        self.objective = training_config.get("objective", "auc")
        self.binary = training_config.get("binary", True)
        self.eval_every = training_config.get("eval_every", 5)
        self.seed = training_config.get("seed", 136)

        # Checkpoint configuration
        checkpoint_config = router.cfg.get("checkpoint", {})
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.checkpoint_root = os.path.join(project_root, checkpoint_config.get("root", "models"))
        self.save_every = checkpoint_config.get("save_every", 25)

        # Model paths
        model_path_config = router.cfg.get("model_path", {})
        self.save_model_path = os.path.join(
            project_root,
            model_path_config.get("save_model_path", "saved_models/gmtrouter/gmtrouter.pt")
        )

        # Data loader
        self.data_loader = GMTRouterDataLoader(router.cfg)

        # Model (will be initialized after loading data)
        self.model = None

        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def train(self):
        """
        Train the GMTRouter model.

        Steps:
        1. Load and validate GMTRouter data
        2. Build heterogeneous graph
        3. Initialize model with graph metadata
        4. Train with pairwise preference learning
        5. Save best model and user embeddings

        Returns:
            dict: Training results
        """
        print("=" * 70)
        print("GMTRouter Training")
        print("=" * 70)

        # Load data
        train_data, val_data = self.router.get_training_data()

        if train_data is None:
            print("Error: No training data loaded. Check data paths in config.")
            return {"status": "failed", "reason": "no_data"}

        # Initialize model with graph metadata
        self._initialize_model(train_data)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )

        # Training loop
        best_metric = 0.0
        best_epoch = 0

        print(f"\nTraining Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Hidden Dim: {self.hidden_dim}")
        print(f"  GNN Layers: {self.num_gnn_layers}")
        print(f"  Objective: {self.objective}")
        print(f"  Binary Classification: {self.binary}")
        print()

        for epoch in range(1, self.epochs + 1):
            # Train for one epoch
            train_loss, train_metric = self._train_epoch(train_data, epoch)

            # Evaluate
            if epoch % self.eval_every == 0 or epoch == self.epochs:
                val_loss, val_metric = self._validate(val_data)

                print(f"Epoch {epoch}/{self.epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train {self.objective.upper()}: {train_metric:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val {self.objective.upper()}: {val_metric:.4f}")

                # Save best model
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_epoch = epoch
                    self._save_checkpoint(epoch, val_metric, best=True)
            else:
                print(f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}, Train {self.objective.upper()}: {train_metric:.4f}")

            # Regular checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, train_metric, best=False)

        print(f"\nTraining completed!")
        print(f"Best {self.objective.upper()}: {best_metric:.4f} at epoch {best_epoch}")

        return {
            "status": "completed",
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "final_epoch": epoch,
            "model_path": self.save_model_path
        }

    def _initialize_model(self, train_data: Dict):
        """
        Initialize GMTRouter model with graph metadata.

        Args:
            train_data: Training data dictionary
        """
        # Extract metadata
        if 'metadata' in train_data:
            # PyTorch Geometric HeteroData
            metadata = train_data['metadata']
        else:
            # Simplified data structure
            # Create metadata from node types
            node_types = ['user', 'session', 'query', 'llm', 'response']
            edge_types = list(train_data.get('edges', {}).keys())

            # Convert edge type strings to tuples
            edge_type_tuples = []
            for edge_type in edge_types:
                parts = edge_type.split('_')
                if len(parts) >= 3:
                    src_type = parts[0]
                    dst_type = parts[-1]
                    rel_type = '_'.join(parts[1:-1])
                    edge_type_tuples.append((src_type, rel_type, dst_type))

            metadata = (node_types, edge_type_tuples)

        # Initialize model
        self.model = GMTRouterModel(
            metadata=metadata,
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

        print(f"Model initialized with metadata: {len(metadata[0])} node types, {len(metadata[1])} edge types")

    def _train_epoch(self, train_data: Dict, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_data: Training data
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, metric)
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Get pairwise comparisons
        metadata = train_data.get('train_metadata', {})
        pairs = metadata.get('pairs', [])

        if len(pairs) == 0:
            print("Warning: No pairwise comparisons found in training data")
            return 0.0, 0.0

        # Sample pairs for this epoch
        num_samples = min(self.prediction_count, len(pairs))
        sampled_pairs = np.random.choice(len(pairs), size=num_samples, replace=False)

        # Get graph data
        if 'x_dict' in train_data:
            x_dict = {k: v.to(self.device) for k, v in train_data['x_dict'].items()}
            edge_index_dict = {k: v.to(self.device) for k, v in train_data['edge_index_dict'].items()}
        else:
            # Simplified data
            x_dict = {
                'user': train_data['user_embeddings'].to(self.device),
                'session': train_data['session_embeddings'].to(self.device),
                'query': train_data['query_embeddings'].to(self.device),
                'llm': train_data['llm_embeddings'].to(self.device),
            }
            edge_index_dict = {}  # Simplified mode doesn't use edge indices

        # Train on sampled pairs
        for pair_idx in sampled_pairs:
            pair = pairs[pair_idx]

            winner = pair['winner']
            loser = pair['loser']

            # Get node IDs
            user_ids = torch.tensor([winner['user_id'], loser['user_id']], dtype=torch.long, device=self.device)
            query_ids = torch.tensor([winner['query_id'], loser['query_id']], dtype=torch.long, device=self.device)
            llm_ids = torch.tensor([winner['llm_id'], loser['llm_id']], dtype=torch.long, device=self.device)

            # Forward pass
            scores = self.model(x_dict, edge_index_dict, user_ids, query_ids, llm_ids)

            # Binary classification loss
            # Label: 1 if first is better (winner), 0 if second is better
            label = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device).unsqueeze(1)

            loss = F.binary_cross_entropy_with_logits(scores, label)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Track predictions for metric calculation
            preds = torch.sigmoid(scores).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

        avg_loss = total_loss / len(sampled_pairs)

        # Calculate metric (AUC or accuracy)
        metric = self._calculate_metric(np.array(all_preds), np.array(all_labels))

        return avg_loss, metric

    def _validate(self, val_data: Dict) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_data: Validation data

        Returns:
            Tuple of (average_loss, metric)
        """
        self.model.eval()

        if val_data is None:
            return 0.0, 0.0

        total_loss = 0.0
        all_preds = []
        all_labels = []

        metadata = val_data.get('val_metadata', {})
        pairs = metadata.get('pairs', [])

        if len(pairs) == 0:
            return 0.0, 0.0

        # Get graph data
        if 'x_dict' in val_data:
            x_dict = {k: v.to(self.device) for k, v in val_data['x_dict'].items()}
            edge_index_dict = {k: v.to(self.device) for k, v in val_data['edge_index_dict'].items()}
        else:
            x_dict = {
                'user': val_data['user_embeddings'].to(self.device),
                'session': val_data['session_embeddings'].to(self.device),
                'query': val_data['query_embeddings'].to(self.device),
                'llm': val_data['llm_embeddings'].to(self.device),
            }
            edge_index_dict = {}

        with torch.no_grad():
            for pair in pairs:
                winner = pair['winner']
                loser = pair['loser']

                user_ids = torch.tensor([winner['user_id'], loser['user_id']], dtype=torch.long, device=self.device)
                query_ids = torch.tensor([winner['query_id'], loser['query_id']], dtype=torch.long, device=self.device)
                llm_ids = torch.tensor([winner['llm_id'], loser['llm_id']], dtype=torch.long, device=self.device)

                scores = self.model(x_dict, edge_index_dict, user_ids, query_ids, llm_ids)

                label = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device).unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(scores, label)

                total_loss += loss.item()

                preds = torch.sigmoid(scores).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label.cpu().numpy())

        avg_loss = total_loss / len(pairs)
        metric = self._calculate_metric(np.array(all_preds), np.array(all_labels))

        return avg_loss, metric

    def _calculate_metric(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate evaluation metric (AUC or accuracy).

        Args:
            preds: Predicted probabilities
            labels: True labels

        Returns:
            Metric value
        """
        if self.objective == "auc":
            try:
                from sklearn.metrics import roc_auc_score
                # Flatten arrays
                preds_flat = preds.flatten()
                labels_flat = labels.flatten()
                return roc_auc_score(labels_flat, preds_flat)
            except Exception as e:
                print(f"Warning: Could not calculate AUC: {e}")
                return 0.0
        else:
            # Accuracy
            preds_binary = (preds > 0.5).astype(int)
            accuracy = (preds_binary == labels).mean()
            return accuracy

    def _save_checkpoint(self, epoch: int, metric: float, best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            metric: Current metric value
            best: Whether this is the best checkpoint
        """
        # Ensure save directory exists
        save_dir = os.path.dirname(self.save_model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
            'config': self.gmt_config
        }

        if best:
            # Save as best model
            torch.save(checkpoint, self.save_model_path)
            print(f"  → Saved best model to {self.save_model_path}")
        else:
            # Save with epoch number
            checkpoint_path = self.save_model_path.replace('.pt', f'_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"  → Saved checkpoint to {checkpoint_path}")
