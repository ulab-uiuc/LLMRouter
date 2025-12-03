import os
import torch
from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.utils import save_model, load_model


class HybridLLMTrainer(BaseTrainer):
    """
    Trainer for HybridLLMRouter supporting three modes:
        deterministic / probabilistic / transformed
    """

    def __init__(self, router, optimizer=None, device="cpu"):
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.query_embedding_list = router.query_embedding_list
        self.router_label_list = router.router_label_list

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.ini_model_path = os.path.join(project_root, router.cfg["model_path"]["ini_model_path"])
        self.save_model_path = os.path.join(project_root, router.cfg["model_path"]["save_model_path"])

        self.model = router.mlp_model
        print(f"[HybridLLMTrainer] Initialized. Mode={router.router_mode}")

    def train(self):
        if os.path.exists(self.ini_model_path) and self.ini_model_path.endswith(".pkl"):
            print(f"[HybridLLMTrainer] Loading init model: {self.ini_model_path}")
            self.model = load_model(self.ini_model_path)

        X = self.query_embedding_list
        y = self.router_label_list

        print(f"[HybridLLMTrainer] Training on {len(X)} samples... mode={self.router.router_mode}")
        self.model.fit(X, y)

        print(f"[HybridLLMTrainer] Saving final model: {self.save_model_path}")
        save_model(self.model, self.save_model_path)





