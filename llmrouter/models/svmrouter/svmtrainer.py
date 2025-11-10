import torch
from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.utils import save_model, load_model
import os


class SVMRouterTrainer(BaseTrainer):
    """
    SVMRouterTrainer
    ------------------
    A simple trainer class for SVMRouter.

    This version initializes paths and model references from router configuration.
    """
    def __init__(self, router, optimizer=None, device="cpu"):
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.query_embedding_list = router.query_embedding_list
        self.model_name_list = router.model_name_list

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.ini_model_path = os.path.join(project_root, router.cfg["model_path"]["ini_model_path"])
        self.save_model_path = os.path.join(project_root, router.cfg["model_path"]["save_model_path"])

        self.model = router.svm_model
        print("[SVMRouterTrainer] Initialized with router.")

    def train(self):
        """
        Train the SVM model using the provided embeddings and labels.
        """
        if os.path.exists(self.ini_model_path) and self.ini_model_path.endswith(".pkl"):
            self.model = load_model(self.ini_model_path)

        self.model.fit(self.query_embedding_list, self.model_name_list)
        save_model(self.model, self.save_model_path)




