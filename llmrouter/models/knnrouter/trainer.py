from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.utils import save_model, load_model
import os

class KNNRouterTrainer(BaseTrainer):
    """
    KNNRouterTrainer
    ------------------
    A simple example trainer for KNNRouter.

    This version defines its own __init__() method without requiring any external parameters.
    It sets default attributes internally and can be extended for customized logic.
    """
    def __init__(self, router, optimizer=None, device="cpu"):
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.query_embedding_list = router.query_embedding_list
        self.model_name_list = router.model_name_list

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        self.ini_model_path = os.path.join(project_root, router.cfg["model_path"]["ini_model_path"])
        self.save_model_path = os.path.join(project_root, router.cfg["model_path"]["save_model_path"])

        self.model = router.knn_model

        print("[KNNRouterTrainer] Initialized with router.")

    def train(self):
        """
        Example placeholder training function for KNNRouterTrainer.
        """
        if os.path.exists(self.ini_model_path) and self.ini_model_path.endswith(".pkl"):
            self.model = load_model(self.ini_model_path)

        self.model.fit(self.query_embedding_list, self.model_name_list)
        save_model(self.model, self.save_model_path)




