from llmrouter.models.base_trainer import BaseTrainer
from llmrouter.utils import save_model, load_model
import os


class KNNMultiRoundRouterTrainer(BaseTrainer):
    """
    KNNMultiRoundRouterTrainer
    --------------------------
    Trainer for KNNMultiRoundRouter.

    This trainer fits a KNN classifier on query embeddings and their corresponding
    best-performing model labels. The training data comes from historical routing
    decisions where the best model for each query is known.
    """

    def __init__(self, router, optimizer=None, device="cpu"):
        """
        Initialize the trainer with a router instance.

        Args:
            router (KNNMultiRoundRouter): The router instance to train.
            optimizer: Not used for KNN (sklearn-based), kept for interface compatibility.
            device: Not used for KNN (CPU-based), kept for interface compatibility.
        """
        super().__init__(router=router, optimizer=optimizer, device=device)

        self.query_embedding_list = router.query_embedding_list
        self.model_name_list = router.model_name_list

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # Get paths from config
        ini_model_path = router.cfg["model_path"].get("ini_model_path", None)
        if ini_model_path:
            self.ini_model_path = os.path.join(project_root, ini_model_path)
        else:
            self.ini_model_path = None

        self.save_model_path = os.path.join(project_root, router.cfg["model_path"]["save_model_path"])

        self.model = router.knn_model

        print("[KNNMultiRoundRouterTrainer] Initialized with router.")

    def train(self):
        """
        Train the KNN classifier on query embeddings and model labels.

        The training process:
        1. Optionally loads a pre-existing model from ini_model_path if available
        2. Fits the KNN classifier on query embeddings and best model labels
        3. Saves the trained model to save_model_path
        """
        # Load initial model if path exists and is a pickle file
        if self.ini_model_path and os.path.exists(self.ini_model_path) and self.ini_model_path.endswith(".pkl"):
            print(f"Loading initial model from: {self.ini_model_path}")
            self.model = load_model(self.ini_model_path)

        # Fit the KNN model on embeddings and labels
        print(f"Training KNN model on {len(self.query_embedding_list)} examples...")
        self.model.fit(self.query_embedding_list, self.model_name_list)
        print(f"KNN model training completed!")

        # Save the trained model
        print(f"Saving trained model to: {self.save_model_path}")
        save_model(self.model, self.save_model_path)
        print(f"Model saved successfully!")

