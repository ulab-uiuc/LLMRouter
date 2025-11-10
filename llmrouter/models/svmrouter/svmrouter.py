from typing import Any, Dict, List, Optional
import os
import pickle
import numpy as np
import torch.nn as nn
import copy
from sklearn.svm import SVC
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding


class SVMRouter(MetaRouter):
    """
    SVMRouter
    ----------
    A routing module that leverages a Support Vector Machine (SVM)
    classifier to select the most suitable language model based on
    query embeddings.

    The router inherits from MetaRouter for consistent interface design.
    If no trained SVM model is found at the specified path, it can fall back
    to random selection.

    YAML Configuration Example:
    ---------------------------
    llm_data:
      GPT4:
        size: "175B"
        embedding: [0.12, 0.33, 0.78, 0.44]
      Claude3:
        size: "52B"
        embedding: [0.10, 0.25, 0.70, 0.50]
    optional:
      svm_model_path: "configs/svm_model.pkl"
      kernel: "rbf"
      C: 1.0
      gamma: "scale"
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the SVMRouter and load configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Steps:
            1. Loads configuration and metadata from MetaRouter.
            2. Builds an SVM classifier with the specified hyperparameters.
            3. Prepares the training embeddings and corresponding model labels.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Initialize SVM classifier
        svm_params = self.cfg["hparam"]
        self.svm_model = SVC(**svm_params)

        # Select best-performing model for each query
        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        # Prepare training data
        query_embedding_id = routing_best["embedding_id"].tolist()
        self.query_embedding_list = [self.query_embedding_data[i].numpy() for i in query_embedding_id]
        self.model_name_list = routing_best["model_name"].tolist()

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the trained SVM model.

        Args:
            query (dict):
                Must contain key "query" with textual input.

        Returns:
            dict: Updated query dictionary containing:
                - "model_name": predicted model name.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.svm_model = load_model(load_model_path)

        # Compute embedding and predict
        query_embedding = [get_longformer_embedding(query["query"]).numpy()]
        model_name = self.svm_model.predict(query_embedding)[0]

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using the trained SVM model.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.svm_model = load_model(load_model_path)

        query_data_output = copy.copy(self.query_data_test)
        for row in query_data_output:
            query_embedding = [get_longformer_embedding(row["query"]).numpy()]
            model_name = self.svm_model.predict(query_embedding)[0]
            row["model_name"] = model_name

        return query_data_output

