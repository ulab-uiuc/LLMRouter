from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import copy
from sklearn.neural_network import MLPClassifier
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding


class MLPRouter(MetaRouter):
    """
    MLPRouter
    ----------
    A routing module that leverages a Multi-Layer Perceptron (MLP)
    classifier to select the most suitable language model based on
    query embeddings.

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
      mlp_model_path: "configs/mlp_model.pkl"
      hidden_layer_sizes: [64, 32]
      activation: "relu"
      solver: "adam"
      max_iter: 300
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the MLPRouter and load configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Steps:
            1. Loads configuration and metadata from MetaRouter.
            2. Builds an MLP classifier with the specified hyperparameters.
            3. Prepares the training embeddings and corresponding model labels.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        mlp_params = self.cfg["hparam"]
        self.mlp_model = MLPClassifier(**mlp_params)

        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        query_embedding_id = routing_best["embedding_id"].tolist()
        self.query_embedding_list = [self.query_embedding_data[i].numpy() for i in query_embedding_id]
        self.model_name_list = routing_best["model_name"].tolist()

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the trained MLP model.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_model_path)

        query_embedding = [get_longformer_embedding(query["query"]).numpy()]
        model_name = self.mlp_model.predict(query_embedding)[0]

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using the trained MLP model.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        self.mlp_model = load_model(load_model_path)

        query_data_output = copy.copy(self.query_data_test)
        for row in query_data_output:
            query_embedding = [get_longformer_embedding(row["query"]).numpy()]
            model_name = self.mlp_model.predict(query_embedding)[0]
            row["model_name"] = model_name

        return query_data_output

