from typing import Any, Dict, List, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding, call_api, generate_task_query, calculate_task_performance


class MLPClassifierNN(nn.Module):
    """
    PyTorch MLP Classifier for routing.
    Replaces sklearn MLPClassifier to enable CUDA support.
    """

    def __init__(self, input_dim: int, hidden_layer_sizes: List[int], num_classes: int, activation: str = "relu"):
        super().__init__()

        self.activation_name = activation
        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.layers = nn.ModuleList(layers)

    def _get_activation(self):
        """Get activation function based on name."""
        if self.activation_name == "relu":
            return F.relu
        elif self.activation_name == "tanh":
            return torch.tanh
        elif self.activation_name == "logistic":
            return torch.sigmoid
        elif self.activation_name == "identity":
            return lambda x: x
        else:
            return F.relu

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        activation = self._get_activation()

        # Hidden layers with activation
        for layer in self.layers[:-1]:
            x = activation(layer(x))

        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.layers[-1](x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class indices."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)


class MLPRouter(MetaRouter):
    """
    MLPRouter
    ----------
    A routing module that leverages a Multi-Layer Perceptron (MLP)
    classifier to select the most suitable language model based on
    query embeddings.

    Now uses PyTorch nn.Module for CUDA support.

    YAML Configuration Example:
    ---------------------------
    hparam:
      hidden_layer_sizes: [128, 64]
      activation: "relu"
      lr: 0.001
      epochs: 100
      batch_size: 32
      alpha: 0.0001  # L2 regularization (weight decay)
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the MLPRouter and load configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Steps:
            1. Loads configuration and metadata from MetaRouter.
            2. Prepares the training embeddings and corresponding model labels.
            3. Builds class mappings for model names.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Get hyperparameters
        self.hparam = self.cfg["hparam"]

        # Extract best model for each query
        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        # Build class mappings
        model_names = routing_best["model_name"].unique().tolist()
        self.model_to_idx = {m: i for i, m in enumerate(model_names)}
        self.idx_to_model = {i: m for m, i in self.model_to_idx.items()}
        self.num_classes = len(model_names)

        # Prepare training data
        query_embedding_id = routing_best["embedding_id"].tolist()
        # Use vectorized stacking for faster conversion
        embeddings_tensor = torch.stack([self.query_embedding_data[i] for i in query_embedding_id])
        self.query_embedding_list = embeddings_tensor  # Keep as tensor
        self.model_name_list = routing_best["model_name"].tolist()
        # Convert labels to indices
        self.label_indices = torch.tensor([self.model_to_idx[m] for m in self.model_name_list], dtype=torch.long)

        # Get input dimension from embeddings
        self.input_dim = self.query_embedding_list.shape[1]

        # Build MLP model
        hidden_layer_sizes = self.hparam.get("hidden_layer_sizes", [128, 64])
        activation = self.hparam.get("activation", "relu")
        self.mlp_model = MLPClassifierNN(
            input_dim=self.input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            num_classes=self.num_classes,
            activation=activation
        )

    def _load_mlp_model(self, device: str = "cpu"):
        """Load trained MLP model from file."""
        state_dict = load_model(self.load_model_path)

        # Handle both old sklearn format and new PyTorch format
        if isinstance(state_dict, dict) and any(k.startswith("layers") for k in state_dict.keys()):
            # PyTorch state dict
            hidden_layer_sizes = self.hparam.get("hidden_layer_sizes", [128, 64])
            activation = self.hparam.get("activation", "relu")
            model = MLPClassifierNN(
                input_dim=self.input_dim,
                hidden_layer_sizes=hidden_layer_sizes,
                num_classes=self.num_classes,
                activation=activation
            )
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model, "pytorch"
        else:
            # Old sklearn model (backward compatibility)
            return state_dict, "sklearn"

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the trained MLP model.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        model, model_type = self._load_mlp_model()

        query_embedding = get_longformer_embedding(query["query"])

        if model_type == "pytorch":
            with torch.no_grad():
                emb_tensor = query_embedding.unsqueeze(0).to(model.device)
                pred_idx = model.predict(emb_tensor).item()
                model_name = self.idx_to_model[pred_idx]
        else:
            # sklearn fallback
            model_name = model.predict([query_embedding.numpy()])[0]

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using the trained MLP model and execute them.

        This method performs end-to-end processing for each query:
        1. Routes the query to get the best model
        2. Applies task-specific prompt formatting if task_name is provided
        3. Calls the routed model via API to get response
        4. Calculates performance metrics if ground truth is available

        Args:
            batch (Any, optional):
                If provided, routes the provided batch. If None, uses self.query_data_test from loaded data.
            task_name (str, optional):
                Task name for prompt formatting (e.g., "mmlu", "gsm8k", "commonsense_qa").

        Returns:
            list of dict:
                A list of query dictionaries with response, tokens, and performance metrics.
        """
        # Load model once
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.load_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        model, model_type = self._load_mlp_model()

        query_data = self._resolve_query_data(batch)
        if query_data is None:
            return []

        query_data_output = []
        for row in query_data:
            # Handle both dict and non-dict inputs
            row_copy, original_query, row_task_name = self._normalize_row(row, task_name)

            # Step 1: Route the query
            query_embedding = get_longformer_embedding(original_query)

            if model_type == "pytorch":
                with torch.no_grad():
                    emb_tensor = query_embedding.unsqueeze(0).to(model.device)
                    pred_idx = model.predict(emb_tensor).item()
                    model_name = self.idx_to_model[pred_idx]
            else:
                # sklearn fallback
                model_name = model.predict([query_embedding.numpy()])[0]

            row_copy["model_name"] = model_name

            # Step 2: Format query if task_name is provided
            if row_task_name:
                try:
                    sample_data = {
                        "query": original_query,
                        "choices": row_copy.get("choices", None) if isinstance(row_copy, dict) else None
                    }
                    formatted_query = generate_task_query(row_task_name, sample_data)
                    row_copy["formatted_query"] = formatted_query
                    query_text_for_execution = formatted_query
                except (ValueError, KeyError) as e:
                    print(f"Warning: Failed to format query with task '{row_task_name}': {e}. Using original query.")
                    query_text_for_execution = original_query
            else:
                query_text_for_execution = original_query

            # Step 3: Call API to get response
            # Get API endpoint and model name from llm_data if available
            api_model_name = model_name
            api_endpoint = None
            service = None
            if hasattr(self, 'llm_data') and self.llm_data and model_name in self.llm_data:
                api_model_name = self.llm_data[model_name].get("model", model_name)
                # Get API endpoint from llm_data, fallback to router config
                api_endpoint = self.llm_data[model_name].get(
                    "api_endpoint",
                    self.cfg.get("api_endpoint")
                )
                # Get service field for service-specific API key selection
                service = self.llm_data[model_name].get("service")

            # If still no endpoint found, try router config
            if api_endpoint is None:
                api_endpoint = self.cfg.get("api_endpoint")

            # Validate that we have an endpoint
            if not api_endpoint:
                raise ValueError(
                    f"API endpoint not found for model '{model_name}'. "
                    f"Please specify 'api_endpoint' in llm_data JSON for this model or in router YAML config."
                )

            request = {
                "api_endpoint": api_endpoint,
                "query": query_text_for_execution,
                "model_name": model_name,
                "api_name": api_model_name
            }
            # Add service field if available (for service-specific API key selection)
            if service:
                request["service"] = service

            try:
                result = call_api(request, max_tokens=1024, temperature=0.7)
                response = result.get("response", "")
                prompt_tokens = result.get("prompt_tokens", 0)
                completion_tokens = result.get("completion_tokens", 0)
                success = "error" not in result
            except Exception as e:
                print(f"Error calling API for query: {e}")
                response = ""
                prompt_tokens = 0
                completion_tokens = 0
                success = False

            row_copy["response"] = response
            row_copy["prompt_tokens"] = prompt_tokens
            row_copy["completion_tokens"] = completion_tokens
            row_copy["input_token"] = prompt_tokens
            row_copy["output_token"] = completion_tokens
            row_copy["success"] = success

            # Step 4: Calculate task performance if ground truth is available
            ground_truth = row_copy.get("ground_truth") or row_copy.get("gt") or row_copy.get("answer")
            metric = row_copy.get("metric")
            if ground_truth:
                task_performance = calculate_task_performance(
                    prediction=response,
                    ground_truth=ground_truth,
                    task_name=row_task_name,
                    metric=metric
                )
                if task_performance is not None:
                    row_copy["task_performance"] = task_performance

            query_data_output.append(row_copy)

        return query_data_output
