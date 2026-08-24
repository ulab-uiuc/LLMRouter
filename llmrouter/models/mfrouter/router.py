from typing import Any, Dict, List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding, call_api, generate_task_query, calculate_task_performance


class BilinearMF(nn.Module):
    """
    Bilinear Matrix Factorization model used by MFRouter.
    Implements:
        δ(M, q) = w2^T ( v_m ⊙ (W1 * v_q) )
    """

    def __init__(self, dim: int, num_models: int, text_dim: int):
        super().__init__()

        # Latent model embeddings
        self.P = nn.Embedding(num_models, dim)

        # Text projection (Longformer embedding → router latent space)
        self.text_proj = nn.Linear(text_dim, dim, bias=False)

        # Final scoring layer
        self.classifier = nn.Linear(dim, 1, bias=False)

    @property
    def device(self):
        return self.P.weight.device

    def project_text(self, q_emb: torch.Tensor) -> torch.Tensor:
        """Project raw Longformer embedding into latent routing space."""
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        proj = self.text_proj(q_emb)
        return proj.squeeze(0)

    def forward(self, model_win, model_loss, q_emb):
        """Pairwise scoring: δ(win, q) − δ(loss, q)."""
        v_win = F.normalize(self.P(model_win), p=2, dim=-1)
        v_loss = F.normalize(self.P(model_loss), p=2, dim=-1)
        h = v_win - v_loss

        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)

        interaction = h * q_emb
        logit = self.classifier(interaction).squeeze(-1)
        return logit

    def score_all(self, q_emb: torch.Tensor):
        """Return δ(M, q) for all models."""
        P_all = F.normalize(self.P.weight, p=2, dim=-1)
        interaction = P_all * q_emb
        logits = self.classifier(interaction).squeeze(-1)
        return logits


class MFRouter(MetaRouter):
    """
    MFRouter (RouteLLM-style Bilinear Matrix Factorization Router)
    Predicts best model for each query:
        best = argmax_M δ(M, q)
    """

    def __init__(self, yaml_path: str):
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Hyperparameters
        hparam = self.cfg["hparam"]
        self.dim = hparam.get("latent_dim", 128)
        self.text_dim = hparam.get("text_dim", 768)

        # Routing data
        routing = self.routing_data_train.reset_index(drop=True)

        # Build model index mappings
        models = routing["model_name"].unique().tolist()
        self.model_to_idx = {m: i for i, m in enumerate(models)}
        self.idx_to_model = {i: m for m, i in self.model_to_idx.items()}

        # Construct pairwise samples with embedding_id for fast lookup
        self.pairs = []
        grouped = routing.groupby("query")
        for q, df in grouped:
            best_row = df.loc[df["performance"].idxmax()]
            winner_id = self.model_to_idx[best_row["model_name"]]
            embedding_id = int(best_row["embedding_id"])

            for _, row in df.iterrows():
                loser_id = self.model_to_idx[row["model_name"]]
                if loser_id != winner_id:
                    self.pairs.append({
                        "query": q,
                        "embedding_id": embedding_id,
                        "winner": winner_id,
                        "loser": loser_id
                    })
    # ---------------------------------------------------------
    # Query embedding using Longformer
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        emb = get_longformer_embedding(text).numpy()
        return torch.tensor(emb, dtype=torch.float32)

    # ---------------------------------------------------------
    # Load MF model from file (MATCH MLPRouter)
    # ---------------------------------------------------------
    def _load_mf_model(self):
        state_dict = load_model(self.load_model_path)
        model = BilinearMF(self.dim, len(self.model_to_idx), self.text_dim)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # ---------------------------------------------------------
    # Route a single query
    # ---------------------------------------------------------
    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.load_model_path = os.path.join(project_root, self.cfg["model_path"]["save_model_path"])
        model = self._load_mf_model()

        q_emb = self.embed_query(query["query"]).to(model.device)
        q_emb_proj = model.project_text(q_emb)

        scores = model.score_all(q_emb_proj)
        best_id = torch.argmax(scores).item()

        out = copy.copy(query)
        out["model_name"] = self.idx_to_model[best_id]
        return out

    # ---------------------------------------------------------
    # Route a batch of queries and execute them
    # ---------------------------------------------------------
    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using the trained MF model and execute them.

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
        self.load_model_path = os.path.join(project_root, self.cfg["model_path"]["save_model_path"])
        model = self._load_mf_model()

        # Determine which data to use
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
        else:
            if hasattr(self, "query_data_test") and self.query_data_test is not None:
                query_data = copy.copy(self.query_data_test)
            else:
                print("Warning: No batch provided and no test data available for batch routing.")
                return []

        query_data_output = []
        for row in query_data:
            # Handle both dict and non-dict inputs
            if isinstance(row, dict):
                row_copy = copy.copy(row)
                original_query = row_copy.get("query", "")
                row_task_name = row_copy.get("task_name", task_name)
            else:
                row_copy = {"query": str(row)}
                original_query = str(row)
                row_task_name = task_name

            # Step 1: Route the query
            q_emb = self.embed_query(original_query).to(model.device)
            q_emb_proj = model.project_text(q_emb)
            scores = model.score_all(q_emb_proj)
            best_id = torch.argmax(scores).item()
            model_name = self.idx_to_model[best_id]
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

