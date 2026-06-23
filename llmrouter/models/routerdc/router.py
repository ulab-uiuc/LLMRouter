"""
DCRouter Router
---------------
Router implementation for the DCRouter routing strategy.

This module provides the DCRouter class that integrates with the
LLMRouter framework.

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework.
"""

import os
import yaml
import copy
import torch
from typing import Any, Dict, List, Optional
from transformers import DebertaV2Model, DebertaV2Tokenizer
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import call_api, generate_task_query, calculate_task_performance
from .dcmodel import RouterModule
from .dcdataset import DCDataset
from .dcdata_utils import preprocess_data


def _env_or(default_value: str, *env_keys: str) -> str:
    """
    Get environment variable or return default value.
    
    Args:
        default_value: Default value if no env var is found
        *env_keys: Environment variable keys to check
    
    Returns:
        Environment variable value or default
    """
    for k in env_keys:
        v = os.environ.get(k)
        if v and len(v.strip()) > 0:
            return v.strip()
    return default_value


class DCRouter(MetaRouter):
    """
    DCRouter
    --------
    Router that uses dual-contrastive learning strategy for LLM routing decisions.

    DCRouter uses a pre-trained encoder (e.g., mDeBERTa) combined with learnable
    LLM embeddings to make routing decisions. The model is trained with three
    contrastive learning objectives:
    1. Sample-LLM contrastive loss
    2. Sample-Sample contrastive loss (task-level)
    3. Cluster contrastive loss
    """

    def __init__(self, yaml_path: str):
        """
        Initialize DCRouter.

        Args:
            yaml_path (str): Path to YAML config file
        """
        # Load configuration
        with open(yaml_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Resolve project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # Prepare data
        self._prepare_data()

        # Remove HF_ENDPOINT to use official HuggingFace source (avoid mirror SSL issues)
        os.environ.pop("HF_ENDPOINT", None)

        # Initialize tokenizer and backbone
        backbone_model = self.cfg['model_path']['backbone_model']
        # Use DebertaV2Tokenizer directly (requires sentencepiece)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            backbone_model,
            truncation_side='left',
            padding=True
        )
        encoder_model = DebertaV2Model.from_pretrained(backbone_model)

        # Load datasets
        hparam = self.cfg['hparam']
        self.train_dataset = DCDataset(
            data=self.train_data_processed,
            source_max_token_len=hparam.get('source_max_token_len', 512),
            target_max_token_len=hparam.get('target_max_token_len', 512),
            dataset_id=0
        )
        self.train_dataset.register_tokenizer(self.tokenizer)

        self.test_dataset = DCDataset(
            data=self.test_data_processed,
            source_max_token_len=hparam.get('source_max_token_len', 512),
            target_max_token_len=hparam.get('target_max_token_len', 512),
            dataset_id=1
        )
        self.test_dataset.register_tokenizer(self.tokenizer)

        num_llms = len(self.train_dataset.router_node)

        # Create RouterModule
        model = RouterModule(
            backbone=encoder_model,
            hidden_state_dim=hparam['hidden_state_dim'],
            node_size=num_llms,
            similarity_function=hparam['similarity_function']
        )

        # Save cfg before calling super().__init__() since it will be reset
        saved_cfg = self.cfg
        
        # Initialize parent class (pass None to avoid duplicate data loading)
        # DCRouter handles its own data loading, so we pass yaml_path=None
        super().__init__(model=model, yaml_path=None)
        
        # Restore cfg since MetaRouter.__init__ resets it to {}
        self.cfg = saved_cfg
        
        # Load additional data using DataLoader (reuse MetaRouter's data loading logic)
        # This ensures consistency with other routers and uses config-based paths
        # DataLoader automatically handles:
        # - Relative/absolute path resolution from config
        # - File existence checking
        # - Error handling with warnings
        from llmrouter.data import DataLoader
        loader = DataLoader(project_root=self.project_root)
        # Load llm_data and llm_embedding_data from config
        # (routing_data is handled separately by _prepare_data)
        loader.load_data(self.cfg, self)
        
        # Load metric weights if provided in config
        weights_dict = self.cfg.get("metric", {}).get("weights", {})
        self.metric_weights = list(weights_dict.values())

    def _prepare_data(self):
        """Prepare and preprocess data."""
        data_path_config = self.cfg['data_path']
        train_data_raw = os.path.join(self.project_root, data_path_config['routing_data_train'])
        test_data_raw = os.path.join(self.project_root, data_path_config['routing_data_test'])

        hparam = self.cfg['hparam']
        n_clusters = hparam.get('n_clusters', 3)

        # Preprocess training data
        self.train_data_processed = preprocess_data(
            input_path=train_data_raw,
            add_cluster_id=True,
            n_clusters=n_clusters
        )

        # Preprocess test data
        self.test_data_processed = preprocess_data(
            input_path=test_data_raw,
            add_cluster_id=False,
            n_clusters=n_clusters
        )

    def route(self, batch):
        """
        Perform routing on a batch of data.

        Args:
            batch (dict): A batch containing tokenized inputs

        Returns:
            dict: A dictionary with routing outputs
        """
        # Extract temperature if provided, default to 1.0
        temperature = batch.get("temperature", 1.0)

        # Prepare inputs for the model
        input_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }

        # Forward pass through RouterModule
        scores, hidden_state = self.model(t=temperature, **input_kwargs)

        # Get predicted LLM indices (argmax)
        predictions = torch.argmax(scores, dim=1)

        return {
            "scores": scores,
            "hidden_state": hidden_state,
            "predictions": predictions,
        }

    def forward(self, batch):
        """
        PyTorch-compatible forward method.
        
        This delegates to route() for compatibility with nn.Module.
        
        Args:
            batch (dict): A batch containing tokenized inputs
            
        Returns:
            dict: A dictionary with routing outputs
        """
        return self.route(batch)

    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using DCRouter and execute them.

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
                If provided, queries will be formatted using task-specific prompts before execution.
                If None, queries are executed as-is. Can also be extracted from each row's 'task_name' field.

        Returns:
            list of dict:
                A list of query dictionaries, each updated with:
                    - "query": original query text (preserved)
                    - "formatted_query": formatted query if task_name was provided (optional)
                    - "model_name": predicted model name
                    - "response": final answer from the routed model
                    - "prompt_tokens": total prompt tokens used
                    - "completion_tokens": total completion tokens used
                    - "input_token": total input tokens (alias for prompt_tokens)
                    - "output_token": total output tokens (alias for completion_tokens)
                    - "task_performance": evaluation score (0.0-1.0) if ground truth available
                    - "success": whether the API call succeeded
        """
        from torch.utils.data import DataLoader

        # Load model if exists
        hparam = self.cfg['hparam']
        device = hparam.get('device', 'cpu')

        # Try to load checkpoint
        save_dir = os.path.join(self.project_root, os.path.dirname(self.cfg['model_path']['save_model_path']))
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(save_dir, 'best_training_model.pth')

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(device)
        self.model.eval()

        # Determine which data to use
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
            # For custom batch, we need to route using the model
            use_custom_batch = True
        else:
            if hasattr(self, "query_data_test") and self.query_data_test is not None:
                query_data = copy.copy(self.query_data_test)
                use_custom_batch = True
            else:
                # Fall back to using test_dataset
                use_custom_batch = False
                query_data = []

        query_data_output = []

        if use_custom_batch:
            # Route custom batch (from batch parameter or query_data_test)
            for row in query_data:
                # Handle both dict and non-dict inputs
                if isinstance(row, dict):
                    row_copy = copy.copy(row)
                    original_query = row_copy.get("query", "")
                    # Use task_name from row if available, otherwise use parameter
                    row_task_name = row_copy.get("task_name", task_name)
                else:
                    row_copy = {"query": str(row)}
                    original_query = str(row)
                    row_task_name = task_name

                # Step 1: Route the query to get model_name
                query_tokens = self.tokenizer(
                    original_query,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

                batch_input = {
                    "input_ids": query_tokens["input_ids"],
                    "attention_mask": query_tokens["attention_mask"],
                    "temperature": hparam.get('inference_temperature', 1.0),
                }

                with torch.no_grad():
                    outputs = self.route(batch_input)

                predicted_llm_idx = outputs["predictions"][0].item()
                predicted_llm = self.test_dataset.router_node[predicted_llm_idx]
                row_copy["model_name"] = predicted_llm

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
                api_model_name = predicted_llm
                api_endpoint = None
                service = None
                
                # Priority 1: Get from llm_data (model-specific endpoint)
                if hasattr(self, 'llm_data') and self.llm_data and predicted_llm in self.llm_data:
                    api_model_name = self.llm_data[predicted_llm].get("model", predicted_llm)
                    api_endpoint = self.llm_data[predicted_llm].get("api_endpoint")
                    # Get service field for service-specific API key selection
                    service = self.llm_data[predicted_llm].get("service")
                
                # Priority 2: Get from router config
                if not api_endpoint:
                    api_endpoint = self.cfg.get("api_endpoint")
                
                # Priority 3: Get from environment variables (like automix)
                if not api_endpoint:
                    api_endpoint = _env_or(
                        "https://integrate.api.nvidia.com/v1",  # Default NVIDIA API
                        "OPENAI_API_BASE",
                        "NVIDIA_API_BASE",
                    )
                
                # Validate that we have an endpoint
                if not api_endpoint:
                    raise ValueError(
                        f"API endpoint not found for model '{predicted_llm}'. "
                        f"Please specify 'api_endpoint' in llm_data JSON, router YAML config, "
                        f"or set OPENAI_API_BASE/NVIDIA_API_BASE environment variable."
                    )

                request = {
                    "api_endpoint": api_endpoint,
                    "query": query_text_for_execution,
                    "model_name": predicted_llm,
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

        else:
            # Use test_dataset with DataLoader (original logic)
            test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=hparam.get('inference_batch_size', 64),
                shuffle=False
            )

            with torch.no_grad():
                sample_idx = 0
                for batch_data in test_dataloader:
                    inputs, scores, _, _ = batch_data
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    scores = scores.to(device)

                    batch_input = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "temperature": hparam.get('inference_temperature', 1.0),
                    }

                    outputs = self.route(batch_input)
                    predictions = outputs["predictions"]

                    # Process each sample in the batch
                    for i in range(len(predictions)):
                        if sample_idx < len(self.test_dataset.data):
                            predicted_llm_idx = predictions[i].item()
                            predicted_llm = self.test_dataset.router_node[predicted_llm_idx]

                            # Get original data from test_dataset
                            test_sample = self.test_dataset.data[sample_idx]
                            query_text = test_sample['question']

                            row_copy = {
                                "query": query_text,
                                "model_name": predicted_llm
                            }

                            # Preserve other fields from test_sample if available
                            for key in ['ground_truth', 'gt', 'answer', 'choices', 'metric']:
                                if key in test_sample:
                                    row_copy[key] = test_sample[key]

                            row_task_name = test_sample.get("task_name", task_name)

                            # Step 2: Format query if task_name is provided
                            if row_task_name:
                                try:
                                    sample_data = {
                                        "query": query_text,
                                        "choices": row_copy.get("choices", None)
                                    }
                                    formatted_query = generate_task_query(row_task_name, sample_data)
                                    row_copy["formatted_query"] = formatted_query
                                    query_text_for_execution = formatted_query
                                except (ValueError, KeyError) as e:
                                    print(f"Warning: Failed to format query with task '{row_task_name}': {e}. Using original query.")
                                    query_text_for_execution = query_text
                            else:
                                query_text_for_execution = query_text

                            # Step 3: Call API to get response
                            # Get API endpoint and model name from llm_data if available
                            api_model_name = predicted_llm
                            api_endpoint = None
                            service = None
                            if hasattr(self, 'llm_data') and self.llm_data and predicted_llm in self.llm_data:
                                api_model_name = self.llm_data[predicted_llm].get("model", predicted_llm)
                                # Get API endpoint from llm_data, fallback to router config
                                api_endpoint = self.llm_data[predicted_llm].get(
                                    "api_endpoint",
                                    self.cfg.get("api_endpoint")
                                )
                                # Get service field for service-specific API key selection
                                service = self.llm_data[predicted_llm].get("service")
                            
                            # If still no endpoint found, try router config
                            if api_endpoint is None:
                                api_endpoint = self.cfg.get("api_endpoint")
                            
                            # Validate that we have an endpoint
                            if not api_endpoint:
                                raise ValueError(
                                    f"API endpoint not found for model '{predicted_llm}'. "
                                    f"Please specify 'api_endpoint' in llm_data JSON for this model or in router YAML config."
                                )

                            request = {
                                "api_endpoint": api_endpoint,
                                "query": query_text_for_execution,
                                "model_name": predicted_llm,
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
                            sample_idx += 1

        return query_data_output

    def route_single(self, data):
        """
        Route a single query.

        Args:
            data (dict): Query data with 'query' key

        Returns:
            dict: Routing result
        """
        hparam = self.cfg['hparam']
        device = hparam.get('device', 'cpu')

        # Load model if exists
        save_dir = os.path.join(self.project_root, os.path.dirname(self.cfg['model_path']['save_model_path']))
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(save_dir, 'best_training_model.pth')

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(device)
        self.model.eval()

        # Tokenize query
        query_text = data["query"]
        query_tokens = self.tokenizer(
            query_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        batch = {
            "input_ids": query_tokens["input_ids"],
            "attention_mask": query_tokens["attention_mask"],
            "temperature": hparam.get('inference_temperature', 1.0),
        }

        with torch.no_grad():
            outputs = self.route(batch)

        predicted_llm_idx = outputs["predictions"][0].item()
        predicted_llm = self.test_dataset.router_node[predicted_llm_idx]

        query_output = copy.copy(data)
        query_output["model_name"] = predicted_llm
        return query_output
