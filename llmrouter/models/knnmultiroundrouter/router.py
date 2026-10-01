from typing import Any, Dict, List, Optional
import os
import torch.nn as nn
import copy
from sklearn.neighbors import KNeighborsClassifier
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding, call_api, generate_task_query, calculate_task_performance

# Optional imports for local LLM inference
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AutoTokenizer = None


class KNNMultiRoundRouter(MetaRouter):
    """
    KNNMultiRoundRouter
    -------------------
    A routing module that leverages a K-Nearest Neighbors (KNN) classifier
    to select the most similar language model based on query embeddings.
    
    This router is designed for multi-round scenarios where queries are decomposed
    into sub-queries, and each sub-query is routed independently. The router works
    seamlessly with the decomposition → route → execute → aggregate pipeline.

    The router inherits from MetaRouter for consistent interface design.
    If no trained KNN model is found at the specified path, it can fall back
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
      knn_model_path: "configs/knn_model.pkl"
      n_neighbors: 5
      metric: "cosine"
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the KNNMultiRoundRouter and load configuration.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        The initialization performs the following steps:
            1. Loads configuration and metadata from MetaRouter.
            2. Builds a KNN classifier using the specified hyperparameters.
            3. Prepares the training embeddings and corresponding model labels.
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Initialize KNN classifier with user-defined hyperparameters
        knn_params = self.cfg["hparam"]
        self.knn_model = KNeighborsClassifier(**knn_params)

        # Select the best-performing model for each query
        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        # Prepare embedding and label arrays for KNN training
        query_embedding_id = routing_best["embedding_id"].tolist()
        self.query_embedding_list = [self.query_embedding_data[i].numpy() for i in query_embedding_id]
        self.model_name_list = routing_best["model_name"].tolist()
        
        # Initialize prompts for decomposition and aggregation
        from llmrouter.prompts import load_prompt_template
        self.DECOMP_PROMPT = load_prompt_template("agent_decomp")
        self.AGENT_PROMPT = load_prompt_template("agent_prompt")
        self.DECOMP_COT_PROMPT = load_prompt_template("agent_decomp_cot")
        
        # Configuration for local LLM (for decomposition and aggregation)
        # Use .get() with defaults to handle missing config gracefully
        self.base_model = self.cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
        self.local_llm = None
        self.local_tokenizer = None
        self.use_local_llm = self.cfg.get("use_local_llm", False) and VLLM_AVAILABLE
        
        # API configuration for execution
        # Note: API keys are handled via environment variable API_KEYS in call_api()
        # api_endpoint will be retrieved from llm_data per model, with fallback to config
        self.api_endpoint = self.cfg.get("api_endpoint")  # Used as fallback only
        
        # Lazy loading flag for KNN model
        # Note: Model path is constructed lazily in _load_knn_model_if_needed() to avoid
        # accessing load_model_path during training when it doesn't exist
        self.knn_model_loaded = False

    def _load_knn_model_if_needed(self):
        """Load KNN model once at inference time with proper error handling."""
        if self.knn_model_loaded:
            return
        
        # Construct model path dynamically (only when needed, not during __init__)
        # This allows training to work when load_model_path doesn't exist in config
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = self.cfg.get("model_path", {}).get("save_model_path")
        
        if not load_model_path:
            raise RuntimeError(
                "load_model_path not found in config. "
                "Please ensure the model is trained and load_model_path is specified in the config."
            )
        
        knn_model_path = os.path.join(project_root, load_model_path)
        
        try:
            self.knn_model = load_model(knn_model_path)
            self.knn_model_loaded = True
            print(f"[KNNMultiRoundRouter] Loaded KNN model from {knn_model_path}")
        except Exception as e:
            print(f"Error loading KNN model from {knn_model_path}: {e}")
            raise RuntimeError(
                f"Failed to load KNN model. Please ensure the model is trained and saved at {knn_model_path}"
            )

    def _route_sub_query(self, sub_query: str) -> str:
        """
        Route a single sub-query using KNN to select the best model.
        
        This method is useful for testing and verification purposes, allowing
        you to test routing decisions without executing the full pipeline.
        
        Args:
            sub_query: Sub-query string to route
            
        Returns:
            str: Model name that the sub-query is routed to
        """
        # Load KNN model if not already loaded
        self._load_knn_model_if_needed()
        
        # Get embedding for the sub-query
        query_embedding = [get_longformer_embedding(sub_query).numpy()]
        
        # Use KNN to predict the best model
        model_name = self.knn_model.predict(query_embedding)[0]
        
        return model_name

    def route_single(self, query):
        """
        Route a single query through the full pipeline: decompose → route → execute → aggregate.

        This method performs end-to-end processing:
        1. Decomposes the initial query into sub-queries
        2. Routes each sub-query using KNN
        3. Executes each sub-query with the routed model
        4. Aggregates all responses into a final answer

        Args:
            query (str or dict):
                For chat mode (simple interaction):
                    - str: The query text. Returns only the response string.
                For evaluation mode (with metrics):
                    - dict: Must contain "query" key, optional "task_name", "ground_truth", "metric".
                      Returns full dict with response, tokens, and performance metrics.

        Returns:
            str (chat mode) or dict (evaluation mode):
                Chat mode: Returns the final response string directly.
                Evaluation mode: Returns dict containing:
                    - "query": original query text
                    - "response": final aggregated answer
                    - "prompt_tokens": total prompt tokens used
                    - "completion_tokens": total completion tokens used
                    - "input_token": total input tokens (alias for prompt_tokens)
                    - "output_token": total output tokens (alias for completion_tokens)
                    - "task_performance": evaluation score (0.0-1.0) if ground truth available
                    - "success": whether the pipeline succeeded
        """
        # Load KNN model if not already loaded
        self._load_knn_model_if_needed()

        # Handle both string (chat mode) and dict (evaluation mode)
        if isinstance(query, str):
            # Chat mode: simple query string, return response only
            original_query = query
            task_name = None
            chat_mode = True
            query_dict = {"query": original_query}
        else:
            # Evaluation mode: full dict with metrics
            original_query = query.get("query", "")
            task_name = query.get("task_name", None)
            chat_mode = False
            query_dict = query
        
        # Step 1: Decompose query into sub-queries
        sub_queries = self._decompose_query(original_query)
        
        # Step 2: Route and execute each sub-query
        sub_responses = []
        for sub_query in sub_queries:
            # Route the sub-query using KNN
            model_name = self._route_sub_query(sub_query)
            
            # Execute the sub-query with the routed model
            execution_result = self._execute_sub_query(sub_query, model_name)
            sub_responses.append(execution_result)
        
        # Step 3: Aggregate responses into final answer
        final_answer = self._aggregate_responses(original_query, sub_queries, sub_responses, task_name)

        # Chat mode: return response string only
        if chat_mode:
            return final_answer

        # Evaluation mode: return full dict with metrics
        # Calculate token counts
        prompt_tokens = sum(r.get("prompt_tokens", 0) for r in sub_responses)
        completion_tokens = sum(r.get("completion_tokens", 0) for r in sub_responses)

        # Calculate task performance if ground truth is available
        ground_truth = query_dict.get("ground_truth") or query_dict.get("gt") or query_dict.get("answer")
        metric = query_dict.get("metric")
        task_performance = calculate_task_performance(
            prediction=final_answer,
            ground_truth=ground_truth,
            task_name=task_name,
            metric=metric
        )

        # Return final result with full metrics
        query_output = copy.copy(query_dict)
        query_output["response"] = final_answer
        query_output["prompt_tokens"] = prompt_tokens
        query_output["completion_tokens"] = completion_tokens
        query_output["input_token"] = prompt_tokens
        query_output["output_token"] = completion_tokens
        if task_performance is not None:
            query_output["task_performance"] = task_performance
        query_output["success"] = all(r.get("success", False) for r in sub_responses)
        return query_output

    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries through the full pipeline: decompose → route → execute → aggregate.
        
        This method performs the same end-to-end processing as route_single() for each query:
        1. Applies task-specific prompt formatting if task_name is provided
        2. Decomposes each initial query into sub-queries
        3. Routes each sub-query using KNN
        4. Executes each sub-query with the routed model
        5. Aggregates all responses into a final answer

        Args:
            batch (Any, optional):
                If provided, routes the provided batch. If None, uses self.query_data_test from loaded data.
            task_name (str, optional):
                Task name for prompt formatting (e.g., "mmlu", "gsm8k", "commonsense_qa").
                If provided, queries will be formatted using task-specific prompts before routing.
                If None, queries are routed as-is. Can also be extracted from each row's 'task_name' field.

        Returns:
            list of dict:
                A list of query dictionaries, each updated with:
                    - "query": original query text (preserved)
                    - "formatted_query": formatted query if task_name was provided (optional)
                    - "response": final aggregated answer
                    - "prompt_tokens": total prompt tokens used
                    - "completion_tokens": total completion tokens used
                    - "input_token": total input tokens (alias for prompt_tokens)
                    - "output_token": total output tokens (alias for completion_tokens)
                    - "task_performance": evaluation score (0.0-1.0) if ground truth available
                    - "success": whether the pipeline succeeded
        """
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
                # Use task_name from row if available, otherwise use parameter
                row_task_name = row_copy.get("task_name", task_name)
            else:
                row_copy = {"query": str(row)}
                original_query = str(row)
                row_task_name = task_name

            # Format query if task_name is provided
            if row_task_name:
                try:
                    # Prepare sample_data dict for generate_task_query
                    sample_data = {
                        "query": original_query,
                        "choices": row_copy.get("choices", None) if isinstance(row_copy, dict) else None
                    }
                    formatted_query = generate_task_query(row_task_name, sample_data)
                    row_copy["formatted_query"] = formatted_query
                    query_text_for_routing = formatted_query
                except (ValueError, KeyError) as e:
                    # If formatting fails, fall back to original query
                    print(f"Warning: Failed to format query with task '{row_task_name}': {e}. Using original query.")
                    query_text_for_routing = original_query
            else:
                query_text_for_routing = original_query

            # Use route_single to process the full pipeline (decompose → route → execute → aggregate)
            routing_result = self.route_single({
                "query": query_text_for_routing,
                "task_name": row_task_name
            })
            
            # Update row with routing and execution results
            row_copy.update(routing_result)
            query_data_output.append(row_copy)

        return query_data_output

    def _initialize_local_llm(self):
        """Initialize local LLM for decomposition and aggregation if not already initialized."""
        if not self.use_local_llm or self.local_llm is not None:
            return
        
        try:
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            self.local_llm = LLM(
                model=self.base_model,
                trust_remote_code=True,
                dtype="float16",
                tensor_parallel_size=1
            )
            print(f"✅ Local LLM initialized: {self.base_model}")
        except Exception as e:
            print(f"⚠️  Failed to initialize local LLM: {e}. Will use API calls instead.")
            self.use_local_llm = False
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a query into sub-queries.
        
        Args:
            query: Original query to decompose
            
        Returns:
            List of sub-queries
        """
        decomp_prompt = self.DECOMP_PROMPT.format(query=query)
        
        if self.use_local_llm:
            self._initialize_local_llm()
            if self.local_llm is not None:
                # Use local LLM
                prompt_text = self.local_tokenizer.apply_chat_template(
                    [{"role": "user", "content": decomp_prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=256)
                outputs = self.local_llm.generate([prompt_text], sampling_params)
                decomp_output = outputs[0].outputs[0].text.strip()
            else:
                decomp_output = ""
        else:
            # Fallback: use API call (would need API key configured)
            # For now, return single query if API not available
            return [query]
        
        # Parse sub-queries from output
        decomp_lines = decomp_output.strip().split("\n")
        sub_queries = []
        for line in decomp_lines:
            line = line.strip()
            if line and len(line) > 2:  # Skip empty or very short lines
                sub_queries.append(line)
        
        # If no sub-queries generated, use original query
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def _execute_sub_query(self, sub_query: str, model_name: str) -> Dict[str, Any]:
        """
        Execute a sub-query using the routed model via API.
        
        Args:
            sub_query: Sub-query to execute
            model_name: Model name to use
            
        Returns:
            Dict with response, tokens, etc.
        """
        agent_prompt = self.AGENT_PROMPT.format(query=sub_query)
        
        # Get API endpoint and model name from llm_data if available
        api_model_name = model_name
        api_endpoint = None
        service = None
        if hasattr(self, 'llm_data') and self.llm_data:
            if model_name in self.llm_data:
                # Direct lookup - works when model_name is a key in llm_data
                api_model_name = self.llm_data[model_name].get("model", model_name)
                # Get API endpoint from llm_data, fallback to router config
                api_endpoint = self.llm_data[model_name].get(
                    "api_endpoint",
                    self.cfg.get("api_endpoint")
                )
                # Get service field for service-specific API key selection
                service = self.llm_data[model_name].get("service")
            else:
                # Fallback: Search by "model" field - handles cases where model_name is an API name
                # or doesn't exactly match a key (e.g., normalization differences)
                for key, value in self.llm_data.items():
                    if value.get("model") == model_name or key == model_name:
                        api_model_name = value.get("model", model_name)
                        # Get API endpoint from llm_data, fallback to router config
                        api_endpoint = value.get(
                            "api_endpoint",
                            self.cfg.get("api_endpoint")
                        )
                        # Get service field for service-specific API key selection
                        service = value.get("service")
                        break
        
        # If still no endpoint found, try router config
        if api_endpoint is None:
            api_endpoint = self.cfg.get("api_endpoint")
        
        # Validate that we have an endpoint
        if not api_endpoint:
            raise ValueError(
                f"API endpoint not found for model '{model_name}'. "
                f"Please specify 'api_endpoint' in llm_data JSON for this model or in router YAML config."
            )
        
        # Use call_api from utils
        request = {
            "api_endpoint": api_endpoint,
            "query": agent_prompt,
            "model_name": model_name,
            "api_name": api_model_name
        }
        # Add service field if available (for service-specific API key selection)
        if service:
            request["service"] = service
        
        try:
            result = call_api(request, max_tokens=512, temperature=0.000001)
            return {
                "response": result.get("response", ""),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "success": "error" not in result
            }
        except Exception as e:
            print(f"Error executing sub-query with {model_name}: {e}")
            return {
                "response": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "success": False
            }
    
    def _aggregate_responses(
        self, 
        original_query: str, 
        sub_queries: List[str], 
        sub_responses: List[Dict[str, Any]],
        task_name: Optional[str] = None
    ) -> str:
        """
        Aggregate sub-query responses into final answer.
        
        Args:
            original_query: Original query
            sub_queries: List of sub-queries
            sub_responses: List of response dicts from sub-queries
            task_name: Optional task name for prompt selection
            
        Returns:
            Final aggregated answer
        """
        # Format auxiliary information
        input_info = ""
        for sub_q, sub_resp in zip(sub_queries, sub_responses):
            input_info += f"Sub-query: {sub_q}\n\n"
            input_info += f"Response: {sub_resp.get('response', '')}\n\n"
        
        # Select prompt based on task type
        mc_tasks = {"commonsense_qa", "openbook_qa", "arc_challenge", "mmlu", "gpqa"}
        if task_name and task_name in mc_tasks:
            # Multiple choice prompt
            agg_prompt = f"""You are given a multiple-choice question and supporting sub-answers. Use the information only if helpful.

Question: {original_query}

Supporting information:
{input_info}

Rules:
- Select exactly one option: A, B, C, D, or E.
- Output only the letter in <answer> tags. No explanation.

Format:
<answer>A</answer>"""
        else:
            # Standard decomposition prompt
            agg_prompt = self.DECOMP_COT_PROMPT.format(query=original_query, info=input_info)
        
        if self.use_local_llm:
            self._initialize_local_llm()
            if self.local_llm is not None:
                # Use local LLM
                prompt_text = self.local_tokenizer.apply_chat_template(
                    [{"role": "user", "content": agg_prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1024)
                outputs = self.local_llm.generate([prompt_text], sampling_params)
                final_answer = outputs[0].outputs[0].text.strip()
            else:
                final_answer = ""
        else:
            # Fallback: use API call with base model
            # Get endpoint for base_model from llm_data or config
            base_api_endpoint = None
            service = None
            if hasattr(self, 'llm_data') and self.llm_data:
                if self.base_model in self.llm_data:
                    # Direct lookup - works when base_model is a key in llm_data
                    base_api_endpoint = self.llm_data[self.base_model].get("api_endpoint", self.api_endpoint)
                    # Get service field for service-specific API key selection
                    service = self.llm_data[self.base_model].get("service")
                else:
                    # Fallback: Search by "model" field - handles cases where base_model is an API name
                    # or doesn't exactly match a key (e.g., normalization differences)
                    for key, value in self.llm_data.items():
                        if value.get("model") == self.base_model or key == self.base_model:
                            base_api_endpoint = value.get("api_endpoint", self.api_endpoint)
                            # Get service field for service-specific API key selection
                            service = value.get("service")
                            break
            else:
                base_api_endpoint = self.api_endpoint
            
            if not base_api_endpoint:
                raise ValueError(
                    f"API endpoint not found for base_model '{self.base_model}'. "
                    f"Please specify 'api_endpoint' in llm_data JSON for this model or in router YAML config."
                )
            
            request = {
                "api_endpoint": base_api_endpoint,
                "query": agg_prompt,
                "model_name": self.base_model,
                "api_name": self.base_model
            }
            # Add service field if available (for service-specific API key selection)
            if service:
                request["service"] = service
            try:
                result = call_api(request, max_tokens=1024, temperature=0.0)
                final_answer = result.get("response", "")
            except Exception as e:
                print(f"Error aggregating responses: {e}")
                final_answer = ""
        
        return final_answer
