from typing import Any, Dict, List, Optional
import os
import torch.nn as nn
import copy
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import call_api, generate_task_query, calculate_task_performance


class CausalLMRouter(MetaRouter):
    """
    CausalLMRouter: A routing module that uses a finetuned Causal Language Model
    to predict the best LLM for a given query.

    The model is finetuned to predict the optimal LLM name based on query content.
    During inference, vLLM is used for efficient batch generation.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize CausalLMRouter.

        Args:
            yaml_path: Path to YAML configuration file
        """
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # Get model config
        self.model_config = self.cfg.get("hparam", {})
        self.base_model_name = self.model_config.get("base_model", "meta-llama/Llama-2-7b-hf")

        # Get available LLM names
        self.model_names = self.routing_data_train["model_name"].unique().tolist()

        # Prepare training data
        self._prepare_training_data()

        # vLLM model (initialized during inference)
        self.vllm_model = None

    def _prepare_training_data(self):
        """Prepare training data: query -> best LLM name pairs."""
        # Get best LLM for each query based on performance
        routing_best = self.routing_data_train.loc[
            self.routing_data_train.groupby("query")["performance"].idxmax()
        ].reset_index(drop=True)

        self.query_list = routing_best["query"].tolist()
        self.best_llm_list = routing_best["model_name"].tolist()

    def _build_prompt(self, query: str) -> str:
        """
        Build prompt for the causal LM.

        Args:
            query: User query string

        Returns:
            Formatted prompt string
        """
        llm_options = ", ".join(self.model_names)

        prompt = f"""You are an intelligent router that selects the best Large Language Model (LLM) for a given query.

Available LLMs: {llm_options}

Based on the query content, complexity, and requirements, predict which LLM would provide the best response.

Query: {query}

Best LLM:"""

        return prompt

    def _build_training_prompt(self, query: str, best_llm: str) -> str:
        """
        Build training prompt with the answer.

        Args:
            query: User query string
            best_llm: Ground truth best LLM name

        Returns:
            Formatted prompt with answer for training
        """
        prompt = self._build_prompt(query)
        return f"{prompt} {best_llm}"

    def get_training_data(self) -> List[Dict[str, str]]:
        """
        Get formatted training data for finetuning.

        Returns:
            List of dicts with 'prompt' and 'completion' keys
        """
        training_data = []
        for query, best_llm in zip(self.query_list, self.best_llm_list):
            training_data.append({
                "prompt": self._build_prompt(query),
                "completion": f" {best_llm}",
                "full_text": self._build_training_prompt(query, best_llm)
            })
        return training_data

    def _load_vllm_model(self):
        """Load model using vLLM for efficient inference."""
        if self.vllm_model is not None:
            return

        from vllm import LLM, SamplingParams

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        load_model_path = os.path.join(
            project_root,
            self.cfg["model_path"]["load_model_path"]
        )

        # Load finetuned model with vLLM
        self.vllm_model = LLM(
            model=load_model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.model_config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=self.model_config.get("gpu_memory_utilization", 0.9)
        )

        # Sampling parameters for generation
        self.sampling_params = SamplingParams(
            max_tokens=self.model_config.get("max_new_tokens", 32),
            temperature=self.model_config.get("temperature", 0.1),
            top_p=self.model_config.get("top_p", 0.95),
            stop=["\n", "Query:", "Available"]
        )

    def _parse_llm_name(self, generated_text: str) -> str:
        """
        Parse LLM name from generated text.

        Args:
            generated_text: Raw generated text from model

        Returns:
            Parsed LLM name (or first available if not found)
        """
        generated_text = generated_text.strip()

        # Try to find exact match
        for llm_name in self.model_names:
            if llm_name.lower() in generated_text.lower():
                return llm_name

        # Try to find partial match
        for llm_name in self.model_names:
            if any(part.lower() in generated_text.lower() for part in llm_name.split("-")):
                return llm_name

        # Default to first LLM if no match found
        return self.model_names[0]

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query using the finetuned causal LM.

        Args:
            query: Dict containing 'query' key with the query string

        Returns:
            Dict with added 'model_name' key
        """
        self._load_vllm_model()

        prompt = self._build_prompt(query["query"])
        outputs = self.vllm_model.generate([prompt], self.sampling_params)

        generated_text = outputs[0].outputs[0].text
        model_name = self._parse_llm_name(generated_text)

        query_output = copy.copy(query)
        query_output["model_name"] = model_name
        return query_output

    def route_batch(self, batch: Optional[Any] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries using vLLM and execute them.

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
        self._load_vllm_model()

        query_data = self._resolve_query_data(batch)
        if query_data is None:
            return []

        # Build prompts for all queries (for routing only)
        prompts = []
        for row in query_data:
            # Handle both dict and non-dict inputs
            if isinstance(row, dict):
                original_query = row.get("query", "")
            else:
                original_query = str(row)
            prompts.append(self._build_prompt(original_query))

        # Batch generation with vLLM for routing
        outputs = self.vllm_model.generate(prompts, self.sampling_params)

        query_data_output = []
        for i, row in enumerate(query_data):
            # Handle both dict and non-dict inputs
            row_copy, original_query, row_task_name = self._normalize_row(row, task_name)

            # Step 1: Get routed model name
            generated_text = outputs[i].outputs[0].text
            model_name = self._parse_llm_name(generated_text)
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
