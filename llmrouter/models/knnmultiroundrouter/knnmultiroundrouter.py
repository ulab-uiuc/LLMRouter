from typing import Any, Dict, List, Optional
import os
import torch.nn as nn
import copy
import re
from sklearn.neighbors import KNeighborsClassifier
from llmrouter.models.meta_router import MetaRouter
from llmrouter.utils import load_model, get_longformer_embedding, call_api, generate_task_query
from llmrouter.utils.evaluation import exact_match_score, cem_score, f1_score

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
        self.DECOMP_PROMPT = """Given the query '{query}', decompose it into as many as 4 meaningful sub-queries (minimum 1, maximum 4). \
Try to cover the full scope of the original query by breaking it down into multiple specific and distinct sub-tasks \
whenever possible. Aim for the maximum number of high-quality sub-queries without introducing redundancy. \
Each sub-query should be clear, self-contained, and semantically coherent. \

**Output formatting rules (must be followed strictly):**
    - Output only the decomposed sub-queries, one per line.
    - Do not include any other text, explanations, or headers in your output.
    - Each line should contain only one sub-query.
"""
        
        self.AGENT_PROMPT = """You are a helpful assistant. \
You are participating in a multi-agent reasoning process, where a base model delegates sub-questions to specialized models like you. \
\nYour task is to do your **absolute best** to either: \n
    + Answer the question directly, if possible, and provide a brief explanation; or \n
    + Offer helpful and relevant context, background knowledge, or insights related to the question, even if you cannot fully answer it. \

If you are completely unable to answer the question or provide any relevant or helpful information, you must: \n
    + Clearly state that you are unable to assist with this question, and \n
    + Explicitly instruct the base model to consult other LLMs for further assistance. \

**Important Constraints**: \n
    + Keep your response clear, concise, and informative (preferably under 512 tokens). Your response will help guide the base model's reasoning and next steps. \n
    + Stay strictly on-topic. Do not include irrelevant or generic content. \

\n\nHere is the sub-question for you to assist with: {query}\n"""
        
        self.DECOMP_COT_PROMPT = """You are given a question along with auxiliary information, which consists of several sub-questions derived from the original question and their respective answers. Use this information to answer the original question if relevant, but make your own reasoning step by step before arriving at the final answer. 

Important: Your final answer MUST be clearly marked and enclosed within <answer> and </answer> tags at the end of your response. No other part of the output should be inside these tags.

Auxiliary Information: {info}

Question: {query}
Let's think step by step.
"""
        
        # Configuration for local LLM (for decomposition and aggregation)
        # Use .get() with defaults to handle missing config gracefully
        self.base_model = self.cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
        self.local_llm = None
        self.local_tokenizer = None
        self.use_local_llm = self.cfg.get("use_local_llm", False) and VLLM_AVAILABLE
        
        # API configuration for execution
        # Note: API keys are handled via environment variable API_KEYS in call_api()
        self.api_endpoint = self.cfg.get("api_endpoint", "https://integrate.api.nvidia.com/v1")
        
        # Load KNN model path for routing
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.knn_model_path = os.path.join(project_root, self.cfg["model_path"]["load_model_path"])
        
        # Lazy loading flag for KNN model
        self.knn_model_loaded = False

    def _calculate_task_performance(
        self, 
        prediction: str, 
        ground_truth: Optional[str], 
        task_name: Optional[str] = None,
        metric: Optional[str] = None
    ) -> Optional[float]:
        """
        Calculate task performance score for a prediction against ground truth.
        
        Args:
            prediction: The model's response/prediction
            ground_truth: Ground truth answer (optional)
            task_name: Task name to determine metric if not provided
            metric: Evaluation metric to use (optional, will be inferred from task_name if not provided)
            
        Returns:
            Performance score (0.0 to 1.0) or None if ground_truth is not available
        """
        if not ground_truth:
            return None
        
        # Determine metric based on task_name if not provided
        if metric is None and task_name:
            # Map task names to metrics
            if task_name in ["natural_qa", "trivia_qa", "squad", "boolq"]:
                metric = "cem"
            elif task_name in ["mmlu", "gpqa", "commonsense_qa", "openbook_qa", "arc_challenge"]:
                metric = "em_mc"
            elif task_name == "gsm8k":
                metric = "gsm8k"
            elif task_name == "math":
                metric = "math"
            else:
                metric = "cem"  # Default to CEM
        
        # Evaluate based on metric
        try:
            if metric == "em":
                return float(exact_match_score(prediction, ground_truth))
            elif metric == "em_mc":
                return float(exact_match_score(prediction, ground_truth, normal_method="mc"))
            elif metric == "cem":
                return float(cem_score(prediction, ground_truth))
            elif metric == "gsm8k":
                # GSM8K evaluation: extract number from ground truth and prediction
                ground_truth_processed = ground_truth.split("####")[-1].replace(',', '').replace('$', '').replace('.', '').strip()
                answer = re.findall(r"(\-?[0-9\.\,]+)", prediction)
                if len(answer) == 0:
                    return 0.0
                invalid_str = ['', '.']
                final_answer = None
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
                if final_answer is None:
                    return 0.0
                final_answer = final_answer.replace(',', '').replace('$', '').replace('.', '').strip()
                return 1.0 if final_answer == ground_truth_processed else 0.0
            elif metric == "math":
                # MATH evaluation: extract from \boxed{} format
                # Simple version - check if ground truth appears in prediction
                # For full implementation, would need more complex parsing
                gt_clean = ground_truth.replace("\\boxed{", "").replace("}", "").strip()
                if gt_clean in prediction:
                    return 1.0
                # Try to find boxed answer in prediction
                boxed_match = re.search(r"\\boxed\{([^}]+)\}", prediction)
                if boxed_match:
                    pred_answer = boxed_match.group(1).strip()
                    if pred_answer == gt_clean:
                        return 1.0
                return 0.0
            elif metric == "f1":
                f1, _, _ = f1_score(prediction, ground_truth)
                return float(f1)
            else:
                # Default to CEM
                return float(cem_score(prediction, ground_truth))
        except Exception as e:
            print(f"Warning: Error calculating task_performance: {e}")
            return None

    def _load_knn_model_if_needed(self):
        """Load KNN model once at inference time with proper error handling."""
        if self.knn_model_loaded:
            return
        
        try:
            self.knn_model = load_model(self.knn_model_path)
            self.knn_model_loaded = True
            print(f"[KNNMultiRoundRouter] Loaded KNN model from {self.knn_model_path}")
        except Exception as e:
            print(f"Error loading KNN model from {self.knn_model_path}: {e}")
            raise RuntimeError(f"Failed to load KNN model. Please ensure the model is trained and saved at {self.knn_model_path}")

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query through the full pipeline: decompose → route → execute → aggregate.
        
        This method performs end-to-end processing:
        1. Decomposes the initial query into sub-queries
        2. Routes each sub-query using KNN
        3. Executes each sub-query with the routed model
        4. Aggregates all responses into a final answer

        Args:
            query (dict):
                A single query dictionary. Must contain the key:
                    - "query": textual input to be processed.
                Optional keys:
                    - "task_name": Task name for prompt selection during aggregation

        Returns:
            dict:
                Updated query dictionary containing:
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
        
        original_query = query.get("query", "")
        task_name = query.get("task_name", None)
        
        # Step 1: Decompose query into sub-queries
        sub_queries = self._decompose_query(original_query)
        
        # Step 2: Route and execute each sub-query
        sub_responses = []
        for sub_query in sub_queries:
            # Route the sub-query using KNN
            query_embedding = [get_longformer_embedding(sub_query).numpy()]
            model_name = self.knn_model.predict(query_embedding)[0]
            
            # Execute the sub-query with the routed model
            execution_result = self._execute_sub_query(sub_query, model_name)
            sub_responses.append(execution_result)
        
        # Step 3: Aggregate responses into final answer
        final_answer = self._aggregate_responses(original_query, sub_queries, sub_responses, task_name)
        
        # Calculate token counts
        prompt_tokens = sum(r.get("prompt_tokens", 0) for r in sub_responses)
        completion_tokens = sum(r.get("completion_tokens", 0) for r in sub_responses)
        
        # Calculate task performance if ground truth is available
        ground_truth = query.get("ground_truth") or query.get("gt") or query.get("answer")
        metric = query.get("metric")
        task_performance = self._calculate_task_performance(
            prediction=final_answer,
            ground_truth=ground_truth,
            task_name=task_name,
            metric=metric
        )
        
        # Return final result
        query_output = copy.copy(query)
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
        
        # Use call_api from utils
        request = {
            "api_endpoint": self.api_endpoint,
            "query": agent_prompt,
            "model_name": model_name,
            "api_name": model_name  # Assuming model_name is the API name
        }
        
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
            request = {
                "api_endpoint": self.api_endpoint,
                "query": agg_prompt,
                "model_name": self.base_model,
                "api_name": self.base_model
            }
            try:
                result = call_api(request, max_tokens=1024, temperature=0.0)
                final_answer = result.get("response", "")
            except Exception as e:
                print(f"Error aggregating responses: {e}")
                final_answer = ""
        
        return final_answer
