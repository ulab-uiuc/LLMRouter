"""
Chatbot Interface for LLMRouter

This script provides a Gradio-based chat interface that uses LLMRouter
to route queries to appropriate models and generate responses.
"""

import argparse
import os
import yaml
from typing import Dict, Any, Optional

import gradio as gr

# Import router classes
from llmrouter.models import (
    KNNRouter,
    SVMRouter,
    MLPRouter,
    MFRouter,
    EloRouter,
    DCRouter,
    SmallestLLM,
    LargestLLM,
)
from llmrouter.models.llmmultiroundrouter import LLMMultiRoundRouter
from llmrouter.models.knnmultiroundrouter import KNNMultiRoundRouter
try:
    from llmrouter.models.router_r1 import RouterR1
except ImportError:
    RouterR1 = None
from llmrouter.utils import call_api, get_longformer_embedding
import torch
import numpy as np

# Router registry: maps router method names to their classes
ROUTER_REGISTRY = {
    "knnrouter": KNNRouter,
    "svmrouter": SVMRouter,
    "mlprouter": MLPRouter,
    "mfrouter": MFRouter,
    "elorouter": EloRouter,
    "dcrouter": DCRouter,
    "smallest_llm": SmallestLLM,
    "largest_llm": LargestLLM,
    "llmmultiroundrouter": LLMMultiRoundRouter,
    "knnmultiroundrouter": KNNMultiRoundRouter,
}

# Add RouterR1 if available
if RouterR1 is not None:
    ROUTER_REGISTRY["router_r1"] = RouterR1
    ROUTER_REGISTRY["router-r1"] = RouterR1

# Routers that have answer_query method (full pipeline)
ROUTERS_WITH_ANSWER_QUERY = {
    "llmmultiroundrouter",
    "knnmultiroundrouter",
}

# Routers that require special handling
# RouterR1 needs model_id, api_base, api_key for route_single
ROUTERS_REQUIRING_SPECIAL_ARGS = {
    "router_r1",
    "router-r1",
}

# Routers that are not supported for chat interface
# GraphRouter requires a model parameter and doesn't have route_single
UNSUPPORTED_ROUTERS = {
    "graphrouter",
    "graph_router",
}


def prepare_query_full_context(message: str, history: list) -> str:
    """
    Prepare query for Full Context Mode: combine all history + current query.
    
    Args:
        message: Current user message
        history: Chat history as list of (user, assistant) tuples
        
    Returns:
        Combined query string
    """
    # Build full context from history
    context_parts = []
    for human, assistant in history:
        context_parts.append(f"Previous Query: {human}")
        context_parts.append(f"Previous Response: {assistant}")
    
    # Add current query
    context_parts.append(f"Current Query: {message}")
    
    # Combine into single query
    combined_query = "\n\n".join(context_parts)
    return combined_query


def prepare_query_current_only(message: str, history: list) -> str:
    """
    Prepare query for Current Query Mode: only the current query.
    
    Args:
        message: Current user message
        history: Chat history (unused in this mode)
        
    Returns:
        Current query string
    """
    return message


def prepare_query_retrieval(message: str, history: list, top_k: int = 3) -> str:
    """
    Prepare query for Retrieval Mode: find top-k similar historical queries.
    
    Args:
        message: Current user message
        history: Chat history as list of (user, assistant) tuples
        top_k: Number of similar queries to retrieve
        
    Returns:
        Combined query string with retrieved context
    """
    if not history:
        # No history, just return current query
        return message
    
    try:
        # Get embedding for current query
        current_embedding = get_longformer_embedding(message)
        if isinstance(current_embedding, torch.Tensor):
            current_embedding = current_embedding.numpy()
        
        # Ensure current_embedding is 1D
        if len(current_embedding.shape) > 1:
            current_embedding = current_embedding.flatten()
        
        # Get embeddings for all historical queries
        historical_queries = [human for human, _ in history]
        historical_responses = [assistant for _, assistant in history]
        
        if not historical_queries:
            return message
        
        # Compute embeddings for historical queries
        historical_embeddings = get_longformer_embedding(historical_queries)
        if isinstance(historical_embeddings, torch.Tensor):
            historical_embeddings = historical_embeddings.numpy()
        
        # Handle case where historical_embeddings might be 1D (single query)
        if len(historical_embeddings.shape) == 1:
            historical_embeddings = historical_embeddings.reshape(1, -1)
        
        # Compute cosine similarity
        # Normalize embeddings
        current_norm = current_embedding / (np.linalg.norm(current_embedding) + 1e-8)
        historical_norms = historical_embeddings / (np.linalg.norm(historical_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities: (num_queries, embedding_dim) @ (embedding_dim,) -> (num_queries,)
        similarities = np.dot(historical_norms, current_norm)
        
        # Ensure similarities is 1D array
        if similarities.ndim == 0:
            similarities = np.array([similarities])
        elif len(similarities.shape) > 1:
            similarities = similarities.flatten()
        
        # Get top-k indices (limit to available history)
        actual_top_k = min(top_k, len(historical_queries))
        top_k_indices = np.argsort(similarities)[-actual_top_k:][::-1]
        
        # Build retrieved context
        retrieved_parts = []
        for idx in top_k_indices:
            retrieved_parts.append(f"Similar Query: {historical_queries[idx]}")
            retrieved_parts.append(f"Response: {historical_responses[idx]}")
        
        # Combine retrieved context with current query
        retrieved_context = "\n\n".join(retrieved_parts)
        combined_query = f"{retrieved_context}\n\nCurrent Query: {message}"
        
        return combined_query
        
    except Exception as e:
        # Fallback to current query only if retrieval fails
        print(f"Warning: Retrieval mode failed, falling back to current query only: {e}")
        return message


def prepare_query_by_mode(message: str, history: list, mode: str, top_k: int = 3) -> str:
    """
    Prepare query based on the selected mode.
    
    Args:
        message: Current user message
        history: Chat history as list of (user, assistant) tuples
        mode: One of "full_context", "current_only", "retrieval"
        top_k: Number of similar queries for retrieval mode
        
    Returns:
        Prepared query string
    """
    mode_lower = mode.lower()
    
    if mode_lower == "full_context" or mode_lower == "full":
        return prepare_query_full_context(message, history)
    elif mode_lower == "current_only" or mode_lower == "current":
        return prepare_query_current_only(message, history)
    elif mode_lower == "retrieval" or mode_lower == "retrieve":
        return prepare_query_retrieval(message, history, top_k)
    else:
        # Default to current_only if mode is unknown
        print(f"Warning: Unknown mode '{mode}', defaulting to 'current_only'")
        return prepare_query_current_only(message, history)


def load_router(router_name: str, config_path: str, load_model_path: Optional[str] = None):
    """
    Load a router instance based on router name and config.
    
    Args:
        router_name: Name of the router method (e.g., "knnrouter", "llmmultiroundrouter")
        config_path: Path to YAML configuration file
        load_model_path: Optional path to override model_path.load_model_path in config
        
    Returns:
        Router instance
    """
    router_name_lower = router_name.lower()
    
    # Check if router is unsupported
    if router_name_lower in UNSUPPORTED_ROUTERS:
        raise ValueError(
            f"Router '{router_name}' is not supported for chat interface. "
            f"Supported routers: {list(ROUTER_REGISTRY.keys())}"
        )
    
    if router_name_lower not in ROUTER_REGISTRY:
        raise ValueError(
            f"Unknown router: {router_name}. Available routers: {list(ROUTER_REGISTRY.keys())}"
        )
    
    router_class = ROUTER_REGISTRY[router_name_lower]
    
    # Override model path in config if provided
    if load_model_path:
        # Read config, modify, write to temp file
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if "model_path" not in config:
            config["model_path"] = {}
        config["model_path"]["load_model_path"] = load_model_path
        
        # Write to temp config file
        import tempfile
        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, temp_config)
        temp_config.close()
        config_path = temp_config.name
    
    # Initialize router
    # Note: RouterR1 might need special handling, but test shows it can be initialized with just yaml_path
    try:
        router = router_class(yaml_path=config_path)
    except TypeError as e:
        # If initialization fails, it might need additional parameters
        if "required positional argument" in str(e) or "missing" in str(e).lower():
            raise ValueError(
                f"Router '{router_name}' requires additional initialization parameters. "
                f"Error: {str(e)}"
            ) from e
        raise
    
    return router


def predict(
    message: str,
    history: list,
    router_instance: Any,
    router_name: str,
    temperature: float = 0.8,
    mode: str = "current_only",
    top_k: int = 3,
):
    """
    Predict response using the router.
    
    Args:
        message: User message
        history: Chat history as list of (user, assistant) tuples
        router_instance: Loaded router instance
        router_name: Router method name
        temperature: Temperature for generation (if applicable)
        mode: Query mode - "full_context", "current_only", or "retrieval"
        top_k: Number of similar queries for retrieval mode
        
    Returns:
        Generated response text (string)
    """
    router_name_lower = router_name.lower()
    
    # Prepare query based on mode
    query_for_router = prepare_query_by_mode(message, history, mode, top_k)
    
    # Check if router has answer_query method (full pipeline)
    if router_name_lower in ROUTERS_WITH_ANSWER_QUERY and hasattr(router_instance, "answer_query"):
        # Use full pipeline: decompose + route + execute + aggregate
        # For multi-round routers, we still use the prepared query
        try:
            final_answer = router_instance.answer_query(query_for_router, return_intermediate=False)
            return final_answer
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Handle RouterR1 specially (requires model_id, api_base, api_key)
    if router_name_lower in ROUTERS_REQUIRING_SPECIAL_ARGS:
        try:
            # Get required parameters from config
            cfg = router_instance.cfg
            model_id = cfg.get("model_id", "ulab-ai/Router-R1-Qwen2.5-3B-Instruct")
            api_base = cfg.get("api_base") or cfg.get("api_endpoint", "https://integrate.api.nvidia.com/v1")
            api_key = cfg.get("api_key") or os.environ.get("API_KEYS", "").split(",")[0] if os.environ.get("API_KEYS") else ""
            
            if not api_key:
                return "Error: RouterR1 requires api_key in config or API_KEYS environment variable"
            
            # RouterR1's route_single returns None (prints output), so we need to handle it differently
            # For now, indicate that RouterR1 needs special implementation
            return f"Error: RouterR1 requires special handling. Please use RouterR1's native interface. " \
                   f"Required: model_id={model_id}, api_base={api_base}"
            
        except Exception as e:
            return f"Error with RouterR1: {str(e)}"
    
    # Otherwise, use route_single to get routing decision, then call model
    try:
        # Route the query - use the prepared query based on mode
        query_input = {"query": query_for_router}
        routing_result = router_instance.route_single(query_input)
        
        # Extract model name from routing result
        # DCRouter returns "predicted_llm", others return "model_name"
        model_name = (
            routing_result.get("model_name") 
            or routing_result.get("predicted_llm")
            or routing_result.get("predicted_llm_name")
        )
        
        if not model_name:
            return f"Error: Router did not return a model name. Routing result: {routing_result}"
        
        # Get API endpoint from router config
        api_endpoint = router_instance.cfg.get("api_endpoint", "https://integrate.api.nvidia.com/v1")
        
        # Get the actual API model name from llm_data if available
        # The router returns the model key, but we need the full model path for the API
        api_model_name = model_name  # Default to model_name
        if hasattr(router_instance, 'llm_data') and router_instance.llm_data:
            if model_name in router_instance.llm_data:
                # Use the "model" field from llm_data which contains the full API path
                api_model_name = router_instance.llm_data[model_name].get("model", model_name)
            else:
                # If model_name not found, try to find it by matching model field
                for key, value in router_instance.llm_data.items():
                    if value.get("model") == model_name or key == model_name:
                        api_model_name = value.get("model", model_name)
                        break
        
        # Build prompt with chat history
        prompt = "You are a helpful AI assistant.\n\n"
        for human, assistant in history:
            prompt += f"User: {human}\nAssistant: {assistant}\n\n"
        prompt += f"User: {message}\nAssistant:"
        
        # Call the routed model via API
        request = {
            "api_endpoint": api_endpoint,
            "query": prompt,
            "model_name": model_name,  # Keep original for router identification
            "api_name": api_model_name,  # Use full API model path
        }
        
        result = call_api(request, max_tokens=1024, temperature=temperature)
        
        response = result.get("response", "No response generated")
        
        # Add model name prefix
        model_prefix = f"[{model_name}]\n"
        return model_prefix + response
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


def main():
    """Main entry point for the chat interface."""
    parser = argparse.ArgumentParser(description="Chatbot Interface for LLMRouter")
    parser.add_argument(
        "--router",
        type=str,
        required=True,
        help="Router method name (e.g., knnrouter, llmmultiroundrouter, mfrouter)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Optional path to override model_path.load_model_path in config",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Temperature for text generation (default: 0.8)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server to (default: None, all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server to (default: 8001)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="current_only",
        choices=["full_context", "current_only", "retrieval"],
        help="Query mode: 'full_context' (all history), 'current_only' (single query), 'retrieval' (top-k similar) (default: current_only)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of similar queries to retrieve in retrieval mode (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Load router
    print(f"Loading router: {args.router}")
    print(f"Using config: {args.config}")
    if args.load_model_path:
        print(f"Overriding model path: {args.load_model_path}")
    
    router_instance = load_router(args.router, args.config, args.load_model_path)
    print("✅ Router loaded successfully!")
    
    # Create predict function with router instance bound
    def predict_with_router(message, history, temperature, mode, top_k):
        return predict(message, history, router_instance, args.router, temperature, mode, top_k)
    
    # Create and launch chat interface
    interface = gr.ChatInterface(
        predict_with_router,
        additional_inputs=[
            gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=2,
                value=args.temp,
                step=0.1,
            ),
            gr.Radio(
                label="Query Mode",
                choices=["full_context", "current_only", "retrieval"],
                value=args.mode,
                info="Full Context: all history + current query | Current Only: single query | Retrieval: top-k similar queries",
            ),
            gr.Slider(
                label="Top-K (Retrieval Mode)",
                minimum=1,
                maximum=10,
                value=args.top_k,
                step=1,
                info="Number of similar queries to retrieve (only used in retrieval mode)",
            ),
        ],
        title=f"LLMRouter Chat - {args.router}",
        description=f"Chat interface using {args.router} router | Mode: {args.mode}",
    )
    
    interface.queue().launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()