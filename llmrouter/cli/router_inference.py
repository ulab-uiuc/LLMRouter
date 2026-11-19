"""
Router Inference Script for LLMRouter

This script provides non-interactive inference functionality for LLMRouter.
It supports single query inference, batch inference from file, and various output modes.
"""

import atexit
import argparse
import json
import os
import multiprocessing as mp
import sys
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

def _configure_multiprocessing() -> None:
    """Ensure CUDA-safe multiprocessing for vLLM workers."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


_configure_multiprocessing()

# Import router classes
from llmrouter.models import (
    KNNRouter,
    SVMRouter,
    MLPRouter,
    RACERRouter,
    MFRouter,
    EloRouter,
    DCRouter,
    HybridLLMRouter,
    GraphRouter,
    CausalLMRouter,
    SmallestLLM,
    LargestLLM,
    AutomixRouter,
    GMTRouter,
    PersonalizedRouter,
    GnosisLatentRouter,
)
from llmrouter.models.llmmultiroundrouter import LLMMultiRoundRouter
from llmrouter.models.knnmultiroundrouter import KNNMultiRoundRouter
try:
    from llmrouter.models import RouterR1
except ImportError:
    RouterR1 = None
from llmrouter.utils import call_api


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# Router registry: maps router method names to their classes
ROUTER_REGISTRY = {
    "racerrouter": RACERRouter,
    "knnrouter": KNNRouter,
    "svmrouter": SVMRouter,
    "mlprouter": MLPRouter,
    "mfrouter": MFRouter,
    "elorouter": EloRouter,
    "dcrouter": DCRouter,
    "routerdc": DCRouter,
    "smallest_llm": SmallestLLM,
    "largest_llm": LargestLLM,
    "llmmultiroundrouter": LLMMultiRoundRouter,
    "knnmultiroundrouter": KNNMultiRoundRouter,
    "automixrouter": AutomixRouter,
    "mhlc": GnosisLatentRouter,
    "gnosis_latent_router": GnosisLatentRouter,
    "gnosislatentrouter": GnosisLatentRouter,
}

# Add optional routers if available
if HybridLLMRouter is not None:
    ROUTER_REGISTRY["hybrid_llm"] = HybridLLMRouter
    ROUTER_REGISTRY["hybridllm"] = HybridLLMRouter

if GraphRouter is not None:
    ROUTER_REGISTRY["graphrouter"] = GraphRouter
    ROUTER_REGISTRY["graph_router"] = GraphRouter

if CausalLMRouter is not None:
    ROUTER_REGISTRY["causallm_router"] = CausalLMRouter
    ROUTER_REGISTRY["causallmrouter"] = CausalLMRouter

if GMTRouter is not None:
    ROUTER_REGISTRY["gmtrouter"] = GMTRouter
    ROUTER_REGISTRY["gmt_router"] = GMTRouter

if PersonalizedRouter is not None:
    ROUTER_REGISTRY["personalizedrouter"] = PersonalizedRouter
    ROUTER_REGISTRY["personalized_router"] = PersonalizedRouter

# Add RouterR1 if available
if RouterR1 is not None:
    ROUTER_REGISTRY["router_r1"] = RouterR1
    ROUTER_REGISTRY["router-r1"] = RouterR1

# Routers that have full pipeline in route_single (multi-round/agentic routers)
# These routers return response directly from route_single, no separate API call needed
MULTI_ROUND_ROUTERS = {
    "llmmultiroundrouter",
    "knnmultiroundrouter",
}

# Routers that require special handling
ROUTERS_REQUIRING_SPECIAL_ARGS = {
    "router_r1",
    "router-r1",
}

# Routers that are not supported
UNSUPPORTED_ROUTERS = {}


# ============================================================================
# Plugin System Integration
# ============================================================================
# Automatically discover and register custom routers from plugin directories
try:
    from llmrouter.plugin_system import discover_and_register_plugins

    # Discover plugins (verbose=False by default, set to True for debugging)
    plugin_registry = discover_and_register_plugins(verbose=False)

    # Register custom routers into ROUTER_REGISTRY
    for router_name, router_class in plugin_registry.discovered_routers.items():
        # Handle both (router, trainer) tuple and single router class
        if isinstance(router_class, tuple):
            ROUTER_REGISTRY[router_name] = router_class[0]  # Only router class for inference
        else:
            ROUTER_REGISTRY[router_name] = router_class

except ImportError:
    # Plugin system not available, continue without custom routers
    pass
# ============================================================================


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
            f"Router '{router_name}' is not supported for inference. "
            f"Supported routers: {list(ROUTER_REGISTRY.keys())}"
        )

    if router_name_lower not in ROUTER_REGISTRY:
        raise ValueError(
            f"Unknown router: {router_name}. Available routers: {list(ROUTER_REGISTRY.keys())}"
        )

    router_class = ROUTER_REGISTRY[router_name_lower]

    if load_model_path and router_class is GnosisLatentRouter:
        router = router_class(yaml_path=config_path)
        router.load_router(load_model_path)
        return router

    # Override model path in config if provided
    if load_model_path:
        # Read config, modify, write to temp file
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if "model_path" not in config:
            config["model_path"] = {}
        config["model_path"]["load_model_path"] = load_model_path

        # Write to temp config file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as temp_config:
            yaml.safe_dump(config, temp_config)
            config_path = temp_config.name
        atexit.register(_safe_unlink, config_path)

    # Initialize router
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


def route_query(
    query: str,
    router_instance: Any,
    router_name: str,
) -> Dict[str, Any]:
    """
    Route a single query and return routing decision.

    Args:
        query: Input query string
        router_instance: Loaded router instance
        router_name: Router method name

    Returns:
        Dictionary containing routing result
    """
    router_name_lower = router_name.lower()

    # Multi-round routers and RouterR1 don't support --route-only
    # because they execute the full pipeline internally
    if router_name_lower in ROUTERS_REQUIRING_SPECIAL_ARGS:
        return {
            "success": False,
            "query": query,
            "error": f"Router '{router_name}' does not support --route-only; run without --route-only.",
        }

    if router_name_lower in MULTI_ROUND_ROUTERS:
        return {
            "success": False,
            "query": query,
            "error": f"Router '{router_name}' is a multi-round router with full pipeline; --route-only is not supported.",
        }

    try:
        # Route the query
        query_input = {"query": query}
        routing_result = router_instance.route_single(query_input)

        # Extract model name from routing result
        model_name = (
            routing_result.get("model_name")
            or routing_result.get("predicted_llm")
            or routing_result.get("predicted_llm_name")
        )

        if not model_name:
            return {
                "success": False,
                "error": "Router did not return a model name",
                "routing_result": routing_result,
            }

        return {
            "success": True,
            "query": query,
            "model_name": model_name,
            "routing_result": routing_result,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def infer_query(
    query: str,
    router_instance: Any,
    router_name: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Perform full inference: route query + call API + return response.

    Args:
        query: Input query string
        router_instance: Loaded router instance
        router_name: Router method name
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation

    Returns:
        Dictionary containing inference result
    """
    router_name_lower = router_name.lower()

    # Check if router is a multi-round router (full pipeline in route_single)
    if router_name_lower in MULTI_ROUND_ROUTERS:
        # Multi-round routers do full pipeline: decompose + route + execute + aggregate
        # Their route_single returns response directly (string in chat mode, dict in eval mode)
        try:
            result = router_instance.route_single(query)
            # route_single returns string for simple query, dict for evaluation mode
            if isinstance(result, str):
                return {
                    "success": True,
                    "query": query,
                    "response": result,
                    "model_name": "multi-round-pipeline",
                    "method": "multi_round_router",
                }
            else:
                # Dict result with response, tokens, etc.
                return {
                    "success": result.get("success", True),
                    "query": query,
                    "response": result.get("response", ""),
                    "model_name": "multi-round-pipeline",
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                    "method": "multi_round_router",
                }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # Handle RouterR1 specially (requires model_id, api_base, api_key)
    if router_name_lower in ROUTERS_REQUIRING_SPECIAL_ARGS:
        try:
            # Get required parameters from config
            cfg = getattr(router_instance, "cfg", {}) or {}
            hparam = cfg.get("hparam", {}) or {}
            api_base = hparam.get("api_base") or getattr(router_instance, "api_base", None)
            api_key = hparam.get("api_key") or getattr(router_instance, "api_key", None)

            if not api_key or not api_base:
                return {
                    "success": False,
                    "query": query,
                    "error": "RouterR1 requires api_key and api_base in yaml config",
                }

            # RouterR1's route_single returns the response
            result = router_instance.route_single({"query": query})
            return {
                "success": True,
                "query": query,
                "response": result,
                "method": "router_r1",
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # Otherwise, use route_single to get routing decision, then call model
    try:
        # Route the query
        query_input = {"query": query}
        routing_result = router_instance.route_single(query_input)

        # Extract model name from routing result
        model_name = (
            routing_result.get("model_name")
            or routing_result.get("predicted_llm")
            or routing_result.get("predicted_llm_name")
        )

        if not model_name:
            return {
                "success": False,
                "query": query,
                "error": "Router did not return a model name",
                "routing_result": routing_result,
            }

        # Gnosis generates the target response before scoring its latent
        # trajectory. Reuse an accepted response instead of generating twice.
        if (
            ROUTER_REGISTRY.get(router_name_lower) is GnosisLatentRouter
            and routing_result.get("response") is not None
        ):
            return {
                "success": True,
                "query": query,
                "model_name": model_name,
                "response": routing_result["response"],
                "routing_result": routing_result,
                "method": routing_result.get("method", router_name_lower),
            }

        # Get API endpoint and model name from llm_data if available
        api_model_name = model_name  # Default to model_name
        api_endpoint = None
        service = None
        
        if hasattr(router_instance, 'llm_data') and router_instance.llm_data:
            if model_name in router_instance.llm_data:
                # Use the "model" field from llm_data which contains the full API path
                api_model_name = router_instance.llm_data[model_name].get("model", model_name)
                # Get API endpoint from llm_data, fallback to router config
                api_endpoint = router_instance.llm_data[model_name].get(
                    "api_endpoint",
                    router_instance.cfg.get("api_endpoint")
                )
                # Get service field for service-specific API key selection
                service = router_instance.llm_data[model_name].get("service")
            else:
                # If model_name not found, try to find it by matching model field
                for key, value in router_instance.llm_data.items():
                    if value.get("model") == model_name or key == model_name:
                        api_model_name = value.get("model", model_name)
                        # Get API endpoint from llm_data, fallback to router config
                        api_endpoint = value.get(
                            "api_endpoint",
                            router_instance.cfg.get("api_endpoint")
                        )
                        # Get service field for service-specific API key selection
                        service = value.get("service")
                        break
        
        # If still no endpoint found, try router config
        if api_endpoint is None:
            api_endpoint = router_instance.cfg.get("api_endpoint")
        
        # Validate that we have an endpoint
        if not api_endpoint:
            return {
                "success": False,
                "query": query,
                "error": f"API endpoint not found for model '{model_name}'. Please specify 'api_endpoint' in llm_data JSON for this model or in router YAML config.",
                "routing_result": routing_result,
            }

        # Call the routed model via API
        request = {
            "api_endpoint": api_endpoint,
            "query": query,
            "model_name": model_name,  # Keep original for router identification
            "api_name": api_model_name,  # Use full API model path
        }
        # Add service field if available (for service-specific API key selection)
        if service:
            request["service"] = service

        result = call_api(request, max_tokens=max_tokens, temperature=temperature)

        if result.get("error"):
            return {
                "success": False,
                "query": query,
                "model_name": model_name,
                "error": result["error"],
                "routing_result": routing_result,
            }

        response = result.get("response", "No response generated")

        return {
            "success": True,
            "query": query,
            "model_name": model_name,
            "api_model_name": api_model_name,
            "response": response,
            "routing_result": routing_result,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def load_queries_from_file(file_path: str) -> List[str]:
    """
    Load queries from a file.
    Supports:
    - Plain text file (one query per line)
    - JSON file (list of strings or list of dicts with "query" field)
    - JSONL file (one JSON object per line with "query" field)

    Args:
        file_path: Path to input file

    Returns:
        List of query strings
    """
    file_ext = Path(file_path).suffix.lower()

    with open(file_path, "r", encoding="utf-8") as f:
        if file_ext == ".json":
            data = json.load(f)
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    return data
                elif all(isinstance(item, dict) and "query" in item for item in data):
                    return [item["query"] for item in data]
                else:
                    raise ValueError("JSON file must contain list of strings or list of dicts with 'query' field")
            else:
                raise ValueError("JSON file must contain a list")

        elif file_ext == ".jsonl":
            queries = []
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "query" in obj:
                        queries.append(obj["query"])
                    elif isinstance(obj, str):
                        queries.append(obj)
                    else:
                        raise ValueError("JSONL line must be dict with 'query' field or string")
            return queries

        else:
            # Plain text file, one query per line
            return [line.strip() for line in f if line.strip()]


def save_results_to_file(results: List[Dict[str, Any]], output_path: str, output_format: str = "json"):
    """
    Save results to a file.

    Args:
        results: List of result dictionaries
        output_path: Path to output file
        output_format: Output format - "json" or "jsonl"
    """
    with open(output_path, "w", encoding="utf-8") as f:
        if output_format == "jsonl":
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:  # json
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point for router inference."""
    parser = argparse.ArgumentParser(
        description="Router Inference Script for LLMRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query inference
  python router_inference.py --router knnrouter --config config.yaml --query "What is machine learning?"

  # Batch inference from file
  python router_inference.py --router knnrouter --config config.yaml --input queries.txt --output results.json

  # Route only (no API call)
  python router_inference.py --router knnrouter --config config.yaml --query "Hello" --route-only
        """
    )

    # Required arguments
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

    # Query input
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        type=str,
        help="Single query string for inference",
    )
    query_group.add_argument(
        "--input",
        type=str,
        help="Path to input file containing queries (supports .txt, .json, .jsonl)",
    )

    # Optional arguments
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Optional path to override model_path.load_model_path in config",
    )
    parser.add_argument(
        "--route-only",
        action="store_true",
        help="Only perform routing without calling API (faster)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file (default: print to stdout)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format for batch inference (default: json)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Temperature for text generation (default: 0.8)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for generation (default: 1024)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load router
    if args.verbose:
        print(f"Loading router: {args.router}", file=sys.stderr)
        print(f"Using config: {args.config}", file=sys.stderr)
        if args.load_model_path:
            print(f"Overriding model path: {args.load_model_path}", file=sys.stderr)

    try:
        router_instance = load_router(args.router, args.config, args.load_model_path)
        if args.verbose:
            print("Router loaded successfully!", file=sys.stderr)
    except Exception as e:
        print(f"Error loading router: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine queries
    if args.query:
        queries = [args.query]
    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        try:
            queries = load_queries_from_file(args.input)
            if args.verbose:
                print(f"Loaded {len(queries)} queries from {args.input}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading queries from file: {e}", file=sys.stderr)
            sys.exit(1)

    # Process queries
    results = []
    for i, query in enumerate(queries):
        if args.verbose:
            print(f"\nProcessing query {i+1}/{len(queries)}: {query[:50]}...", file=sys.stderr)

        if args.route_only:
            result = route_query(query, router_instance, args.router)
        else:
            result = infer_query(
                query,
                router_instance,
                args.router,
                temperature=args.temp,
                max_tokens=args.max_tokens,
            )

        results.append(result)

        if args.verbose:
            if result["success"]:
                if args.route_only:
                    print(f"Routed to: {result.get('model_name')}", file=sys.stderr)
                else:
                    print(f"Response generated", file=sys.stderr)
            else:
                print(f"  └ Error: {result.get('error')}", file=sys.stderr)

    # Output results
    if args.output:
        try:
            save_results_to_file(results, args.output, args.output_format)
            if args.verbose:
                print(f"\nResults saved to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving results: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print to stdout
        if len(results) == 1:
            # Single query: print nicely formatted
            result = results[0]
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Multiple queries: print as JSON array
            print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
