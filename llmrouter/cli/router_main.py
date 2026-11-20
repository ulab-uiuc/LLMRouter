"""
LLMRouter Main CLI Entry Point

This script provides a unified command-line interface for LLMRouter.
It integrates training, inference, and chat functionalities through subcommands.
"""

import argparse
import sys
from typing import List, Optional


def print_banner():
    """Print LLMRouter banner."""
    banner = """
    TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW
    Q                        LLMRouter                          Q
    Q          Intelligent Model Routing for LLMs               Q
    ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
    """
    print(banner)


def train_command(args):
    """Execute the train subcommand."""
    from llmrouter.cli.router_train import train_router, get_device

    # Determine device
    device = get_device(args.device if args.device != "auto" else None)

    # Train router
    verbose = not args.quiet
    try:
        train_router(
            router_name=args.router,
            config_path=args.config,
            device=device,
            verbose=verbose,
        )
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        if verbose:
            import traceback
            print("\nTraceback:", file=sys.stderr)
            traceback.print_exc()
        sys.exit(1)


def infer_command(args):
    """Execute the infer subcommand."""
    import os
    from llmrouter.cli.router_inference import (
        load_router,
        route_query,
        infer_query,
        load_queries_from_file,
        save_results_to_file,
    )

    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load router
    if args.verbose:
        print(f"Loading router: {args.router}", file=sys.stderr)
        print(f"Using config: {args.config}", file=sys.stderr)

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
                print(f"Error: {result.get('error')}", file=sys.stderr)

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
        import json
        if len(results) == 1:
            print(json.dumps(results[0], indent=2, ensure_ascii=False))
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))


def chat_command(args):
    """Execute the chat subcommand."""
    import os
    try:
        import gradio as gr
    except ImportError:
        print("Error: gradio is required for chat interface. Install it with:", file=sys.stderr)
        print("  pip install gradio", file=sys.stderr)
        sys.exit(1)

    from llmrouter.cli.router_chat import load_router, predict

    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load router
    print(f"Loading router: {args.router}")
    print(f"Using config: {args.config}")
    if args.load_model_path:
        print(f"Overriding model path: {args.load_model_path}")

    try:
        router_instance = load_router(args.router, args.config, args.load_model_path)
        print("Router loaded successfully!")
    except Exception as e:
        print(f"Error loading router: {e}", file=sys.stderr)
        sys.exit(1)

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

    interface.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


def list_routers_command(args):
    """Execute the list-routers subcommand."""
    _ = args  # Unused, kept for API consistency
    from llmrouter.cli.router_train import ROUTER_TRAINER_REGISTRY, UNSUPPORTED_ROUTERS
    from llmrouter.cli.router_inference import ROUTER_REGISTRY

    print("\n" + "=" * 70)
    print("AVAILABLE ROUTERS")
    print("=" * 70)

    # All routers (for inference)
    print("\nRouters available for INFERENCE:")
    print("-" * 70)
    for router_name in sorted(ROUTER_REGISTRY.keys()):
        router_class = ROUTER_REGISTRY[router_name]
        print(f"  {router_name:25s} - {router_class.__name__}")

    # Trainable routers
    print("\nRouters available for TRAINING:")
    print("-" * 70)
    for router_name in sorted(ROUTER_TRAINER_REGISTRY.keys()):
        router_class, trainer_class = ROUTER_TRAINER_REGISTRY[router_name]
        print(f"  {router_name:25s} - {router_class.__name__} / {trainer_class.__name__}")

    # Non-trainable routers
    if UNSUPPORTED_ROUTERS:
        print("\nRouters NOT available for training:")
        print("-" * 70)
        for router_name, reason in sorted(UNSUPPORTED_ROUTERS.items()):
            print(f"  {router_name:25s} - {reason}")

    print("\n" + "=" * 70)


def version_command(args):
    """Execute the version subcommand."""
    _ = args  # Unused, kept for API consistency
    try:
        import llmrouter
        version = getattr(llmrouter, "__version__", "unknown")
    except Exception:
        version = "unknown"

    print(f"LLMRouter version: {version}")


def gnosis_latent_router_command(args):
    """Execute a Gnosis Latent Router pipeline stage."""
    import json

    from llmrouter.models.gnosis_latent_router.pipeline import (
        run_gnosis_pipeline,
    )

    try:
        result = run_gnosis_pipeline(
            config_path=args.config,
            action=args.gnosis_action,
            query=getattr(args, "query", None),
            workspace=getattr(args, "workspace", None),
            dry_run=getattr(args, "dry_run", False),
            route_only=getattr(args, "route_only", False),
            temperature=getattr(args, "temp", 0.8),
            max_tokens=getattr(args, "max_tokens", 1024),
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("inference", {}).get("success") is False:
        sys.exit(1)


def serve_command(args):
    """Execute the serve subcommand."""
    import os

    # Check dependencies
    try:
        import fastapi
        import uvicorn
        import httpx
    except ImportError:
        print("Error: FastAPI dependencies required. Install with:", file=sys.stderr)
        print("  pip install fastapi uvicorn httpx", file=sys.stderr)
        sys.exit(1)

    from openclaw_router import run_server, create_app, OpenClawConfig

    # Load config
    config = None
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        config = OpenClawConfig.from_yaml(args.config)
    else:
        config = OpenClawConfig()

    # Override config with CLI args
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.router:
        config.router.strategy = "llmrouter"
        config.router.llmrouter_name = args.router
    if args.router_config:
        config.router.llmrouter_config = args.router_config
    if args.no_prefix:
        config.show_model_prefix = False
        config.show_model_suffix = False

    # Create and run app
    app = create_app(config=config)
    run_server(app, host=config.host, port=config.port)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    try:
        import llmrouter
        cli_version = getattr(llmrouter, "__version__", "unknown")
    except Exception:
        cli_version = "unknown"

    parser = argparse.ArgumentParser(
        description="LLMRouter - Intelligent Model Routing for Large Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a router
  llmrouter train --router knnrouter --config configs/model_config_train/knnrouter.yaml

  # Perform inference
  llmrouter infer --router knnrouter --config config.yaml --query "What is AI?"

  # Launch chat interface
  llmrouter chat --router knnrouter --config config.yaml

  # Start OpenAI-compatible API server (OpenClaw Router for OpenClaw integration)
  llmrouter serve --config configs/openclaw_example.yaml

  # List all available routers
  llmrouter list-routers

For more information on each subcommand, use:
  llmrouter <subcommand> --help
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"LLMRouter CLI v{cli_version}",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Available commands",
        dest="command",
        help="Command to execute",
    )

    # ========== TRAIN SUBCOMMAND ==========
    train_parser = subparsers.add_parser(
        "train",
        help="Train a router model",
        description="Train a router model with the specified configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--router",
        type=str,
        required=True,
        help="Router method name (e.g., knnrouter, mlprouter)",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file for training",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Device to use for training (default: auto-detect)",
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    train_parser.set_defaults(func=train_command)

    # ========== INFER SUBCOMMAND ==========
    infer_parser = subparsers.add_parser(
        "infer",
        help="Perform inference with a router",
        description="Perform inference using a trained router model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer_parser.add_argument(
        "--router",
        type=str,
        required=True,
        help="Router method name",
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    # Query input (mutually exclusive)
    query_group = infer_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        type=str,
        help="Single query string for inference",
    )
    query_group.add_argument(
        "--input",
        type=str,
        help="Path to input file containing queries",
    )

    infer_parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Optional path to override model_path.load_model_path in config",
    )
    infer_parser.add_argument(
        "--route-only",
        action="store_true",
        help="Only perform routing without calling API",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file (default: print to stdout)",
    )
    infer_parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format for batch inference (default: json)",
    )
    infer_parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Temperature for text generation (default: 0.8)",
    )
    infer_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for generation (default: 1024)",
    )
    infer_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    infer_parser.set_defaults(func=infer_command)

    # ========== CHAT SUBCOMMAND ==========
    chat_parser = subparsers.add_parser(
        "chat",
        help="Launch interactive chat interface",
        description="Launch a Gradio-based chat interface for interactive conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    chat_parser.add_argument(
        "--router",
        type=str,
        required=True,
        help="Router method name",
    )
    chat_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    chat_parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Optional path to override model_path.load_model_path in config",
    )
    chat_parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Default temperature for text generation (default: 0.8)",
    )
    chat_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server to (default: None, all interfaces)",
    )
    chat_parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server to (default: 8001)",
    )
    chat_parser.add_argument(
        "--mode",
        type=str,
        default="current_only",
        choices=["full_context", "current_only", "retrieval"],
        help="Default query mode (default: current_only)",
    )
    chat_parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of similar queries to retrieve in retrieval mode (default: 3)",
    )
    chat_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link",
    )
    chat_parser.set_defaults(func=chat_command)

    # ========== LIST-ROUTERS SUBCOMMAND ==========
    list_parser = subparsers.add_parser(
        "list-routers",
        help="List all available routers",
        description="Display all available router models and their capabilities",
    )
    list_parser.set_defaults(func=list_routers_command)

    # ========== VERSION SUBCOMMAND ==========
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
        description="Display LLMRouter version information",
    )
    version_parser.set_defaults(func=version_command)

    # ========== SERVE SUBCOMMAND ==========
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start OpenAI-compatible API server (OpenClaw Router)",
        description="Start an OpenAI-compatible API server with intelligent routing. "
                    "Supports built-in strategies (random, rules, round_robin, llm) and "
                    "LLMRouter ML-based routers (knnrouter, mlprouter, thresholdrouter, etc.). "
                    "Can be directly integrated with OpenClaw and other OpenAI-compatible clients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with a config file
  llmrouter serve --config configs/openclaw_example.yaml

  # Use a specific LLMRouter ML-based router
  llmrouter serve --config config.yaml --router knnrouter --router-config configs/knnrouter.yaml

  # Custom port
  llmrouter serve --config config.yaml --port 9000

Routing Strategies (in config.yaml):
  Built-in:
    - random: Random model selection (with optional weights)
    - round_robin: Rotate through models
    - rules: Keyword-based routing
    - llm: Use an LLM to decide

  LLMRouter ML-based (use --router flag or strategy: llmrouter):
    - knnrouter, mlprouter, svmrouter, mfrouter
    - thresholdrouter, randomrouter (custom_routers/)
    - And more...

OpenClaw Integration:
  In ~/.openclaw/openclaw.json, add:
  {
    "models": {
      "providers": {
        "openclaw": {
          "api": "openai-completions",
          "baseUrl": "http://localhost:8000/v1",
          "apiKey": "not-needed",
          "models": [{"id": "auto", "name": "OpenClaw Router"}]
        }
      }
    },
    "agents": {
      "defaults": {
        "model": {"primary": "openclaw/auto"}
      }
    }
  }
        """
    )
    serve_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to bind the server to (default: 8000)",
    )
    serve_parser.add_argument(
        "--router",
        type=str,
        default=None,
        help="Router to use (e.g., randomrouter, thresholdrouter)",
    )
    serve_parser.add_argument(
        "--router-config",
        type=str,
        default=None,
        help="Path to router-specific config file",
    )
    serve_parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Disable model prefix and suffix attribution in responses",
    )
    serve_parser.set_defaults(func=serve_command)

    # ========== GNOSIS LATENT ROUTER SUBCOMMAND ==========
    gnosis_parser = subparsers.add_parser(
        "gnosis-latent-router",
        aliases=["mhlc"],
        help="Generate data, train, benchmark, and run Multi-Head Latent Control (MHLC)",
        description=(
            "Run individual Gnosis Latent Router stages or the complete "
            "generate → label → train → benchmark workflow."
        ),
    )
    gnosis_subparsers = gnosis_parser.add_subparsers(
        dest="gnosis_action", required=True
    )
    for action in ("generate", "label", "train", "infer", "benchmark", "run"):
        action_parser = gnosis_subparsers.add_parser(
            action,
            help=(
                "Run the complete pipeline"
                if action == "run"
                else f"Run the {action} stage"
            ),
        )
        action_parser.add_argument(
            "--config", required=True, help="Gnosis Latent Router YAML config"
        )
        action_parser.add_argument(
            "--workspace",
            help=(
                "Writable root for data, checkpoints, and evaluation outputs "
                "(overrides config and GNOSIS_WORKSPACE)"
            ),
        )
        if action != "infer":
            action_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Print resolved Gnosis command(s) without executing them",
            )
        if action in {"infer", "run"}:
            action_parser.add_argument(
                "--query",
                required=action == "infer",
                help=(
                    "Optional query after benchmark evaluation"
                    if action == "run"
                    else "Query to route"
                ),
            )
            action_parser.add_argument(
                "--route-only",
                action="store_true",
                help="Return the routing decision without calling the fallback API",
            )
            action_parser.add_argument("--temp", type=float, default=0.8, help="Fallback sampling temperature")
            action_parser.add_argument("--max-tokens", type=int, default=1024, help="Fallback output token limit")
        action_parser.add_argument(
            "--verbose", action="store_true", help="Show tracebacks"
        )
        action_parser.set_defaults(func=gnosis_latent_router_command)

    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point for LLMRouter CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # If no subcommand is provided, show help
    if not hasattr(args, "func"):
        print_banner()
        parser.print_help()
        sys.exit(0)

    # Execute the appropriate subcommand
    args.func(args)


if __name__ == "__main__":
    main()
