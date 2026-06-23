#!/usr/bin/env python3
"""
OpenClaw Router - Main Entry Point
Run with: python -m openclaw_router --config config.yaml

Examples:
    python -m openclaw_router --config config.yaml
    python -m openclaw_router --config config.yaml --router knnrouter
    python -m openclaw_router --config config.yaml --router randomrouter --port 9000
"""

import argparse

from .server import create_app, run_server
from .config import OpenClawConfig


def main():
    parser = argparse.ArgumentParser(
        description="OpenClaw Router - OpenAI-compatible API with intelligent routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Routers:
  Built-in:
    - random: Random model selection
    - round_robin: Rotate through models
    - rules: Keyword-based routing
    - llm: Use an LLM to decide

  LLMRouter ML-based:
    - knnrouter, mlprouter, svmrouter, mfrouter
    - thresholdrouter, randomrouter (custom_routers/)
"""
    )
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--router", "-r", help="LLMRouter name (e.g., knnrouter, randomrouter)")
    parser.add_argument("--router-config", help="Router config file path")
    parser.add_argument("--no-prefix", action="store_true", help="Don't add model name prefix to responses")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = OpenClawConfig.from_yaml(args.config)
    else:
        config = OpenClawConfig()

    # Override with CLI args
    config.host = args.host
    config.port = args.port

    # Set router if specified
    if args.router:
        # Built-in strategies
        builtin_strategies = ["random", "round_robin", "rules", "llm"]
        if args.router in builtin_strategies:
            config.router.strategy = args.router
            print(f"[Config] Using built-in strategy: {args.router}")
        else:
            # LLMRouter plugin
            config.router.strategy = "llmrouter"
            config.router.llmrouter_name = args.router
            print(f"[Config] Using router: {args.router}")

    if args.router_config:
        config.router.llmrouter_config = args.router_config

    if args.no_prefix:
        config.show_model_prefix = False

    # Run server
    app = create_app(config=config)
    run_server(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
