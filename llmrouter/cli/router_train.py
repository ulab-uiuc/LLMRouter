"""
Router Training Script for LLMRouter

This script provides a unified CLI interface for training different router models.
It supports all router types that have corresponding trainer implementations.
"""

import argparse
import os
import sys
from typing import Dict, Any, Optional, Tuple

# Import router and trainer classes
from llmrouter.models import (
    # Routers
    KNNRouter,
    SVMRouter,
    MLPRouter,
    RACERRouter,
    RACERTrainer,
    MFRouter,
    EloRouter,
    DCRouter,
    AutomixRouter,
    HybridLLMRouter,
    GraphRouter,
    CausalLMRouter,
    GMTRouter,
    PersonalizedRouter,
    GnosisLatentRouter,
    # Trainers
    KNNRouterTrainer,
    SVMRouterTrainer,
    MLPTrainer,
    MFRouterTrainer,
    EloRouterTrainer,
    DCTrainer,
    AutomixRouterTrainer,
    HybridLLMTrainer,
    GraphTrainer,
    CausalLMTrainer,
    GMTRouterTrainer,
    PersonalizedRouterTrainer,
    GnosisLatentRouterTrainer,
)

# Import multi-round routers
from llmrouter.models.knnmultiroundrouter import KNNMultiRoundRouter
from llmrouter.models.knnmultiroundrouter import KNNMultiRoundRouterTrainer


# Router registry: maps router method names to (router_class, trainer_class) tuples
ROUTER_TRAINER_REGISTRY: Dict[str, Tuple[Any, Any]] = {
    "racerrouter": (RACERRouter, RACERTrainer),
    "knnrouter": (KNNRouter, KNNRouterTrainer),
    "svmrouter": (SVMRouter, SVMRouterTrainer),
    "mlprouter": (MLPRouter, MLPTrainer),
    "mfrouter": (MFRouter, MFRouterTrainer),
    "elorouter": (EloRouter, EloRouterTrainer),
    "dcrouter": (DCRouter, DCTrainer),
    "routerdc": (DCRouter, DCTrainer),
    "automix": (AutomixRouter, AutomixRouterTrainer),
    "automixrouter": (AutomixRouter, AutomixRouterTrainer),
    "hybrid_llm": (HybridLLMRouter, HybridLLMTrainer),
    "hybridllm": (HybridLLMRouter, HybridLLMTrainer),
    "graphrouter": (GraphRouter, GraphTrainer),
    "causallm_router": (CausalLMRouter, CausalLMTrainer),
    "causallmrouter": (CausalLMRouter, CausalLMTrainer),
    "knnmultiroundrouter": (KNNMultiRoundRouter, KNNMultiRoundRouterTrainer),
    "gmtrouter": (GMTRouter, GMTRouterTrainer),
    "gmt_router": (GMTRouter, GMTRouterTrainer),
    "personalizedrouter": (PersonalizedRouter, PersonalizedRouterTrainer),
    "personalized_router": (PersonalizedRouter, PersonalizedRouterTrainer),
    "mhlc": (GnosisLatentRouter, GnosisLatentRouterTrainer),
    "gnosis_latent_router": (GnosisLatentRouter, GnosisLatentRouterTrainer),
    "gnosislatentrouter": (GnosisLatentRouter, GnosisLatentRouterTrainer),
}   

# Routers that do not support training
UNSUPPORTED_ROUTERS = {
    "smallest_llm": "SmallestLLM is a baseline router that does not require training",
    "largest_llm": "LargestLLM is a baseline router that does not require training",
    "llmmultiroundrouter": "LLMMultiRoundRouter does not have a trainer implementation",
    "router_r1": "RouterR1 is a pre-trained model and does not support training via this CLI",
    "router-r1": "RouterR1 is a pre-trained model and does not support training via this CLI",
}

# Filter out routers whose optional deps are unavailable
_optional_missing = []
for _name, (_router_cls, _trainer_cls) in list(ROUTER_TRAINER_REGISTRY.items()):
    if _router_cls is None or _trainer_cls is None:
        _optional_missing.append(_name)
        ROUTER_TRAINER_REGISTRY.pop(_name, None)

for _name in _optional_missing:
    UNSUPPORTED_ROUTERS[_name] = (
        "Optional dependencies missing for this router/trainer; "
        "install the extra requirements and try again."
    )


# ============================================================================
# Plugin System Integration
# ============================================================================
# Automatically discover and register custom routers from plugin directories
try:
    from llmrouter.plugin_system import discover_and_register_plugins

    # Discover plugins (verbose=False by default, set to True for debugging)
    plugin_registry = discover_and_register_plugins(verbose=False)

    # Register custom routers into ROUTER_TRAINER_REGISTRY
    for router_name, (router_class, trainer_class) in plugin_registry.discovered_routers.items():
        if trainer_class is not None:
            # Router has a trainer, add to training registry
            ROUTER_TRAINER_REGISTRY[router_name] = (router_class, trainer_class)
        else:
            # Router has no trainer, mark as unsupported for training
            UNSUPPORTED_ROUTERS[router_name] = (
                "Custom router does not have a trainer implementation"
            )

except ImportError:
    # Plugin system not available, continue without custom routers
    pass
# ============================================================================


def get_device(device_arg: Optional[str] = None) -> str:
    """
    Determine the device to use for training.

    Args:
        device_arg: Optional device argument from CLI

    Returns:
        Device string ("cuda" or "cpu")
    """
    if device_arg:
        return device_arg

    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


def load_router_and_trainer(
    router_name: str,
    config_path: str,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """
    Load router and trainer instances based on router name and config.

    Args:
        router_name: Name of the router method (e.g., "knnrouter", "mlprouter")
        config_path: Path to YAML configuration file
        device: Device to use for training ("cuda" or "cpu")

    Returns:
        Tuple of (router_instance, trainer_instance)
    """
    router_name_lower = router_name.lower()

    # Check if router is unsupported
    if router_name_lower in UNSUPPORTED_ROUTERS:
        raise ValueError(
            f"Router '{router_name}' does not support training.\n"
            f"Reason: {UNSUPPORTED_ROUTERS[router_name_lower]}"
        )

    if router_name_lower not in ROUTER_TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown router: {router_name}.\n"
            f"Supported routers for training: {list(ROUTER_TRAINER_REGISTRY.keys())}"
        )

    router_class, trainer_class = ROUTER_TRAINER_REGISTRY[router_name_lower]

    # Initialize router
    try:
        router_instance = router_class(yaml_path=config_path)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize router '{router_name}'.\n"
            f"Error: {str(e)}"
        ) from e

    # Initialize trainer
    try:
        trainer_instance = trainer_class(router=router_instance, device=device)
    except Exception as e:
        raise ValueError(
            f"Failed to initialize trainer for '{router_name}'.\n"
            f"Error: {str(e)}"
        ) from e

    return router_instance, trainer_instance


def train_router(
    router_name: str,
    config_path: str,
    device: str = "cpu",
    verbose: bool = True,
) -> None:
    """
    Train a router with the given configuration.

    Args:
        router_name: Name of the router method
        config_path: Path to YAML configuration file
        device: Device to use for training
        verbose: Whether to print verbose output
    """
    if verbose:
        print(f"=" * 60)
        print(f"Starting Training for Router: {router_name}")
        print(f"=" * 60)
        print(f"Config file: {config_path}")
        print(f"Device: {device}")
        print(f"=" * 60)

    # Load router and trainer
    if verbose:
        print("\nLoading router and trainer...")

    _, trainer_instance = load_router_and_trainer(
        router_name, config_path, device
    )

    if verbose:
        print("Router and trainer loaded successfully!")

    # Train
    if verbose:
        print(f"\nStarting training for {router_name}...\n")

    try:
        trainer_instance.train()
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}") from e

    if verbose:
        print(f"\nTraining completed for {router_name}!")
        print(f"=" * 60)


def main():
    """Main entry point for router training."""
    parser = argparse.ArgumentParser(
        description="Router Training Script for LLMRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train KNN router
  python router_train.py --router knnrouter --config configs/model_config_train/knnrouter.yaml

  # Train MLP router with GPU
  python router_train.py --router mlprouter --config configs/model_config_train/mlprouter.yaml --device cuda

  # Train MF router quietly
  python router_train.py --router mfrouter --config configs/model_config_train/mfrouter.yaml --quiet

Supported routers for training:
  - knnrouter: K-Nearest Neighbors Router
  - svmrouter: Support Vector Machine Router
  - mlprouter: Multi-Layer Perceptron Router
  - mfrouter: Matrix Factorization Router
  - elorouter: Elo Rating Router
  - dcrouter: Divide-and-Conquer Router
  - automix: Automix Router
  - hybrid_llm: Hybrid LLM Router
  - graphrouter: Graph Router
  - causallm_router: Causal Language Model Router
  - knnmultiroundrouter: KNN Multi-Round Router
        """
    )

    # Required arguments
    parser.add_argument(
        "--router",
        type=str,
        required=True,
        help="Router method name (e.g., knnrouter, mlprouter, mfrouter)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file for training",
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "auto"],
        help="Device to use for training (default: auto-detect)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output (only show errors)",
    )
    parser.add_argument(
        "--list-routers",
        action="store_true",
        help="List all supported routers for training and exit",
    )

    args = parser.parse_args()

    # Handle --list-routers
    if args.list_routers:
        print("Supported routers for training:")
        print("=" * 60)
        for router_name in sorted(ROUTER_TRAINER_REGISTRY.keys()):
            router_class, trainer_class = ROUTER_TRAINER_REGISTRY[router_name]
            print(f"  • {router_name}")
            print(f"    Router: {router_class.__name__}")
            print(f"    Trainer: {trainer_class.__name__}")
            print()
        print("Unsupported routers:")
        print("=" * 60)
        for router_name, reason in sorted(UNSUPPORTED_ROUTERS.items()):
            print(f"  • {router_name}")
            print(f"    Reason: {reason}")
            print()
        sys.exit(0)

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

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
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if verbose:
            import traceback
            print("\nTraceback:", file=sys.stderr)
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
