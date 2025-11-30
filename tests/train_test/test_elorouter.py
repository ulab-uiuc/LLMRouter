import argparse
import os

from llmrouter.models import EloRouter
from llmrouter.models import EloRouterTrainer


def main():
    # Compute project root: same as your original script logic
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Default YAML path for training EloRouter
    default_yaml = os.path.join(
        project_root,
        "configs",
        "model_config_train",
        "elorouter.yaml"
    )

    parser = argparse.ArgumentParser(
        description="Train the EloRouter with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    args = parser.parse_args()

    # Verify YAML exists
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    print(f"📄 Using YAML file: {args.yaml_path}")

    # Initialize router
    router = EloRouter(args.yaml_path)
    print("✅ EloRouter initialized successfully!")

    # Train Elo model
    trainer = EloRouterTrainer(router=router, device="cpu")
    print("🚀 Starting Elo model training...")
    trainer.train()
    print("🎯 Elo training completed successfully!")


if __name__ == "__main__":
    main()


