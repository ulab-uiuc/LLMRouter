import argparse
import os

from llmrouter.models import AutomixRouter, AutomixRouterTrainer


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_yaml = os.path.join(
        project_root, "configs", "model_config_train", "automix.yaml"
    )

    parser = argparse.ArgumentParser(
        description="Train the Automix router with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    args = parser.parse_args()

    # Verify file existence
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    # Initialize the router (automatically handles data preparation)
    print(f"Using YAML file: {args.yaml_path}")
    router = AutomixRouter(args.yaml_path)
    print("AutomixRouter initialized successfully!")

    # Run training
    trainer = AutomixRouterTrainer(router=router)
    print("Starting Automix router training...")
    trainer.train()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
