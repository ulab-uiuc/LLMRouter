import argparse
import os
from llmrouter.models import MLPRouter
from llmrouter.models import MLPTrainer


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # /data/taofeng2/LLMRouter
    default_yaml = os.path.join(project_root, "configs", "model_config_train", "mlprouter.yaml")

    parser = argparse.ArgumentParser(
        description="Train and test the MLPRouter with a YAML configuration file."
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

    # Initialize the router
    print(f"📄 Using YAML file: {args.yaml_path}")
    router = MLPRouter(args.yaml_path)
    print("✅ MLPRouter initialized successfully!")

    # Run training
    trainer = MLPTrainer(router=router, device="cpu")
    print("🚀 Starting MLP model training...")
    trainer.train()
    print("🎯 Training completed successfully!")


if __name__ == "__main__":
    main()


