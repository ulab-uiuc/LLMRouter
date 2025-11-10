import argparse
import os
from llmrouter.models import SVMRouter
from llmrouter.models import SVMRouterTrainer


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # /data/taofeng2/LLMRouter
    default_yaml = os.path.join(project_root, "configs", "model_config_train", "svmrouter.yaml")

    parser = argparse.ArgumentParser(
        description="Test the SVMRouter with a YAML configuration file."
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
    router = SVMRouter(args.yaml_path)
    print("✅ SVMRouter initialized successfully!")

    # Run training
    trainer = SVMRouterTrainer(router=router, device="cpu")
    trainer.train()


if __name__ == "__main__":
    main()



