import argparse
import os

# Import the updated Hybrid LLM router + trainer
from llmrouter.models import HybridLLMRouter, HybridLLMTrainer


def main():
    # Resolve project root: /data/taofeng2/LLMRouter
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # The default YAML used for training (you can rename it as you like)
    default_yaml = os.path.join(
        project_root, "configs", "model_config_train", "hybrid_llm.yaml"
    )

    parser = argparse.ArgumentParser(
        description="Train the HybridLLMRouter with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    args = parser.parse_args()

    # Validate YAML path
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    print(f"📄 Using YAML file: {args.yaml_path}")

    # Initialize HybridLLMRouter
    router = HybridLLMRouter(args.yaml_path)
    print("✅ HybridLLMRouter initialized successfully!")

    # Train model
    trainer = HybridLLMTrainer(router=router, device="cpu")
    print("🚀 Starting Hybrid LLM model training...")
    trainer.train()
    print("🎯 Training completed successfully!")


if __name__ == "__main__":
    main()



