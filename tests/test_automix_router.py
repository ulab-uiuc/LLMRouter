import argparse
import os
import sys

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmrouter.main_automix import load_config, train_and_evaluate


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(__file__))  # LLMRouter
    default_yaml = os.path.join(project_root, "llmrouter", "configs", "model_config_test", "automix_config.yaml")

    parser = argparse.ArgumentParser(
        description="Test the Automix router with a YAML configuration file."
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

    # Load configuration
    print(f"📄 Using YAML file: {args.yaml_path}")
    try:
        config = load_config(args.yaml_path)
        print("✅ Configuration loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return

    # Run training and evaluation
    print("\n🚀 Starting Automix router training and evaluation...")
    try:
        train_and_evaluate(config)
        print("\n✅ Automix router test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training/evaluation: {e}")
        raise


if __name__ == "__main__":
    main()

