import argparse
import os
import sys

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmrouter.main_routerdc import load_config, preprocess_data, train_from_config


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(__file__))  # LLMRouter
    default_yaml = os.path.join(project_root, "llmrouter", "configs", "model_config_test", "routerdc_nq.yaml")

    parser = argparse.ArgumentParser(
        description="Test the RouterDC router with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    parser.add_argument(
        "--preprocess_data",
        action="store_true",
        help="Run data preprocessing before training",
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

    # Run data preprocessing if requested
    if args.preprocess_data:
        print("\n🚀 Starting RouterDC data preprocessing...")
        try:
            preprocess_data(config)
            print("\n✅ Data preprocessing completed successfully!")
            print("You can now run training without --preprocess_data flag")
        except Exception as e:
            print(f"\n❌ Error during data preprocessing: {e}")
            raise
        return

    # Run training
    print("\n🚀 Starting RouterDC router training...")
    try:
        train_from_config(config)
        print("\n✅ RouterDC router test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Hint: If data preprocessing is needed, run with --preprocess_data flag first")
        raise


if __name__ == "__main__":
    main()

