import argparse
import os

from llmrouter.models.RouterDC.main_routerdc import (
    load_config,
    preprocess_data,
    train_from_config,
)


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
    default_yaml = os.path.join(
        project_root, "configs", "model_config_test", "routerdc_nq.yaml"
    )

    parser = argparse.ArgumentParser(
        description="Train the RouterDC router with a YAML configuration file."
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
        help="Run RouterDC data preprocessing before training.",
    )
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(project_root, yaml_path)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    print(f"📄 Using YAML file: {yaml_path}")
    config = load_config(yaml_path)
    print("✅ Configuration loaded successfully!")

    if args.preprocess_data:
        print("\n🚀 Starting RouterDC data preprocessing...")
        preprocess_data(config)
        print("\n✅ Data preprocessing completed!")

    print("\n🚀 Starting RouterDC training run...")
    train_from_config(config)
    print("\n✅ RouterDC train_test completed successfully!")


if __name__ == "__main__":
    main()
