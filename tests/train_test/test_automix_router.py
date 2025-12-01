import argparse
import os

from llmrouter.models.Automix.main_automix import (
    load_config,
    train_and_evaluate,
    convert_default_data,
)


def main():
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

    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    print(f"📄 Using YAML file: {args.yaml_path}")
    config = load_config(args.yaml_path)
    print("✅ Configuration loaded successfully!")

    data_cfg = config["data_path"]
    data_path = data_cfg.get("prepared_data", "data/automix/router_automix_llamapair_ver_outputs.jsonl")
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

    if not os.path.exists(data_path):
        print("⚠️ Automix data not found. Converting default data...")
        new_path = convert_default_data(config)
        config["data_path"]["prepared_data"] = os.path.relpath(new_path, project_root)
        data_path = new_path
        print(f"✅ Converted default data to: {data_path}")

    print("\n🚀 Starting Automix router training and evaluation...")
    results = train_and_evaluate(config)
    if results is None:
        raise RuntimeError("Automix training did not complete successfully.")
    print("\n✅ Automix router train_test completed successfully!")


if __name__ == "__main__":
    main()
