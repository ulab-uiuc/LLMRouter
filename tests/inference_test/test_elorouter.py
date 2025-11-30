import argparse
import os
from llmrouter.models import EloRouter


def main():
    # Resolve project root (same logic as your original test script)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_yaml = os.path.join(
        project_root,
        "configs",
        "model_config_test",
        "elorouter.yaml"
    )

    parser = argparse.ArgumentParser(
        description="Test the EloRouter with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    args = parser.parse_args()

    # Ensure YAML file exists
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    print(f"📄 Using YAML file: {args.yaml_path}")

    # Initialize router (note: no model is loaded here)
    router = EloRouter(args.yaml_path)
    print("✅ EloRouter initialized successfully!")

    # Perform batch inference
    print("🚀 Running batch routing...")
    batch_output = router.route_batch()
    print("🧠 Batch routing result:")
    print(batch_output)

    # Perform single-query inference
    print("🚀 Running single-query routing...")
    single_output = router.route_single({"query": "How are you"})
    print("💬 Single query routing result:")
    print(single_output)


if __name__ == "__main__":
    main()


