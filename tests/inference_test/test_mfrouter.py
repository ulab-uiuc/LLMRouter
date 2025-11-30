import argparse
import os
from llmrouter.models import MFRouter


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # e.g. /data/taofeng2/LLMRouter
    default_yaml = os.path.join(project_root, "configs", "model_config_test", "mfrouter.yaml")

    parser = argparse.ArgumentParser(
        description="Test the MFRouter with a YAML configuration file."
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

    # Initialize router
    print(f"📄 Using YAML file: {args.yaml_path}")
    router = MFRouter(args.yaml_path)
    print("✅ MFRouter initialized successfully!")

    # Run inference on batch
    result_batch = router.route_batch()
    print("🧠 Batch routing result:")
    for i, item in enumerate(result_batch[:5]):  # print only first 5 for brevity
        print(f"[{i}] → {item['model_name']}")

    # Run single query inference
    result_single = router.route_single({"query": "Explain reinforcement learning briefly."})
    print("💬 Single query routing result:")
    print(result_single)


if __name__ == "__main__":
    main()



