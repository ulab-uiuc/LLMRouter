import argparse
import os
from llmrouter.models import HybridLLMRouter


def main():
    # Resolve project root: /data/taofeng2/LLMRouter
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Default YAML for inference (test)
    default_yaml = os.path.join(
        project_root,
        "configs",
        "model_config_test",
        "hybrid_llm.yaml",
    )

    parser = argparse.ArgumentParser(
        description="Test the HybridLLMRouter with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    args = parser.parse_args()

    # Verify YAML path
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    print(f"📄 Using YAML file: {args.yaml_path}")

    # Initialize router
    router = HybridLLMRouter(args.yaml_path)
    print("✅ HybridLLMRouter initialized successfully!")

    # --- Batch routing ---
    print("\n🧠 Running batch routing on test queries...")
    batch_results = router.route_batch()
    print("🧩 Batch routing output:")
    for item in batch_results[:5]:  # print first 5 only
        print(item)

    # --- Single routing ---
    print("\n💬 Running single query routing...")
    result_single = router.route_single({"query": "How are you?"})
    print("🎯 Single query routing result:")
    print(result_single)


if __name__ == "__main__":
    main()


