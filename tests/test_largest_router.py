import argparse
import os
from llmrouter.models.largest_llm import LargestLLM


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(__file__))  # /data/taofeng2/LLMRouter
    default_yaml = os.path.join(project_root, "llmrouter", "configs", "model_config_test", "largest_llm.yaml")

    parser = argparse.ArgumentParser(
        description="Test the LargestLLM router with a YAML configuration file."
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
    llm = LargestLLM(args.yaml_path)
    print("✅ LargestLLM initialized successfully!")

    # Run inference
    result = llm.inference()
    print("\n🚀 Largest model selected:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
