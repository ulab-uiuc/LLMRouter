import argparse
import os
from llmrouter.models import DCRouter


def main():
    # Correct default path based on your folder structure
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_yaml = os.path.join(project_root, "configs", "model_config_test", "dcrouter.yaml")

    parser = argparse.ArgumentParser(
        description="Test the DCRouter with a YAML configuration file."
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
    print(f"Using YAML file: {args.yaml_path}")
    router = DCRouter(args.yaml_path)
    print("DCRouter initialized successfully!")

    # Run batch inference
    result = router.route_batch()
    print("\nBatch routing result:")
    print(f"  Total samples: {result['total']}")
    print(f"  Routing accuracy: {result['routing_accuracy']:.4f}")
    print(f"  Task accuracy: {result['task_accuracy']:.4f}")

    # Run single query inference
    result_single = router.route_single({"query": "How are you"})
    print("\nSingle query routing result:")
    print(f"  Query: {result_single['query']}")
    print(f"  Predicted LLM: {result_single['predicted_llm']}")
    print(f"  Routing scores:")
    for llm_name, score in result_single['routing_scores'].items():
        print(f"    {llm_name}: {score:.4f}")

    print("\nDCRouter inference completed successfully!")


if __name__ == "__main__":
    main()
