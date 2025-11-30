import argparse
import os

from torch.utils.data import DataLoader

from llmrouter.models import RouterDCRouter, RouterDataset
from llmrouter.models.RouterDC.utils import load_tokenizer_and_backbone


def _resolve_path(path: str, root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_yaml = os.path.join(
        project_root, "configs", "model_config_test", "routerdc_nq.yaml"
    )

    parser = argparse.ArgumentParser(
        description="RouterDC inference smoke test."
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
    router = RouterDCRouter(args.yaml_path)
    print("✅ RouterDCRouter initialized successfully!")

    cfg = router.cfg or {}
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    train_data_path = _resolve_path(data_cfg["train_output_path"], project_root)
    data_type = eval_cfg.get("test_data_type", "probability")

    dataset = RouterDataset(
        data_path=train_data_path,
        data_type=data_type,
        dataset_id=0,
    )

    tokenizer = getattr(router.model, "tokenizer", None)
    if tokenizer is None:
        tokenizer, _, _ = load_tokenizer_and_backbone(cfg["model"])
    dataset.register_tokenizer(tokenizer)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    inputs, scores, dataset_ids, cluster_ids = next(iter(loader))

    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "temperature": cfg.get("training", {}).get("temperature", 1.0),
        "true_scores": scores,
        "data_type": data_type,
    }

    outputs = router(batch)
    metrics = router.compute_metrics(outputs, batch)

    print("Routing predictions:", outputs["predictions"])
    print(
        f"Routing accuracy: {metrics['routing_accuracy']:.2f}%, "
        f"Task accuracy: {metrics['task_accuracy']:.2f}%"
    )
    print("✅ RouterDC inference test completed successfully!")


if __name__ == "__main__":
    main()
