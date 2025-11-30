import argparse
import os

import torch
from torch.utils.data import DataLoader

from llmrouter.models.RouterDC import RouterDCRouter, RouterDataset, RouterModule
from llmrouter.models.RouterDC.main_routerdc import load_config
from llmrouter.models.RouterDC.utils import load_tokenizer_and_backbone


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))
    )
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

    yaml_path = args.yaml_path
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(project_root, yaml_path)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    print(f"📄 Using YAML file: {yaml_path}")
    config = load_config(yaml_path)
    print("✅ Configuration loaded successfully!")

    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    train_data_path = data_cfg["train_output_path"]
    if not os.path.isabs(train_data_path):
        train_data_path = os.path.join(project_root, train_data_path)

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Training data not found: {train_data_path}. "
            "Run the RouterDC preprocessing pipeline first."
        )

    tokenizer, backbone, hidden_dim = load_tokenizer_and_backbone(config["model"])
    dataset = RouterDataset(
        data_path=train_data_path,
        data_type=eval_cfg["test_data_type"],
        dataset_id=0,
    )
    dataset.register_tokenizer(tokenizer)

    node_size = len(dataset.router_node)
    model = RouterModule(
        backbone=backbone,
        hidden_state_dim=hidden_dim,
        node_size=node_size,
        similarity_function=config["model"]["similarity_function"],
    )
    router = RouterDCRouter(model=model)

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    inputs, scores, dataset_ids, cluster_ids = next(iter(loader))

    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "temperature": config["training"]["temperature"],
        "true_scores": scores,
        "data_type": eval_cfg["test_data_type"],
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
