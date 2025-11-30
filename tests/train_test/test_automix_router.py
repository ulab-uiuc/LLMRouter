import argparse
import os

import pandas as pd

from llmrouter.models.Automix import (
    AutomixModel,
    AutomixRouter,
    AutomixRouterTrainer,
    POMDP,
    SelfConsistency,
    Threshold,
)
from llmrouter.models.Automix.main_automix import load_config


def build_method(name: str, num_bins: int):
    mapping = {
        "POMDP": POMDP,
        "Threshold": Threshold,
        "SelfConsistency": SelfConsistency,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported Automix routing method: {name}")
    return mapping[name](num_bins=num_bins)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_yaml = os.path.join(
        project_root, "configs", "model_config_test", "automix_config.yaml"
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

    cfg = config["real_data"]
    data_path = cfg["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

    data = pd.read_json(data_path, lines=True, orient="records")
    train_df = data[data["split"] == "train"].copy()

    method = build_method(cfg["routing_method"], cfg["num_bins"])
    model = AutomixModel(
        method=method,
        slm_column=cfg["columns"]["slm"],
        llm_column=cfg["columns"]["llm"],
        verifier_column=cfg["columns"]["verifier"],
        costs=[cfg["costs"]["small_model"], cfg["costs"]["large_model"]],
        verifier_cost=cfg["costs"]["verifier"],
        verbose=cfg["training"]["verbose"],
    )
    model.train_routing(train_df)
    router = AutomixRouter(model=model)

    trainer = AutomixRouterTrainer(router=router, device="cpu")
    trainer.train_on_dataframe(train_df)


if __name__ == "__main__":
    main()
