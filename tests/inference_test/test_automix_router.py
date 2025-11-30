import argparse
import os

import pandas as pd

from llmrouter.models.Automix import (
    AutomixModel,
    AutomixRouter,
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
        description="Automix inference smoke test."
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

    df = pd.read_json(data_path, lines=True, orient="records")
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    method = build_method(cfg["routing_method"], cfg["num_bins"])
    model = AutomixModel(
        method=method,
        slm_column=cfg["columns"]["slm"],
        llm_column=cfg["columns"]["llm"],
        verifier_column=cfg["columns"]["verifier"],
        costs=[cfg["costs"]["small_model"], cfg["costs"]["large_model"]],
        verifier_cost=cfg["costs"]["verifier"],
        verbose=False,
    )
    model.train_routing(train_df)

    router = AutomixRouter(model=model)

    batch_outputs = router.route_batch({"data": test_df})
    print("Batch decisions:", batch_outputs["decisions"])
    print(
        f"Average performance: {batch_outputs['performance']:.4f}, "
        f"Average cost: {batch_outputs['cost']:.2f}"
    )

    single_row = test_df.iloc[0]
    single_result = router.route_single(single_row)
    print("Single inference:", single_result)
    print("✅ Automix inference test completed successfully!")


if __name__ == "__main__":
    main()
