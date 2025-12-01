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
from llmrouter.models.Automix.main_automix import load_config, convert_default_data


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
        project_root, "configs", "model_config_test", "automix.yaml"
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

    df = pd.read_json(data_path, lines=True, orient="records")
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    hparam = config["hparam"]
    method = build_method(hparam["routing_method"], hparam["num_bins"])
    model = AutomixModel(
        method=method,
        slm_column=hparam["columns"]["slm"],
        llm_column=hparam["columns"]["llm"],
        verifier_column=hparam["columns"]["verifier"],
        costs=[hparam["costs"]["small_model"], hparam["costs"]["large_model"]],
        verifier_cost=hparam["costs"]["verifier"],
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
