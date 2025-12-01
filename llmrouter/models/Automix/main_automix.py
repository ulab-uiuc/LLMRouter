"""
Automix Router - Complete Usage Example
========================================

This script demonstrates how to use the Automix router for complete training and inference workflows.

Usage:
    python main_automix.py [--config CONFIG_PATH]

Arguments:
    --config: Path to YAML configuration file (default: configs/model_config_test/automix.yaml)
"""

import os
import sys
import argparse
import shutil
import pandas as pd
import yaml

# Add project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print("Running in: ", PROJECT_ROOT)


from llmrouter.models.Automix import (
    AutomixRouter,
    AutomixRouterTrainer,
    AutomixModel,
    POMDP,
    Threshold,
    SelfConsistency,
    prepare_automix_data,
)
from llmrouter.utils.data_convert import (
    convert_data,
    convert_train_data,
    merge_train_test,
)


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration file

    Args:
        config_path: Path to configuration file. If None, use default path

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(
            PROJECT_ROOT, "configs", "model_config_test", "automix.yaml"
        )
    elif not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Configuration file loaded: {config_path}")
    return config


def get_routing_method(method_name: str, num_bins: int):
    """
    Create routing method instance based on method name

    Args:
        method_name: Method name ("Threshold", "SelfConsistency", "POMDP")
        num_bins: Number of bins

    Returns:
        Routing method instance
    """
    method_map = {
        "Threshold": Threshold,
        "SelfConsistency": SelfConsistency,
        "POMDP": POMDP,
    }

    if method_name not in method_map:
        raise ValueError(
            f"Unknown routing method: {method_name}. "
            f"Available methods: {list(method_map.keys())}"
        )

    return method_map[method_name](num_bins=num_bins)


def convert_default_data(config: dict) -> str:
    """
    Convert default_data to required format
    
    Args:
        config: Configuration dictionary containing data_conversion settings
    
    Returns:
        Path to merged data file
    """
    data_cfg = config["data_path"]
    conv_cfg = data_cfg.get("conversion", {})

    output_dir = data_cfg.get("output_dir", "data/automix")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    default_data_dir = conv_cfg.get("default_data_dir", "data/default_data")
    if not os.path.isabs(default_data_dir):
        default_data_dir = os.path.join(PROJECT_ROOT, default_data_dir)

    prepared_path = data_cfg.get(
        "prepared_data",
        os.path.join(output_dir, conv_cfg.get("merged_file", "train_test_nq_split.jsonl")),
    )
    if not os.path.isabs(prepared_path):
        prepared_path = os.path.join(PROJECT_ROOT, prepared_path)

    os.makedirs(output_dir, exist_ok=True)

    input_train = os.path.join(default_data_dir, conv_cfg.get("train_file", "default_routing_train_data.jsonl"))
    input_test = os.path.join(default_data_dir, conv_cfg.get("test_file", "default_routing_test_data.jsonl"))
    output_train = os.path.join(output_dir, conv_cfg.get("train_output_file", "router_train_data_nq.json"))
    output_test = os.path.join(output_dir, conv_cfg.get("test_output_file", "router_test_data_nq.jsonl"))
    merged_output = os.path.join(output_dir, conv_cfg.get("merged_file", "train_test_nq_split.jsonl"))

    if os.path.exists(prepared_path):
        print(f"Prepared Automix data already exists: {prepared_path}")
        return prepared_path
    
    # Convert test data
    if os.path.exists(input_test):
        print(f"Converting test data: {input_test} -> {output_test}")
        convert_data(
            input_file=input_test,
            output_file=output_test,
            use_llm=False,
        )
    else:
        print(f"Warning: Test data file not found: {input_test}")
        output_test = None

    if os.path.exists(input_train):
        print(f"Converting train data: {input_train} -> {output_train}")
        convert_train_data(
            input_file=input_train,
            output_file=output_train,
        )
    else:
        print(f"Warning: Train data file not found: {input_train}")
        output_train = None

    if output_test and output_train and os.path.exists(output_test) and os.path.exists(output_train):
        print(f"Merging data: {output_test} + {output_train} -> {merged_output}")
        merge_train_test(
            test_file=output_test,
            train_file=output_train,
            output_file=merged_output,
        )
        if prepared_path != merged_output:
            shutil.copy2(merged_output, prepared_path)
        return prepared_path
    else:
        raise FileNotFoundError("Failed to convert Automix data files. Please check input files exist.")


def train_and_evaluate(config: dict):
    """
    Train and evaluate using real data
    Data files need to be prepared first

    Args:
        config: Configuration dictionary loaded from YAML file
    """
    data_cfg = config["data_path"]
    model_cfg = config["model_path"]
    hparam = config["hparam"]
    train_cfg = config.get("train_param", {})
    display_cfg = config.get(
        "display",
        {
            "separator_width": 70,
            "precision": {
                "ibc_lift": 4,
                "performance": 4,
                "cost": 2,
                "percentage": 1,
            },
        },
    )
    sep_width = display_cfg["separator_width"]

    print("=" * sep_width)
    print("Example 1: Training and evaluation with real data")
    print("=" * sep_width)

    # Get paths from configuration
    data_path = data_cfg.get("prepared_data", "data/automix/router_automix_llamapair_ver_outputs.jsonl")
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)

    output_dir = data_cfg.get("output_dir", "data/automix")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    if not os.path.exists(data_path):
        print(f"\nStep 0: Convert default_data to required format")
        print("-" * sep_width)
        try:
            data_path = convert_default_data(config)
            print(f"Data conversion completed: {data_path}")
        except Exception as e:
            print(f"Error converting data: {e}")
            print(f"Please ensure data files exist or run data preparation pipeline first")
            return

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Please prepare data files or run data preparation pipeline first")
        return

    print(f"\nStep 1: Prepare data (get model predictions and self-verification)")
    print("-" * sep_width)

    # Skip this step if data is already prepared
    final_data_path = os.path.join(output_dir, "router_automix_llamapair_ver_outputs.jsonl")
    skip_data_prep = os.path.exists(final_data_path)

    if skip_data_prep:
        print("Detected prepared data, skipping data preparation step")
        df = pd.read_json(final_data_path, lines=True, orient="records")
    else:
        df = prepare_automix_data(
            input_data_path=data_path,
            output_dir=output_dir,
            engine_small=model_cfg["engine_small"],
            engine_large=model_cfg["engine_large"],
        )

    print(f"\nPreparation complete! Dataset size: {len(df)}")
    print(f"Training set size: {len(df[df['split'] == 'train'])}")
    print(f"Test set size: {len(df[df['split'] == 'test'])}")

    print(f"\nStep 2: Create Automix router")
    print("-" * sep_width)

    # Create routing method from configuration
    method = get_routing_method(hparam["routing_method"], hparam["num_bins"])
    print(f"Routing method: {hparam['routing_method']} (num_bins={hparam['num_bins']})")

    # Create model
    model = AutomixModel(
        method=method,
        slm_column=hparam["columns"]["slm"],
        llm_column=hparam["columns"]["llm"],
        verifier_column=hparam["columns"]["verifier"],
        costs=[hparam["costs"]["small_model"], hparam["costs"]["large_model"]],
        verifier_cost=hparam["costs"]["verifier"],
        verbose=train_cfg.get("verbose", True),
    )
    print(
        f"Model configuration: Small model cost={cfg['costs']['small_model']}, "
        f"Large model cost={cfg['costs']['large_model']}, "
        f"Verifier cost={cfg['costs']['verifier']}"
    )

    # Create router
    router = AutomixRouter(model=model)
    print("Router created successfully")

    print(f"\nStep 3: Train router")
    print("-" * sep_width)

    # Create trainer
    trainer = AutomixRouterTrainer(router=router, device=train_cfg.get("device", "cpu"))

    # Split data
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Train and evaluate
    results = trainer.train_and_evaluate(train_df, test_df)

    print(f"\nStep 4: View results")
    print("-" * sep_width)

    # Get display precision configuration
    prec = display_cfg["precision"]

    print("\nTraining set results:")
    print(f"  Best parameter: {results['train']['best_param']}")
    print(f"  IBC Lift: {results['train']['metrics']['ibc_lift']:.{prec['ibc_lift']}f}")
    print(f"  Average performance: {results['train']['metrics']['avg_performance']:.{prec['performance']}f}")
    print(f"  Average cost: {results['train']['metrics']['avg_cost']:.{prec['cost']}f}")

    print("\nTest set results:")
    print(f"  IBC Lift: {results['test']['ibc_lift']:.{prec['ibc_lift']}f}")
    print(f"  Average performance: {results['test']['avg_performance']:.{prec['performance']}f}")
    print(f"  Average cost: {results['test']['avg_cost']:.{prec['cost']}f}")

    # Calculate routing statistics
    test_decisions = results["test"]["route_to_llm"]
    num_routed = int(test_decisions.sum())
    total = len(test_decisions)
    print(f"  Routed to large model: {num_routed}/{total} ({num_routed/total*100:.{prec['percentage']}f}%)")

    print(f"\nStep 5: Inference with trained router")
    print("-" * sep_width)

    # Select a few test samples for inference
    num_samples = train_cfg.get("num_samples", 2)
    sample_data = test_df.head(num_samples)

    for idx, row in sample_data.iterrows():
        decision = router.model.infer(row)
        model_used = "Large model (70B)" if decision else "Small model (13B)"
        print(f"\nQuestion: {row['question'][:60]}...")
        print(f"  Verifier score: {row[cfg['columns']['verifier']]:.3f}")
        print(f"  Small model F1: {row[cfg['columns']['slm']]:.3f}")
        print(f"  Large model F1: {row[cfg['columns']['llm']]:.3f}")
        print(f"  Routing decision: {model_used}")

    return results

    return results

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Automix Router Usage Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: configs/model_config_test/automix.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the configuration file exists or use --config to specify the configuration file path")
        return

    sep_width = config["display"]["separator_width"]

    print("\n" + "=" * sep_width)
    print("Automix Router Usage Example")
    print("=" * sep_width)

    try:
        train_and_evaluate(config)
    except Exception as e:
        print(f"\nReal data example failed: {e}")
        print("Hint: Please ensure data files exist and configuration is correct")

    print("\n" + "=" * sep_width)
    print("Example completed!") 
    print("=" * sep_width)


if __name__ == "__main__":
    main()
