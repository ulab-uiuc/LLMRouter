"""
Automix Router - Complete Usage Example
========================================

This script demonstrates how to use the Automix router for complete training and inference workflows.

Usage:
    python main_automix.py [--config CONFIG_PATH]

Arguments:
    --config: Path to YAML configuration file (default: configs/model_config_test/automix_config.yaml)
"""

import os
import sys
import argparse
import pandas as pd
import yaml

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print("Running in: ", project_root)


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
        # Use default configuration path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config", "automix_config.yaml")

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


def convert_default_data(config: dict, script_dir: str) -> str:
    """
    Convert default_data to required format
    
    Args:
        config: Configuration dictionary containing data_conversion settings
        script_dir: Script directory for resolving relative paths
    
    Returns:
        Path to merged data file
    """
    conv_cfg = config["real_data"]["data_conversion"]
    output_dir = config["real_data"]["output_dir"]
    
    # Handle relative paths
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    default_data_dir = conv_cfg["default_data_dir"]
    if not os.path.isabs(default_data_dir):
        default_data_dir = os.path.join(script_dir, default_data_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file names from config
    input_files = conv_cfg["input_files"]
    output_files = conv_cfg["output_files"]
    
    # Define paths
    test_input = os.path.join(default_data_dir, input_files["test"])
    train_input = os.path.join(default_data_dir, input_files["train"])
    test_output = os.path.join(output_dir, output_files["test"])
    train_output = os.path.join(output_dir, output_files["train"])
    merged_output = os.path.join(output_dir, output_files["merged"])
    
    # Check if merged file already exists
    if os.path.exists(merged_output):
        print(f"Converted data already exists: {merged_output}")
        return merged_output
    
    # Convert test data
    if os.path.exists(test_input):
        print(f"Converting test data: {test_input} -> {test_output}")
        convert_data(
            input_file=test_input,
            output_file=test_output,
            use_llm=False,
        )
    else:
        print(f"Warning: Test data file not found: {test_input}")
        test_output = None
    
    # Convert train data
    if os.path.exists(train_input):
        print(f"Converting train data: {train_input} -> {train_output}")
        convert_train_data(
            input_file=train_input,
            output_file=train_output,
        )
    else:
        print(f"Warning: Train data file not found: {train_input}")
        train_output = None
    
    # Merge data
    if test_output and train_output and os.path.exists(test_output) and os.path.exists(train_output):
        print(f"Merging data: {test_output} + {train_output} -> {merged_output}")
        merge_train_test(
            test_file=test_output,
            train_file=train_output,
            output_file=merged_output,
        )
        return merged_output
    else:
        raise FileNotFoundError("Failed to convert data files. Please check input files exist.")


def train_and_evaluate(config: dict):
    """
    Train and evaluate using real data
    Data files need to be prepared first

    Args:
        config: Configuration dictionary loaded from YAML file
    """
    cfg = config["real_data"]
    display_cfg = config["display"]
    sep_width = display_cfg["separator_width"]

    print("=" * sep_width)
    print("Example 1: Training and evaluation with real data")
    print("=" * sep_width)

    # Get paths from configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle relative paths
    data_path = cfg["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(script_dir, data_path)
    
    output_dir = cfg["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    # If data_path doesn't exist, try to convert from default_data
    if not os.path.exists(data_path):
        print(f"\nStep 0: Convert default_data to required format")
        print("-" * sep_width)
        try:
            data_path = convert_default_data(config, script_dir)
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
    skip_data_prep = os.path.exists(
        os.path.join(output_dir, "router_automix_llamapair_ver_outputs.jsonl")
    )

    if skip_data_prep:
        print("Detected prepared data, skipping data preparation step")
        df = pd.read_json(
            os.path.join(output_dir, "router_automix_llamapair_ver_outputs.jsonl"),
            lines=True,
            orient="records",
        )
    else:
        df = prepare_automix_data(
            input_data_path=data_path,
            output_dir=output_dir,
            engine_small=cfg["engine_small"],
            engine_large=cfg["engine_large"],
        )

    print(f"\nPreparation complete! Dataset size: {len(df)}")
    print(f"Training set size: {len(df[df['split'] == 'train'])}")
    print(f"Test set size: {len(df[df['split'] == 'test'])}")

    print(f"\nStep 2: Create Automix router")
    print("-" * sep_width)

    # Create routing method from configuration
    method = get_routing_method(cfg["routing_method"], cfg["num_bins"])
    print(f"Routing method: {cfg['routing_method']} (num_bins={cfg['num_bins']})")

    # Create model
    model = AutomixModel(
        method=method,
        slm_column=cfg["columns"]["slm"],
        llm_column=cfg["columns"]["llm"],
        verifier_column=cfg["columns"]["verifier"],
        costs=[cfg["costs"]["small_model"], cfg["costs"]["large_model"]],
        verifier_cost=cfg["costs"]["verifier"],
        verbose=cfg["training"]["verbose"],
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
    trainer = AutomixRouterTrainer(router=router, device=cfg["training"]["device"])

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
    num_samples = cfg["inference"]["num_samples"]
    sample_data = test_df.head(num_samples)

    for idx, row in sample_data.iterrows():
        decision = router.model.infer(row)
        model_used = "Large model (70B)" if decision else "Small model (13B)"
        print(f"\nQuestion: {row['question'][:60]}...")
        print(f"  Verifier score: {row[cfg['columns']['verifier']]:.3f}")
        print(f"  Small model F1: {row[cfg['columns']['slm']]:.3f}")
        print(f"  Large model F1: {row[cfg['columns']['llm']]:.3f}")
        print(f"  Routing decision: {model_used}")

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
        help="Path to YAML configuration file (default: configs/model_config_test/automix_config.yaml)",
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
