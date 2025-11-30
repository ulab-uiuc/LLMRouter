"""
RouterDC Training with YAML Configuration
==========================================
Complete training script for RouterDC using the LLMRouter framework.
Usage:
    ### Step 1: Data Preprocessing

    ```bash
    python main_routerdc.py --config configs/model_config_test/routerdc_nq.yaml --preprocess_data
    ```

    ### Step 2: Start Training

    ```bash
    python main_routerdc.py --config configs/routerdc_nq.yaml
    ```
"""

import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print("Running in: ", project_root)

from llmrouter.models.RouterDC import (
    RouterModule,
    RouterDCRouter,
    RouterDCTrainer,
    RouterDataset,
    prepare_routerdc_data,
)
from llmrouter.models.RouterDC.utils import load_tokenizer_and_backbone
from llmrouter.utils.data_convert import (
    convert_data,
    convert_train_data,
    merge_train_test,
)


def setup_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"✅ Configuration file loaded: {config_path}")
    return config


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge command line arguments with config file.
    Command line arguments take precedence.

    Args:
        config (dict): Configuration from YAML
        args (argparse.Namespace): Command line arguments

    Returns:
        dict: Merged configuration
    """
    # Override config with command line args if provided
    if args.data_paths:
        config['data']['train_output_path'] = args.data_paths[0]
    if args.test_data_paths:
        config['data']['test_output_path'] = args.test_data_paths[0]
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.training_steps:
        config['training']['training_steps'] = args.training_steps
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.cluster_loss_weight is not None:
        config['training']['cluster_loss_weight'] = args.cluster_loss_weight
    if args.save_path:
        config['output']['save_path'] = args.save_path

    return config


def convert_default_data(config: dict, script_dir: str) -> tuple:
    """
    Convert default_data to required format
    
    Args:
        config: Configuration dictionary containing data_conversion settings
        script_dir: Script directory for resolving relative paths
    
    Returns:
        Tuple of (train_input_path, test_input_path)
    """
    data_cfg = config['data']
    conv_cfg = data_cfg.get('data_conversion', {})
    
    if not conv_cfg:
        raise ValueError("data_conversion configuration not found in config")
    
    output_dir = conv_cfg.get('output_dir', './data/routerdc')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    default_data_dir = conv_cfg['default_data_dir']
    if not os.path.isabs(default_data_dir):
        default_data_dir = os.path.join(script_dir, default_data_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file names from config
    input_files = conv_cfg['input_files']
    output_files = conv_cfg['output_files']
    
    # Define paths
    test_input = os.path.join(default_data_dir, input_files['test'])
    train_input = os.path.join(default_data_dir, input_files['train'])
    test_output = os.path.join(output_dir, output_files['test'])
    train_output = os.path.join(output_dir, output_files['train'])
    merged_output = os.path.join(output_dir, output_files['merged'])
    
    # Final paths (matching config expectations)
    train_final_path = train_output
    test_final_path = merged_output
    
    # Check if converted files already exist
    if os.path.exists(train_final_path) and os.path.exists(test_final_path):
        print(f"Converted data already exists")
        return train_final_path, test_final_path
    
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
    
    # Merge data (for test set)
    if test_output and os.path.exists(test_output):
        # Use merged output as final test path
        if not os.path.exists(merged_output):
            # If we have train output, merge them
            if train_output and os.path.exists(train_output):
                print(f"Merging data: {test_output} + {train_output} -> {merged_output}")
                merge_train_test(
                    test_file=test_output,
                    train_file=train_output,
                    output_file=merged_output,
                )
            else:
                # Just copy test output to merged output
                shutil.copy2(test_output, merged_output)
        test_final_path = merged_output
    else:
        test_final_path = None
    
    # Train final path is the train_output
    train_final_path = train_output if train_output and os.path.exists(train_output) else None
    
    if not train_final_path or not test_final_path:
        raise FileNotFoundError("Failed to convert data files. Please check input files exist.")
    
    return train_final_path, test_final_path


def preprocess_data(config: dict):
    """
    Run data preprocessing pipeline.

    Args:
        config (dict): Configuration dictionary
    """
    data_cfg = config['data']
    script_dir = project_root

    print("\n" + "=" * config['display']['separator_width'])
    print("Data Preprocessing")
    print("=" * config['display']['separator_width'])

    # Handle relative paths
    train_input_path = data_cfg['train_input_path']
    if not os.path.isabs(train_input_path):
        train_input_path = os.path.join(script_dir, train_input_path)
    
    test_input_path = data_cfg.get('test_input_path')
    if test_input_path and not os.path.isabs(test_input_path):
        test_input_path = os.path.join(script_dir, test_input_path)
    
    train_output_path = data_cfg['train_output_path']
    if not os.path.isabs(train_output_path):
        train_output_path = os.path.join(script_dir, train_output_path)
    
    test_output_path = data_cfg.get('test_output_path')
    if test_output_path and not os.path.isabs(test_output_path):
        test_output_path = os.path.join(script_dir, test_output_path)
    
    # Convert from default_data if input files don't exist
    if not os.path.exists(train_input_path) or (test_input_path and not os.path.exists(test_input_path)):
        print("\n[Step 0] Converting default_data to required format...")
        print("-" * config['display']['separator_width'])
        try:
            converted_train, converted_test = convert_default_data(config, script_dir)
            if not os.path.exists(train_input_path):
                train_input_path = converted_train
                print(f"Using converted train data: {train_input_path}")
            if test_input_path and not os.path.exists(test_input_path):
                test_input_path = converted_test
                print(f"Using converted test data: {test_input_path}")
        except Exception as e:
            print(f"Error converting data: {e}")
            print("Please ensure data files exist or run data preparation pipeline first")
            raise

    prepare_routerdc_data(
        train_input_path=train_input_path,
        train_output_path=train_output_path,
        test_input_path=test_input_path,
        test_output_path=test_output_path,
        n_clusters=data_cfg['n_clusters'],
        max_test_samples=data_cfg.get('max_test_samples'),
        api_key=config.get('api', {}).get('nvidia_api_key'),
        skip_existing=data_cfg.get('skip_existing', True)
    )


def train_from_config(config: dict):
    """
    Train RouterDC using configuration.

    Args:
        config (dict): Configuration dictionary
    """
    # Extract config sections
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    eval_cfg = config['evaluation']
    output_cfg = config['output']
    display_cfg = config['display']

    sep_width = display_cfg['separator_width']

    # Set random seed
    setup_seed(train_cfg['seed'])

    print("\n" + "=" * sep_width)
    print("RouterDC Training - LLMRouter Framework")
    print("=" * sep_width)
    print(f"Device: {train_cfg['device']}")
    print(f"Training steps: {train_cfg['training_steps']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Learning rate: {train_cfg['learning_rate']}")
    print(f"Cluster loss weight: {train_cfg['cluster_loss_weight']}")
    print(f"Sample loss weight: {train_cfg['sample_loss_weight']}")
    print("=" * sep_width)

    # ======================================================================
    # Step 1: Load tokenizer and backbone model
    # ======================================================================
    print(f"\n[1/6] Loading backbone model...")
    tokenizer, encoder_model, hidden_state_dim = load_tokenizer_and_backbone(model_cfg)
    print(f"  ✓ Loaded {model_cfg['backbone']}")

    # ======================================================================
    # Step 2: Prepare datasets
    # ======================================================================
    print(f"\n[2/6] Preparing datasets...")

    # Handle relative paths
    script_dir = project_root
    
    train_data_path = data_cfg['train_output_path']
    if not os.path.isabs(train_data_path):
        train_data_path = os.path.join(script_dir, train_data_path)
    
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"Training data not found: {train_data_path}\n"
            f"Please run data preprocessing first: python main_routerdc.py --config <config_path> --preprocess_data"
        )

    train_dataset = RouterDataset(
        data_path=train_data_path,
        size=data_cfg.get('training_samples_per_dataset'),
        data_type=eval_cfg['test_data_type'],
        dataset_id=0
    )
    train_dataset.register_tokenizer(tokenizer)
    print(f"  ✓ Loaded training data: {train_data_path} ({len(train_dataset)} samples)")

    # Load test data
    test_datasets = []
    test_data_path = data_cfg.get('test_output_path')
    
    if test_data_path and not os.path.isabs(test_data_path):
        test_data_path = os.path.join(script_dir, test_data_path)

    if test_data_path and os.path.exists(test_data_path):
        test_dataset = RouterDataset(
            data_path=test_data_path,
            data_type=eval_cfg['test_data_type'],
            dataset_id=0
        )
        test_dataset.register_tokenizer(tokenizer)
        test_datasets.append((test_dataset, eval_cfg['test_data_type'], test_data_path))
        print(f"  ✓ Loaded test data: {test_data_path} ({len(test_dataset)} samples, type={eval_cfg['test_data_type']})")

    # Load additional test datasets if specified
    additional_test_paths = data_cfg.get('additional_test_paths', [])
    additional_test_types = data_cfg.get('additional_test_types', [])

    for i, (test_path, data_type) in enumerate(zip(additional_test_paths, additional_test_types)):
        if not os.path.isabs(test_path):
            test_path = os.path.join(script_dir, test_path)
        if os.path.exists(test_path):
            dataset = RouterDataset(data_path=test_path, data_type=data_type, dataset_id=i+1)
            dataset.register_tokenizer(tokenizer)
            test_datasets.append((dataset, data_type, test_path))
            print(f"  ✓ Loaded additional test data: {test_path} ({len(dataset)} samples, type={data_type})")

    # Get number of LLMs from first dataset
    num_llms = len(train_dataset.router_node)
    print(f"  Number of LLMs: {num_llms}")
    print(f"  LLM names: {train_dataset.router_node}")

    # ======================================================================
    # Step 3: Create RouterModule (underlying PyTorch model)
    # ======================================================================
    print(f"\n[3/6] Creating RouterModule...")
    model = RouterModule(
        backbone=encoder_model,
        hidden_state_dim=hidden_state_dim,
        node_size=num_llms,
        similarity_function=model_cfg['similarity_function']
    )
    print(f"  ✓ RouterModule created with {num_llms} LLM nodes")

    # ======================================================================
    # Step 4: Wrap model in RouterDCRouter
    # ======================================================================
    print(f"\n[4/6] Creating RouterDCRouter...")
    router = RouterDCRouter(model=model)
    router = router.to(train_cfg['device'])
    print("  ✓ RouterDCRouter created and moved to device")

    # ======================================================================
    # Step 5: Create RouterDCTrainer
    # ======================================================================
    print(f"\n[5/6] Creating RouterDCTrainer...")
    optimizer = torch.optim.AdamW(router.parameters(), lr=train_cfg['learning_rate'])

    # Add training dataset to evaluation list
    eval_datasets_with_train = test_datasets + [(train_dataset, eval_cfg['test_data_type'], train_data_path)]

    save_path = output_cfg['save_path']
    if not os.path.isabs(save_path):
        save_path = os.path.join(project_root, save_path)

    trainer = RouterDCTrainer(
        router=router,
        optimizer=optimizer,
        device=train_cfg['device'],
        top_k=train_cfg['top_k'],
        last_k=train_cfg['last_k'],
        temperature=train_cfg['temperature'],
        sample_loss_weight=train_cfg['sample_loss_weight'],
        cluster_loss_weight=train_cfg['cluster_loss_weight'],
        H=train_cfg['H'],
        gradient_accumulation=train_cfg['gradient_accumulation'],
        eval_datasets=eval_datasets_with_train,
        eval_steps=eval_cfg['eval_steps'],
        save_path=save_path,
    )
    print("  ✓ RouterDCTrainer created")
    print(f"    - Top-k: {train_cfg['top_k']}")
    print(f"    - Last-k: {train_cfg['last_k']}")
    print(f"    - Temperature: {train_cfg['temperature']}")
    print(f"    - Sample loss weight: {train_cfg['sample_loss_weight']}")
    print(f"    - Cluster loss weight: {train_cfg['cluster_loss_weight']}")

    # ======================================================================
    # Step 6: Train the router and evaluate on test sets
    # ======================================================================
    print(f"\n[6/6] Starting training and evaluation...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True
    )

    trainer.train(train_dataloader, training_steps=train_cfg['training_steps'])

    print("\n" + "=" * sep_width)
    print("Training Complete!")
    print("=" * sep_width)
    print(f"Checkpoints saved to: {save_path}")
    print(f"Best test average: {trainer.max_average:.4f}")
    print(f"Best train average: {trainer.max_training_average:.4f}")
    print("=" * sep_width)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train RouterDC using LLMRouter framework with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        default="configs/model_config_test/routerdc_nq.yaml",
        help='Path to YAML configuration file'
    )

    # Data preprocessing
    parser.add_argument(
        '--preprocess_data',
        action='store_true',
        help='Run data preprocessing before training'
    )

    # Override options (optional - these override config file values)
    parser.add_argument(
        '--data_paths',
        nargs='+',
        default=None,
        help='Override training data paths from config'
    )
    parser.add_argument(
        '--test_data_paths',
        nargs='+',
        default=None,
        help='Override test data paths from config'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--training_steps',
        type=int,
        default=None,
        help='Override training steps from config'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    parser.add_argument(
        '--cluster_loss_weight',
        type=float,
        default=None,
        help='Override cluster loss weight from config'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Override save path from config'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        try:
            config = load_config(args.config)
            config = merge_config_with_args(config, args)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure the configuration file exists or use --config to specify the correct configuration file path")
            sys.exit(1)
    else:
        print("Error: Configuration file must be provided")
        print("Usage: python main_routerdc.py --config configs/routerdc_nq.yaml")
        print("\nAvailable configuration files:")
        print("  - configs/routerdc_nq.yaml       (NQ dataset)")
        print("  - configs/routerdc_default.yaml  (Default template)")
        sys.exit(1)

    # Run data preprocessing if requested
    if args.preprocess_data:
        try:
            preprocess_data(config)
            print("\n✅ Data preprocessing completed!")
            print("You can now start training: python main_routerdc.py --config", args.config)
            return
        except Exception as e:
            print(f"\n❌ Data preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Train model
    try:
        train_from_config(config)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
