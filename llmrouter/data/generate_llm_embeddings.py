#!/usr/bin/env python3
"""
LLM Embeddings Generation Script - Step 2b: Generate LLM Candidate Embeddings

This script generates embeddings for LLM candidates from their metadata.
Reads default_llm.json and generates default_llm_embeddings.json.

Input: LLM metadata JSON file (default_llm.json format)
Output: LLM embeddings JSON file (default_llm_embeddings.json format)

Usage:
    python generate_llm_embeddings.py --config config.yaml
    python generate_llm_embeddings.py --input llm_data.json --output llm_embeddings.json
"""

import os
import sys
import json
import argparse
import yaml
from typing import Dict

from llmrouter.utils import setup_environment, get_longformer_embedding
from llmrouter.data.data_loader import DataLoader

# Setup environment
setup_environment()


def generate_llm_embeddings(llm_data: Dict, output_path: str):
    """
    Generate embeddings for LLM candidates.
    
    Args:
        llm_data: Dictionary with LLM metadata (from default_llm.json)
        output_path: Path to save embeddings JSON file
        
    Returns:
        Dictionary with LLM embeddings (same structure as default_llm_embeddings.json)
    """
    print("=== LLM EMBEDDINGS GENERATION ===")
    print(f"Processing {len(llm_data)} LLM candidates...")
    
    llm_embeddings = {}
    
    for model_name, model_info in llm_data.items():
        # Use 'feature' field to generate embedding
        feature_text = model_info.get('feature', '')
        
        if not feature_text:
            print(f"Warning: No 'feature' field for {model_name}, skipping embedding")
            continue
        
        print(f"Generating embedding for {model_name}...")
        
        try:
            # Generate embedding using longformer
            embedding = get_longformer_embedding(feature_text)
            
            # Create output entry with same structure as input + embedding
            llm_embeddings[model_name] = {
                'feature': model_info.get('feature', ''),
                'input_price': model_info.get('input_price', 0.0),
                'output_price': model_info.get('output_price', 0.0),
                'model': model_info.get('model', ''),
                'api_endpoint': model_info.get('api_endpoint', ''),
                'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            }
            
        except Exception as e:
            print(f"Error generating embedding for {model_name}: {e}")
            continue
    
    # Save to JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(llm_embeddings, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(llm_embeddings)} LLM embeddings to {output_path}")
    
    return llm_embeddings


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate LLM Candidate Embeddings")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--input", type=str, default=None,
                       help="Path to input LLM data JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output LLM embeddings JSON file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        loader = DataLoader(project_root)
        data_path = config.get("data_path", {})
        
        # Get paths from config
        input_path = loader.to_abs(data_path.get("llm_data", ""))
        output_path = loader.to_abs(data_path.get("llm_embedding_data", ""))
    else:
        # Use command-line args
        if args.input is None or args.output is None:
            parser.error("Either --config or both --input and --output must be provided")
        input_path = args.input
        output_path = args.output
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Load LLM data
    print(f"Loading LLM data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    try:
        # Generate embeddings
        llm_embeddings = generate_llm_embeddings(llm_data, output_path)
        
        print(f"\n🎉 LLM embeddings generation completed successfully!")
        print(f"📊 Generated embeddings for {len(llm_embeddings)} LLM candidates")
        print(f"📁 Output file: {output_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Embeddings generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during embeddings generation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

