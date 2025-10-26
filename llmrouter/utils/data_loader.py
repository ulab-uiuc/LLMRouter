from typing import Any, Dict, List, Optional
import pandas as pd
import torch
import json
import os


def load_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and return a pandas DataFrame.
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"Loaded CSV: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return None


def load_jsonl(path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load a JSONL (JSON Lines) file and return a list of dictionaries.
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        print(f"Loaded JSONL: {path} ({len(data)} records)")
        return data
    except Exception as e:
        print(f"Failed to load JSONL: {e}")
        return None


def jsonl_to_csv(jsonl_path: str, csv_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Convert a JSONL file to CSV and return the DataFrame.
    If `csv_path` is not provided, the CSV will be saved in the same directory as the JSONL.
    """
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return None

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        df = pd.DataFrame(data)

        if csv_path is None:
            csv_path = os.path.splitext(jsonl_path)[0] + ".csv"

        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Converted JSONL → CSV: {csv_path} ({len(df)} rows)")
        return df

    except Exception as e:
        print(f"Failed to convert JSONL to CSV: {e}")
        return None


def load_pt(path: str) -> Optional[Any]:
    """
    Load a PyTorch .pt file (can be a Tensor or a Dict).
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            print(f"Loaded .pt file (dict) with keys: {list(data.keys())}")
        elif isinstance(data, torch.Tensor):
            print(f"Loaded .pt Tensor with shape: {tuple(data.shape)}")
        else:
            print(f"Loaded .pt object of type: {type(data)}")
        return data
    except Exception as e:
        print(f"Failed to load .pt: {e}")
        return None
