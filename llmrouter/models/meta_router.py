from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from llmrouter.utils import load_csv, load_jsonl, jsonl_to_csv, load_pt
import json
import os
import yaml


class MetaRouter(ABC):
    """
    MetaRouter (Base Class)
    -----------------------
    Initialize with one YAML configuration file.
    Automatically loads:
        - data_path section (paths for query, routing, llm data, etc.)
        - metric section (weights for evaluation)
        - any additional fields defined in the YAML
    """

    def __init__(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.load_data(config)
        weights_dict = config.get("metric", {}).get("weights", {})
        self.weights_list = list(weights_dict.values())

    def load_data(self, config):
        """Load all data files specified in config['data_path']."""
        data_path = config.get("data_path", {})

        # 🧭 Compute project root (two levels up from models/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

        def to_abs(path_str):
            """Convert relative path (in YAML) to absolute path based on project root."""
            if os.path.isabs(path_str):
                return path_str
            return os.path.join(project_root, path_str)

        # -------- query_data_train --------
        if "query_data_train" in data_path:
            path = to_abs(data_path["query_data_train"])
            if os.path.exists(path):
                self.query_data_train = jsonl_to_csv(path)
            else:
                print(f"[Warning] Missing file: {path}")
                self.query_data_train = None
        else:
            self.query_data_train = None

        # -------- query_data_test --------
        if "query_data_test" in data_path:
            path = to_abs(data_path["query_data_test"])
            if os.path.exists(path):
                self.query_data_test = jsonl_to_csv(path)
            else:
                print(f"[Warning] Missing file: {path}")
                self.query_data_test = None
        else:
            self.query_data_test = None

        # -------- query_embedding_data (.pt) --------
        if "query_embedding_data" in data_path:
            path = to_abs(data_path["query_embedding_data"])
            if os.path.exists(path):
                self.query_embedding_data = load_pt(path)
            else:
                print(f"[Warning] Missing file: {path}")
                self.query_embedding_data = None
        else:
            self.query_embedding_data = None

        # -------- routing_data_train --------
        if "routing_data_train" in data_path:
            path = to_abs(data_path["routing_data_train"])
            if os.path.exists(path):
                self.routing_data_train = jsonl_to_csv(path)
            else:
                print(f"[Warning] Missing file: {path}")
                self.routing_data_train = None
        else:
            self.routing_data_train = None

        # -------- routing_data_test --------
        if "routing_data_test" in data_path:
            path = to_abs(data_path["routing_data_test"])
            if os.path.exists(path):
                self.routing_data_test = jsonl_to_csv(path)
            else:
                print(f"[Warning] Missing file: {path}")
                self.routing_data_test = None
        else:
            self.routing_data_test = None

        # -------- llm_data (.json) --------
        if "llm_data" in data_path:
            path = to_abs(data_path["llm_data"])
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.llm_data = json.load(f)
            else:
                print(f"[Warning] Missing file: {path}")
                self.llm_data = None
        else:
            self.llm_data = None

        # -------- llm_embedding_data (.json) --------
        if "llm_embedding_data" in data_path:
            path = to_abs(data_path["llm_embedding_data"])
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.llm_embedding_data = json.load(f)
            else:
                print(f"[Warning] Missing file: {path}")
                self.llm_embedding_data = None
        else:
            self.llm_embedding_data = None

