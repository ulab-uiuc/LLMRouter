import os
import json
from llmrouter.utils import load_csv, load_jsonl, jsonl_to_csv, load_pt


class DataLoader:
    """
    DataLoaderCore
    --------------
    Handles all file-loading logic used by MetaRouter.
    """

    def __init__(self, project_root: str):
        self.project_root = project_root

    def to_abs(self, path_str: str) -> str:
        """Convert relative path (in YAML) to absolute path based on project root."""
        if os.path.isabs(path_str):
            return path_str
        return os.path.join(self.project_root, path_str)

    def load_data(self, config, obj_ref):
        """Attach loaded data fields directly onto the given object (obj_ref)."""
        data_path = config.get("data_path", {})

        def safe_load(path_key, loader_fn, desc):
            if path_key in data_path:
                abs_path = self.to_abs(data_path[path_key])
                if os.path.exists(abs_path):
                    return loader_fn(abs_path)
                else:
                    print(f"[Warning] Missing {desc}: {abs_path}")
            return None

        # Query data
        obj_ref.query_data_train = safe_load("query_data_train", load_jsonl, "query_data_train")
        obj_ref.query_data_test = safe_load("query_data_test", load_jsonl, "query_data_test")

        # Embeddings
        obj_ref.query_embedding_data = safe_load("query_embedding_data", load_pt, "query_embedding_data")

        # Routing data
        obj_ref.routing_data_train = safe_load("routing_data_train", jsonl_to_csv, "routing_data_train")
        obj_ref.routing_data_test = safe_load("routing_data_test", jsonl_to_csv, "routing_data_test")

        # LLM info
        obj_ref.llm_data = safe_load("llm_data", lambda p: json.load(open(p, "r", encoding="utf-8")), "llm_data")
        obj_ref.llm_embedding_data = safe_load(
            "llm_embedding_data", lambda p: json.load(open(p, "r", encoding="utf-8")), "llm_embedding_data"
        )
