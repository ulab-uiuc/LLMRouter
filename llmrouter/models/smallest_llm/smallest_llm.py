from typing import Any, Dict, List, Optional

import torch.nn as nn
from llmrouter.models.meta_router import MetaRouter
import copy


def parse_size(size_str: str) -> float:
    """
    Parse a model size string (e.g., '7B', '13B', '512M') into
    a numeric value in billions.

    Supported suffixes:
        - K: thousands
        - M: millions
        - B: billions
        - T: trillions

    If parsing fails, this function returns 0.0.
    """
    size_str = str(size_str).strip().upper()
    try:
        if size_str.endswith("K"):
            return float(size_str[:-1]) / 1e6
        elif size_str.endswith("M"):
            return float(size_str[:-1]) / 1e3
        elif size_str.endswith("B"):
            return float(size_str[:-1])
        elif size_str.endswith("T"):
            return float(size_str[:-1]) * 1e3
        else:
            # Treat raw numeric string as billions directly
            return float(size_str)
    except Exception:
        return 0.0


class SmallestLLM(MetaRouter):
    """
    SmallestLLM Router
    ------------------
    A heuristic router that always selects the smallest LLM based on the
    'size' field in `self.llm_data`, restricted to models whose size
    string ends with 'B'.

    This router does not perform any learning and does not depend on
    the input batch. It only uses metadata loaded from the YAML config.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the SmallestLLM router.

        Args:
            yaml_path (str):
                Path to the YAML configuration file. The corresponding
                DataLoader is expected to populate `self.llm_data` based
                on this configuration.
        """
        # Use a dummy model because this router does not train or forward
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)
        print("✅ SmallestLLM initialized successfully")

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query to the smallest LLM (by size ending with 'B').

        This method ignores the content of the input query and purely relies on
        `self.llm_data`, which should be populated during MetaRouter initialization.

        Args:
            query (dict):
                A single query dictionary. The content is unused here but required
                for interface compatibility with multi-query routing methods.

        Returns:
            dict:
                A dictionary containing:
                    - "model_name": name of the selected model
                    - "model_size": size string of the selected model
                    - "model_info": full metadata entry from `self.llm_data`
        """
        # --- Validate LLM metadata ---
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError(
                "LLM data not loaded or missing in YAML configuration. "
                "Expected `self.llm_data` to be populated by DataLoader."
            )

        # --- Filter models whose size ends with 'B' ---
        filtered_names = [
            name
            for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str)
               and info["size"].upper().endswith("B")
        ]

        if not filtered_names:
            raise ValueError(
                "No models with size ending in 'B' found in `llm_data`."
            )

        # --- Select the smallest model among candidates ---
        smallest_model_name = min(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B")),
        )

        query_output = copy.copy(query)
        query_output["model_name"] = smallest_model_name

        return query_output

    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Select the smallest LLM (by size) whose size string ends with 'B'.

        This method ignores the input batch and purely relies on
        `self.llm_data`, which should be attached by DataLoader during
        MetaRouter initialization.

        Args:
            batch (Any, optional):
                Unused input. Kept for interface compatibility.

        Returns:
            list of dict:
                Each dictionary contains:
                    - "model_name": name of the selected model
                    - "model_size": size string of the selected model
                    - "model_info": full metadata entry from `self.llm_data`
        """
        if not hasattr(self, "llm_data") or not self.llm_data:
            raise ValueError(
                "LLM data not loaded or missing in YAML configuration. "
                "Expected `self.llm_data` to be populated by DataLoader."
            )

        # Filter only models whose size ends with 'B'
        filtered_names = [
            name
            for name, info in self.llm_data.items()
            if isinstance(info.get("size", ""), str)
            and info["size"].upper().endswith("B")
        ]

        if not filtered_names:
            raise ValueError(
                "No models with size ending in 'B' found in `llm_data`."
            )

        # Find the smallest model among the filtered candidates
        smallest_model_name = min(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B")),
        )

        query_data_output = copy.copy(self.query_data_test)
        for row in query_data_output:
            row["model_name"] = smallest_model_name

        return query_data_output

