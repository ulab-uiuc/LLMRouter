import copy
import os
import yaml
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from llmrouter.data import DataLoader


class MetaRouter(nn.Module, ABC):
    """
    MetaRouter (Base Class)
    -----------------------
    Unified abstraction for all LLM routers.

    Responsibilities:
        - Hold an underlying PyTorch model (nn.Module)
        - Optionally load configuration and data
        - Provide a standard routing interface: `route()` / `forward()`
        - Provide basic utilities: metrics, save/load

    Training logic is intentionally decoupled and handled by Trainer classes.
    """

    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        """
        Args:
            model (nn.Module):
                The underlying PyTorch model that performs routing computation.
            yaml_path (str | None):
                Optional path to a YAML config file. If provided, configuration
                and data will be loaded during initialization.
            resources (Any, optional):
                Optional shared resources or context (e.g., tokenizer, env, etc.).
        """
        super().__init__()
        self.model = model
        self.resources = resources
        self.cfg = {}
        self.metric_weights = []

        if yaml_path is not None:
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"YAML file not found: {yaml_path}")

            with open(yaml_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

            # Compute project root (two levels up from models/)
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../..")
            )

            # Load data via DataLoader (side-effect: attach datasets to `self`)
            loader = DataLoader(project_root)
            loader.load_data(self.cfg, self)

            # Load metric weights if provided
            weights_dict = self.cfg.get("metric", {}).get("weights", {})
            self.metric_weights = list(weights_dict.values())

            print("✅ MetaRouter initialized successfully (YAML + data loaded).")

    # ------------------------------------------------------------------
    # Core abstract method: subclasses must define routing behavior
    # ------------------------------------------------------------------

    @abstractmethod
    def route_batch(self, batch):
        """
        Define how routing decisions are computed.

        Args:
            batch (Any):
                Input batch for routing. The exact structure (dict, tensor, etc.)
                is defined by each specific router implementation.

        Returns:
            Any:
                Routing outputs such as logits, scores, or selected model indices.
        """
        raise NotImplementedError

    @abstractmethod
    def route_single(self, batch):
        """
        Define how routing decisions are computed.

        Args:
            batch (Any):
                Input batch for routing. The exact structure (dict, tensor, etc.)
                is defined by each specific router implementation.

        Returns:
            Any:
                Routing outputs such as logits, scores, or selected model indices.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers for concrete routers
    # ------------------------------------------------------------------

    def _resolve_query_data(self, batch):
        """Resolve the rows to route.

        Uses an explicit ``batch`` when provided, otherwise falls back to the
        loaded ``query_data_test``. Returns a list of rows, or ``None`` when
        neither source is available (callers should then return an empty result).
        """
        if batch is not None:
            return batch if isinstance(batch, list) else [batch]
        if getattr(self, "query_data_test", None) is not None:
            return copy.copy(self.query_data_test)
        print("Warning: No batch provided and no test data available for batch routing.")
        return None

    @staticmethod
    def _normalize_row(row, task_name):
        """Normalize one routing input row.

        Returns ``(row_copy, original_query, row_task_name)``. A dict row is
        shallow-copied and its ``query``/``task_name`` read out; any non-dict
        row is wrapped as ``{"query": str(row)}``. The per-row task name falls
        back to the batch-level ``task_name`` when absent.
        """
        if isinstance(row, dict):
            row_copy = copy.copy(row)
            original_query = row_copy.get("query", "")
            row_task_name = row_copy.get("task_name", task_name)
        else:
            row_copy = {"query": str(row)}
            original_query = str(row)
            row_task_name = task_name
        return row_copy, original_query, row_task_name

    def forward(self, batch):
        """
        PyTorch-compatible forward method.

        This simply delegates to `route_batch()`, so that the router can be used
        like a regular nn.Module in training loops.
        """
        return self.route_batch(batch)

    # ------------------------------------------------------------------
    # Optional shared utilities
    # ------------------------------------------------------------------

    def compute_metrics(self, outputs, batch) -> dict:
        """
        Optional metric computation function.

        Subclasses can override this method to define common evaluation metrics
        (e.g., accuracy, cost, latency) based on routing outputs.

        Args:
            outputs (Any):
                Model or routing outputs from `route()`.
            batch (Any):
                Original input batch, possibly containing labels and meta info.

        Returns:
            dict:
                A dictionary of metric_name -> value.
        """
        return {}

    def save_router(self, path: str):
        """
        Save the entire router state dict to disk.

        Args:
            path (str):
                Target file path for saving the state.
        """
        torch.save(self.state_dict(), path)
        print(f"💾 Router state saved to: {path}")

    def load_router(self, path: str):
        """
        Load the router state dict from disk.

        Args:
            path (str):
                Source file path of a previously saved state.
        """
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)
        print(f"📂 Router state loaded from: {path}")
