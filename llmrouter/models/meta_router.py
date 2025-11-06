import os
import yaml
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from llmrouter.data.data_loader import DataLoader


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
    def route(self, batch):
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

    def forward(self, batch):
        """
        PyTorch-compatible forward method.

        This simply delegates to `route()`, so that the router can be used
        like a regular nn.Module in training loops.
        """
        return self.route(batch)

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
