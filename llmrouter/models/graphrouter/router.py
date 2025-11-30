import torch.nn as nn
from llmrouter.models.meta_router import MetaRouter


class GraphRouter(MetaRouter):
    """
    GraphRouter
    -----------
    Example router that performs graph-based routing.

    This class:
        - Inherits MetaRouter to reuse configuration and utilities
        - Implements the `route()` method using the underlying `model`
    """

    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        """
        Args:
            model (nn.Module):
                Underlying graph-based PyTorch model (e.g., a GNN).
            yaml_path (str | None):
                Optional path to YAML config for this router.
            resources (Any, optional):
                Additional shared resources, if needed.
        """
        super().__init__(model=model, yaml_path=yaml_path, resources=resources)

    def route(self, batch):
        """
        Perform routing on a batch of graph data.

        Args:
            batch (dict):
                A batch containing:
                    - "graph": graph structure or tensors
                    - possibly other fields

        Returns:
            dict:
                A dictionary with routing-related outputs, e.g.:
                    - "logits": raw model outputs for each candidate LLM
        """
        graph = batch["graph"]
        logits = self.model(graph)
        return {"logits": logits}
