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


class LargestLLM(MetaRouter):
    """
    LargestLLM Router
    -----------------
    A heuristic router that always selects the largest LLM based on the
    'size' field in `self.llm_data`, restricted to models whose size
    string ends with 'B'.

    This router does not perform any learning and does not depend on
    the input batch. It only uses metadata loaded from the YAML config.
    """

    def __init__(self, yaml_path: str):
        """
        Initialize the LargestLLM router.

        Args:
            yaml_path (str):
                Path to the YAML configuration file. The corresponding
                DataLoader is expected to populate `self.llm_data` based
                on this configuration.
        """
        # Use a dummy model because this router does not train or forward
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)
        print("✅ LargestLLM initialized successfully")

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query to the largest LLM (by size ending with 'B').

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

        # --- Select the largest model among candidates ---
        largest_model_name = max(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B")),
        )

        query_output = copy.copy(query)
        query_output["model_name"]=largest_model_name

        return query_output


    def route_batch(self, batch: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Select the largest LLM (by size) whose size string ends with 'B'.

        This method ignores the input batch and purely relies on
        `self.llm_data`, which should be attached by DataLoader during
        MetaRouter initialization.

        Args:
            batch (Any, optional):
                Unused input. Kept for interface compatibility.

        Returns:
            dict:
                A dictionary containing:
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

        # Find the largest model among the filtered candidates
        largest_model_name = max(
            filtered_names,
            key=lambda k: parse_size(self.llm_data[k].get("size", "0B")),
        )

        query_data_output = copy.copy(self.query_data_test)
        for row in query_data_output:
            row["model_name"]=largest_model_name


        return query_data_output

    # LLMRouter /
    # ├── README.md
    # ├── LICENSE
    # ├── pyproject.toml  # Build configuration for pip/poetry
    # ├── setup.cfg  # Supplementary setup() configuration
    # ├── requirements.txt  # Dependencies list
    # ├──.gitignore
    # │
    # ├── llmrouter /  # Main library source (import llmrouter after installation)
    # │   ├── __init__.py
    # │   │
    # │   ├── config /  # Global configuration and registration system
    # │   │   ├── __init__.py
    # │   │   ├── defaults.py  # Default parameters, paths, API keys
    # │   │   ├── registry.py  # Model/template registry (register_model, register_router)
    # │   │   ├── templates /  # Prompt templates for different agent roles
    # │   │   │   ├── base_user.json
    # │   │   │   ├── planner.json
    # │   │   │   ├── executor.json
    # │   │   │   └── summarizer.json
    # │   │   └── schemas /  # JSON Schemas for validation
    # │   │       ├── dataset_schema.json
    # │   │       └── router_schema.json
    # │   │
    # │   ├── data /  # Data processing and loading modules
    # │   │   ├── __init__.py
    # │   │   ├── loader.py  # Load LLMFusionBench or custom datasets
    # │   │   ├── processor.py  # Embedding generation, normalization, context building
    # │   │   ├── splitter.py  # Random / OOD splits
    # │   │   ├── formatter.py  # Format converters (standard JSON interface)
    # │   │   ├── downloader.py  # Automatic benchmark data downloader
    # │   │   └── example_data /  # Example data for demos and tests
    # │   │       ├── qa.json
    # │   │       ├── code.json
    # │   │       ├── math.json
    # │   │       └── routing_sample.json
    # │   │
    # │   ├── models /  # Router and model implementations
    # │   │   ├── __init__.py
    # │   │   ├── meta_router.py  # MetaRouter (abstract router base class, defines fit/route/evaluate)
    # │   │   ├── user_aware.py  # PersonalizedRouter, GMTRouter
    # │   │   ├── user_agnostic.py  # Router-KNN, Router-SVM, Router-MLP, Best/Smallest LLM
    # │   │   ├── router_dc.py  # RouterDC
    # │   │   ├── graph_router.py  # GraphRouter
    # │   │   ├── hybrid_router.py  # HybridLLM, FrugalGPT, ICL-Router
    # │   │   ├── embedding_router.py  # Embedding-based router
    # │   │   ├── multi_round.py  # Multi-round routers (Router-KNN-MR, Router-R1)
    # │   │   └── agentic_router.py  # Agentic routers (GraphPlanner, R2-Reasoner)
    # │   │
    # │   ├── evaluation /  # Evaluation and metrics module
    # │   │   ├── __init__.py
    # │   │   ├── metrics.py  # P0–P2 metrics (performance, cost, preference) [{"query":,"response","},]
    # │   │   ├── cost.py  # Token cost calculation
    # │   │   ├── judge.py  # LLM-as-a-Judge scoring
    # │   │   ├── analysis.py  # Pareto frontier and load balancing analysis
    # │   │   └── reports /  # Stored evaluation results and plots
    # │   │       ├── run_2025_10.json
    # │   │       └── pareto_plot.png
    # │   │
    # │   ├── agentic /  # Agent-level modules
    # │   │   ├── __init__.py
    # │   │   ├── planner.py  # GraphPlanner (task decomposition)
    # │   │   ├── executor.py  # Execution agent
    # │   │   ├── summarizer.py  # Summary agent
    # │   │   └── roles.py  # Role registry (executor / planner / summarizer)
    # │   │
    # │   ├── utils /  # General utilities
    # │   │   ├── __init__.py
    # │   │   ├── io.py  # File I/O helpers
    # │   │   ├── logging.py  # Logging utilities
    # │   │   ├── registry_utils.py  # Decorators for registry registration
    # │   │   ├── embedding.py  # Vector math and embedding utilities
    # │   │   ├── visualization.py  # Visualization (graph, Pareto, t-SNE)
    # │   │   └── decorators.py  # @timeit, @cache_route, @safe_execute
    # │   │
    # │   ├── cli /  # Command-line interface (CLI) entry points
    # │   │   ├── __init__.py
    # │   │   ├── main.py  # Main CLI entry (e.g., `llmrouter`)
    # │   │   ├── train.py  # CLI command: `llmrouter train --config configs/router/mlp.yaml`
    # │   │   ├── eval.py  # CLI command: `llmrouter eval`
    # │   │   ├── list_models.py  # CLI command: `llmrouter models`
    # │   │   └── visualize.py  # CLI command: `llmrouter viz`
    # │   │
    # │   └── examples /  # Example scripts and tutorials
    # │       ├── run_meta_router.py
    # │       ├── run_graph_router.py
    # │       ├── run_agentic_router.py
    # │       ├── run_user_router.py
    # │       └── evaluate_all.py
    # │
    # ├── tests /  # Unit and integration tests
    # │   ├── test_loader.py
    # │   ├── test_router_base.py
    # │   ├── test_eval_metrics.py
    # │   ├── test_meta_router.py
    # │   └── test_cli.py
    # │
    # └── docs /  # Documentation
    # ├── index.md
    # ├── quickstart.md
    # ├── api_reference.md
    # ├── developer_guide.md
    # └── assets /
    # ├── architecture.png
    # └── data_flow.pdf