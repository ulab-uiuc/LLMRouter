

# 📁 Project Structure — `LLMRouter`

```bash
LLMRouter/
├── README.md
├── LICENSE
├── pyproject.toml              # Build configuration for pip/poetry
├── setup.cfg                   # Supplementary setup() configuration
├── requirements.txt            # Dependencies list
├── .gitignore
│
├── llmrouter/                  # Main library source (import llmrouter after installation)
│   ├── __init__.py
│   │
│   ├── config/                 # Global configuration and registration system
│   │   ├── __init__.py
│   │   ├── defaults.py         # Default parameters, paths, API keys
│   │   ├── registry.py         # Model/template registry (register_model, register_router)
│   │   ├── templates/          # Prompt templates for different agent roles
│   │   │   ├── base_user.json
│   │   │   ├── planner.json
│   │   │   ├── executor.json
│   │   │   └── summarizer.json
│   │   └── schemas/            # JSON Schemas for validation
│   │       ├── dataset_schema.json
│   │       └── router_schema.json
│   │
│   ├── data/                   # Data processing and loading modules
│   │   ├── __init__.py
│   │   ├── loader.py           # Load LLMFusionBench or custom datasets
│   │   ├── processor.py        # Embedding generation, normalization, context building
│   │   ├── splitter.py         # Random / OOD splits
│   │   ├── formatter.py        # Format converters (standard JSON interface)
│   │   ├── downloader.py       # Automatic benchmark data downloader
│   │   └── example_data/       # Example data for demos and tests
│   │       ├── qa.json
│   │       ├── code.json
│   │       ├── math.json
│   │       └── routing_sample.json
│   │
│   ├── models/                 # Router and model implementations
│   │   ├── __init__.py
│   │   ├── meta_router.py      # MetaRouter (abstract router base class, defines fit/route/evaluate)
│   │   ├── user_aware.py       # PersonalizedRouter, GMTRouter
│   │   ├── user_agnostic.py    # Router-KNN, Router-SVM, Router-MLP, Best/Smallest LLM
│   │   ├── router_dc.py        # RouterDC
│   │   ├── graph_router.py     # GraphRouter
│   │   ├── hybrid_router.py    # HybridLLM, FrugalGPT, ICL-Router
│   │   ├── embedding_router.py # Embedding-based router
│   │   ├── multi_round.py      # Multi-round routers (Router-KNN-MR, Router-R1)
│   │   └── agentic_router.py   # Agentic routers (GraphPlanner, R2-Reasoner)
│   │
│   ├── evaluation/             # Evaluation and metrics module
│   │   ├── __init__.py
│   │   ├── metrics.py          # P0–P2 metrics (performance, cost, preference)
│   │   ├── cost.py             # Token cost calculation
│   │   ├── judge.py            # LLM-as-a-Judge scoring
│   │   ├── analysis.py         # Pareto frontier and load balancing analysis
│   │   └── reports/            # Stored evaluation results and plots
│   │       ├── run_2025_10.json
│   │       └── pareto_plot.png
│   │
│   ├── agentic/                # Agent-level modules
│   │   ├── __init__.py
│   │   ├── planner.py          # GraphPlanner (task decomposition)
│   │   ├── executor.py         # Execution agent
│   │   ├── summarizer.py       # Summary agent
│   │   └── roles.py            # Role registry (executor / planner / summarizer)
│   │
│   ├── utils/                  # General utilities
│   │   ├── __init__.py
│   │   ├── io.py               # File I/O helpers
│   │   ├── logging.py          # Logging utilities
│   │   ├── registry_utils.py   # Decorators for registry registration
│   │   ├── embedding.py        # Vector math and embedding utilities
│   │   ├── visualization.py    # Visualization (graph, Pareto, t-SNE)
│   │   └── decorators.py       # @timeit, @cache_route, @safe_execute
│   │
│   ├── cli/                    # Command-line interface (CLI) entry points
│   │   ├── __init__.py
│   │   ├── main.py             # Main CLI entry (e.g., `llmrouter`)
│   │   ├── train.py            # CLI command: `llmrouter train --config configs/router/mlp.yaml`
│   │   ├── eval.py             # CLI command: `llmrouter eval`
│   │   ├── list_models.py      # CLI command: `llmrouter models`
│   │   └── visualize.py        # CLI command: `llmrouter viz`
│   │
│   └── examples/               # Example scripts and tutorials
│       ├── run_meta_router.py
│       ├── run_graph_router.py
│       ├── run_agentic_router.py
│       ├── run_user_router.py
│       └── evaluate_all.py
│
├── tests/                      # Unit and integration tests
│   ├── test_loader.py
│   ├── test_router_base.py
│   ├── test_eval_metrics.py
│   ├── test_meta_router.py
│   └── test_cli.py
│
└── docs/                       # Documentation
    ├── index.md
    ├── quickstart.md
    ├── api_reference.md
    ├── developer_guide.md
    └── assets/
        ├── architecture.png
        └── data_flow.pdf



# ⚙️ Set up and initialization (wit pip install)
1. **Create a virtual environment**
```bash
   python -m venv myenv
```

2. **Activate the virtual environment**

   - On macOS/Linux:
```bash
     source myenv/bin/activate
```
   
   - On Windows:
```bash
     myenv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install mkdocs-material
```

### Running Locally

Start the development server:
```bash
mkdocs serve
```

The site will be available at `http://127.0.0.1:8000/`

