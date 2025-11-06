"""
Automix Router Package
======================

This package provides the Automix routing strategy for the LLMRouter framework.

Automix is a cost-effective routing method that uses self-verification to decide
when to route queries from a small language model to a larger, more capable model.

Main Components:
---------------
- **AutomixRouter**: Main router class (inherits from MetaRouter)
- **AutomixRouterTrainer**: Trainer for Automix (inherits from BaseTrainer)
- **AutomixModel**: PyTorch model wrapper for Automix logic
- **Routing Methods**: Threshold, POMDP, SelfConsistency, etc.
- **Data Pipeline**: Functions for data preparation and verification

Usage Example:
-------------
```python
from llmrouter.models.Automix import (
    AutomixRouter,
    AutomixRouterTrainer,
    AutomixModel,
    POMDP,
    Threshold,
    SelfConsistency,
    prepare_automix_data,
)

# 1. Prepare data (run predictions and verification)
df = prepare_automix_data(
    input_data_path="./data/queries.jsonl",
    output_dir="./data",
)

# 2. Create routing method
method = POMDP(num_bins=8)

# 3. Create model and router
model = AutomixModel(method=method)
router = AutomixRouter(model=model)

# 4. Create trainer
trainer = AutomixRouterTrainer(router=router)

# 5. Train on data
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']
results = trainer.train_and_evaluate(train_df, test_df)

print(f"Test IBC Lift: {results['test']['ibc_lift']:.4f}")
print(f"Test Performance: {results['test']['avg_performance']:.4f}")
```

Original Source:
---------------
Based on the Automix paper and implementation:
- Paper: "Automix: Automatically Mixing Language Models"
- Original code: automix/colabs/

Adapted for LLMRouter framework with:
- PyTorch nn.Module interface
- MetaRouter/BaseTrainer integration
- Improved documentation and type hints
"""

# Import routing methods
from .methods import (
    Threshold,
    DoubleThreshold,
    TripleThreshold,
    SelfConsistency,
    POMDPSimple,
    GreedyPOMDP,
    AutomixUnion,
    FixedAnswerRouting,
    POMDP,
)

# Import core classes
from .model import AutomixModel
from .router import AutomixRouter
from .trainer import AutomixRouterTrainer

# Import data pipeline functions
from .data_pipeline import (
    prepare_automix_data,
    solve_queries,
    self_verify,
    init_providers,
    normalize_answer,
    compute_f1,
    f1_score_single,
    categorize_rows,
)

# Define public API
__all__ = [
    # Routing methods
    "Threshold",
    "DoubleThreshold",
    "TripleThreshold",
    "SelfConsistency",
    "POMDPSimple",
    "GreedyPOMDP",
    "AutomixUnion",
    "FixedAnswerRouting",
    "POMDP",
    # Core classes
    "AutomixModel",
    "AutomixRouter",
    "AutomixRouterTrainer",
    # Data pipeline
    "prepare_automix_data",
    "solve_queries",
    "self_verify",
    "init_providers",
    "normalize_answer",
    "compute_f1",
    "f1_score_single",
    "categorize_rows",
]

# Package metadata
__version__ = "1.0.0"
__author__ = "LLMRouter Team"
__description__ = "Automix routing strategy for LLMRouter framework"
