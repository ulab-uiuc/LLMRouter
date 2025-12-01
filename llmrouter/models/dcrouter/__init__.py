"""
DCRouter - Dual-Contrastive Learning for LLM Routing
====================================================

DCRouter is a routing strategy that uses dual-contrastive learning to make
intelligent routing decisions between multiple LLMs.

Key components:
    - RouterModule: Core model with encoder + learnable LLM embeddings
    - DCRouter: Router implementation (inherits MetaRouter)
    - DCTrainer: Trainer with dual-contrastive learning (inherits BaseTrainer)
    - DCDataset: Dataset class for loading training data
    - Utility functions: AverageMeter, accuracy

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework.

Usage example:
    >>> from transformers import AutoTokenizer, DebertaV2Model
    >>> from llmrouter.models.dcrouter import (
    ...     RouterModule, DCRouter, DCTrainer, DCDataset
    ... )
    >>>
    >>> # 1. Load backbone model
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    >>> encoder = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")
    >>>
    >>> # 2. Create RouterModule
    >>> model = RouterModule(
    ...     backbone=encoder,
    ...     hidden_state_dim=768,
    ...     node_size=6,  # number of LLMs
    ...     similarity_function="cos"
    ... )
    >>>
    >>> # 3. Wrap in DCRouter
    >>> router = DCRouter(model=model)
    >>>
    >>> # 4. Create trainer
    >>> trainer = DCTrainer(
    ...     router=router,
    ...     top_k=3,
    ...     last_k=3,
    ...     cluster_loss_weight=1.0
    ... )
    >>>
    >>> # 5. Prepare data
    >>> dataset = DCDataset(data_path="train.json")
    >>> dataset.register_tokenizer(tokenizer)
    >>>
    >>> # 6. Train
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    >>> trainer.train(dataloader, training_steps=1000)
"""

from .dcmodel import RouterModule
from .dcrouter import DCRouter
from .dctrainer import DCTrainer
from .dcdataset import DCDataset
from .dcutils import AverageMeter, accuracy

__all__ = [
    # Core components
    "RouterModule",
    "DCRouter",
    "DCTrainer",
    "DCDataset",
    # Utilities
    "AverageMeter",
    "accuracy",
]

__version__ = "1.0.0"
