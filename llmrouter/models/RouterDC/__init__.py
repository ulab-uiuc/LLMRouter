"""
RouterDC - Dual-Contrastive Learning for LLM Routing
=====================================================

RouterDC is a routing strategy that uses dual-contrastive learning to make
intelligent routing decisions between multiple LLMs.

Key components:
    - RouterModule: Core model with encoder + learnable LLM embeddings
    - RouterDCRouter: Router implementation (inherits MetaRouter)
    - RouterDCTrainer: Trainer with dual-contrastive learning (inherits BaseTrainer)
    - RouterDataset: Dataset class for loading training data
    - Utility functions: AverageMeter, accuracy

Original source: RouterDC/train_router_mdeberta.py
Adapted for LLMRouter framework.

Usage example:
    >>> from transformers import AutoTokenizer, DebertaV2Model
    >>> from llmrouter.models.RouterDC import (
    ...     RouterModule, RouterDCRouter, RouterDCTrainer, RouterDataset
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
    >>> # 3. Wrap in RouterDCRouter
    >>> router = RouterDCRouter(model=model)
    >>>
    >>> # 4. Create trainer
    >>> trainer = RouterDCTrainer(
    ...     router=router,
    ...     top_k=3,
    ...     last_k=3,
    ...     cluster_loss_weight=1.0
    ... )
    >>>
    >>> # 5. Prepare data
    >>> dataset = RouterDataset(data_path="train.json")
    >>> dataset.register_tokenizer(tokenizer)
    >>>
    >>> # 6. Train
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    >>> trainer.train(dataloader, training_steps=1000)
"""

from .model import RouterModule
from .router import RouterDCRouter
from .trainer import RouterDCTrainer
from .dataset import RouterDataset
from .utils import AverageMeter, accuracy, load_tokenizer_and_backbone
from .data_pipeline import (
    convert_format,
    add_clusters,
    convert_train_data,
    generate_test_scores,
    prepare_routerdc_data,
)

__all__ = [
    # Core components
    "RouterModule",
    "RouterDCRouter",
    "RouterDCTrainer",
    "RouterDataset",
    # Utilities
    "AverageMeter",
    "accuracy",
    "load_tokenizer_and_backbone",
    # Data preprocessing
    "convert_format",
    "add_clusters",
    "convert_train_data",
    "generate_test_scores",
    "prepare_routerdc_data",
]

__version__ = "1.0.0"
