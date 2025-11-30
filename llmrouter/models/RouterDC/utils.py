"""
RouterDC Utilities
------------------
Utility functions for RouterDC training and evaluation.

This module provides helper classes and functions used throughout
the RouterDC implementation.

Original source: RouterDC/utils/meters.py
Adapted for LLMRouter framework.
"""

from typing import Optional, List
import torch
import torch.nn as nn


class AverageMeter:
    """
    AverageMeter
    ------------
    Computes and stores the average and current value.

    This is useful for tracking metrics like loss during training.

    Usage:
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter('Loss', ':3.2f')
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
        >>> print(losses)  # Print formatted average
    """

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        """
        Initialize AverageMeter.

        Args:
            name (str): Name of the metric being tracked
            fmt (str): Format string for printing (default: ':f')
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update statistics with a new value.

        Args:
            val (float): New value to add
            n (int): Number of samples this value represents (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        """
        Format meter as string.

        Returns:
            str: Formatted string showing current and average values
        """
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """
    Compute the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor):
            Classification outputs, shape (N, C) where C = number of classes
        target (torch.Tensor):
            Ground truth labels, shape (N) where each value is 0 <= target[i] < C
        topk (tuple[int]):
            A tuple of top-N numbers to compute accuracy for (default: (1,))

    Returns:
        list[torch.Tensor]:
            List of top-N accuracies (N in topk), each as a percentage

    Example:
        >>> output = torch.randn(32, 10)  # 32 samples, 10 classes
        >>> target = torch.randint(0, 10, (32,))  # 32 labels
        >>> acc1, acc5 = accuracy(output, target, topk=(1, 5))
        >>> print(f"Top-1 Accuracy: {acc1.item():.2f}%")
        >>> print(f"Top-5 Accuracy: {acc5.item():.2f}%")
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top-k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        # Compute accuracy for each k
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class _SimpleBatch(dict):
    """Dictionary that also exposes attributes like HuggingFace BatchEncoding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


class TinyTokenizer:
    """
    Lightweight tokenizer used for offline RouterDC tests.

    Tokenizes by splitting on whitespace and mapping tokens to a tiny vocab.
    """

    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[UNK]": 2}

    def _tokenize(self, text: str) -> List[int]:
        tokens = text.lower().split()
        token_ids = [1]  # CLS token
        for token in tokens:
            idx = self.vocab.get(token)
            if idx is None:
                idx = len(self.vocab)
                self.vocab[token] = idx
            token_ids.append(idx % self.vocab_size)
        return token_ids

    def __call__(
        self,
        text: str,
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        **_: dict,
    ):
        token_ids = self._tokenize(text) if add_special_tokens else text.split()
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding == "max_length" and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        if return_tensors == "pt":
            ids_tensor = torch.tensor([token_ids], dtype=torch.long)
            mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
        else:
            raise ValueError("TinyTokenizer only supports return_tensors='pt'")

        return _SimpleBatch(input_ids=ids_tensor, attention_mask=mask_tensor)


class TinyBackbone(nn.Module):
    """
    Minimal encoder used for offline RouterDC execution.
    """

    def __init__(self, vocab_size: int = 2048, hidden_state_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_state_dim)
        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=True)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        encoded, _ = self.gru(embeddings)
        return {"last_hidden_state": encoded}


def load_tokenizer_and_backbone(model_cfg: dict):
    """
    Load tokenizer and backbone based on configuration.

    Supports:
        - HuggingFace backbones (requires transformers)
        - Local tiny backbone for tests (backbone == \"routerdc-tiny\")
    """
    model_name = model_cfg["backbone"]
    hidden_dim = model_cfg.get("hidden_state_dim", 768)

    if model_name.lower() in {"routerdc-tiny", "routerdc_debug", "debug"}:
        tokenizer = TinyTokenizer()
        backbone = TinyBackbone(
            vocab_size=tokenizer.vocab_size,
            hidden_state_dim=hidden_dim,
        )
        return tokenizer, backbone, hidden_dim

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - only triggered without transformers
        raise ImportError(
            "transformers is required for RouterDC backbones other than 'routerdc-tiny'."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left",
        padding=True,
    )
    backbone = AutoModel.from_pretrained(model_name)
    return tokenizer, backbone, backbone.config.hidden_size
