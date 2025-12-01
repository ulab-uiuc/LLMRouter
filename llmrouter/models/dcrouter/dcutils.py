"""
DCRouter Utilities
------------------
Utility functions for DCRouter training and evaluation.

This module provides helper classes and functions used throughout
the DCRouter implementation.

Original source: RouterDC/utils/meters.py
Adapted for LLMRouter framework.
"""

from typing import Optional, List
import torch


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
