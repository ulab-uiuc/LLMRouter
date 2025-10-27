"""
Tensor processing utilities for LLMRouter scripts
"""

import ast
import re
import torch

def to_tensor(s: str) -> torch.Tensor:
    """Convert string representation to tensor"""
    s = s.strip()
    if s.startswith("tensor("):
        s = s[len("tensor("):].rstrip(")")
    s = re.sub(r"device='[^']*'", "", s)
    s = re.sub(r"dtype=\w+\.\w+", "", s)
    s = s.replace(", ,", ",").strip().rstrip(",")
    data = ast.literal_eval(s)
    return torch.tensor(data, dtype=torch.float32)
