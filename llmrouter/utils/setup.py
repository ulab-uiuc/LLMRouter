"""
Setup utilities for LLMRouter scripts
"""

import os

def setup_environment():
    """Setup common environment variables and paths"""
    # All required modules are now local, no need for external paths

    # Set environment variables (only if not already set by user)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # Default to using all available GPUs; users can override by setting CUDA_VISIBLE_DEVICES
        pass
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
