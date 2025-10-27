"""
Setup utilities for LLMRouter scripts
"""

import os
import sys

def setup_environment():
    """Setup common environment variables and paths"""
    # All required modules are now local, no need for external paths
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
