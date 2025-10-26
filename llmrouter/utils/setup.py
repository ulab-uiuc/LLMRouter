"""
Setup utilities for LLMRouter scripts
"""

import os
import sys

def setup_environment():
    """Setup common environment variables and paths"""
    # Add paths for imports
    sys.path.append('/data/taofeng2/router_planner/embedding_based_router')
    sys.path.append('/data/taofeng2/router_planner/embedding_based_router/zijie_baseline')
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
