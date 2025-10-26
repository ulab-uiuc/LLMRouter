"""
Progress tracking utilities for LLMRouter scripts
"""

import time
import threading
from tqdm import tqdm

class ProgressTracker:
    """Progress tracking for parallel processing"""
    def __init__(self, total: int, desc: str = "Processing"):
        self.pbar = tqdm(total=total, desc=desc)
        self.lock = threading.Lock()
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
    
    def update(self, success: bool = True, model_name: str = None):
        with self.lock:
            if success:
                self.completed += 1
            else:
                self.errors += 1
            
            # Calculate rate
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            
            self.pbar.update(1)
            postfix = {
                'completed': self.completed,
                'errors': self.errors,
                'rate': f"{rate:.1f}/s"
            }
            if model_name:
                postfix['model'] = model_name[:20]
            self.pbar.set_postfix(postfix)
    
    def close(self):
        self.pbar.close()
        elapsed = time.time() - self.start_time
        print(f"Total processing time: {elapsed:.1f}s")
        print(f"Average rate: {self.completed / elapsed:.1f} requests/second")
