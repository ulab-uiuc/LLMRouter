"""
Data processing utilities for LLMRouter scripts
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

from .embeddings import parallel_embedding_task
from .tensor_utils import to_tensor
from .dataframe_utils import clean_df

def process_final_data(df_all):
    """
    Process final data to match process.py format
    
    Args:
        df_all: DataFrame with all data including query_embedding column
        
    Returns:
        tuple: (df_train_indexed, df_test_indexed, embedding_dict)
    """
    print("=== FINAL PROCESSING ===")
    
    # Step 1: Extract unique query embedding
    print("Extracting unique query embeddings...")
    embedding_df = (
        df_all.groupby(['task_name', 'query', 'gt', 'metric'], as_index=False)
              .first()[['task_name', 'query', 'gt', 'metric', 'query_embedding']]
    )
    
    # Step 2: Generate embedding_id and save pt
    print("Generating embedding IDs and saving tensors...")
    embedding_df['embedding_id'] = range(len(embedding_df))
    
    embedding_dict = {}
    for _, row in embedding_df.iterrows():
        embedding_id = int(row['embedding_id'])
        query_embedding = row['query_embedding']
        
        # Handle numpy array embeddings directly
        if isinstance(query_embedding, np.ndarray):
            embedding_dict[embedding_id] = torch.tensor(query_embedding, dtype=torch.float32)
        else:
            # Handle string embeddings (fallback)
            embedding_dict[embedding_id] = to_tensor(query_embedding)
    
    torch.save(embedding_dict, "query_embeddings.pt")
    print(f"✅ Saved {len(embedding_dict)} vectors to query_embeddings.pt")
    
    # Step 3: Merge embedding_id back to main data
    print("Merging embedding IDs...")
    df_all_indexed = df_all.merge(
        embedding_df[['task_name', 'query', 'gt', 'metric', 'embedding_id']],
        on=['task_name', 'query', 'gt', 'metric'],
        how='left'
    )
    
    # Step 4: Clean DataFrame
    print("Cleaning DataFrame...")
    df_all_indexed = clean_df(df_all_indexed)
    
    # Step 5: Split into train/test (80/20 split)
    print("Splitting into train/test sets...")
    total_size = len(df_all_indexed)
    train_size = int(0.8 * total_size)
    
    df_train_indexed = df_all_indexed.iloc[:train_size]
    df_test_indexed = df_all_indexed.iloc[train_size:]
    
    # Step 6: Save as JSONL files
    print("Saving final files...")
    df_train_indexed.to_json("default_routing_train_data.jsonl", orient="records", lines=True, force_ascii=False)
    df_test_indexed.to_json("default_routing_test_data.jsonl", orient="records", lines=True, force_ascii=False)
    
    print(f"✅ Final files saved:")
    print(f"  - default_routing_train_data.jsonl ({len(df_train_indexed)} records)")
    print(f"  - default_routing_test_data.jsonl ({len(df_test_indexed)} records)")
    print(f"  - query_embeddings.pt ({len(embedding_dict)} embeddings)")
    
    return df_train_indexed, df_test_indexed, embedding_dict

def generate_embeddings_for_data(data, desc="Generating embeddings"):
    """
    Generate embeddings for a list of data items
    
    Args:
        data: List of data items with 'query' field
        desc: Description for progress bar
        
    Returns:
        list: List of (id, embedding, success) tuples
    """
    print("Generating embeddings...")
    
    task_args = [(id, row['query']) for id, row in enumerate(data)]
    ret_1 = []
    with ThreadPool(100) as p:
        for r in tqdm(p.imap_unordered(parallel_embedding_task, task_args), 
                     total=len(task_args), desc=desc, ncols=100):
            ret_1.append(r)
    
    ret_1.sort(key=lambda x: x[0], reverse=False)
    fail_count = sum(1 for r in ret_1 if not r[-1])
    print(f"Embedding generation complete: Success: {len(ret_1) - fail_count}, Fail: {fail_count}")
    
    return ret_1
