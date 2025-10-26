"""
DataFrame processing utilities for LLMRouter scripts
"""

import pandas as pd

def clean_df(df):
    """Clean and standardize DataFrame columns"""
    df = df.drop(columns=['query_embedding'], errors='ignore')
    df = df.drop(columns=['formatted_query'], errors='ignore')
    df["user_id"] = None
    df["fig_id"] = None
    df.rename(columns={"prompt_tokens": "input_tokens"}, inplace=True)
    df.rename(columns={"completion_tokens": "output_tokens"}, inplace=True)
    df.rename(columns={"gt": "ground_truth"}, inplace=True)
    return df
