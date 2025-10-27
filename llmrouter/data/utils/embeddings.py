"""
Embedding utilities for LLMRouter scripts
"""

from sentence_transformers import SentenceTransformer

def get_bert_representation(texts):
    """Get BERT embeddings for texts"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        return embeddings[0] if len(texts) == 1 else embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return ""

def parallel_embedding_task(data):
    """Parallel task for generating embeddings"""
    success = True
    id, query_t = data
    try:
        query_t_embedding = get_bert_representation([query_t])
    except Exception as e:
        print(e)
        query_t_embedding = ""
        success = False
    return id, query_t_embedding, success
