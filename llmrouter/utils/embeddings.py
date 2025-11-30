"""
Embedding utilities for LLMRouter scripts (Longformer version)
"""

from transformers import AutoTokenizer, AutoModel
import torch


# -------------------------
# 1. Model and tokenizer initialization
# -------------------------
model_name: str = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Automatically select device
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Longformer model loaded on {device}.")


def get_longformer_embedding(texts):
    """
    Get Longformer embeddings for given texts.

    Args:
        texts (str or list[str]): Input text(s) to encode.

    Returns:
        torch.Tensor or list[torch.Tensor]:
            - Single embedding (torch.Tensor) if only one input text.
            - Batch embeddings (torch.Tensor) if multiple input texts.
    """
    try:
        # -------------------------
        # 2. Input handling
        # -------------------------
        if isinstance(texts, str):
            texts = [texts]

        # -------------------------
        # 3. Tokenization
        # -------------------------
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        ).to(device)

        # -------------------------
        # 4. Model forward pass
        # -------------------------
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state: torch.Tensor = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # -------------------------
        # 5. Mean pooling
        # -------------------------
        attention_mask: torch.Tensor = inputs["attention_mask"]
        mask_expanded: torch.Tensor = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sentence_embeddings: torch.Tensor = (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1)

        # -------------------------
        # 6. Return result (move to CPU for safety)
        # -------------------------
        sentence_embeddings = sentence_embeddings.cpu()

        if len(texts) == 1:
            return sentence_embeddings[0]
        else:
            return sentence_embeddings

    except Exception as e:
        print(f"Error generating Longformer embeddings: {e}")
        return ""


def parallel_embedding_task(data):
    """
    Parallel task for generating Longformer embeddings.

    Args:
        data (tuple): (id, query_text)

    Returns:
        tuple: (id, query_embedding, success_flag)
    """
    success: bool = True
    id, query_t = data
    query_t_embedding = ""

    try:
        # Compute embedding
        query_t_embedding = get_longformer_embedding(query_t)
    except Exception as e:
        print(f"Error in parallel embedding task (id={id}): {e}")
        success = False
        query_t_embedding = ""

    return id, query_t_embedding, success
