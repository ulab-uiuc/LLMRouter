"""Generate query embeddings (and LLM-candidate embeddings) for the benchmark data.

Mirrors the original xRouteBench embedding setup: Qwen/Qwen3-Embedding-0.6B,
last-token pooling, L2-normalized. Produces, per dataset:

    benchmark_pipeline/data/<dataset>/query_embeddings_train.pt
    benchmark_pipeline/data/<dataset>/query_embeddings_test.pt

Embeddings are indexed by the `embedding_id` column of the routing data, so
tensor row i corresponds to embedding_id == i.

Also produces the candidate-description embeddings:
    benchmark_pipeline/data/llm_candidates/default_llm_embeddings.json

Usage:
    python generate_embeddings.py                       # all datasets + candidates
    python generate_embeddings.py --datasets video
    python generate_embeddings.py --device cuda:1 --batch-size 16
"""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

DATASETS = [
    "llmrouter_generic", "memory_locomo", "memory_longmemeval", "timeseries",
    "video", "multimodal_geometry3k", "multimodal_mathvista", "personalized",
]
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch, seq_lens]


def embed_texts(texts, tokenizer, model, device, batch_size=8, max_length=512):
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu())
        if (i // batch_size) % 50 == 0:
            print(f"  {i + len(chunk)}/{len(texts)}", flush=True)
    return torch.cat(out, dim=0)


def query_to_text(q):
    """personalized stores queries as chat-message lists; flatten to text."""
    if isinstance(q, list):
        return "\n".join(m.get("content", "") for m in q if isinstance(m, dict))
    return str(q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-candidates", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                              padding_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Pin the dtype rather than inherit it. transformers 4.x loaded in float32 whatever the
    # checkpoint declared; transformers 5.x honours the checkpoint, and
    # Qwen3-Embedding-0.6B declares bfloat16. numpy has no bfloat16, so the saved tensors
    # then take down every router that reaches sklearn through .numpy().
    # Cast rather than pass a dtype argument, because the keyword was renamed between the
    # two versions this package supports.
    model = model.to(args.device).to(torch.float32).eval()

    for ds in args.datasets:
        for split in ("train", "test"):
            src = os.path.join(DATA_DIR, ds, f"routing_data_{split}.jsonl")
            dst = os.path.join(DATA_DIR, ds, f"query_embeddings_{split}.pt")
            if not os.path.exists(src):
                print(f"skip {ds}/{split}: {src} missing (run download_data.py first)")
                continue
            if os.path.exists(dst):
                print(f"skip {ds}/{split}: embeddings already exist")
                continue
            df = pd.read_json(src, lines=True)
            # one embedding per embedding_id, ordered by id
            uniq = (df[["embedding_id", "query"]]
                    .drop_duplicates("embedding_id")
                    .sort_values("embedding_id"))
            texts = [query_to_text(q) for q in uniq["query"]]
            print(f"{ds}/{split}: embedding {len(texts)} unique queries ...")
            emb = embed_texts(texts, tokenizer, model, args.device, args.batch_size)
            torch.save(emb, dst)
            print(f"  -> {dst}  shape={tuple(emb.shape)}")

    if not args.skip_candidates:
        pool_path = os.path.join(DATA_DIR, "llm_candidates", "default_llm.json")
        out_path = os.path.join(DATA_DIR, "llm_candidates", "default_llm_embeddings.json")
        if os.path.exists(pool_path) and not os.path.exists(out_path):
            pool = json.load(open(pool_path))
            names = list(pool.keys())
            texts = [pool[n].get("feature", n) for n in names]
            print(f"llm_candidates: embedding {len(texts)} model descriptions ...")
            emb = embed_texts(texts, tokenizer, model, args.device, args.batch_size)
            json.dump({n: emb[i].tolist() for i, n in enumerate(names)},
                      open(out_path, "w"))
            print(f"  -> {out_path}")

    print("\nNext step: python run_pipeline.py --datasets all --routers local")


if __name__ == "__main__":
    main()
