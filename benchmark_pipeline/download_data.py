"""Download xRouteBench routing data from Hugging Face into the local layout
expected by the LLMRouter training/inference pipeline.

Creates, per dataset:
    benchmark_pipeline/data/<dataset>/routing_data_train.jsonl
    benchmark_pipeline/data/<dataset>/routing_data_test.jsonl
plus the shared candidate pool:
    benchmark_pipeline/data/llm_candidates/default_llm.json

Usage:
    python download_data.py                    # all datasets
    python download_data.py --datasets video memory_locomo
"""

import argparse
import json
import os

HF_REPO = "ulab-ai/xRouteBench"
DATASETS = [
    "llmrouter_generic", "memory_locomo", "memory_longmemeval", "timeseries",
    "video", "multimodal_geometry3k", "multimodal_mathvista", "personalized",
]
# Fields that were JSON-encoded during the parquet conversion; decode them back.
MAYBE_JSON_FIELDS = ["ground_truth", "choices", "query"]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def maybe_json_decode(value):
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS)
    args = parser.parse_args()

    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")

    for ds_name in args.datasets:
        out_dir = os.path.join(DATA_DIR, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        ds = load_dataset(HF_REPO, ds_name, token=token)
        for split in ("train", "test"):
            out_path = os.path.join(out_dir, f"routing_data_{split}.jsonl")
            with open(out_path, "w") as f:
                for row in ds[split]:
                    for field in MAYBE_JSON_FIELDS:
                        if field in row:
                            row[field] = maybe_json_decode(row[field])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{ds_name}/{split}: {len(ds[split])} rows -> {out_path}")

    # Candidate pool with pricing
    pool = load_dataset(HF_REPO, "llm_candidates", token=token)["train"]
    os.makedirs(os.path.join(DATA_DIR, "llm_candidates"), exist_ok=True)
    llm_json = {}
    for row in pool:
        llm_json[row["model_name"]] = {
            "size": row["size"],
            "feature": row["description"],
            "input_price": row["input_price_per_1m"],
            "output_price": row["output_price_per_1m"],
            "model": row["api_model_id"],
            "service": row["service"],
        }
    pool_path = os.path.join(DATA_DIR, "llm_candidates", "default_llm.json")
    with open(pool_path, "w") as f:
        json.dump(llm_json, f, indent=2, ensure_ascii=False)
    print(f"llm_candidates: {len(llm_json)} models -> {pool_path}")
    print("\nNext step: python generate_embeddings.py")


if __name__ == "__main__":
    main()
