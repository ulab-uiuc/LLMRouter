"""Prepare a two-question MHLC pipeline with a small, fixed-answer Qwen backbone."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    Qwen3_5ForConditionalGeneration,
)

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def save_tiny_qwen(path, source):
    # Reuse the real processor and chat template. Only two tiny decoder layers are needed.
    revision = None if Path(source).is_dir() else REVISION
    config = AutoConfig.from_pretrained(source, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16),
        video_processor=Qwen3VLVideoProcessor(patch_size=16),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
    )
    config.text_config.update(dict(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=64,
        linear_value_head_dim=64, max_position_embeddings=4096,
        layer_types=["linear_attention", "full_attention"],
    ))
    config.text_config.rope_parameters["mrope_section"] = [3, 3, 2]
    config.vision_config.update(dict(
        depth=1, hidden_size=128, num_heads=2, out_hidden_size=128, intermediate_size=256,
    ))
    torch.manual_seed(12)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    # Make the toy answer "4" on both questions so the labeler sees one success
    # and one failure, irrespective of the randomly initialized decoder weights.
    (four,) = tokenizer.encode("4", add_special_tokens=False)
    with torch.no_grad():
        model.get_input_embeddings().weight[:, 0] = 1
        model.get_output_embeddings().weight[four, 0] = 128
    model.save_pretrained(path)
    processor.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="An empty directory for this run")
    parser.add_argument("--source", default=MODEL, help="Qwen3.5-4B ID or a local config/tokenizer snapshot")
    parser.add_argument("--gpu-memory-utilization", default=0.35, type=float)
    args = parser.parse_args()
    work = args.workspace.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        parser.error(f"Choose an empty directory for this run: {work}")

    model = str(work / "model")
    save_tiny_qwen(model, args.source)
    pq.write_table(pa.Table.from_pylist([
        {"prompt": "What is 2 + 2?", "solution": "4"},
        {"prompt": "What is 2 + 3?", "solution": "5"},
    ]), work / "arithmetic.parquet")

    gpu = args.gpu_memory_utilization
    recipe = {
        "model_name_or_path": model, "model_family": "qwen3_5", "thinking_mode": "off",
        "dataset_path": "run/verified", "output_dir": "head", "max_seq_len": 2048,
        "num_epochs": 1, "per_device_batch_size": 1, "grad_accum_steps": 16,
        "num_workers": 0, "log_every": 1, "wandb_enabled": False, "trust_remote_code": False,
        "vllm_enforce_eager": True, "vllm_gpu_memory_utilization": gpu, "vllm_max_num_seqs": 2,
    }
    (work / "training.yaml").write_text(yaml.safe_dump(recipe))
    config = yaml.safe_load(Path(__file__).with_name("smoke.yaml").read_text())
    router = config["gnosis_latent_router"]
    router["target_model"] = model
    router["pipeline"]["variables"]["gpu_memory_utilization"] = gpu
    router["inference"]["generator"]["gpu_memory_utilization"] = gpu
    (work / "router.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {work / 'router.yaml'}")


if __name__ == "__main__":
    main()
