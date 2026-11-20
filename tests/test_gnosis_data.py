"""Exercise MHLC's data stages on local records without a backbone or a judge."""
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("qwen_vl_utils")

from datasets import Dataset

from llmrouter.models.gnosis_latent_router.backend import (
    combined_all_datagen_multimodel as generate,
    combined_all_labeling_multimodel as label,
)


def test_generate_verify_and_resume(tmp_path, monkeypatch):
    source = Dataset.from_list([
        {"question": "Capital of France?", "answer": {"value": "Paris", "aliases": ["Paris, France"]}},
    ])
    source_path = tmp_path / "questions.parquet"
    source.to_parquet(str(source_path))

    class Generator:
        def __init__(self, **config):
            assert config["revision"] == "test-revision"

        def chat(self, conversations, **kwargs):
            assert len(conversations) == 1
            return [SimpleNamespace(outputs=[SimpleNamespace(text=r"\boxed{Paris}")])]

    monkeypatch.setattr(generate, "LLM", Generator)
    generation_args = [
        "generate", "--save-root", str(tmp_path), "--model-revision", "test-revision",
        "--source", "triviaqa", "--source-file", str(source_path),
        "--total-qa-pairs", "1", "--seed", "42",
    ]
    monkeypatch.setattr(sys, "argv", generation_args)
    generate.main()
    generate.main()
    assert len(list((tmp_path / "raw").glob("*.parquet"))) == 1
    monkeypatch.setattr(sys, "argv", [*generation_args[:-1], "43"])
    with pytest.raises(ValueError, match="seed"):
        generate.main()

    monkeypatch.setattr(sys, "argv", ["label", "--run-root", str(tmp_path)])
    label.main()
    label.main()
    result = Dataset.from_parquet(str(tmp_path / "verified/shard-000001.parquet"))[0]
    assert (result["question"], result["original_answer"]) == ("Capital of France?", "Paris")
    assert result["completion"] == r"\boxed{Paris}"
    assert result["correctness_score"] == 1.0
    assert result["judge_used"] is False
    stats = json.loads((tmp_path / "verification_stats.json").read_text())
    assert stats["num_rows"] == stats["skipped_shards"] == 1


def test_judge_must_supply_a_real_score():
    job = {"row_idx": 0, "kind": "general"}
    for invalid in ('{"total_score": null}', "{}", "", '{"total_score": NaN}'):
        with pytest.raises(ValueError, match="total_score"):
            label._apply_judge_batch([{}], [job], [invalid], {"num_judge_calls": 0})
    row = {}
    label._apply_judge_batch([row], [job], ['{"total_score":0.75}'], {"num_judge_calls": 0})
    assert row["correctness_score"] == pytest.approx(0.75)


def test_dapo_math_verification_uses_plain_numeric_reference():
    pytest.importorskip("math_verify")
    assert label._math_score(r"\boxed{2+2}", "4") == 1.0
    assert label._math_score(r"\boxed{2+3}", "4") == 0.0
    assert label._math_score("4", "4") == 1.0
    assert label._math_score("4", "5") == 0.0
    assert label._math_score("unanswered", "4") is None


def test_image_survives_raw_and_verified_shards(tmp_path):
    from PIL import Image

    row = {
        **generate._default_raw_row(),
        "subset_name": "vqav2", "images": [Image.new("RGB", (3, 4), "blue")],
        "question": "What color?", "original_answer": "blue", "completion": "blue",
    }
    raw = generate._save_shard([row], tmp_path / "raw", 1)
    encoded = label.load_rows(raw)[0]
    assert encoded["images"][0]["bytes"]
    prepared, job = label._prepare_row_for_eval(encoded)
    assert job is None
    verified = tmp_path / "verified" / raw.name
    label.save_verified_shard([prepared], verified)
    result = Dataset.from_parquet(str(verified))[0]
    assert result["images"][0].size == (3, 4)
    assert result["correctness_score"] == 1
