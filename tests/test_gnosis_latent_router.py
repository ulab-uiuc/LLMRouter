import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from llmrouter.cli.router_inference import infer_query
from llmrouter.cli.router_main import create_parser
from llmrouter.models.gnosis_latent_router.adapter import (
    command_for_stage,
    gnosis_backend_root,
    load_gnosis_runtime_module,
    run_gnosis_stage,
)
from llmrouter.models.gnosis_latent_router.config import load_gnosis_config
from llmrouter.models.gnosis_latent_router.pipeline import run_gnosis_pipeline
from llmrouter.models.gnosis_latent_router.router import GnosisLatentRouter
from llmrouter.models.gnosis_latent_router.trainer import GnosisLatentRouterTrainer
from llmrouter.models.gnosis_latent_router.backend.vllm_hidden_states import (
    ExtractedHiddenStates,
    align_extracted_batch,
    normalize_vllm_dtype,
)


def write_config(tmp_path: Path) -> Path:
    config = tmp_path / "gnosis.yaml"
    config.write_text(
        f"""
gnosis_latent_router:
  target_model: Qwen/Qwen3.5-4B
  target_model_name: qwen3.5-4b
  fallback_model: qwen3.5-9b
  threshold: 0.5
  pipeline:
    workspace: {tmp_path}
    python: exact-python
    run_stages: [generate, label, train, benchmark]
    variables:
      run_root: data/mixed-120k
      checkpoint: trained/aux_head_final.pt
    stages:
      generate:
        command:
          - "{{python}}"
          - "{{backend_root}}/combined_all_datagen_multimodel.py"
          - --model-id
          - "{{target_model}}"
          - --save-root
          - "{{run_root}}"
      label:
        command:
          - "{{python}}"
          - "{{backend_root}}/combined_all_labeling_multimodel.py"
      train:
        command:
          - "{{python}}"
          - "{{backend_root}}/train_head_standalone_vllm_regression_weighted_multimodel.py"
          - --config
          - "{{backend_root}}/recipes/training/qwen3_5_4b_think_off.yaml"
      benchmark:
        command:
          - "{{python}}"
          - "{{backend_root}}/multi_agenT_bench/run_multi_agent_generate_then_eval.py"
          - --model1_aux_head_ckpt
          - "{{checkpoint}}"
  inference:
    runtime_module: multi_agenT_bench/compact_multi_agent_shared_optimized_v4_textbench.py
    checkpoint: "{{checkpoint}}"
    generator:
      model_name_or_path: Qwen/Qwen3.5-4B
    sampling:
      max_new_tokens: 16
    auxiliary:
      model_name_or_path: Qwen/Qwen3.5-4B
      head_input_mode: null
""",
        encoding="utf-8",
    )
    return config


def test_config_and_stage_commands_are_passed_through(tmp_path):
    config = load_gnosis_config(str(write_config(tmp_path)))
    backend = gnosis_backend_root()

    command, cwd, _ = command_for_stage(config, "generate")
    assert command == [
        "exact-python",
        str(backend / "combined_all_datagen_multimodel.py"),
        "--model-id",
        "Qwen/Qwen3.5-4B",
        "--save-root",
        "data/mixed-120k",
    ]
    assert cwd == tmp_path


def test_stage_streams_output_and_preserves_workspace_and_environment(tmp_path):
    config = load_gnosis_config(str(write_config(tmp_path)))
    config["pipeline"]["stages"]["generate"] = {
        "command": [sys.executable, "-c", "import os; print(os.getcwd(), os.environ['RUN_NAME'])"],
        "environment": {"RUN_NAME": "{target_model_name}"},
    }
    result = run_gnosis_stage(config, "generate")
    log_path = Path(result["log_path"])
    assert log_path == tmp_path / "logs/gnosis/generate.log"
    log_text = log_path.read_text()
    assert f"{tmp_path} qwen3.5-4b" in log_text
    assert "return_code=0" in log_text


def test_cyclic_stage_variable_fails_before_execution(tmp_path):
    config = load_gnosis_config(str(write_config(tmp_path)))
    config["pipeline"]["variables"]["run_root"] = "data/{run_root}"
    with pytest.raises(ValueError, match="Cyclic pipeline variable"):
        command_for_stage(config, "generate")


def test_generation_refuses_to_mix_with_downloaded_release(tmp_path):
    config = load_gnosis_config(str(write_config(tmp_path)))
    config["pipeline"]["downloaded_release_marker"] = (
        "{run_root}/verified/.mhlc_artifact.json"
    )
    marker = tmp_path / "data/mixed-120k/verified/.mhlc_artifact.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="downloaded verified release"):
        run_gnosis_stage(config, "generate")


def test_benchmark_requires_configured_checkpoint(tmp_path):
    config = load_gnosis_config(str(write_config(tmp_path)))

    with pytest.raises(FileNotFoundError, match="Gnosis checkpoint not found"):
        run_gnosis_stage(config, "benchmark")


def test_complete_pipeline_dry_run_preserves_stage_order(tmp_path):
    config_path = write_config(tmp_path)
    result = run_gnosis_pipeline(
        config_path=str(config_path),
        action="run",
        dry_run=True,
    )
    assert [stage["stage"] for stage in result["stages"]] == [
        "generate",
        "label",
        "train",
        "benchmark",
    ]
    assert all(stage["dry_run"] for stage in result["stages"])


def test_runtime_import_keeps_unrelated_modules_and_search_path(tmp_path, monkeypatch):
    config = load_gnosis_config(str(write_config(tmp_path)))
    unrelated = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "aux_head_shared_utils", unrelated)
    search_path = sys.path.copy()

    module = load_gnosis_runtime_module(config)

    assert module.__name__.startswith("llmrouter.models.gnosis_latent_router.backend.")
    assert sys.modules["aux_head_shared_utils"] is unrelated
    assert sys.path == search_path


def test_qwen_config_uses_40k_mix_and_integrated_recipe():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "configs/model_config_train/gnosis_qwen35_4b.yaml"
    config = load_gnosis_config(str(path))
    commands = config["pipeline"]["stages"]
    generation = [str(value) for value in commands["generate"]["command"]]
    training = [str(value) for value in commands["train"]["command"]]
    benchmark = [str(value) for value in commands["benchmark"]["command"]]

    recipe_path = config["pipeline"]["variables"]["training_recipe"].replace(
        "{backend_root}", str(gnosis_backend_root())
    )
    recipe = yaml.safe_load(Path(recipe_path).read_text())
    resolved, _, _ = command_for_stage(config, "generate")
    run_root = resolved[resolved.index("--save-root") + 1]
    assert resolved[resolved.index("--model-id") + 1] == recipe["model_name_or_path"]
    assert resolved[resolved.index("--model-revision") + 1] == recipe["model_revision"]
    assert resolved[resolved.index("--thinking-mode") + 1] == recipe["thinking_mode"]
    assert recipe["dataset_path"] == f"{run_root}/verified"
    assert config["pipeline"]["variables"]["checkpoint"] == f"{recipe['output_dir']}/aux_head_final.pt"
    assert generation[generation.index("--total-qa-pairs") + 1] == "40000"
    assert "{training_recipe}" in training
    assert "run_multi_agent_generate_then_eval.py" in " ".join(benchmark)
    assert "{checkpoint}" in benchmark
    assert "Mixed_Sources_40k" in json.dumps(config)
    assert "Mixed_Sources_120k" not in json.dumps(config)


def test_integrated_backend_contains_the_complete_pipeline():
    backend = gnosis_backend_root()
    expected = {
        "combined_all_datagen_multimodel.py",
        "combined_all_labeling_multimodel.py",
        "train_head_standalone_vllm_regression_weighted_multimodel.py",
        "vllm_hidden_states.py",
        "aux_head_shared_utils.py",
        "feature_extractors.py",
        "aux_eval_metrics_utils.py",
        "recipes/training/qwen3_5_4b_think_off.yaml",
        "recipes/training/qwen3_5_4b_think_on.yaml",
        "multi_agenT_bench/compact_multi_agent_generate.py",
        "multi_agenT_bench/compact_multi_agent_evaluate.py",
        "multi_agenT_bench/compact_multi_agent_shared_optimized_v4_textbench.py",
        "multi_agenT_bench/run_multi_agent_generate_then_eval.py",
    }
    assert all((backend / relative).is_file() for relative in expected)


def test_vllm_dtype_aliases_and_hidden_state_alignment():
    assert normalize_vllm_dtype("bf16") == "bfloat16"
    assert normalize_vllm_dtype("fp16") == "float16"
    assert normalize_vllm_dtype("fp32") == "float32"

    extracted = [
        ExtractedHiddenStates(
            token_ids=torch.tensor([10, 11, 12]),
            hidden_states=torch.arange(24, dtype=torch.float32).reshape(3, 2, 4),
            layer_ids=[4, 8],
        ),
        ExtractedHiddenStates(
            token_ids=torch.tensor([20, 21]),
            hidden_states=torch.arange(16, dtype=torch.float32).reshape(2, 2, 4),
            layer_ids=[4, 8],
        ),
    ]
    hidden, mask, layer_ids = align_extracted_batch(
        extracted=extracted,
        input_ids=torch.tensor([[10, 11, 12], [0, 20, 21]]),
        attention_mask=torch.tensor([[1, 1, 1], [0, 1, 1]]),
        head_token_mask=torch.tensor([[0, 1, 1], [0, 0, 1]]),
    )
    assert hidden.shape == (2, 3, 2, 4)
    assert mask.tolist() == [[0, 1, 1], [0, 1, 0]]
    assert layer_ids == [4, 8]


def test_hidden_state_alignment_rejects_tokenizer_drift():
    with pytest.raises(RuntimeError, match="tokenized the completed chat differently"):
        align_extracted_batch(
            extracted=[
                ExtractedHiddenStates(
                    token_ids=torch.tensor([10, 99]),
                    hidden_states=torch.zeros(2, 1, 4),
                    layer_ids=[8],
                )
            ],
            input_ids=torch.tensor([[10, 11]]),
            attention_mask=torch.tensor([[1, 1]]),
            head_token_mask=torch.tensor([[0, 1]]),
        )


def test_router_uses_gnosis_generation_and_head_runtimes(tmp_path, monkeypatch):
    config_path = write_config(tmp_path)
    checkpoint = tmp_path / "trained/aux_head_final.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test")
    created = {"events": []}

    class Generator:
        def __init__(self, config):
            created["generator_config"] = config

        def generate(self, *, messages, image, sampling_cfg):
            created["events"].append("generate")
            created["messages"] = messages
            created["image"] = image
            created["sampling"] = sampling_cfg
            return SimpleNamespace(
                text="generated once",
                prompt_tokens=3,
                completion_tokens=2,
            )

        def unload(self, drop_processor=False):
            created["events"].append("unload_generator")
            created["generator_unloaded"] = drop_processor

    class Scorer:
        def __init__(self, config):
            created["aux_config"] = config

        def required_hidden_state_layer_ids(self):
            return [32]

        def attach_vllm_runtime(self, runtime):
            created["shared_runtime"] = runtime

        def score_messages(self, *, messages, image):
            created["events"].append("score")
            created["scored_messages"] = messages
            return SimpleNamespace(prob_correct=0.75)

        def unload(self, drop_processor=False):
            created["events"].append("unload_scorer")
            created["scorer_unloaded"] = drop_processor

    fake_module = SimpleNamespace(
        VLLMRuntimeConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        SamplingConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        AuxHeadRuntimeConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        VLLMChatRuntime=Generator,
        AuxHeadRuntime=Scorer,
    )
    monkeypatch.setattr(
        "llmrouter.models.gnosis_latent_router.router.load_gnosis_runtime_module",
        lambda config: fake_module,
    )

    router = GnosisLatentRouter(str(config_path))
    result = router.route_single(
        {
            "query": "hello",
            "system_prompt": "system",
            "image": "image-ref",
        }
    )
    assert result["accepted"] is True
    assert result["response"] == "generated once"
    assert result["gnosis_score"] == 0.75
    assert created["scored_messages"][-1] == {
        "role": "assistant",
        "content": "generated once",
    }
    assert created["messages"][1]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "hello"},
    ]
    assert created["aux_config"].head_input_mode is None
    assert created["aux_config"].aux_head_ckpt == str(
        tmp_path / "trained/aux_head_final.pt"
    )
    assert created["generator_config"].enable_hidden_state_extraction is True
    assert created["generator_config"].hidden_state_layer_ids == [32]
    assert created["shared_runtime"] is router._generator
    assert created["events"] == [
        "generate",
        "score",
    ]
    router.threshold = 0.8
    rejected = router.route_single({"query": "hello"})
    assert rejected["accepted"] is False
    assert rejected["model_name"] == "qwen3.5-9b"
    assert "response" not in rejected
    router.unload()
    assert created["generator_unloaded"] is True
    assert created["scorer_unloaded"] is True


def test_registered_inference_reuses_accepted_target_response():
    class Router:
        cfg = {}
        llm_data = None

        def route_single(self, query_input):
            return {
                "model_name": "target",
                "response": "already generated",
                "method": "gnosis_latent_router",
            }

    result = infer_query("hello", Router(), "gnosis_latent_router")
    assert result["success"] is True
    assert result["response"] == "already generated"


def test_trainer_delegates_to_integrated_stage(tmp_path, monkeypatch):
    checkpoint = tmp_path / "trained/aux_head_final.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test")
    router = GnosisLatentRouter(str(write_config(tmp_path)))
    called = {}

    def fake_run(config, stage_name, **kwargs):
        called["stage"] = stage_name
        called["workspace"] = kwargs["workspace"]
        return {"stage": stage_name}

    monkeypatch.setattr(
        "llmrouter.models.gnosis_latent_router.trainer.run_gnosis_stage",
        fake_run,
    )
    result = GnosisLatentRouterTrainer(router, device="cpu").train()
    assert called == {"stage": "train", "workspace": tmp_path}
    assert result["backend"] == "gnosis_latent_router"


def test_cli_exposes_workspace_and_dry_run_options():
    args = create_parser().parse_args(
        [
            "gnosis-latent-router",
            "label",
            "--config",
            "gnosis.yaml",
            "--workspace",
            "/work",
            "--dry-run",
        ]
    )
    assert args.gnosis_action == "label"
    assert args.workspace == "/work"
    assert args.dry_run is True
