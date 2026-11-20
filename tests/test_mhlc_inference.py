import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from llmrouter.cli import router_chat, router_inference, router_train
from llmrouter.models.gnosis_latent_router import GnosisLatentRouter
from llmrouter.models.gnosis_latent_router.backend.multi_agenT_bench import (
    compact_multi_agent_shared_optimized_v4_textbench as runtime,
    run_multi_agent_generate_then_eval as benchmark,
)
from llmrouter.models.gnosis_latent_router.backend.vllm_hidden_states import (
    FEATURE_BACKEND,
    FEATURE_BACKEND_VERSION,
)


def test_standard_interfaces_for_checkpoint_and_fallback(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "gnosis_latent_router:\n"
        "  target_model: target\n"
        "  target_model_name: target\n"
        "  fallback_model: fallback\n"
        "  pipeline: {workspace: .}\n"
        "data_path: {llm_data: catalog.json}\n"
    )
    (tmp_path / "catalog.json").write_text(json.dumps(
        {"fallback": {"api_endpoint": "http://localhost/v1", "model": "served-id"}}
    ))
    head = tmp_path / "head.pt"
    head.write_bytes(b"checkpoint")
    assert router_train.ROUTER_TRAINER_REGISTRY["mhlc"][0] is GnosisLatentRouter
    router = router_chat.load_router("mhlc", str(config), str(head))
    assert router._checkpoint_path() == head
    assert router.workspace_root == tmp_path
    exported = tmp_path / "exported.pt"
    router.save_router(str(exported))
    assert exported.read_bytes() == head.read_bytes()

    route = {"model_name": "target", "response": "generated once"}
    monkeypatch.setattr(router, "route_single", lambda query: route)
    assert router_chat.predict("question", [], router, "mhlc") == "[target]\ngenerated once"

    route = {"model_name": "fallback", "candidate_response": "rejected"}
    received = []

    def fallback(request, **kwargs):
        received.append(request)
        return {"response": "fallback answer"}

    monkeypatch.setattr(router_inference, "call_api", fallback)
    decision = router_inference.route_query("question", router, "mhlc")
    assert decision["model_name"] == "fallback"
    assert received == []
    assert router_inference.infer_query("question", router, "mhlc")["response"] == "fallback answer"
    assert len(received) == 1
    assert received[0]["api_name"] == "served-id"
    monkeypatch.setattr(router_inference, "call_api", lambda *a, **kw: {"error": "offline"})
    assert router_inference.infer_query("question", router, "mhlc")["success"] is False


def test_head_metadata_controls_token_mask_and_backbone(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"cfg": {
        "feature_backend": FEATURE_BACKEND, "feature_backend_version": FEATURE_BACKEND_VERSION,
        "model_name_or_path": "target", "model_revision": "pinned", "model_family": "qwen3_5",
        "thinking_mode": "off", "head_input_mode": "completion_only", "max_pixels": 100000, "max_seq_len": 512,
    }}, path)
    cfg = runtime.AuxHeadRuntimeConfig(
        enabled=True, aux_head_ckpt=str(path), model_name_or_path="target",
        model_family="qwen3_5", thinking_mode="off", revision="pinned",
    )
    _, metadata = runtime.AuxHeadRuntime(cfg)._load_checkpoint_and_config()
    assert metadata["head_input_mode"] == "completion_only"
    assert (cfg.max_pixels, cfg.max_seq_len) == (100000, 512)
    cfg.revision = "other"
    with pytest.raises(ValueError, match="revision"):
        runtime.AuxHeadRuntime(cfg)._load_checkpoint_and_config()
    cfg.revision, cfg.thinking_mode = "pinned", "on"
    with pytest.raises(ValueError, match="thinking mode"):
        runtime.AuxHeadRuntime(cfg)._load_checkpoint_and_config()


def test_benchmark_routes_and_scores_fresh_rows(tmp_path, monkeypatch):
    """Exercise both benchmark commands with deterministic model outputs on CPU."""
    pytest.importorskip("matplotlib")
    from llmrouter.models.gnosis_latent_router.backend.multi_agenT_bench import (
        compact_multi_agent_evaluate as evaluate,
        compact_multi_agent_generate as generate,
    )

    monkeypatch.chdir(tmp_path)
    table = tmp_path / "questions.csv"
    table.write_text("id,question,answer\n1,What color is a clear sky?,blue\n")
    monkeypatch.setitem(generate.DATASET_CONFIGS, "triviaqa", {"data_mode": "csv", "data_path": str(table)})
    head = tmp_path / "head.pt"
    torch.save({"cfg": {"feature_backend": FEATURE_BACKEND, "feature_backend_version": FEATURE_BACKEND_VERSION}}, head)
    monkeypatch.setattr(runtime, "model_dimensions", lambda *args, **kwargs: (16, 2))
    monkeypatch.setattr(runtime.AuxHeadRuntime, "load", lambda self: None)
    monkeypatch.setattr(runtime.AuxHeadRuntime, "score_messages", lambda *args, **kwargs: SimpleNamespace(prob_correct=0.65))

    def generate_batch(self, *, messages_list, **kwargs):
        assert self.cfg.model_name_or_path in {"small", "large"}
        if self.cfg.model_name_or_path == "small":
            assert self.cfg.revision == "rev"
        answer = "blue" if self.cfg.model_name_or_path == "small" else "red"
        return [runtime.GenerationResult(text=r"\boxed{" + answer + "}", prompt_tokens=8,
                                         completion_tokens=3, generation_time_sec=0.1) for _ in messages_list]

    monkeypatch.setattr(runtime.VLLMChatRuntime, "generate_batch", generate_batch)
    strategy = "m1_after_finish_handoff_fresh_m2"

    def run_stage(cmd, *, cwd):
        assert cwd == tmp_path
        monkeypatch.setattr(sys, "argv", cmd[1:])
        if Path(cmd[1]).name == "compact_multi_agent_generate.py":
            generate.main()
        else:
            evaluate.main()

    monkeypatch.setattr(benchmark, "_run", run_stage)
    command = [
        "benchmark", "--benchmark", "triviaqa", "--model1_name_or_path", "small",
        "--model2_name_or_path", "large", "--model1_aux_head_ckpt", str(head),
        "--model1_revision", "rev", "--model1_aux_thresholds", "0.5,0.8",
        "--strategy_names", strategy, "--output_base_root", "results",
        "--model1_params", "4", "--model2_params", "9",
    ]
    monkeypatch.setattr(sys, "argv", command)
    benchmark.main()
    root = next((tmp_path / "results").rglob("run_manifest.json")).parent
    accepted, escalated = [
        json.loads(next(root.glob(f"thr1_{value}__*/{strategy}/results_scored.jsonl")).read_text())
        for value in ("0p50", "0p80")
    ]
    assert (accepted["final_model_name"], accepted["benchmark_correct"]) == ("model1", 1)
    assert (escalated["final_model_name"], escalated["benchmark_correct"]) == ("model2", 0)
    head.write_bytes(b"modified head")
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(ValueError, match="checkpoint_sha256 changed"):
        benchmark.main()


def test_judge_batch_matches_scalar_labels():
    from llmrouter.models.gnosis_latent_router.backend.multi_agenT_bench import compact_multi_agent_evaluate as evaluator

    class Judge:
        def __init__(self):
            self.batch_sizes = []

        def generate(self, **kwargs):
            return runtime.GenerationResult(text=r"\boxed{correct}", prompt_tokens=8, completion_tokens=2)

        def generate_batch(self, *, messages_list, **kwargs):
            self.batch_sizes.append(len(messages_list))
            return [self.generate() for _ in messages_list]

    judge = Judge()
    sampling = runtime.SamplingConfig()
    rows = [{"question": "What color?", "gold_answer": "blue", "raw_response": "blue", "id": i} for i in range(3)]
    scored, skipped = evaluator.evaluate_saved_rows_batched("simplevqa", rows, judge, sampling, 2, False, "judging")
    assert judge.batch_sizes == [2, 1]
    assert skipped == []
    assert scored == [runtime.evaluate_saved_row("simplevqa", row, judge, sampling) for row in rows]
