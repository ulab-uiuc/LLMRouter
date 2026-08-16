import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from custom_routers.taskawarerouter.router import TaskAwareRouter

CONFIG = "custom_routers/taskawarerouter/config.yaml"


# ── Test 1: keyword fallback works when judge fails ───────────────────
def test_keyword_fallback():
    router = TaskAwareRouter(yaml_path=CONFIG)
    result = router._keyword_fallback("build an enterprise scale pipeline")
    assert result["task"] == "coding"
    assert result["complexity"] == "complex"


# ── Test 2: unknown task → default model, never crashes ───────────────
def test_unknown_task_default_model():
    router = TaskAwareRouter(yaml_path=CONFIG)
    model = router._pick_model("unknown_task", "simple")
    assert model in router.llm_names


# ── Test 3: successful routing returns required keys ──────────────────
def test_route_single_returns_required_keys():
    router = TaskAwareRouter(yaml_path=CONFIG)
    result = router.route_single({"query": "write a python function"})
    assert "model_name" in result
    assert "predicted_llm" in result
    assert "task" in result
    assert "complexity" in result