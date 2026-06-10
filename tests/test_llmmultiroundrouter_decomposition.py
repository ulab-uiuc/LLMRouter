import torch.nn as nn
from unittest.mock import patch

from llmrouter.models.llmmultiroundrouter.router import LLMMultiRoundRouter


def _build_router(cfg=None):
    router = LLMMultiRoundRouter.__new__(LLMMultiRoundRouter)
    nn.Module.__init__(router)
    router.cfg = cfg or {}
    router.use_local_llm = False
    router.base_model = "deepseek-v4-pro"
    router.api_endpoint = "https://api.example.com"
    router.llm_data = {
        "deepseek-flash": {
            "model": "deepseek-v4-flash",
            "api_endpoint": "https://api.example.com",
            "service": "DeepSeek",
        },
        "deepseek-pro": {
            "model": "deepseek-v4-pro",
            "api_endpoint": "https://api.example.com",
            "service": "DeepSeek",
        },
    }
    router.DECOMP_ROUTE_PROMPT = "{query}"
    return router


def test_decompose_and_route_uses_large_default_max_tokens():
    router = _build_router()
    captured = {}

    def fake_call_api(request, max_tokens, temperature):
        captured["request"] = request
        captured["max_tokens"] = max_tokens
        captured["temperature"] = temperature
        return {"response": "simple task: deepseek-flash"}

    with patch(
        "llmrouter.models.llmmultiroundrouter.router.call_api",
        side_effect=fake_call_api,
    ):
        result = router._decompose_and_route("test query")

    assert captured["max_tokens"] == 2048
    assert captured["temperature"] == 0.0
    assert result == [("simple task", "deepseek-v4-flash")]


def test_decompose_and_route_allows_token_override_from_config():
    router = _build_router({"decomposition_max_tokens": 1024})
    captured = {}

    def fake_call_api(request, max_tokens, temperature):
        captured["max_tokens"] = max_tokens
        return {"response": "complex task: deepseek-pro"}

    with patch(
        "llmrouter.models.llmmultiroundrouter.router.call_api",
        side_effect=fake_call_api,
    ):
        result = router._decompose_and_route("test query")

    assert captured["max_tokens"] == 1024
    assert result == [("complex task", "deepseek-v4-pro")]
