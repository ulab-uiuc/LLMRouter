from unittest.mock import patch

import torch.nn as nn

from custom_routers.llmjudgerouter.router import LLMJudgeRouter


def _build_router():
    router = LLMJudgeRouter.__new__(LLMJudgeRouter)
    nn.Module.__init__(router)
    router.small_model = "deepseek-flash"
    router.large_model = "deepseek-pro"
    router.judge_api_base = "http://127.0.0.1:11434/v1"
    router.judge_api_key = None
    router.judge_model = "qwen3:0.6b"
    router.timeout_s = 5
    router.max_tokens = 64
    router.temperature = 0
    router.reason_max_chars = 80
    router.max_signals = 3
    router.fallback_to_large_on_judge_error = True
    router.prompt_budget_chars_per_token = 4
    router.prompt_budget_output_buffer = 128
    return router


def test_llm_judge_router_uses_openai_compatible_chat_completions():
    router = _build_router()
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"model":"deepseek-flash","confidence":0.82,'
                                '"reason":"simple factual request","signals":["fact","brief"]}'
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    with patch("custom_routers.llmjudgerouter.router.httpx.Client", FakeClient):
        result = router.route_single({"query": "What is Goldbach's conjecture?"})

    assert captured["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert captured["json"]["max_tokens"] == 64
    assert captured["json"]["model"] == "qwen3:0.6b"
    assert result["model_name"] == "deepseek-flash"
    assert result["routing_confidence"] == 0.82
    assert result["routing_reason"] == "simple factual request"
    assert result["routing_signals"] == ["fact", "brief"]
    assert isinstance(result["routing_judge_latency_ms"], int)


def test_llm_judge_router_uses_large_model_when_prompt_budget_is_risky():
    router = _build_router()
    router.max_tokens = 64
    router.prompt_budget_output_buffer = 4

    result = router.route_single({"query": "x" * 100})

    assert result["model_name"] == "deepseek-pro"
    assert result["routing_reason"] == "judge_budget_risk"
    assert result["routing_signals"] == ["judge_budget"]
    assert result["routing_judge_latency_ms"] == 0


def test_llm_judge_router_falls_back_to_large_model_on_judge_error():
    router = _build_router()

    with patch.object(LLMJudgeRouter, "_judge", side_effect=ValueError("empty judge output")):
        result = router.route_single({"query": "证明哥德巴赫猜想"})

    assert result["model_name"] == "deepseek-pro"
    assert result["routing_reason"] == "judge_error_fallback"
    assert result["routing_signals"] == ["judge_error"]
    assert result["routing_judge_latency_ms"] is None
