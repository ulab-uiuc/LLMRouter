import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter


class LLMJudgeRouter(MetaRouter):
    def __init__(self, yaml_path: Optional[str] = None):
        super().__init__(model=nn.Identity(), yaml_path=yaml_path)

        hparam = (self.cfg or {}).get("hparam", {}) or {}
        judge = (self.cfg or {}).get("judge", {}) or {}

        self.small_model = hparam.get("small_model", "small-model")
        self.large_model = hparam.get("large_model", "large-model")
        self.judge_api_base = str(judge.get("api_base", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.judge_api_key = self._resolve_api_key(judge.get("api_key"))
        self.judge_model = judge.get("model", "qwen3:0.6b")
        self.timeout_s = float(judge.get("timeout_s", 5))
        self.max_tokens = int(judge.get("max_tokens", 64))
        self.temperature = float(judge.get("temperature", 0))
        self.reason_max_chars = int(judge.get("reason_max_chars", 80))
        self.max_signals = int(judge.get("max_signals", 3))
        self.fallback_to_large_on_judge_error = bool(judge.get("fallback_to_large_on_judge_error", True))
        self.prompt_budget_chars_per_token = int(judge.get("prompt_budget_chars_per_token", 4))
        self.prompt_budget_output_buffer = int(judge.get("prompt_budget_output_buffer", 128))

    def _resolve_api_key(self, value: Optional[str]) -> Optional[str]:
        if not isinstance(value, str):
            return None
        if value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1])
        return value

    def _extract_first_json_object(self, text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise ValueError("no json object found")
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        raise ValueError("unclosed json object")

    def _compact_reason(self, text: str) -> Optional[str]:
        one_line = " ".join((text or "").strip().split())
        if not one_line:
            return None
        if len(one_line) <= self.reason_max_chars:
            return one_line
        return one_line[:self.reason_max_chars]

    def _normalize_signals(self, signals: Any) -> Optional[List[str]]:
        if not isinstance(signals, list):
            return None
        normalized = []
        seen = set()
        for item in signals:
            if not isinstance(item, str):
                continue
            signal = " ".join(item.strip().split())
            if not signal or signal in seen:
                continue
            seen.add(signal)
            normalized.append(signal[:32])
            if len(normalized) >= self.max_signals:
                break
        return normalized or None

    def _extract_assistant_text(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts).strip()
        return ""

    def _judge(self, query: str) -> Dict[str, Any]:
        # Keep the prompt short and generic so downstream users can tune it easily.
        prompt = (
            "You are a MaaS routing judge.\n"
            f"Pick exactly one model: {self.small_model} or {self.large_model}.\n"
            f"Use {self.small_model} for simple requests.\n"
            f"Use {self.large_model} only for clearly harder reasoning or reliability needs.\n"
            f"If unsure, pick {self.small_model}.\n"
            "Do not answer the user.\n"
            "Return JSON only:\n"
            "{"
            f"\"model\":\"{self.small_model}\","
            "\"confidence\":0.0,"
            "\"reason\":\"short\","
            "\"signals\":[\"tag\"]"
            "}\n"
            f"The model field must be exactly {self.small_model} or {self.large_model}."
        )
        estimated_prompt_tokens = max(
            1,
            (len(prompt) + len(query)) // max(1, self.prompt_budget_chars_per_token),
        )
        if estimated_prompt_tokens > self.prompt_budget_output_buffer:
            return {
                "model": self.large_model,
                "reason": "judge_budget_risk",
                "signals": ["judge_budget"],
                "confidence": None,
                "raw": None,
                "judge_latency_ms": 0,
            }

        headers = {"Content-Type": "application/json"}
        if self.judge_api_key:
            headers["Authorization"] = f"Bearer {self.judge_api_key}"

        body = {
            "model": self.judge_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        judge_start = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                f"{self.judge_api_base}/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        judge_latency_ms = int((time.perf_counter() - judge_start) * 1000)

        text = self._extract_assistant_text(data)
        if not text:
            raise ValueError("empty judge output")

        try:
            obj = json.loads(text)
        except Exception:
            obj = json.loads(self._extract_first_json_object(text))

        if not isinstance(obj, dict):
            raise ValueError("judge output is not a json object")

        model = str(obj.get("model", "")).strip()
        if model not in (self.small_model, self.large_model):
            raise ValueError(f"invalid model from judge: {model}")

        confidence = obj.get("confidence")
        if isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0:
            confidence = float(confidence)
        else:
            confidence = None

        return {
            "model": model,
            "reason": self._compact_reason(obj.get("reason") or ""),
            "signals": self._normalize_signals(obj.get("signals")),
            "confidence": confidence,
            "raw": text,
            "judge_latency_ms": judge_latency_ms,
        }

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        query = query_input.get("query", "") if isinstance(query_input, dict) else str(query_input)
        try:
            judged = self._judge(query)
        except Exception:
            if not self.fallback_to_large_on_judge_error:
                raise
            judged = {
                "model": self.large_model,
                "reason": "judge_error_fallback",
                "signals": ["judge_error"],
                "confidence": None,
                "judge_latency_ms": None,
            }
        selected = judged["model"]
        return {
            "query": query,
            "model_name": selected,
            "predicted_llm": selected,
            "predicted_llm_name": selected,
            "method": "llm_judge",
            "routing_reason": judged.get("reason"),
            "routing_confidence": judged.get("confidence"),
            "routing_signals": judged.get("signals"),
            "routing_judge_latency_ms": judged.get("judge_latency_ms"),
        }

    def route_batch(self, batch):
        return [self.route_single(item) for item in (batch or [])]
