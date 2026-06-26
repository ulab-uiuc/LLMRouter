# ── Standard library imports ──────────────────────────────────────────
import os
import json


import urllib.request
import urllib.error

import yaml

# ── LLMRouter imports ─────────────────────────────────────────────────
from llmrouter.models.meta_router import MetaRouter
import torch.nn as nn

# ── HTTP library to call judge LLM ────────────────────────────────────
import urllib.request


# ═════════════════════════════════════════════════════════════════════
class TaskAwareRouter(MetaRouter):
    """
    Routes queries to the right LLM based on:
      1. Task type  (coding / design / planning / research)
      2. Complexity (simple / complex)

    A small cheap LLM judges both signals from the query.
    If judge fails, falls back to safe defaults.
    """

    # ── Startup ───────────────────────────────────────────────────────
    def __init__(self, yaml_path: str):
        super().__init__(model=nn.Identity(), yaml_path=yaml_path)

        # Real model names loaded from config → no hardcoding
        self.llm_names = list(self.llm_data.keys())

        # Judge model = smallest cheapest available
        import yaml
        with open(yaml_path, "r") as f:
             _cfg = yaml.safe_load(f)
             router_cfg = _cfg.get("router", {})
             self.judge_model     = router_cfg.get("judge_model", self.llm_names[0])
             self.judge_api_model = router_cfg.get("judge_api_model", "")
             self.judge_timeout   = router_cfg.get("judge_timeout", 30)

        # Task → [simple model, complex model]
        # Built from actual available models by price
        self.task_map = {
    "coding": {
        "simple":  "mistral-7b-instruct-v0.3",
        "complex": "llama3-70b-instruct",
    },
    "design": {
        "simple":  "llama-3.1-8b-instruct",
        "complex": "mixtral-8x22b-instruct-v0.1",
    },
    "planning": {
        "simple":  "qwen2.5-7b-instruct",
        "complex": "llama-3.3-nemotron-super-49b-v1",
    },
    "research": {
        "simple":  "qwen2.5-7b-instruct",
        "complex": "mixtral-8x22b-instruct-v0.1",
    },
    "language": {
        "simple":  "llama-3.1-8b-instruct",
        "complex": "mixtral-8x22b-instruct-v0.1",
    },
}

        # Keyword fallback if judge LLM fails
        self.complex_keywords = [
            "advanced", "complex", "production", "scale",
            "optimize", "architect", "integrate", "enterprise",
            "microservices", "distributed", "secure", "pipeline"
        ]

    # ── Judge LLM: classify task + complexity in one call ─────────────
    def _llm_judge(self, query: str) -> dict:
        """
        Sends query to smallest model.
        Asks it to return task type and complexity as JSON.
        """
        prompt = f"""You are a query classifier. Given a user query, return ONLY a JSON object with two fields:
- "task": one of ["coding", "design", "planning", "research","language"]
- "complexity": one of ["simple", "complex"]

Rules:
- coding    = writing code, debugging, APIs, backend, frontend implementation
- design    = UI layout, visual design, wireframes, user experience
- planning  = project plans, roadmaps, strategy, timelines
- research  = finding information, comparing options, summarizing topics,"what is", "which is better"
- language  = translation, grammar correction, rewriting, editing, summarizing text
- simple    = straightforward, single step, clear requirement
- complex   = multi-step, production grade, architecture level, ambiguous
Important:
- "compare", "best", "options", "which", "recommend" → always research
- "build", "create", "implement", "fix", "write" → always coding
- "design", "layout", "wireframe", "UI" → always design
- "plan", "roadmap", "strategy", "timeline" → always planning
- language = translation, grammar, writing, summarizing text, editing

Query: {query}

Return ONLY the JSON. No explanation. No markdown. Example: {{"task": "coding", "complexity": "simple"}}"""

        # Build API request
        judge_model_info = self.llm_data.get(self.judge_model, {})
        api_endpoint      = judge_model_info.get("api_endpoint", "")
        model_name        = self.judge_api_model    
        api_key           = os.environ.get("API_KEYS", "")

        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{api_endpoint}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.judge_timeout) as resp:
            result  = json.loads(resp.read().decode())
            content = result["choices"][0]["message"]["content"].strip()
            return json.loads(content)

    # ── Keyword fallback if judge fails ───────────────────────────────
    def _keyword_fallback(self, query: str) -> dict:
        """
        Simple rule based fallback.
        Used only when judge LLM is unavailable.
        """
        complexity = "simple"
        for keyword in self.complex_keywords:
            if keyword in query.lower():
                complexity = "complex"
                break
        return {"task": "coding", "complexity": complexity}

    # ── Pick model from task map ──────────────────────────────────────
    def _pick_model(self, task: str, complexity: str) -> str:
        """
        Looks up task_map → returns model name.
        Falls back to first available model if not found.
        """
        task_entry = self.task_map.get(task, {})
        preferred  = task_entry.get(complexity, "")

        if preferred and preferred in self.llm_data:
            return preferred

        # Safety net → never crash
        return self.llm_names[0]

    # ── Main function LLMRouter calls ─────────────────────────────────
    def route_single(self, query_input: dict) -> dict:
        """
        Entry point. Called by LLMRouter for every query.
        """
        query = query_input.get("query", "")

        # Step 1 → judge task + complexity
        try:
            judgment   = self._llm_judge(query)
            task       = judgment.get("task", "coding")
            complexity = judgment.get("complexity", "simple")
        except urllib.error.HTTPError as e:
            print(f"[TaskAwareRouter] Judge HTTP error: {e.code} {e.reason}")
            judgment   = self._keyword_fallback(query)
            task       = judgment["task"]
            complexity = judgment["complexity"]
        except urllib.error.URLError as e:
             print(f"[TaskAwareRouter] Judge connection error: {e.reason}") 
             judgment   = self._keyword_fallback(query)
             task       = judgment["task"]
             complexity = judgment["complexity"]
        except json.JSONDecodeError as e:
            print(f"[TaskAwareRouter] Judge response parse error: {e}")
            judgment   = self._keyword_fallback(query)
            task       = judgment["task"]
            complexity = judgment["complexity"]

        # Step 2 → pick right model
        model = self._pick_model(task, complexity)

        # Step 3 → return result
        return {
            "query":         query,
            "task":          task,
            "complexity":    complexity,
            "model_name":    model,
            "predicted_llm": model,
        }

    # ── Batch routing ─────────────────────────────────────────────────
    def route_batch(self, batch: list) -> list:
        """Run route_single for every query in a list."""
        return [self.route_single(q) for q in batch]