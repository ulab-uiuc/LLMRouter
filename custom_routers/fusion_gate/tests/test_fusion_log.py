"""Offline tests for the fusion log sink (UMB-125).

Fully offline: no network, no torch, no real home directory. All writes go to a
per-test temp directory. The target modules are loaded by file path (not via the
package ``__init__``, which imports torch through router.py), so this suite runs
in a torch-free environment. Run with either:

    python -m pytest custom_routers/fusion_gate/tests/test_fusion_log.py
    python custom_routers/fusion_gate/tests/test_fusion_log.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Load the target modules directly by file path so these tests stay offline and
# free of the package __init__ (which imports torch via router.py). Both modules
# are torch-free. Each is registered in sys.modules before execution so the
# relative ``from .executor import FusionResult`` inside fusion_log.py — and the
# dataclass field types in executor.py — resolve. This mirrors test_executor.py.
_PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(str(_PLUGIN_DIR), filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# executor must be registered under the package-relative name that fusion_log's
# ``from .executor import FusionResult`` resolves to, so the dataclass identities
# match. fusion_log.py is loaded as a package submodule (package=fusion_gate_pkg)
# whose .executor points at the executor we just loaded.
_pkg = type(sys)("fusion_gate_pkg")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["fusion_gate_pkg"] = _pkg

_executor = _load_module("fusion_gate_pkg.executor", "executor.py")
_fusion_log = _load_module("fusion_gate_pkg.fusion_log", "fusion_log.py")

FusionResult = _executor.FusionResult
DEFAULT_SINK_PATH = _fusion_log.DEFAULT_SINK_PATH
log_fusion = _fusion_log.log_fusion
to_training_rows = _fusion_log.to_training_rows


def _sample_result() -> FusionResult:
    """A representative fusion result with a 3-model panel and a judge."""
    return FusionResult(
        answer="Synthesized final answer.",
        analysis={
            "consensus": "All three models agree on the core claim.",
            "contradictions": [],
            "blind_spots": ["edge case X"],
        },
        responses=[
            {"model": "qwen2.5-7b-instruct", "content": "Answer from qwen."},
            {"model": "llama-3.1-8b-instruct", "content": "Answer from llama."},
            {"model": "mistral-7b-instruct-v0.3", "content": "Answer from mistral."},
        ],
        panel=[
            "qwen2.5-7b-instruct",
            "llama-3.1-8b-instruct",
            "mistral-7b-instruct-v0.3",
        ],
        judge="llama3-70b-instruct",
        cost=0.0042,
        raw={"id": "gen-123", "authorization": "Bearer sk-secretkey-should-never-leak"},
    )


class TestLogFusion(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.sink = self.tmp_path / "nested" / "fusion_log.jsonl"
        self.query = "Explain the tradeoffs of consensus algorithms."

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_entry_appended(self) -> None:
        """One call writes exactly one JSONL line; a second call appends."""
        result = _sample_result()

        returned = log_fusion(result, self.query, sink_path=str(self.sink), token=512, cost=0.0042)
        self.assertEqual(returned, self.sink)
        self.assertTrue(self.sink.exists())

        lines = self.sink.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)

        log_fusion(result, self.query, sink_path=str(self.sink), token=256, cost=0.002)
        lines = self.sink.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)

    def test_entry_required_fields(self) -> None:
        """Logged entry carries the full structured schema."""
        log_fusion(_sample_result(), self.query, sink_path=str(self.sink), token=512, cost=0.0042)
        entry = json.loads(self.sink.read_text(encoding="utf-8").splitlines()[0])

        for field in ("ts", "strategy", "query", "panel", "judge", "responses", "analysis", "token", "cost"):
            self.assertIn(field, entry, f"missing field: {field}")

        self.assertEqual(entry["strategy"], "fusion")
        self.assertEqual(entry["query"], self.query)
        self.assertEqual(entry["judge"], "llama3-70b-instruct")
        self.assertEqual(entry["token"], 512)
        self.assertEqual(entry["cost"], 0.0042)
        self.assertEqual(len(entry["responses"]), 3)
        self.assertEqual(len(entry["panel"]), 3)
        self.assertIn("consensus", entry["analysis"])

    def test_cost_falls_back_to_result(self) -> None:
        """When cost arg is None, the entry uses result.cost."""
        log_fusion(_sample_result(), self.query, sink_path=str(self.sink))
        entry = json.loads(self.sink.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(entry["cost"], 0.0042)
        self.assertIsNone(entry["token"])

    def test_no_key_leakage(self) -> None:
        """Raw provider payload and credential shapes never reach the file."""
        log_fusion(_sample_result(), self.query, sink_path=str(self.sink), token=1)
        text = self.sink.read_text(encoding="utf-8")

        self.assertNotIn("sk-secretkey-should-never-leak", text)
        self.assertNotIn("authorization", text.lower())
        self.assertNotIn("bearer", text.lower())
        self.assertNotIn("gen-123", text)  # raw payload is dropped entirely

        entry = json.loads(text.splitlines()[0])
        self.assertNotIn("raw", entry)

    def test_inline_secret_in_response_redacted(self) -> None:
        """Inline credential shapes inside response content are redacted."""
        result = FusionResult(
            responses=[
                {"model": "qwen2.5-7b-instruct", "content": "Use key sk-abcdef0123456789 to call it."},
            ],
            panel=["qwen2.5-7b-instruct"],
            judge=None,
        )
        log_fusion(result, self.query, sink_path=str(self.sink))
        text = self.sink.read_text(encoding="utf-8")
        self.assertNotIn("sk-abcdef0123456789", text)
        self.assertIn("[REDACTED]", text)

    def test_default_sink_path_uses_llmrouter_home(self) -> None:
        """The documented default lands under ~/.llmrouter (not asserted to disk)."""
        self.assertTrue(DEFAULT_SINK_PATH.endswith("openclaw_memory.jsonl"))
        self.assertIn(".llmrouter", DEFAULT_SINK_PATH)


class TestToTrainingRows(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "Explain the tradeoffs of consensus algorithms."

    def test_responses_decompose_to_n_rows(self) -> None:
        """N panel responses produce N per-model rows."""
        result = _sample_result()
        rows = to_training_rows(result, self.query)
        self.assertEqual(len(rows), len(result.responses))

        models = [r["model_name"] for r in rows]
        self.assertEqual(
            models,
            ["qwen2.5-7b-instruct", "llama-3.1-8b-instruct", "mistral-7b-instruct-v0.3"],
        )

    def test_row_required_schema_fields(self) -> None:
        """Each row carries the FusionFactory-consumable schema."""
        rows = to_training_rows(_sample_result(), self.query)
        for row in rows:
            for field in ("query", "model_name", "model", "response", "performance"):
                self.assertIn(field, row, f"missing field: {field}")
            self.assertEqual(row["query"], self.query)
            self.assertEqual(row["model_name"], row["model"])
            self.assertIsNone(row["performance"])

    def test_rows_no_key_leakage(self) -> None:
        """Inline secrets in response content are scrubbed in training rows."""
        result = FusionResult(
            responses=[
                {"model": "qwen2.5-7b-instruct", "content": "token Bearer sk-leak0123456789abc here"},
            ],
            panel=["qwen2.5-7b-instruct"],
        )
        rows = to_training_rows(result, self.query)
        blob = json.dumps(rows)
        self.assertNotIn("sk-leak0123456789abc", blob)
        self.assertIn("[REDACTED]", blob)

    def test_empty_responses_yield_no_rows(self) -> None:
        """No panel responses -> no rows (fail-safe, not an error)."""
        result = FusionResult(responses=[], panel=[])
        self.assertEqual(to_training_rows(result, self.query), [])


if __name__ == "__main__":
    unittest.main()
