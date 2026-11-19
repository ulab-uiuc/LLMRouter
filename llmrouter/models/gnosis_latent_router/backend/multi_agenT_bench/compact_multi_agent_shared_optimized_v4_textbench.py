#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Compact shared utilities for a 2-model multi-agent VLM benchmark pipeline.

This module intentionally combines everything that is shared across the three
benchmarks so that the user only has to keep track of three files total:
  1) compact_multi_agent_shared.py
  2) compact_multi_agent_generate.py
  3) compact_multi_agent_evaluate.py

What this file includes
-----------------------
- Generic helpers: JSONL, image loading, boxed-answer extraction, normalization.
- vLLM chat runtime.
- Aux-head runtime aligned with the user's current aux-head eval path.
- Two-model orchestration with built-in single-agent and multi-agent strategies.
- Benchmark-specific dataset loading, prompting, saved-row formatting, and final
  evaluation for:
    * MathVista
    * ScreenSpot-Pro
    * SimpleVQA

Design choices
--------------
- Generation and evaluation are separate stages.
- Generation loads model1 + model2 (+ aux heads if provided), but not the judge.
- Evaluation loads only the judge model when the benchmark needs it.
- The optional "check every N tokens" mode is implemented with chunked decoding.
  This is the cleanest practical way to do mid-generation aux checks with vLLM.
"""

import base64
import copy
import gc
import json
import os
import random
import re
import string
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from transformers import AutoProcessor

import torch
from llmrouter.models.gnosis_latent_router.backend.aux_head_shared_utils import (
    AuxHeadModule,
    ChatBatchBuilder,
    build_messages_from_prompt_completion,
    dtype_from_str,
    get_device,
    set_max_pixels,
)
from llmrouter.models.gnosis_latent_router.backend.vllm_hidden_states import (
    FEATURE_BACKEND,
    FEATURE_BACKEND_VERSION,
    VLLMHiddenStateExtractor,
    align_extracted_batch,
    model_dimensions,
    normalize_vllm_dtype,
)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

BENCHMARK_ALIASES = {
    "mathvista": "mathvista",
    "mathverse": "mathverse",
    "charxiv_reasoning": "charxiv_reasoning",
    "charxiv": "charxiv_reasoning",
    "screenspot_pro": "screenspot_pro",
    "screenspot-pro": "screenspot_pro",
    "screenspot": "screenspot_pro",
    "simplevqa": "simplevqa",
    "simple_vqa": "simplevqa",
    "triviaqa": "triviaqa",
    "trivia_qa": "triviaqa",
    "trivia": "triviaqa",
    "math": "math",
    "mmlu_pro": "mmlu_pro",
    "mmlu-pro": "mmlu_pro",
    "mmlupro": "mmlu_pro",
    "mmlu": "mmlu_pro",
}


def canonicalize_benchmark_name(benchmark: str) -> str:
    key = str(benchmark or '').strip().lower()
    return BENCHMARK_ALIASES.get(key, key)


def infer_model_family_for_runtime(model_id: str, requested_family: str = "auto") -> str:
    requested_family = str(requested_family or "auto").strip().lower()
    if requested_family != "auto":
        return requested_family
    mid = str(model_id or "").strip().lower()
    if "qwen3.5" in mid:
        return "qwen3_5"
    if "qwen3-vl" in mid:
        return "qwen3_vl"
    if "gemma-4" in mid:
        return "gemma4"
    if "qwen3" in mid:
        return "qwen3"
    return "other"


def resolve_thinking_enabled_for_runtime(model_id: str, model_family: str, thinking_mode: Any) -> bool:
    if isinstance(thinking_mode, bool):
        return thinking_mode

    mode = str(thinking_mode or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
        return False

    mid = str(model_id or "").strip().lower()
    if model_family == "qwen3_5":
        if any(x in mid for x in ["qwen3.5-0.8b", "qwen3.5-2b"]):
            return False
        return True
    if model_family == "qwen3_vl":
        return "thinking" in mid
    if model_family == "qwen3":
        if "instruct" in mid and "thinking" not in mid:
            return False
        return True
    if model_family == "gemma4":
        return False
    return False


def _remove_gemma_think_prefix(text: str) -> str:
    text = text or ""
    return re.sub(r"^\s*<\|think\|>\s*\n?", "", text, count=1)


def _patch_gemma_messages(messages: List[Dict[str, Any]], thinking_enabled: bool) -> List[Dict[str, Any]]:
    patched: List[Dict[str, Any]] = []
    for msg in messages:
        cloned = dict(msg)
        if isinstance(cloned.get("content"), list):
            cloned["content"] = list(cloned["content"])
        patched.append(cloned)

    if not patched or patched[0].get("role") != "system":
        if thinking_enabled:
            patched.insert(0, {"role": "system", "content": "<|think|>"})
        return patched

    system_content = patched[0].get("content", "")
    if not isinstance(system_content, str):
        return patched

    system_content = _remove_gemma_think_prefix(system_content)
    if thinking_enabled:
        system_content = "<|think|>\n" + system_content if system_content else "<|think|>"
    patched[0]["content"] = system_content
    return patched


def _patch_chat_template_callable(bound_callable, model_family: str, thinking_enabled: bool):
    def patched(messages, *args, **kwargs):
        if model_family == "gemma4":
            messages = _patch_gemma_messages(messages, thinking_enabled)
        elif model_family in {"qwen3_5", "qwen3"}:
            kwargs = dict(kwargs)
            kwargs.setdefault("enable_thinking", thinking_enabled)
        return bound_callable(messages, *args, **kwargs)
    return patched


def patch_processor_for_runtime_prompting(processor, model_family: str, thinking_enabled: bool):
    patched_targets = []

    if hasattr(processor, "apply_chat_template"):
        processor.apply_chat_template = _patch_chat_template_callable(
            processor.apply_chat_template, model_family, thinking_enabled
        )
        patched_targets.append("processor.apply_chat_template")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        tokenizer.apply_chat_template = _patch_chat_template_callable(
            tokenizer.apply_chat_template, model_family, thinking_enabled
        )
        patched_targets.append("tokenizer.apply_chat_template")

    runtime_info = {
        "resolved_model_family": model_family,
        "resolved_thinking_enabled": bool(thinking_enabled),
        "patched_targets": patched_targets,
    }
    return processor, runtime_info


def try_set_max_pixels(processor_like, max_pixels: int) -> None:
    targets = []
    if processor_like is not None:
        targets.append(processor_like)
        ip = getattr(processor_like, "image_processor", None)
        if ip is not None:
            targets.append(ip)
    for target in targets:
        try:
            if hasattr(target, "max_pixels"):
                setattr(target, "max_pixels", int(max_pixels))
                return
        except Exception:
            pass
        for attr in ("size", "image_size"):
            obj = getattr(target, attr, None)
            if isinstance(obj, dict):
                obj.setdefault("max_pixels", int(max_pixels))


def resolve_attn_implementation_for_runtime(attn_implementation: str, model_family: str) -> str:
    attn = str(attn_implementation or "").strip()
    if model_family == "gemma4" and attn == "flash_attention_3":
        return "sdpa"
    return attn


GEMMA_THOUGHT_BLOCK_RE = re.compile(
    r"^\s*(?:<\|think\|>\s*)?(?:<\|channel\|?>\s*thought\b.*?(?:<\|channel\|>|<channel\|>|<\|/channel\|>))\s*",
    flags=re.IGNORECASE | re.DOTALL,
)


def debug_print(enabled: bool, *parts, prefix: str = "[DEBUG]", flush: bool = True, **kwargs) -> None:
    if not enabled:
        return
    try:
        print(prefix, *parts, flush=flush, **kwargs)
    except Exception:
        safe = " ".join(str(p) for p in parts)
        print(prefix, safe, flush=flush)


def _short_debug_text(x: object, max_chars: int = 220) -> str:
    s = str(x or "").replace("\n", "\\n")
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...<truncated>"
# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _maybe_add_forward_key(dst: Dict[str, Any], batch: Dict[str, Any], key: str) -> None:
    if key not in batch:
        return
    v = batch[key]
    if v is None:
        return
    if isinstance(v, bool):
        return
    if torch.is_tensor(v):
        if v.ndim == 0 and v.dtype == torch.bool:
            return
        if v.numel() == 0:
            return
    dst[key] = v


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{ln}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_think(text: str) -> str:
    text = str(text or "").strip()
    text = GEMMA_THOUGHT_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()


def _repair_common_boxed_escape_corruption(text: str) -> str:
    s = str(text or "")
    # Common JSON/string corruption: "\boxed" written with a single backslash can
    # turn \b into a backspace control character.
    s = s.replace("\x08oxed", r"\boxed")
    s = s.replace("\u0008oxed", r"\boxed")
    return s


def extract_last_boxed(text: str) -> Optional[str]:
    text = strip_think(text)
    text = _repair_common_boxed_escape_corruption(text)
    key = r"\boxed"
    idx = text.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return None
    if text[i] == "{":
        depth = 0
        start_inner = i + 1
        j = i
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_inner:j].strip()
            j += 1
        tail = text[start_inner:].strip()
        if not tail:
            return None
        tail = re.split(r"[\n\r]", tail, maxsplit=1)[0].strip()
        tail = re.split(r"(?<!\\)[,;:!?]\s", tail, maxsplit=1)[0].strip()
        return tail if tail else None
    m = re.match(r"([^\s.,;:!?]+)", text[i:])
    return m.group(1).strip() if m else None

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.casefold().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(string.punctuation + " ")
    return s

def normalize_text_loose(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[\.,;:!?\-_/\\()\[\]{}'\"`~|]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


REFUSAL_PATTERNS = [
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bnot sure\b",
    r"\bcannot determine\b",
    r"\bcan't determine\b",
    r"\bunable to determine\b",
    r"\bneed more information\b",
    r"\bi cannot answer\b",
    r"\bi can't answer\b",
    r"\bunknown\b",
    r"\bn/?a\b",
    r"\bno answer\b",
]


def is_refusal(text: str) -> bool:
    s = strip_think(text)
    s_norm = normalize_text_loose(s)
    return any(re.search(p, s_norm, flags=re.IGNORECASE) for p in REFUSAL_PATTERNS)


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------


def _open_image_from_path(path_str: str) -> Image.Image:
    p = Path(path_str)
    if not p.exists():
        raise RuntimeError(f"Image path does not exist: {path_str}")
    with Image.open(p) as img:
        return img.convert("RGB")


def _open_image_from_bytes(raw: bytes) -> Image.Image:
    with Image.open(BytesIO(raw)) as img:
        return img.convert("RGB")


def _looks_like_base64_image_string(s: str) -> bool:
    s = s.strip()
    if len(s) < 64 or any(ch.isspace() for ch in s):
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-")
    return set(s).issubset(allowed)


def load_image_any(x: Any) -> Optional[Image.Image]:
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, dict):
        if x.get("bytes") is not None:
            raw = x["bytes"]
            if isinstance(raw, bytes):
                return _open_image_from_bytes(raw)
            if isinstance(raw, bytearray):
                return _open_image_from_bytes(bytes(raw))
            if isinstance(raw, list):
                return _open_image_from_bytes(bytes(raw))
            raise RuntimeError(f"Unsupported image bytes type: {type(raw)}")
        if x.get("path") is not None:
            return _open_image_from_path(str(x["path"]))
        if x.get("array") is not None:
            arr = np.asarray(x["array"])
            return Image.fromarray(arr).convert("RGB")
        raise RuntimeError(f"Unsupported image dict keys: {sorted(x.keys())}")
    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert("RGB")
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("data:image/") and "," in s:
            _, b64 = s.split(",", 1)
            return _open_image_from_bytes(base64.b64decode(b64, validate=False))
        p = s[7:] if s.startswith("file://") else s
        try:
            if len(p) <= 1024 and Path(p).exists():
                return _open_image_from_path(p)
        except OSError:
            pass
        if _looks_like_base64_image_string(s):
            return _open_image_from_bytes(base64.b64decode(s, validate=False))
        raise RuntimeError(f"Unsupported image string: {s[:120]!r}")
    if isinstance(x, (bytes, bytearray)):
        return _open_image_from_bytes(bytes(x))
    raise RuntimeError(f"Unsupported image type: {type(x)}")


# -----------------------------------------------------------------------------
# Runtime configs
# -----------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    greedy: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = -1
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    max_new_tokens: int = 15000
    stop: Optional[List[str]] = None


@dataclass
class VLLMRuntimeConfig:
    model_name_or_path: str
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    max_model_len: int = 32000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.55
    max_num_seqs: int = 8
    enforce_eager: bool = False
    trust_remote_code: bool = False
    limit_mm_images: int = 1
    model_family: str = "auto"
    thinking_mode: Any = "auto"
    enable_hidden_state_extraction: bool = False
    hidden_state_layer_ids: Optional[List[int]] = None
    hidden_states_storage_path: Optional[str] = None


@dataclass
class AuxHeadRuntimeConfig:
    enabled: bool = False
    model_name_or_path: str = ""
    revision: Optional[str] = None
    aux_head_ckpt: str = ""
    trust_remote_code: bool = True
    dtype: str = "bf16"
    max_seq_len: int = 32000
    max_pixels: int = 200000
    regression_threshold: float = 0.5
    head_input_mode: Optional[str] = None
    hidden_layer_selection: Optional[str] = None
    hidden_layer_index: Optional[int] = None
    hidden_layer_indices: Optional[List[int]] = None
    model_family: str = "auto"
    thinking_mode: Any = "auto"
    hidden_states_storage_path: Optional[str] = None


@dataclass
class ModelBundle:
    name: str
    generator_cfg: VLLMRuntimeConfig
    sampling_cfg: SamplingConfig
    aux_cfg: AuxHeadRuntimeConfig


@dataclass
class AuxPolicy:
    enabled: bool = False
    threshold: float = 0.5
    trigger_mode: str = "after_finish"  # options: after_finish, every_n_tokens
    check_every_n_tokens: int = 128
    action_below_threshold: str = "accept"  # accept, retry, self_repair, handoff_fresh, handoff_with_context
    next_model: Optional[str] = None
    max_self_repairs: int = 1


@dataclass
class StrategyConfig:
    name: str
    entry_model: str
    model_policies: Dict[str, AuxPolicy]
    max_total_handoffs: int = 2
    max_total_repairs: int = 2


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    aux_scored_tokens: int = 0
    aux_calls: int = 0
    generation_calls: int = 0
    generation_time_sec: float = 0.0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += int(other.prompt_tokens)
        self.completion_tokens += int(other.completion_tokens)
        self.aux_scored_tokens += int(other.aux_scored_tokens)
        self.aux_calls += int(other.aux_calls)
        self.generation_calls += int(other.generation_calls)
        self.generation_time_sec += float(other.generation_time_sec)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    generation_time_sec: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuxScore:
    pred: int
    prob_correct: float
    probs: List[float]
    raw: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# vLLM runtime
# -----------------------------------------------------------------------------


class VLLMChatRuntime:
    def __init__(self, cfg: VLLMRuntimeConfig) -> None:
        self.cfg = cfg
        self._processor = None
        self._llm = None
        self._resolved_gpu_memory_utilization: Optional[float] = None
        self._hidden_extractor: Optional[VLLMHiddenStateExtractor] = None

    def _parse_vllm_memory_error(self, exc: Exception) -> Optional[Tuple[float, float, float]]:
        msg = str(exc)
        m = re.search(r"Free memory on device .*?\(([-+]?\d*\.?\d+)\s*/\s*([-+]?\d*\.?\d+) GiB\).*?desired GPU memory utilization \(\s*([-+]?\d*\.?\d+)", msg)
        if not m:
            return None
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    def _suggest_lower_gpu_util(self, current_util: float, exc: Exception) -> Optional[float]:
        parsed = self._parse_vllm_memory_error(exc)
        if parsed is None:
            return None
        free_gib, total_gib, _ = parsed
        free_ratio = free_gib / max(total_gib, 1e-6)
        next_util = min(current_util - 0.03, free_ratio - 0.03)
        next_util = round(next_util, 3)
        if next_util < 0.5:
            return None
        return next_util

    def _make_llm_with_retry(self):
        from vllm import LLM
        extra_kwargs: Dict[str, Any] = {}
        if self.cfg.enable_hidden_state_extraction:
            layer_ids = self.cfg.hidden_state_layer_ids
            if not layer_ids:
                _, num_hidden_layers = model_dimensions(
                    self.cfg.model_name_or_path,
                    revision=self.cfg.revision,
                    trust_remote_code=self.cfg.trust_remote_code,
                )
                layer_ids = [num_hidden_layers]
            self._hidden_extractor = VLLMHiddenStateExtractor(
                layer_ids=layer_ids,
                storage_path=self.cfg.hidden_states_storage_path,
            )
            extra_kwargs.update(self._hidden_extractor.llm_kwargs())
        util = float(self.cfg.gpu_memory_utilization)
        while True:
            try:
                llm_kwargs = dict(
                    model=self.cfg.model_name_or_path,
                    revision=self.cfg.revision,
                    trust_remote_code=self.cfg.trust_remote_code,
                    dtype=normalize_vllm_dtype(self.cfg.dtype),
                    tensor_parallel_size=self.cfg.tensor_parallel_size,
                    gpu_memory_utilization=util,
                    max_model_len=self.cfg.max_model_len,
                    max_num_seqs=self.cfg.max_num_seqs,
                    enforce_eager=self.cfg.enforce_eager,
                    limit_mm_per_prompt={"image": int(self.cfg.limit_mm_images), "video": 0},
                )
                llm_kwargs.update(extra_kwargs)
                llm = LLM(**llm_kwargs)
                self._resolved_gpu_memory_utilization = util
                return llm
            except Exception as exc:
                next_util = self._suggest_lower_gpu_util(util, exc)
                if next_util is None:
                    raise
                print(
                    f"[vLLM] Lowering gpu_memory_utilization for {self.cfg.model_name_or_path} from {util:.3f} to {next_util:.3f} after startup failure: {exc}",
                    flush=True,
                )
                util = next_util

    def configure_hidden_state_extraction(
        self,
        layer_ids: Sequence[int],
        storage_path: Optional[str] = None,
    ) -> None:
        requested = [int(x) for x in layer_ids]
        if self._llm is not None:
            active = list(self._hidden_extractor.layer_ids) if self._hidden_extractor else []
            if active != requested:
                raise RuntimeError(
                    "Cannot change vLLM hidden-state layers after the engine is loaded: "
                    f"active={active}, requested={requested}"
                )
            return
        self.cfg.enable_hidden_state_extraction = True
        self.cfg.hidden_state_layer_ids = requested
        if storage_path is not None:
            self.cfg.hidden_states_storage_path = storage_path

    @property
    def processor(self):
        if self._processor is None:
            processor = AutoProcessor.from_pretrained(
                self.cfg.model_name_or_path,
                revision=self.cfg.revision,
                trust_remote_code=self.cfg.trust_remote_code,
            )
            resolved_family = infer_model_family_for_runtime(
                self.cfg.model_name_or_path,
                self.cfg.model_family,
            )
            resolved_thinking_enabled = resolve_thinking_enabled_for_runtime(
                self.cfg.model_name_or_path,
                resolved_family,
                self.cfg.thinking_mode,
            )
            processor, _ = patch_processor_for_runtime_prompting(
                processor,
                resolved_family,
                resolved_thinking_enabled,
            )
            self._processor = processor
        return self._processor

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._make_llm_with_retry()
        return self._llm

    def _normalize_content(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            return content
        raise TypeError(f"Unsupported message content type: {type(content)}")

    def build_prompt(self, messages: List[Dict[str, Any]], continue_final_message: bool = False) -> str:
        normalized = [{"role": m["role"], "content": self._normalize_content(m["content"])} for m in messages]
        try:
            return self.processor.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
            )
        except TypeError:
            if continue_final_message:
                prompt = self.processor.apply_chat_template(normalized[:-1], tokenize=False, add_generation_prompt=True)
                last = normalized[-1]
                tail = "".join(item.get("text", "") for item in last["content"] if item.get("type") == "text")
                return str(prompt) + str(tail)
            return self.processor.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)

    def count_prompt_tokens_from_text(self, prompt: str) -> int:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            return 0
        return int(len(tok(prompt, add_special_tokens=False)["input_ids"]))

    def generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        image: Optional[Image.Image],
        sampling_cfg: SamplingConfig,
        continue_final_message: bool = False,
    ) -> GenerationResult:
        results = self.generate_batch(
            messages_list=[messages],
            images=[image],
            sampling_cfg=sampling_cfg,
            continue_final_messages=[continue_final_message],
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected one generation result, got {len(results)}")
        return results[0]

    def generate_batch(
        self,
        *,
        messages_list: List[List[Dict[str, Any]]],
        images: List[Optional[Image.Image]],
        sampling_cfg: SamplingConfig,
        continue_final_messages: Optional[List[bool]] = None,
    ) -> List[GenerationResult]:
        from vllm import SamplingParams

        if len(messages_list) != len(images):
            raise RuntimeError(f"messages_list/images length mismatch: {len(messages_list)} vs {len(images)}")
        if continue_final_messages is None:
            continue_final_messages = [False] * len(messages_list)
        if len(continue_final_messages) != len(messages_list):
            raise RuntimeError(
                f"continue_final_messages/messages_list length mismatch: {len(continue_final_messages)} vs {len(messages_list)}"
            )
        if not messages_list:
            return []

        prompts: List[str] = []
        requests: List[Any] = []
        for messages, image, cont in zip(messages_list, images, continue_final_messages):
            prompt = self.build_prompt(messages, continue_final_message=bool(cont))
            prompts.append(prompt)
            request: Any = {"prompt": prompt, "multi_modal_data": {"image": image}} if image is not None else prompt
            requests.append(request)

        if getattr(sampling_cfg, "greedy", False):
            sp = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=float(getattr(sampling_cfg, "repetition_penalty", 1.0)),
                presence_penalty=float(getattr(sampling_cfg, "presence_penalty", 0.0)),
                max_tokens=sampling_cfg.max_new_tokens,
                stop=getattr(sampling_cfg, "stop", None),
                n=1,
            )
        else:
            sp = SamplingParams(
                temperature=float(getattr(sampling_cfg, "temperature", 0.7)),
                top_p=float(getattr(sampling_cfg, "top_p", 0.8)),
                top_k=int(getattr(sampling_cfg, "top_k", -1)),
                repetition_penalty=float(getattr(sampling_cfg, "repetition_penalty", 1.0)),
                presence_penalty=float(getattr(sampling_cfg, "presence_penalty", 0.0)),
                max_tokens=sampling_cfg.max_new_tokens,
                stop=getattr(sampling_cfg, "stop", None),
                n=1,
            )
        t_generate = time.time()
        outputs = self.llm.generate(requests, sp, use_tqdm=False)
        generate_elapsed = float(time.time() - t_generate)
        if len(outputs) != len(requests):
            raise RuntimeError(f"Invalid vLLM output count: expected {len(requests)}, got {len(outputs)}")
        if self._hidden_extractor is not None:
            self._hidden_extractor.discard_outputs(outputs)

        per_example_generation_time = generate_elapsed / max(len(requests), 1)
        results: List[GenerationResult] = []
        tok = getattr(self.processor, "tokenizer", None)
        for out, prompt, cont in zip(outputs, prompts, continue_final_messages):
            if not hasattr(out, "outputs") or len(out.outputs) != 1:
                raise RuntimeError("Invalid vLLM output structure in batch generation")
            completion = out.outputs[0]
            text = str(completion.text)
            try:
                prompt_tokens = int(len(getattr(out, "prompt_token_ids")))
            except Exception:
                prompt_tokens = self.count_prompt_tokens_from_text(prompt)
            try:
                completion_tokens = int(len(getattr(completion, "token_ids")))
            except Exception:
                completion_tokens = int(len(tok(text, add_special_tokens=False)["input_ids"])) if tok is not None else 0
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    generation_time_sec=per_example_generation_time,
                    raw={"prompt": prompt, "continue_final_message": bool(cont)},
                )
            )
        return results

    def extract_hidden_states(
        self,
        *,
        prompts: Sequence[str | list[int]],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Accessing llm initializes the engine and its extractor together.
        llm = self.llm
        if self._hidden_extractor is None:
            raise RuntimeError(
                "This vLLM runtime was not configured for hidden-state extraction"
            )
        return self._hidden_extractor.extract(
            llm,
            prompts=prompts,
            images=images,
            use_tqdm=False,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def unload(self, drop_processor: bool = False) -> None:
        llm = self._llm
        self._llm = None
        if llm is not None:
            try:
                eng = getattr(llm, "llm_engine", None)
                if eng is not None and hasattr(eng, "shutdown"):
                    eng.shutdown()
            except Exception:
                pass
            try:
                del llm
            except Exception:
                pass
        if drop_processor:
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Aux-head runtime
# -----------------------------------------------------------------------------


def _normalize_hidden_state_index(idx: int, num_hidden_states: int) -> int:
    idx = int(idx)
    n = int(num_hidden_states)
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise ValueError(
            f"hidden state index {idx} is out of range for {n} hidden states "
            f"(valid raw hidden_states indices: 0..{n - 1}, negatives allowed)"
        )
    return idx


def _normalize_transformer_layer_index(idx: int, num_hidden_layers: int) -> int:
    n = int(num_hidden_layers)
    idx = int(idx)
    if idx < 0:
        idx = n + idx
    if idx < 0 or idx >= n:
        raise ValueError(
            f"transformer layer index {idx} is out of range for {n} layers "
            f"(valid transformer-layer indices: 0..{n - 1}, negatives allowed)"
        )
    return idx + 1


def _resolve_selected_hidden_layer_indices_for_inference(
    *,
    num_hidden_layers: int,
    selected_hidden_layer_indices: Optional[List[int]],
    hidden_layer_selection: Optional[str],
    hidden_layer_index: Optional[int],
    hidden_layer_indices: Optional[List[int]],
) -> Optional[List[int]]:
    if selected_hidden_layer_indices is not None:
        return [
            _normalize_hidden_state_index(i, num_hidden_layers + 1)
            for i in selected_hidden_layer_indices
        ]

    sel = hidden_layer_selection
    if sel is None or str(sel).strip().lower() in {"", "none", "default"}:
        return None

    sel = str(sel).strip().lower()

    if sel == "first":
        return [_normalize_transformer_layer_index(0, num_hidden_layers)]
    if sel == "middle":
        middle_idx = max(0, (int(num_hidden_layers) - 1) // 2)
        return [_normalize_transformer_layer_index(middle_idx, num_hidden_layers)]
    if sel == "last":
        return [_normalize_transformer_layer_index(-1, num_hidden_layers)]
    if sel == "index":
        if hidden_layer_index is None:
            raise ValueError("hidden_layer_selection='index' requires hidden_layer_index to be set.")
        return [_normalize_transformer_layer_index(hidden_layer_index, num_hidden_layers)]
    if sel == "indices":
        if not hidden_layer_indices:
            raise ValueError("hidden_layer_selection='indices' requires hidden_layer_indices to be set.")
        return [
            _normalize_transformer_layer_index(i, num_hidden_layers)
            for i in hidden_layer_indices
        ]
    if sel == "all":
        return list(range(1, int(num_hidden_layers) + 1))

    raise ValueError(
        f"Unsupported hidden_layer_selection={hidden_layer_selection!r}. "
        f"Use one of: first, middle, last, index, indices, all, or leave it null."
    )


AUX_HEAD_DEFAULTS = {
    "num_labels": 1,
    "head_input_mode": "completion_text_only",
    "hidden_encoder_type": "lite",
    "selected_hidden_layer_indices": None,
    "hidden_layer_selection": "last",
    "hidden_layer_index": None,
    "hidden_layer_indices": None,
}


class AuxHeadRuntime:
    def __init__(
        self,
        cfg: AuxHeadRuntimeConfig,
        vllm_runtime: Optional[VLLMChatRuntime] = None,
    ) -> None:
        self.cfg = cfg
        self._vllm_runtime = vllm_runtime
        self._loaded = False
        self._torch = None
        self._device = None
        self._fp_dtype = None
        self._processor = None
        self._head = None
        self._aux_head_cfg = dict(AUX_HEAD_DEFAULTS)
        self._runtime_prompting: Dict[str, Any] = {}
        self._checkpoint = None
        self._required_layer_ids: Optional[List[int]] = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.aux_head_ckpt)

    def attach_vllm_runtime(self, runtime: VLLMChatRuntime) -> None:
        self._vllm_runtime = runtime
        runtime.configure_hidden_state_extraction(
            self.required_hidden_state_layer_ids(),
            storage_path=self.cfg.hidden_states_storage_path,
        )

    def _load_checkpoint_and_config(self):
        if self._checkpoint is not None:
            return self._checkpoint, self._aux_head_cfg
        ckpt = torch.load(self.cfg.aux_head_ckpt, map_location="cpu", weights_only=True)
        ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        if not isinstance(ckpt_cfg, dict):
            ckpt_cfg = {}
        backend = ckpt_cfg.get("feature_backend")
        version = ckpt_cfg.get("feature_backend_version")
        if backend != FEATURE_BACKEND or version not in {1, FEATURE_BACKEND_VERSION}:
            raise ValueError(
                "This checkpoint was not trained with the vLLM Gnosis feature backend. "
                "Retrain it with the integrated Gnosis trainer before routing inference."
            )
        backbone = ckpt_cfg.get("model_name_or_path")
        if backbone and backbone != self.cfg.model_name_or_path:
            raise ValueError(f"Capability head was trained for {backbone}, requested {self.cfg.model_name_or_path}")
        revision = ckpt_cfg.get("model_revision")
        if revision and revision != self.cfg.revision:
            raise ValueError(f"Capability head requires target backbone revision {revision}, requested {self.cfg.revision}")
        if backbone and "thinking_mode" in ckpt_cfg:
            trained_family = infer_model_family_for_runtime(backbone, ckpt_cfg.get("model_family"))
            runtime_family = infer_model_family_for_runtime(backbone, self.cfg.model_family)
            trained_thinking = resolve_thinking_enabled_for_runtime(backbone, trained_family, ckpt_cfg["thinking_mode"])
            runtime_thinking = resolve_thinking_enabled_for_runtime(backbone, runtime_family, self.cfg.thinking_mode)
            if (trained_family, trained_thinking) != (runtime_family, runtime_thinking):
                raise ValueError("Capability head requires the model family and thinking mode used in training")
        if version >= 2:
            self.cfg.max_pixels = int(ckpt_cfg.get("max_pixels", self.cfg.max_pixels))
            trained_length = int(ckpt_cfg.get("max_seq_len", self.cfg.max_seq_len))
            self.cfg.max_seq_len = min(self.cfg.max_seq_len, trained_length)
        aux_head_cfg = dict(AUX_HEAD_DEFAULTS)
        for key in (
            "num_labels",
            "head_input_mode",
            "hidden_encoder_type",
            "selected_hidden_layer_indices",
            "hidden_layer_selection",
            "hidden_layer_index",
            "hidden_layer_indices",
        ):
            if key in ckpt_cfg:
                aux_head_cfg[key] = ckpt_cfg[key]
        if self.cfg.head_input_mode is not None:
            aux_head_cfg["head_input_mode"] = str(self.cfg.head_input_mode)
        if self.cfg.hidden_layer_selection is not None:
            aux_head_cfg["hidden_layer_selection"] = self.cfg.hidden_layer_selection
        if self.cfg.hidden_layer_index is not None:
            aux_head_cfg["hidden_layer_index"] = int(self.cfg.hidden_layer_index)
        if self.cfg.hidden_layer_indices is not None:
            aux_head_cfg["hidden_layer_indices"] = [int(x) for x in self.cfg.hidden_layer_indices]
        aux_head_cfg["num_labels"] = int(aux_head_cfg["num_labels"])
        self._checkpoint = ckpt
        self._aux_head_cfg = aux_head_cfg
        return ckpt, aux_head_cfg

    def required_hidden_state_layer_ids(self) -> List[int]:
        if self._required_layer_ids is not None:
            return list(self._required_layer_ids)
        _, aux_head_cfg = self._load_checkpoint_and_config()
        _, num_hidden_layers = model_dimensions(
            self.cfg.model_name_or_path,
            revision=self.cfg.revision,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        selected = _resolve_selected_hidden_layer_indices_for_inference(
            num_hidden_layers=num_hidden_layers,
            selected_hidden_layer_indices=aux_head_cfg.get("selected_hidden_layer_indices"),
            hidden_layer_selection=aux_head_cfg.get("hidden_layer_selection"),
            hidden_layer_index=aux_head_cfg.get("hidden_layer_index"),
            hidden_layer_indices=aux_head_cfg.get("hidden_layer_indices"),
        )
        self._required_layer_ids = selected or [num_hidden_layers]
        self._aux_head_cfg["selected_hidden_layer_indices"] = list(self._required_layer_ids)
        return list(self._required_layer_ids)

    def load(self) -> None:
        if self._loaded:
            return
        if not self.enabled:
            raise RuntimeError("Attempted to load aux head although it is disabled")

        self._torch = torch
        self._ChatBatchBuilder = ChatBatchBuilder
        self._device = get_device()
        self._fp_dtype = dtype_from_str(self.cfg.dtype)

        resolved_family = infer_model_family_for_runtime(
            self.cfg.model_name_or_path,
            self.cfg.model_family,
        )
        resolved_thinking_enabled = resolve_thinking_enabled_for_runtime(
            self.cfg.model_name_or_path,
            resolved_family,
            self.cfg.thinking_mode,
        )
        processor = (
            self._vllm_runtime.processor
            if self._vllm_runtime is not None
            else AutoProcessor.from_pretrained(
                self.cfg.model_name_or_path,
                revision=self.cfg.revision,
                trust_remote_code=self.cfg.trust_remote_code,
            )
        )
        ckpt, aux_head_cfg = self._load_checkpoint_and_config()
        if ckpt["cfg"]["feature_backend_version"] == 1:
            try_set_max_pixels(processor, self.cfg.max_pixels)
        else:
            set_max_pixels(processor, self.cfg.max_pixels)
        if self._vllm_runtime is None:
            processor, runtime_prompting = patch_processor_for_runtime_prompting(
                processor,
                resolved_family,
                resolved_thinking_enabled,
            )
        else:
            runtime_prompting = {"patched_targets": ["shared_vllm_processor"]}

        hidden_size, num_hidden_layers = model_dimensions(
            self.cfg.model_name_or_path,
            revision=self.cfg.revision,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        resolved_selected_hidden_layer_indices = self.required_hidden_state_layer_ids()
        head = AuxHeadModule(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            hidden_encoder_type=aux_head_cfg["hidden_encoder_type"],
            num_labels=aux_head_cfg["num_labels"],
            selected_hidden_layer_indices=resolved_selected_hidden_layer_indices,
        ).to(self._device)
        state = ckpt["head_state"] if isinstance(ckpt, dict) and "head_state" in ckpt else ckpt
        head.load_state_dict(state)
        head.eval()

        self._processor = processor
        self._head = head
        self._aux_head_cfg = aux_head_cfg
        self._runtime_prompting = {
            "resolved_model_id": self.cfg.model_name_or_path,
            "revision": self.cfg.revision,
            "feature_backend": FEATURE_BACKEND,
            "feature_backend_version": ckpt["cfg"]["feature_backend_version"],
            "resolved_model_family": resolved_family,
            "resolved_thinking_enabled": bool(resolved_thinking_enabled),
            "patched_targets": runtime_prompting["patched_targets"],
        }
        if self._vllm_runtime is None:
            raise RuntimeError("AuxHeadRuntime requires a shared VLLMChatRuntime")
        self.attach_vllm_runtime(self._vllm_runtime)
        self._loaded = True

    def unload(self, drop_processor: bool = False) -> None:
        head = self._head
        self._head = None
        if drop_processor:
            self._processor = None
        if head is not None:
            try:
                del head
            except Exception:
                pass
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _logits_to_binary_outputs(self, logits) -> Tuple[int, float, List[float]]:
        torch = self._torch
        logits = logits.float()
        num_labels = int(self._aux_head_cfg["num_labels"])
        if num_labels == 1:
            scores = torch.sigmoid(logits)
            if scores.ndim == 2 and scores.shape[-1] == 1:
                scores = scores[:, 0]
            score = float(scores[0].item())
            pred = int(score >= float(self.cfg.regression_threshold))
            return pred, score, [float(1.0 - score), float(score)]
        probs = torch.softmax(logits, dim=-1)
        pred = int(probs.argmax(dim=-1)[0].item())
        prob_correct = float(probs[0, 1].item())
        return pred, prob_correct, probs[0].detach().cpu().tolist()

    def score_messages(self, *, messages: List[Dict[str, Any]], image: Optional[Image.Image]) -> AuxScore:
        self.load()
        torch = self._torch
        batch_builder = self._ChatBatchBuilder(
            processor=self._processor,
            max_seq_len=min(self.cfg.max_seq_len, self._vllm_runtime.cfg.max_model_len),
            head_input_mode=self._aux_head_cfg["head_input_mode"],
            feature_backend_version=self._checkpoint["cfg"]["feature_backend_version"],
        )
        batch = batch_builder.build_from_messages([messages], [image])
        prompts = batch.pop("gnosis_prompt_texts")
        images = batch.pop("gnosis_images")
        extracted = self._vllm_runtime.extract_hidden_states(
            prompts=prompts,
            images=images,
            mm_processor_kwargs=batch["gnosis_mm_processor_kwargs"],
        )
        hidden_batch, token_mask, layer_ids = align_extracted_batch(
            extracted=extracted,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            head_token_mask=batch["head_token_mask"],
        )
        hidden_batch = hidden_batch.to(self._device)
        if self._fp_dtype is not None:
            hidden_batch = hidden_batch.to(self._fp_dtype)
        token_mask = token_mask.to(self._device)
        with torch.inference_mode():
            logits = self._head.forward_extracted(
                extracted_hidden_states=hidden_batch,
                extracted_layer_ids=layer_ids,
                token_mask=token_mask,
            )
        pred, prob_correct, probs = self._logits_to_binary_outputs(logits)
        return AuxScore(
            pred=pred,
            prob_correct=prob_correct,
            probs=probs,
            raw={"aux_head_cfg": dict(self._aux_head_cfg), "runtime_prompting": dict(self._runtime_prompting)},
        )

    def score_single(self, *, prompt_text: str, image: Optional[Image.Image], response_text: str) -> AuxScore:
        messages = build_messages_from_prompt_completion(str(prompt_text), str(response_text), has_image=(image is not None))
        return self.score_messages(messages=messages, image=image)


# -----------------------------------------------------------------------------
# Sampling presets / strategy helpers
# -----------------------------------------------------------------------------


def auto_sampling_from_model_name(
    model_name: str,
    profiles: Mapping[str, Dict[str, Any]],
    model_family: str = "auto",
    thinking_mode: Any = "auto",
) -> SamplingConfig:
    name = str(model_name).lower()
    resolved_family = infer_model_family_for_runtime(model_name, model_family)
    resolved_thinking_enabled = resolve_thinking_enabled_for_runtime(model_name, resolved_family, thinking_mode)
    if resolved_thinking_enabled:
        prof = dict(profiles["thinking"])
    elif ("instruct" in name) or (resolved_family in {"qwen3_5", "qwen3", "qwen3_vl", "gemma4"}):
        prof = dict(profiles["instruct"])
    else:
        prof = dict(profiles["default"])
    return SamplingConfig(
        greedy=bool(prof.get("greedy", False)),
        temperature=float(prof.get("temperature", 0.7)),
        top_p=float(prof.get("top_p", 0.8)),
        top_k=int(prof.get("top_k", -1)),
        repetition_penalty=float(prof.get("repetition_penalty", 1.0)),
        presence_penalty=float(prof.get("presence_penalty", 0.0)),
        max_new_tokens=int(prof.get("max_new_tokens", prof.get("out_seq_length", 15000))),
    )


def build_default_two_model_suite(
    *,
    threshold1: float,
    threshold2: float,
    chunk_tokens: int,
    enable_model2_aux: bool,
) -> List[StrategyConfig]:
    return [
        StrategyConfig(
            name="single_agent_model1",
            entry_model="model1",
            model_policies={"model1": AuxPolicy(enabled=False)},
        ),
        StrategyConfig(
            name="single_agent_model2",
            entry_model="model2",
            model_policies={"model2": AuxPolicy(enabled=False)},
        ),
        StrategyConfig(
            name="m1_after_finish_self_repair",
            entry_model="model1",
            model_policies={
                "model1": AuxPolicy(enabled=True, threshold=threshold1, trigger_mode="after_finish", action_below_threshold="self_repair", max_self_repairs=1),
            },
        ),
        StrategyConfig(
            name="m1_after_finish_retry",
            entry_model="model1",
            model_policies={
                "model1": AuxPolicy(enabled=True, threshold=threshold1, trigger_mode="after_finish", action_below_threshold="retry", max_self_repairs=1),
            },
        ),
        StrategyConfig(
            name="m1_after_finish_handoff_fresh_m2",
            entry_model="model1",
            model_policies={
                "model1": AuxPolicy(enabled=True, threshold=threshold1, trigger_mode="after_finish", action_below_threshold="handoff_fresh", next_model="model2"),
                "model2": AuxPolicy(enabled=False),
            },
        ),
        StrategyConfig(
            name="m1_after_finish_handoff_context_m2",
            entry_model="model1",
            model_policies={
                "model1": AuxPolicy(enabled=True, threshold=threshold1, trigger_mode="after_finish", action_below_threshold="handoff_with_context", next_model="model2"),
                "model2": AuxPolicy(enabled=False),
            },
        ),
        StrategyConfig(
            name="m1_after_1000tok_handoff_context_m2",
            entry_model="model1",
            model_policies={
                "model1": AuxPolicy(enabled=True, threshold=threshold1, trigger_mode="every_n_tokens", check_every_n_tokens=1000, action_below_threshold="handoff_with_context", next_model="model2"),
                "model2": AuxPolicy(enabled=False),
            },
        ),
        StrategyConfig(
            name=f"m1_every_{chunk_tokens}_handoff_context_m2",
            entry_model="model1",
            model_policies={
                "model1": AuxPolicy(enabled=True, threshold=threshold1, trigger_mode="every_n_tokens", check_every_n_tokens=chunk_tokens, action_below_threshold="handoff_with_context", next_model="model2"),
                "model2": AuxPolicy(enabled=False),
            },
        ),
        StrategyConfig(
            name=f"m1_every_{chunk_tokens}_handoff_context_m2_with_m2_aux",
            entry_model="model1",
            model_policies={
                "model1": AuxPolicy(enabled=True, threshold=threshold1, trigger_mode="every_n_tokens", check_every_n_tokens=chunk_tokens, action_below_threshold="handoff_with_context", next_model="model2"),
                "model2": AuxPolicy(enabled=enable_model2_aux, threshold=threshold2, trigger_mode="after_finish", action_below_threshold="self_repair", max_self_repairs=1),
            },
            max_total_handoffs=2,
            max_total_repairs=2,
        ),
    ]


def filter_strategies(strategies: Sequence[StrategyConfig], names_csv: str) -> List[StrategyConfig]:
    names = [x.strip() for x in str(names_csv).split(",") if x.strip()]
    if not names or names == ["all"]:
        return list(strategies)
    by_name = {s.name: s for s in strategies}
    missing = [x for x in names if x not in by_name]
    if missing:
        raise RuntimeError(f"Unknown strategy names: {missing}. Available: {list(by_name.keys())}")
    return [by_name[x] for x in names]


# -----------------------------------------------------------------------------
# Benchmark-specific loading, prompting, scoring
# -----------------------------------------------------------------------------


def _get_example_image(ex: Dict[str, Any]) -> Optional[Image.Image]:
    if ex.get("decoded_image") is not None:
        return load_image_any(ex["decoded_image"])
    if ex.get("image") is not None:
        return load_image_any(ex["image"])
    if ex.get("images") is not None:
        imgs = ex["images"]
        if isinstance(imgs, list):
            return load_image_any(imgs[0]) if imgs else None
        return load_image_any(imgs)
    if ex.get("img_path") is not None:
        return load_image_any(ex["img_path"])
    return None


# ----- MathVista -----

def mathvista_get_row_query(row: Dict[str, Any]) -> str:
    q = row.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()
    question = str(row.get("question", "")).strip()
    if not question:
        raise RuntimeError("MathVista row missing both query and question")
    choices = row.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        opts = [f"({chr(ord('A') + i)}) {c}" for i, c in enumerate(choices)]
        question += "\nChoices: " + " ".join(opts)
    return question


def mathvista_build_prompt(row: Dict[str, Any]) -> str:
    base = mathvista_get_row_query(row)
    choices = row.get("choices")
    mcq_note = ""
    if isinstance(choices, list) and len(choices) > 0:
        mcq_note = (
            "\nIf this is multiple choice, do NOT return only the option letter like A, B, C, or D. "
            "Return the actual final answer text/content itself inside \\boxed{...}."
        )
    return base + "\n\nPlease solve the problem carefully." + mcq_note + " Your final answer must appear only once, at the end, inside \\boxed{...}."


def mathvista_build_judge_prompt(row: Dict[str, Any], boxed_answer: Optional[str]) -> str:
    question = str(row.get("question", "")).strip() or mathvista_get_row_query(row)
    gt = str(row.get("answer", "")).strip()
    final_answer = "" if boxed_answer is None else str(boxed_answer).strip()
    choices = row.get("choices")
    choices_text = ""
    if isinstance(choices, list) and len(choices) > 0:
        formatted = [f"{chr(ord('A') + i)}: {c}" for i, c in enumerate(choices)]
        choices_text = "Choices:\n" + "\n".join(formatted) + "\n\n"
    return (
        "You are grading whether a model's final answer matches the gold answer for a math question.\n"
        "Focus only on the final extracted answer and the gold answer.\n"
        "Different formatting, syntax, spacing, punctuation, capitalization, equivalent notation, or minor expression style should NOT matter.\n"
        "For multiple-choice questions, treat the option letter and the corresponding option text as equivalent.\n"
        "If the final answer and gold answer are mathematically or semantically the same, return only \\boxed{1}.\n"
        "Otherwise return only \\boxed{0}.\n"
        "Do not explain anything. Output only \\boxed{1} or \\boxed{0}.\n\n"
        f"Question: {question}\n\n{choices_text}Gold answer: {gt}\n\nModel final extracted answer: {final_answer}\n"
    )


def mathvista_parse_judge_label(text: str) -> int:
    boxed = extract_last_boxed(text)
    if boxed is None:
        raise RuntimeError(f"Judge did not return a boxed label: {text!r}")
    norm = boxed.strip().lower()
    if norm in {"1", "correct", "yes", "true"}:
        return 1
    if norm in {"0", "incorrect", "no", "false"}:
        return 0
    raise RuntimeError(f"Unsupported MathVista judge label: {boxed!r}")


# ----- MathVerse -----

def mathverse_get_row_query(row: Dict[str, Any]) -> str:
    for key in ("query_cot", "query_wo", "question_for_eval", "query", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError("MathVerse row missing question/query fields")


def mathverse_get_eval_question(row: Dict[str, Any]) -> str:
    for key in ("question_for_eval", "query_wo", "query_cot", "question", "query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError("MathVerse row missing evaluation question fields")


def mathverse_build_prompt(row: Dict[str, Any]) -> str:
    q = mathverse_get_row_query(row)
    # if "\\boxed{" in q or "provide the correct option letter" in q.lower() or "please first conduct reasoning" in q.lower():
    #     return q
    return q + "\n\nPlease solve the problem carefully. Your final answer must appear only once, at the end, inside \\boxed{...}."


def mathverse_build_judge_prompt(row: Dict[str, Any], boxed_answer: Optional[str]) -> str:
    question = mathverse_get_eval_question(row)
    gt = str(row.get("answer", row.get("gold_answer", ""))).strip()
    final_answer = "" if boxed_answer is None else str(boxed_answer).strip()
    return (
        "You are grading whether a model's final answer matches the gold answer for a visual math question.\n"
        "Focus only on the final extracted answer and the gold answer.\n"
        "Different formatting, spacing, punctuation, capitalization, or equivalent option notation should NOT matter.\n"
        "If the final answer and gold answer are mathematically or semantically the same, return only \\boxed{1}.\n"
        "Otherwise return only \\boxed{0}.\n"
        "Do not explain anything. Output only \\boxed{1} or \\boxed{0}.\n\n"
        f"Question: {question}\n\nGold answer: {gt}\n\nModel final extracted answer: {final_answer}\n"
    )


# ----- ScreenSpot-Pro -----


def screenspot_resolve_choice_arg(value: str, allowed: Sequence[str]) -> List[str]:
    if value == "all":
        return list(allowed)
    out = [v.strip() for v in str(value).split(",") if v.strip()]
    bad = [v for v in out if v not in allowed]
    if bad:
        raise RuntimeError(f"Unsupported values {bad}; allowed={allowed} or 'all'")
    return out


def screenspot_resolve_dirs(cfg: Dict[str, Any]) -> Tuple[Path, Path, Optional[str]]:
    from huggingface_hub import snapshot_download

    def _validate_pair(ann_dir: Path, img_dir: Path, base: Optional[Path]) -> Optional[Tuple[Path, Path, Optional[str]]]:
        ann_dir = ann_dir.expanduser().resolve()
        img_dir = img_dir.expanduser().resolve()
        if ann_dir.is_dir() and img_dir.is_dir():
            return ann_dir, img_dir, (str(base.resolve()) if base is not None else None)
        return None

    if cfg.get("screenspot_test") and cfg.get("screenspot_imgs"):
        ann_dir = Path(cfg["screenspot_test"])
        img_dir = Path(cfg["screenspot_imgs"])
        resolved = _validate_pair(ann_dir, img_dir, None)
        if resolved is None:
            if not ann_dir.expanduser().resolve().is_dir():
                raise RuntimeError(f"screenspot_test directory not found: {ann_dir.expanduser().resolve()}")
            raise RuntimeError(f"screenspot_imgs directory not found: {img_dir.expanduser().resolve()}")
        return resolved

    if cfg.get("screenspot_root"):
        root = Path(cfg["screenspot_root"]).expanduser().resolve()
        if root.exists():
            direct_candidates = [
                (root / "annotations", root / "images"),
                (root / "annotation", root / "images"),
                (root / "annotations", root / "imgs"),
                (root / "test", root / "images"),
                (root / "annotations", root / "screenshots"),
            ]
            for ann_dir, img_dir in direct_candidates:
                resolved = _validate_pair(ann_dir, img_dir, root)
                if resolved is not None:
                    return resolved

            recursive_pairs: List[Tuple[Path, Path]] = []
            try:
                ann_dirs = [p for p in root.rglob('*') if p.is_dir() and p.name.lower() in {"annotations", "annotation", "test"}]
                img_dirs = [p for p in root.rglob('*') if p.is_dir() and p.name.lower() in {"images", "imgs", "screenshots"}]
                for ann_dir in ann_dirs:
                    for img_dir in img_dirs:
                        if ann_dir.parent == img_dir.parent:
                            recursive_pairs.append((ann_dir, img_dir))
            except Exception:
                recursive_pairs = []

            seen = set()
            dedup_pairs: List[Tuple[Path, Path]] = []
            for ann_dir, img_dir in recursive_pairs:
                key = (str(ann_dir.resolve()), str(img_dir.resolve()))
                if key not in seen:
                    seen.add(key)
                    dedup_pairs.append((ann_dir, img_dir))
            for ann_dir, img_dir in dedup_pairs:
                resolved = _validate_pair(ann_dir, img_dir, ann_dir.parent)
                if resolved is not None:
                    return resolved

            debug_print(True, f"[DEBUG] screenspot_root did not contain a usable annotations/images pair under: {root}. Falling back to HF snapshot download.")
        else:
            debug_print(True, f"[DEBUG] screenspot_root does not exist: {root}. Falling back to HF snapshot download.")

    snapshot_dir = Path(snapshot_download(
        repo_id=cfg["dataset_repo_id"],
        repo_type="dataset",
        allow_patterns=["annotations/*.json", "images/**"],
    )).resolve()
    ann_dir, img_dir = snapshot_dir / "annotations", snapshot_dir / "images"
    resolved = _validate_pair(ann_dir, img_dir, snapshot_dir)
    if resolved is None:
        raise RuntimeError(f"Dataset snapshot missing annotations/ or images/: {snapshot_dir}")
    return resolved


def screenspot_get_prompt_to_evaluate(ex: Dict[str, Any]) -> str:
    for key in ("prompt_to_evaluate", "instruction", "instruction_cn", "action", "description", "query", "question", "prompt_text", "prompt", "text"):
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError(f"ScreenSpot-Pro row missing prompt/instruction. keys={list(ex.keys())}")


def screenspot_build_prompt(ex: Dict[str, Any]) -> str:
    instruction = screenspot_get_prompt_to_evaluate(ex)
    gt_type = str(ex.get("gt_type", "positive")).strip().lower()
    base = (
        "You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. "
        "If you are not sure about where to click, guess a most likely one.\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        "{\"type\": \"function\", \"function\": {\"name\": \"computer_use\", \"description\": \"Use a mouse to interact with a computer.\\n* The screen's resolution is 1000x1000.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \\n* You can only use the left_click action to interact with the computer.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `left_click`: Click the left mouse button with coordinate (x, y).\", \"enum\": [\"left_click\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.\", \"type\": \"array\"}}, \"required\": [\"action\"], \"type\": \"object\"}}}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [x, y]}}\n"
        "</tool_call>\n\n"
    )
    tail = ""
    if gt_type == "negative":
        tail = (
            "If the target element is not present in the screenshot, return exactly <tool_call>\n"
            "{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [-1, -1]}}\n"
            "</tool_call>.\n\n"
        )
    return base + tail + instruction


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL | re.IGNORECASE)


def _extract_last_tool_call_json(text: str) -> Optional[str]:
    matches = _TOOL_CALL_RE.findall(strip_think(text))
    return matches[-1].strip() if matches else None


def _find_first_two_numbers(text: str) -> Optional[Tuple[float, float]]:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def screenspot_parse_response(raw_response: str) -> Dict[str, Any]:
    text = strip_think(raw_response)
    lowered = text.lower()
    if re.search(r"\bnegative\b", lowered):
        return {"result": "negative", "point": None, "boxed_answer": extract_last_boxed(text), "tool_call": _extract_last_tool_call_json(text)}
    tool_call_json = _extract_last_tool_call_json(text)
    if tool_call_json is not None:
        try:
            action = json.loads(tool_call_json)
            args = action.get("arguments", {})
            coords = args.get("coordinate")
            nums = [float(x) for x in coords]
            if len(nums) == 2:
                x, y = nums
            elif len(nums) == 4:
                x1, y1, x2, y2 = nums
                x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            else:
                raise ValueError("bad coordinate length")
            if x == -1 and y == -1:
                return {"result": "negative", "point": None, "boxed_answer": extract_last_boxed(text), "tool_call": tool_call_json}
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                point = [x, y]
            elif 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
                point = [x / 1000.0, y / 1000.0]
            else:
                raise ValueError("bad coordinate range")
            return {"result": "positive", "point": point, "boxed_answer": extract_last_boxed(text), "tool_call": tool_call_json}
        except Exception:
            pass
    boxed = extract_last_boxed(text)
    if boxed is not None:
        s = str(boxed).strip().lower()
        if s in {"negative", "none", "not_found", "not found", "absent"}:
            return {"result": "negative", "point": None, "boxed_answer": boxed, "tool_call": tool_call_json}
        pair = _find_first_two_numbers(str(boxed))
        if pair is not None:
            x, y = pair
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                return {"result": "positive", "point": [x, y], "boxed_answer": boxed, "tool_call": tool_call_json}
            if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
                return {"result": "positive", "point": [x / 1000.0, y / 1000.0], "boxed_answer": boxed, "tool_call": tool_call_json}
    pair = _find_first_two_numbers(text)
    if pair is not None:
        x, y = pair
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return {"result": "positive", "point": [x, y], "boxed_answer": boxed, "tool_call": tool_call_json}
        if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
            return {"result": "positive", "point": [x / 1000.0, y / 1000.0], "boxed_answer": boxed, "tool_call": tool_call_json}
    return {"result": "wrong_format", "point": None, "boxed_answer": boxed, "tool_call": tool_call_json}


def _screenspot_coerce_img_size(x: Any) -> Tuple[float, float]:
    if x is None:
        raise RuntimeError("ScreenSpot row missing img_size")
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        w, h = float(x[0]), float(x[1])
    elif isinstance(x, dict):
        if "width" in x and "height" in x:
            w, h = float(x["width"]), float(x["height"])
        elif "w" in x and "h" in x:
            w, h = float(x["w"]), float(x["h"])
        else:
            raise RuntimeError(f"Unsupported ScreenSpot img_size dict keys: {sorted(x.keys())}")
    else:
        raise RuntimeError(f"Unsupported ScreenSpot img_size format: {type(x).__name__}: {x}")
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid ScreenSpot img_size values: {(w, h)}")
    return w, h


def _screenspot_coerce_bbox_xyxy(x: Any) -> List[float]:
    if x is None:
        raise RuntimeError("ScreenSpot positive row missing bbox")
    if isinstance(x, (list, tuple)) and len(x) >= 4:
        return [float(x[0]), float(x[1]), float(x[2]), float(x[3])]
    if isinstance(x, dict):
        if all(k in x for k in ("x1", "y1", "x2", "y2")):
            return [float(x["x1"]), float(x["y1"]), float(x["x2"]), float(x["y2"])]
        if all(k in x for k in ("left", "top", "right", "bottom")):
            return [float(x["left"]), float(x["top"]), float(x["right"]), float(x["bottom"])]
        if all(k in x for k in ("x", "y", "width", "height")):
            x1 = float(x["x"])
            y1 = float(x["y"])
            return [x1, y1, x1 + float(x["width"]), y1 + float(x["height"])]
        raise RuntimeError(f"Unsupported ScreenSpot bbox dict keys: {sorted(x.keys())}")
    raise RuntimeError(f"Unsupported ScreenSpot bbox format: {type(x).__name__}: {x}")


def screenspot_bbox_to_normalized_xyxy(bbox_xyxy: List[float], img_size: Tuple[int, int]) -> List[float]:
    w, h = img_size
    return [bbox_xyxy[0] / w, bbox_xyxy[1] / h, bbox_xyxy[2] / w, bbox_xyxy[3] / h]


def screenspot_eval_saved_row(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed = screenspot_parse_response(row["raw_response"])
    img_size = _screenspot_coerce_img_size(row.get("img_size"))
    gt_type = str(row.get("gt_type", "positive")).lower()
    if gt_type == "positive":
        bbox = _screenspot_coerce_bbox_xyxy(row.get("bbox"))
        norm_bbox = screenspot_bbox_to_normalized_xyxy(bbox, img_size)
        point = parsed["point"]
        if point is None:
            correctness = "wrong_format"
        elif norm_bbox[0] <= point[0] <= norm_bbox[2] and norm_bbox[1] <= point[1] <= norm_bbox[3]:
            correctness = "correct"
        else:
            correctness = "wrong"
    else:
        if parsed["result"] == "negative":
            correctness = "correct"
        elif parsed["result"] == "positive":
            correctness = "wrong"
        else:
            correctness = "wrong_format"
    return {
        "parsed_result": parsed["result"],
        "parsed_point": parsed["point"],
        "boxed_answer": parsed["boxed_answer"],
        "tool_call": parsed["tool_call"],
        "correctness": correctness,
        "benchmark_correct": int(correctness == "correct"),
        "benchmark_score": float(correctness == "correct"),
    }


def screenspot_metric_block(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct_num = sum(1 for r in results if r["correctness"] == "correct")
    wrong_format_num = sum(1 for r in results if r["correctness"] == "wrong_format")
    text_results = [r for r in results if r.get("ui_type") == "text"]
    icon_results = [r for r in results if r.get("ui_type") == "icon"]
    text_correct = sum(1 for r in text_results if r["correctness"] == "correct")
    icon_correct = sum(1 for r in icon_results if r["correctness"] == "correct")
    total = len(results)
    return {
        "num_correct_action": correct_num,
        "num_total": total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / total if total else 0.0,
        "text_acc": text_correct / len(text_results) if text_results else 0.0,
        "icon_acc": icon_correct / len(icon_results) if icon_results else 0.0,
    }


def screenspot_group_metrics(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    values = sorted({r.get(key) for r in rows if r.get(key) is not None})
    for value in values:
        subset = [r for r in rows if r.get(key) == value]
        out[str(value)] = screenspot_metric_block(subset)
    return out


def screenspot_summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "overall": screenspot_metric_block(rows),
        "by_group": screenspot_group_metrics(rows, "group"),
        "by_application": screenspot_group_metrics(rows, "application"),
        "by_platform": screenspot_group_metrics(rows, "platform"),
        "by_ui_type": screenspot_group_metrics(rows, "ui_type"),
        "by_gt_type": screenspot_group_metrics(rows, "gt_type"),
    }


# ----- SimpleVQA -----

def simplevqa_load_named_split(dataset_name: str, split_name: str):
    from datasets import Dataset, DatasetDict, load_dataset
    ds = load_dataset(dataset_name)
    if isinstance(ds, DatasetDict):
        if split_name not in ds:
            raise RuntimeError(f"Split {split_name!r} not found in dataset. Available: {list(ds.keys())}")
        ds = ds[split_name]
    else:
        from datasets import Dataset
        if not isinstance(ds, Dataset):
            raise RuntimeError(f"Unsupported dataset object type: {type(ds)}")
        if split_name not in {"train", "validation", "test", "all"}:
            raise RuntimeError(f"Dataset loaded as single Dataset; split={split_name!r} is unsupported")
    return ds


def simplevqa_filter_english_only(ds):
    lang_key = None
    for k in ("language", "lang", "Language"):
        if k in ds.column_names:
            lang_key = k
            break
    if lang_key is None:
        raise RuntimeError(f"Could not find language column. Columns: {ds.column_names}")
    ds = ds.filter(lambda ex: str(ex[lang_key]).strip().upper() == "EN")
    if len(ds) == 0:
        raise RuntimeError("English-only filter produced an empty dataset")
    return ds


def simplevqa_get_example_id(ex: Dict[str, Any], fallback_idx: int) -> str:
    for k in ("pid", "id", "question_id", "data_id", "uid"):
        if k in ex and ex[k] is not None:
            return str(ex[k])
    return f"simplevqa_{fallback_idx:06d}"


def simplevqa_get_question(ex: Dict[str, Any]) -> str:
    q = ex.get("question") or ex.get("query") or ex.get("prompt") or ex.get("text")
    if not isinstance(q, str) or not q.strip():
        raise RuntimeError(f"Could not find question text in SimpleVQA example keys={list(ex.keys())}")
    return q.strip()


def simplevqa_build_prompt(ex: Dict[str, Any]) -> str:
    q = simplevqa_get_question(ex)
    return (
        "Answer the visual question concisely. Your final answer must appear only once, at the end, inside \\boxed{...}. "
        "If you do not use \\boxed{...}, then put the final answer clearly after your thinking is finished. "
        "Do not put explanations after the final answer.\n"
        f"Question: {q}"
    )


def simplevqa_get_answer(ex: Dict[str, Any]) -> str:
    for k in ("answer", "answers", "gt_answer", "label"):
        if k in ex and ex[k] is not None:
            v = ex[k]
            if isinstance(v, list):
                return ", ".join(str(x) for x in v)
            return str(v)
    raise RuntimeError(f"Could not find SimpleVQA answer in keys={list(ex.keys())}")


def simplevqa_strip_after_think(text: str) -> Optional[str]:
    text = str(text or "").strip()
    if "</think>" not in text:
        return None
    tail = text.split("</think>")[-1].strip()
    return tail if tail else None


def simplevqa_extract_final_answer(text: str) -> Tuple[Optional[str], str]:
    boxed = extract_last_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip(), "boxed"
    post_think = simplevqa_strip_after_think(text)
    if post_think is not None and post_think.strip():
        return post_think.strip(), "post_think"
    if "</think>" not in str(text or ""):
        return None, "no_endthink"
    return None, "empty_after_think"


def simplevqa_build_judge_prompt(question: str, gold_answer: str, final_answer: Optional[str]) -> str:
    candidate_text = final_answer if final_answer is not None else "<NO_FINAL_ANSWER>"
    return (
        "You are judging a visual question answering prediction.\n"
        "Given the question, the ground-truth answer, and the model's FINAL extracted answer, return exactly one label in \\boxed{}:\n"
        "- \\boxed{correct} if the final extracted answer is semantically correct\n"
        "- \\boxed{incorrect} if the final extracted answer is wrong\n"
        "- \\boxed{not_attempted} if there is no final answer or the model refused / did not answer\n"
        "Be strict about the final extracted answer only. Ignore any other text.\n"
        f"Question: {question}\nGround truth answer: {gold_answer}\nModel final extracted answer: {candidate_text}\n"
        "Return only one boxed label."
    )


def simplevqa_parse_judge_label(text: str) -> str:
    boxed = extract_last_boxed(text)
    if boxed is None:
        raise RuntimeError(f"Judge did not return a boxed label: {text!r}")
    label = normalize_text_loose(boxed)
    if label in {"correct", "incorrect", "not attempted", "not_attempted"}:
        return label.replace(" ", "_")
    raise RuntimeError(f"Unsupported SimpleVQA judge label: {boxed!r}")


def simplevqa_metrics(labels: Sequence[str]) -> Dict[str, Any]:
    total = len(labels)
    correct = sum(1 for x in labels if x == "correct")
    incorrect = sum(1 for x in labels if x == "incorrect")
    not_attempted = sum(1 for x in labels if x == "not_attempted")
    attempted = correct + incorrect
    acc_given_attempted = (correct / attempted) if attempted > 0 else 0.0
    f1 = 0.0
    if (acc_given_attempted + (correct / total if total else 0.0)) > 0:
        f1 = 2.0 * acc_given_attempted * (correct / total) / (acc_given_attempted + (correct / total))
    return {
        "is_correct": correct / total if total else 0.0,
        "is_incorrect": incorrect / total if total else 0.0,
        "is_not_attempted": not_attempted / total if total else 0.0,
        "is_given_attempted": attempted / total if total else 0.0,
        "accuracy_given_attempted": acc_given_attempted,
        "f1": f1,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "total": total,
    }


EVAL_TAIL_FALLBACK_MAX_TOKENS = 1000


def _maybe_get_runtime_tokenizer(runtime: Optional[Any]):
    if runtime is None:
        return None
    processor = getattr(runtime, "processor", None)
    return getattr(processor, "tokenizer", None) if processor is not None else None


def _tail_text_by_tokens(text: str, tokenizer: Optional[Any], max_tokens: int = EVAL_TAIL_FALLBACK_MAX_TOKENS) -> str:
    s = strip_think(text)
    if not s:
        return ""
    if tokenizer is not None:
        try:
            ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            if len(ids) > int(max_tokens):
                ids = ids[-int(max_tokens):]
            return str(tokenizer.decode(ids, skip_special_tokens=False)).strip()
        except Exception:
            pass
    parts = s.split()
    if len(parts) <= int(max_tokens):
        return s.strip()
    return " ".join(parts[-int(max_tokens):]).strip()


def _judge_tail_candidate_text(text: str, judge_runtime: Optional[Any], max_tokens: int = EVAL_TAIL_FALLBACK_MAX_TOKENS) -> str:
    tok = _maybe_get_runtime_tokenizer(judge_runtime)
    return _tail_text_by_tokens(text, tok, max_tokens=max_tokens)


def _candidate_or_placeholder(candidate_text: Optional[str]) -> str:
    s = str(candidate_text or "").strip()
    return s if s else "<NO_FINAL_ANSWER>"


def _build_textbench_rm_judge_prompt(
    benchmark: str,
    row: Dict[str, Any],
    candidate_text: Optional[str],
    candidate_source: str,
) -> str:
    benchmark = canonicalize_benchmark_name(benchmark)
    question = str(row.get("question") or row.get("prompt_text") or "").strip()
    gold = str(row.get("gold_answer_raw") if row.get("gold_answer_raw") is not None else row.get("gold_answer") or "").strip()
    choices = row.get("choices")
    choices_text = ""
    if isinstance(choices, list) and choices:
        formatted = [f"{chr(ord('A') + i)}: {c}" for i, c in enumerate(choices)]
        choices_text = "Choices:\n" + "\n".join(formatted) + "\n\n"
    benchmark_note = ""
    if benchmark == "triviaqa":
        benchmark_note = "Treat aliases, punctuation differences, capitalization, and minor formatting variants as equivalent when they express the same answer.\n"
    elif benchmark == "math":
        benchmark_note = "Treat mathematically equivalent expressions, values, and formats as the same answer.\n"
    elif benchmark == "mmlu_pro":
        benchmark_note = "For multiple-choice questions, treat the option letter and the corresponding option text as equivalent.\n"
    if candidate_source.startswith("tail_"):
        source_note = (
            f"The explicit final-answer parse failed, so the model's last {EVAL_TAIL_FALLBACK_MAX_TOKENS} tokens are provided below. "
            "Infer the model's intended final answer from that tail only.\n"
        )
    elif candidate_source == "boxed":
        source_note = "The candidate answer below came from the model's final boxed answer.\n"
    else:
        source_note = f"The candidate answer below came from: {candidate_source}.\n"
    return (
        "You are grading whether a model's final answer matches the gold answer for a benchmark question.\n"
        "Focus on the candidate answer and the gold answer only.\n"
        + benchmark_note
        + source_note
        + "If the candidate answer is semantically correct, return only \\boxed{1}. Otherwise return only \\boxed{0}.\n"
        "Do not explain anything. Output only \\boxed{1} or \\boxed{0}.\n\n"
        f"Question: {question}\n\n{choices_text}Gold answer: {gold}\n\nModel candidate answer: {_candidate_or_placeholder(candidate_text)}\n"
    )


def _build_screenspot_rm_judge_prompt(row: Dict[str, Any], candidate_text: Optional[str], candidate_source: str) -> str:
    gt_type = str(row.get("gt_type", "positive")).strip().lower()
    prompt_text = str(row.get("prompt_to_evaluate") or row.get("prompt_text") or "").strip()
    img_size = row.get("img_size")
    bbox = row.get("bbox")
    bbox_text = f"Ground-truth bbox: {bbox}\n" if gt_type == "positive" and bbox is not None else ""
    source_note = (
        f"The structured parser could not reliably parse the response, so the model's last {EVAL_TAIL_FALLBACK_MAX_TOKENS} tokens are provided below. "
        "Infer whether the model's intended action is correct from that tail."
        if candidate_source.startswith("tail_")
        else f"Candidate source: {candidate_source}."
    )
    negative_note = (
        "This is a negative example: the correct behavior is to indicate that the target is absent / not found, not to click anywhere."
        if gt_type != "positive"
        else "This is a positive example: the correct behavior is to click the target element."
    )
    return (
        "You are grading whether a UI-grounding / screen-spotting response is correct.\n"
        "Use the screenshot, the instruction, and the candidate answer below.\n"
        f"{negative_note}\n"
        f"{source_note}\n"
        "If the model's intended answer/action is correct, return only \\boxed{1}. Otherwise return only \\boxed{0}.\n"
        "Do not explain anything. Output only \\boxed{1} or \\boxed{0}.\n\n"
        f"Instruction: {prompt_text}\n"
        f"Ground-truth type: {gt_type}\n"
        f"Image size: {img_size}\n"
        f"{bbox_text}"
        f"Model candidate answer: {_candidate_or_placeholder(candidate_text)}\n"
    )

def response_final_answer_status(benchmark: str, text: str) -> Dict[str, Any]:
    benchmark = canonicalize_benchmark_name(benchmark)
    raw_text = str(text or "")

    if benchmark in {"mathvista", "mathverse", "triviaqa", "math", "mmlu_pro"}:
        boxed = extract_last_boxed(raw_text)
        has_final = boxed is not None and bool(str(boxed).strip()) and (not is_refusal(raw_text))
        return {
            "has_final_answer": bool(has_final),
            "reason": "boxed" if has_final else ("refusal" if is_refusal(raw_text) else "missing_boxed_answer"),
            "final_answer": boxed,
        }

    if benchmark == "screenspot_pro":
        parsed = screenspot_parse_response(raw_text)
        result = str(parsed.get("result", "wrong_format"))
        has_final = result in {"positive", "negative"} and (not is_refusal(raw_text))
        return {
            "has_final_answer": bool(has_final),
            "reason": result if has_final else ("refusal" if is_refusal(raw_text) else result),
            "final_answer": parsed.get("point") if result == "positive" else result,
        }

    if benchmark in {"simplevqa", "charxiv_reasoning"}:
        final_answer, final_source = simplevqa_extract_final_answer(raw_text)
        has_final = final_answer is not None and bool(str(final_answer).strip()) and (not is_refusal(raw_text))
        return {
            "has_final_answer": bool(has_final),
            "reason": final_source if has_final else ("refusal" if is_refusal(raw_text) else final_source),
            "final_answer": final_answer,
        }

    raise RuntimeError(f"Unknown benchmark: {benchmark}")


def response_has_usable_final_answer(benchmark: str, text: str) -> bool:
    return bool(response_final_answer_status(benchmark, text).get("has_final_answer", False))


# -----------------------------------------------------------------------------
# Benchmark dispatch helpers
# -----------------------------------------------------------------------------


# ----- CharXiv Reasoning -----

def get_prompt_text(ex: Dict[str, Any]) -> str:
    return str(ex["prompt_text"])


PROMPT_CANDIDATES = ["prompt", "question", "query", "input", "instruction"]
ANSWER_CANDIDATES = ["answer", "final_answer", "target", "label", "answers"]
SOLUTION_CANDIDATES = ["solution", "rationale", "steps", "explanation", "cot"]
ORIGINAL_SOURCE_CANDIDATES = ["original_source", "source"]


def _textbench_pick_split(ds: Any, desired: str = "train"):
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        if desired and desired in ds:
            return ds[desired]
        for s in ["train", "validation", "dev", "val", "test"]:
            if s in ds:
                return ds[s]
        return ds[next(iter(ds.keys()))]
    return ds


def _textbench_load_dataset_from_cfg(dataset_cfg: Dict[str, Any]):
    from datasets import load_dataset, load_from_disk
    data_mode = str(dataset_cfg.get("data_mode", "hf") or "hf").strip().lower()
    dataset_name = dataset_cfg.get("dataset_name") or dataset_cfg.get("dataset_id") or ""
    dataset_config_name = dataset_cfg.get("dataset_config_name") or dataset_cfg.get("dataset_config") or dataset_cfg.get("config_name")
    split = dataset_cfg.get("split") or dataset_cfg.get("dataset_split") or "train"
    data_path = dataset_cfg.get("data_path") or ""

    if data_mode == "hf":
        if not dataset_name:
            raise RuntimeError("HF text benchmark config requires dataset_name or dataset_id")
        if dataset_config_name:
            return load_dataset(dataset_name, dataset_config_name, split=split)
        return load_dataset(dataset_name, split=split)
    if data_mode == "disk":
        if not data_path:
            raise RuntimeError("disk text benchmark config requires data_path")
        ds = load_from_disk(data_path)
        return _textbench_pick_split(ds, desired=str(split))
    if data_mode == "csv":
        if not data_path:
            raise RuntimeError("csv text benchmark config requires data_path")
        return load_dataset("csv", data_files=data_path)["train"]
    if data_mode == "parquet":
        if not data_path:
            raise RuntimeError("parquet text benchmark config requires data_path")
        return load_dataset("parquet", data_files=data_path)["train"]
    raise RuntimeError(f"Unsupported text benchmark data_mode: {data_mode}")


def _textbench_normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, dict):
        for k in ["value", "normalized_value", "text", "answer", "label"]:
            if k in x and str(x[k]).strip():
                return str(x[k]).strip()
        for k in ["aliases", "normalized_aliases"]:
            if k in x and isinstance(x[k], list) and x[k]:
                vals = [str(v).strip() for v in x[k] if str(v).strip()]
                if vals:
                    return " | ".join(vals)
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, (list, tuple)):
        vals = [_textbench_normalize_text(v) for v in x]
        vals = [v for v in vals if v]
        return " | ".join(vals)
    return str(x).strip()


def _textbench_maybe_letter(s: str) -> Optional[str]:
    s = str(s or "").strip()
    if len(s) == 1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return s.upper()
    m = re.search(r"\b([A-Z])\b", s.upper())
    return m.group(1) if m else None


def _textbench_extract_choices(ex: Dict[str, Any]) -> List[str]:
    candidates = [
        ex.get("choices"), ex.get("options"), ex.get("mcq_choices"), ex.get("candidates")
    ]
    for val in candidates:
        if isinstance(val, list) and val:
            return [_textbench_normalize_text(v) for v in val]
        if isinstance(val, dict) and val:
            out = []
            for k in sorted(val.keys()):
                v = val[k]
                if isinstance(v, (list, tuple)):
                    if v:
                        out.append(_textbench_normalize_text(v[0]))
                else:
                    out.append(_textbench_normalize_text(v))
            if out:
                return out
    # Common MMLU/GPQA patterns
    if all(str(ex.get(k, "")).strip() for k in ["A", "B", "C", "D"]):
        out = [_textbench_normalize_text(ex[k]) for k in ["A", "B", "C", "D"]]
        if str(ex.get("E", "")).strip():
            out.append(_textbench_normalize_text(ex["E"]))
        return out
    return []


def _textbench_format_choices_block(choices: Sequence[str]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))


def _textbench_build_question_text(ex: Dict[str, Any]) -> str:
    q = ""
    for c in PROMPT_CANDIDATES:
        if c in ex and str(ex[c]).strip():
            q = str(ex[c]).strip()
            break
    if not q:
        raise RuntimeError(f"Could not find prompt/question in example keys={sorted(ex.keys())}")
    choices = _textbench_extract_choices(ex)
    if choices:
        q = q.rstrip() + "\n\nChoices:\n" + _textbench_format_choices_block(choices)
    return q


def _textbench_extract_raw_gold(ex: Dict[str, Any], benchmark: str) -> Any:
    benchmark = canonicalize_benchmark_name(benchmark)
    if benchmark == "math":
        for c in SOLUTION_CANDIDATES + ANSWER_CANDIDATES:
            if c in ex and ex[c] is not None and str(ex[c]).strip():
                return ex[c]
        return None
    for c in ANSWER_CANDIDATES:
        if c in ex and ex[c] is not None and str(ex[c]).strip():
            return ex[c]
    return None


def _textbench_extract_gold_answer(ex: Dict[str, Any], benchmark: str) -> str:
    benchmark = canonicalize_benchmark_name(benchmark)
    for c in ANSWER_CANDIDATES:
        if c in ex and ex[c] is not None and str(ex[c]).strip():
            val = ex[c]
            if benchmark == "triviaqa" and isinstance(val, dict):
                aliases = val.get("aliases") or val.get("normalized_aliases")
                if isinstance(aliases, list) and aliases:
                    vals = [str(a).strip() for a in aliases if str(a).strip()]
                    if vals:
                        return " | ".join(vals)
            return _textbench_normalize_text(val)
    if benchmark == "math":
        for c in SOLUTION_CANDIDATES:
            if c in ex and ex[c] is not None and str(ex[c]).strip():
                return _textbench_normalize_text(ex[c])
    return ""


def _textbench_expand_gold_for_mcq(gold: str, choices: Sequence[str]) -> str:
    gold = _textbench_normalize_text(gold)
    if not gold:
        return gold
    letter = _textbench_maybe_letter(gold)
    if letter is not None:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(choices):
            return f"{letter} | {choices[idx]}"
    return gold


def textbench_build_prompt(ex: Dict[str, Any], benchmark: str) -> str:
    benchmark = canonicalize_benchmark_name(benchmark)
    base = _textbench_build_question_text(ex)
    if benchmark == "triviaqa":
        suffix = (
            "\n\nThis is a trivia question. "
            "Answer carefully and put your final answer only once at the end inside \\boxed{...}."
        )
    elif benchmark == "math":
        suffix = (
            "\n\nPlease reason carefully and put your final answer only once at the end inside\\boxed{...}."
        )
    elif benchmark == "mmlu_pro":
        suffix = (
            "\n\nPlease solve the multiple-choice question carefully. "
            "Put the single best choice letter inside\\boxed{...}."
        )
    else:
        raise RuntimeError(f"Unsupported text benchmark: {benchmark}")
    return base + suffix


def _textbench_extract_pred_answer(text: str, benchmark: str) -> Tuple[str, bool]:
    benchmark = canonicalize_benchmark_name(benchmark)
    raw = str(text or "")
    boxed = extract_last_boxed(raw)
    if boxed is not None and str(boxed).strip():
        pred = _textbench_normalize_text(boxed)
        if benchmark == "mmlu_pro":
            letter = _textbench_maybe_letter(pred)
            return ((letter or pred), True)
        return pred, True
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    pred = _textbench_normalize_text(last)
    if benchmark == "mmlu_pro":
        letter = _textbench_maybe_letter(pred)
        return ((letter or pred), False)
    return pred, False


def _textbench_match_prediction(benchmark: str, pred: str, gold: str, row: Dict[str, Any]) -> Optional[float]:
    benchmark = canonicalize_benchmark_name(benchmark)
    pred_n = normalize_text_loose(pred)
    gold_n = normalize_text_loose(gold)
    if not pred_n:
        return 0.0
    if benchmark == "triviaqa":
        gold_aliases = [normalize_text_loose(x) for x in str(gold or "").split("|") if normalize_text_loose(x)]
        return 1.0 if pred_n in gold_aliases else 0.0
    if benchmark == "mmlu_pro":
        pred_letter = _textbench_maybe_letter(pred_n) or pred_n[:1].upper()
        gold_letter = _textbench_maybe_letter(gold_n) or gold_n[:1].upper()
        return 1.0 if pred_letter == gold_letter else 0.0
    return 1.0 if pred_n == gold_n else 0.0


def _direct_eval_batch(benchmark: str, row_chunk: Sequence[Dict[str, Any]], gen_texts: Sequence[str]) -> List[Tuple[Optional[float], str, str, bool, str]]:
    benchmark = canonicalize_benchmark_name(benchmark)
    out: List[Tuple[Optional[float], str, str, bool, str]] = []
    for row, text in zip(row_chunk, gen_texts):
        pred, parsed = _textbench_extract_pred_answer(str(text or ""), benchmark)
        gold = str(row.get("gold_answer") or "")
        correct = _textbench_match_prediction(benchmark, pred, gold, row)
        out.append((correct, pred, gold, parsed, "fallback_exact_match"))
    return out


# -----------------------------------------------------------------------------
# Benchmark dispatch helpers
# -----------------------------------------------------------------------------


# ----- CharXiv Reasoning -----

def charxiv_get_question(ex: Dict[str, Any]) -> str:
    q = ex.get("reasoning_q") or ex.get("question") or ex.get("query") or ex.get("prompt") or ex.get("text")
    if not isinstance(q, str) or not q.strip():
        raise RuntimeError(f"Could not find CharXiv reasoning question in keys={list(ex.keys())}")
    return q.strip()


def charxiv_get_answer(ex: Dict[str, Any]) -> str:
    a = ex.get("reasoning_a") or ex.get("answer") or ex.get("gold_answer")
    if a is None:
        raise RuntimeError(f"Could not find CharXiv reasoning answer in keys={list(ex.keys())}")
    return str(a).strip()


def charxiv_build_prompt(ex: Dict[str, Any]) -> str:
    q = charxiv_get_question(ex)
    return (
        "Answer the chart reasoning question carefully and concisely. "
        "Your final answer must appear only once, at the end, inside \\boxed{...}. "
        "Do not put explanations after the final answer.\n"
        f"Question: {q}"
    )


def charxiv_build_judge_prompt(question: str, gold_answer: str, final_answer: Optional[str]) -> str:
    candidate_text = final_answer if final_answer is not None else "<NO_FINAL_ANSWER>"
    return (
        "You are judging a chart reasoning prediction.\n"
        "Given the question, the ground-truth answer, and the model's FINAL extracted answer, return exactly one label in \\boxed{}:\n"
        "- \\boxed{correct} if the final extracted answer is semantically correct\n"
        "- \\boxed{incorrect} if the final extracted answer is wrong\n"
        "- \\boxed{not_attempted} if there is no final answer or the model refused / did not answer\n"
        "Be strict about the final extracted answer only. Ignore any other text.\n"
        f"Question: {question}\nGround truth answer: {gold_answer}\nModel final extracted answer: {candidate_text}\n"
        "Return only one boxed label."
    )


def benchmark_needs_judge(benchmark: str) -> bool:
    benchmark = canonicalize_benchmark_name(benchmark)
    return benchmark in {"mathvista", "mathverse", "simplevqa", "charxiv_reasoning", "screenspot_pro", "triviaqa", "math", "mmlu_pro"}


def build_initial_messages(benchmark: str, ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    benchmark = canonicalize_benchmark_name(benchmark)
    if benchmark in {"mathvista", "mathverse", "screenspot_pro", "simplevqa", "charxiv_reasoning", "triviaqa", "math", "mmlu_pro"}:
        content: List[Dict[str, Any]] = []
        if _get_example_image(ex) is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": str(ex["prompt_text"])})
        return [{"role": "user", "content": content}]
    raise RuntimeError(f"Unknown benchmark: {benchmark}")


def get_example_image_for_benchmark(ex: Dict[str, Any]) -> Optional[Image.Image]:
    return _get_example_image(ex)

def _load_hf_dataset_from_cfg(dataset_cfg: Dict[str, Any]):
    from datasets import load_dataset
    dataset_name = dataset_cfg["dataset_name"]
    dataset_config_name = dataset_cfg.get("dataset_config_name") or dataset_cfg.get("config_name")
    split = dataset_cfg.get("split")

    if dataset_config_name:
        if split:
            return load_dataset(dataset_name, dataset_config_name, split=split)
        return load_dataset(dataset_name, dataset_config_name)

    if split:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name)

def load_examples_for_benchmark(benchmark: str, dataset_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    benchmark = canonicalize_benchmark_name(benchmark)
    if benchmark == "mathvista":
        from datasets import Image as HFImage, load_dataset
        ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["split"])
        if int(dataset_cfg.get("max_samples", -1)) > 0:
            ds = ds.select(range(min(int(dataset_cfg["max_samples"]), len(ds))))
        image_col = "decoded_image" if "decoded_image" in ds.column_names else "image"
        if image_col not in ds.column_names:
            raise RuntimeError(f"No image column found. Columns: {ds.column_names}")
        if not isinstance(ds.features[image_col], HFImage):
            ds = ds.cast_column(image_col, HFImage())
        rows = []
        for i in range(len(ds)):
            row = ds[i]
            rows.append({
                "sample_idx": i,
                "decoded_image": load_image_any(row[image_col]),
                "question": row.get("question"),
                "query": row.get("query"),
                "choices": row.get("choices"),
                "answer": row.get("answer"),
                "unit": row.get("unit"),
                "precision": row.get("precision"),
                "answer_type": row.get("answer_type"),
                "question_type": row.get("question_type"),
                "metadata": row.get("metadata"),
                "prompt_text": mathvista_build_prompt(row),
            })
        return rows

    if benchmark == "mathverse":
        from datasets import Image as HFImage, load_dataset
        ds = _load_hf_dataset_from_cfg(dataset_cfg)
        if int(dataset_cfg.get("max_samples", -1)) > 0:
            ds = ds.select(range(min(int(dataset_cfg["max_samples"]), len(ds))))
        image_col = "decoded_image" if "decoded_image" in ds.column_names else "image"
        if image_col not in ds.column_names:
            raise RuntimeError(f"No image column found. Columns: {ds.column_names}")
        if not isinstance(ds.features[image_col], HFImage):
            ds = ds.cast_column(image_col, HFImage())
        rows = []
        for i in range(len(ds)):
            row = ds[i]
            rows.append({
                "sample_idx": i,
                "decoded_image": load_image_any(row[image_col]),
                "question": row.get("question_for_eval") or row.get("query_wo") or row.get("query_cot"),
                "query": row.get("query_cot") or row.get("query_wo") or row.get("question_for_eval"),
                "choices": row.get("choices"),
                "answer": row.get("answer"),
                "question_type": row.get("question_type"),
                "metadata": row.get("metadata"),
                "prompt_text": mathverse_build_prompt(row),
            })
        return rows

    if benchmark == "screenspot_pro":
        ann_dir, img_dir, snapshot_dir = screenspot_resolve_dirs(dataset_cfg)
        GT_TYPES = ["positive", "negative"]
        INSTRUCTION_STYLES = ["instruction", "action", "description"]
        LANGUAGES = ["en", "cn"]
        task_value = str(dataset_cfg.get("task", "all"))
        task_filenames = sorted(p.stem for p in ann_dir.glob("*.json")) if task_value == "all" else [x.strip() for x in task_value.split(",") if x.strip()]
        inst_styles = screenspot_resolve_choice_arg(str(dataset_cfg.get("inst_style", "instruction")), INSTRUCTION_STYLES)
        gt_types = screenspot_resolve_choice_arg(str(dataset_cfg.get("gt_type", "positive")), GT_TYPES)
        languages = screenspot_resolve_choice_arg(str(dataset_cfg.get("language", "en")), LANGUAGES)
        tasks_to_run: List[Dict[str, Any]] = []
        for task_filename in task_filenames:
            dataset_path = ann_dir / f"{task_filename}.json"
            if not dataset_path.exists():
                raise RuntimeError(f"Missing annotation file: {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            for inst_style in inst_styles:
                for gt_type in gt_types:
                    for lang in languages:
                        for task_instance in task_data:
                            sample = copy.deepcopy(task_instance)
                            sample["task_filename"] = task_filename
                            sample["gt_type"] = gt_type
                            sample["instruction_style"] = inst_style
                            sample["language"] = lang
                            if lang == "cn":
                                if inst_style != "instruction" or gt_type != "positive":
                                    raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                                prompt = sample.get("instruction_cn")
                            else:
                                if inst_style == "instruction":
                                    prompt = sample.get("instruction")
                                else:
                                    prompt = sample.get(inst_style) or sample.get("instruction")
                            if not isinstance(prompt, str) or not prompt.strip():
                                raise RuntimeError(f"Sample in {dataset_path.name} missing prompt text for inst_style={inst_style}, lang={lang}")
                            img_filename = sample.get("img_filename")
                            if not isinstance(img_filename, str) or not img_filename.strip():
                                raise RuntimeError(f"Sample in {dataset_path.name} missing img_filename")
                            img_path = (img_dir / img_filename).resolve()
                            if not img_path.exists():
                                raise RuntimeError(f"ScreenSpot image file not found: {img_path}")
                            sample["img_path"] = str(img_path)
                            sample["prompt_to_evaluate"] = prompt.strip()
                            sample["snapshot_dir"] = snapshot_dir
                            sample["prompt_text"] = screenspot_build_prompt(sample)
                            sample["metadata"] = {
                                "annotation_dir": str(ann_dir),
                                "image_dir": str(img_dir),
                                "snapshot_dir": snapshot_dir,
                            }
                            tasks_to_run.append(sample)
        if int(dataset_cfg.get("max_samples", -1)) > 0:
            tasks_to_run = tasks_to_run[: min(int(dataset_cfg["max_samples"]), len(tasks_to_run))]
        for i, sample in enumerate(tasks_to_run):
            sample["sample_idx"] = i
        return tasks_to_run

    if benchmark == "simplevqa":
        ds = simplevqa_load_named_split(dataset_cfg["dataset_name"], dataset_cfg["split"])
        ds = simplevqa_filter_english_only(ds)
        if int(dataset_cfg.get("max_samples", -1)) > 0:
            ds = ds.select(range(min(int(dataset_cfg["max_samples"]), len(ds))))
        rows = []
        for i in range(len(ds)):
            ex = ds[i]
            rows.append({
                "dataset_index": i,
                "id": simplevqa_get_example_id(ex, i),
                "question": simplevqa_get_question(ex),
                "prompt_text": simplevqa_build_prompt(ex),
                "gold_answer": simplevqa_get_answer(ex),
                "decoded_image": _get_example_image(ex),
            })
        return rows

    if benchmark == "charxiv_reasoning":
        from datasets import Image as HFImage, load_dataset
        ds = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg["split"])
        if int(dataset_cfg.get("max_samples", -1)) > 0:
            ds = ds.select(range(min(int(dataset_cfg["max_samples"]), len(ds))))
        image_col = "decoded_image" if "decoded_image" in ds.column_names else "image"
        if image_col not in ds.column_names:
            raise RuntimeError(f"No image column found. Columns: {ds.column_names}")
        if not isinstance(ds.features[image_col], HFImage):
            ds = ds.cast_column(image_col, HFImage())
        rows = []
        for i in range(len(ds)):
            ex = ds[i]
            rows.append({
                "dataset_index": i,
                "id": str(ex.get("original_id", ex.get("figure_path", f"charxiv_{i:06d}"))),
                "question": charxiv_get_question(ex),
                "prompt_text": charxiv_build_prompt(ex),
                "gold_answer": charxiv_get_answer(ex),
                "decoded_image": load_image_any(ex[image_col]),
                "category": ex.get("category"),
                "year": ex.get("year"),
                "reasoning_q_source": ex.get("reasoning_q_source"),
                "reasoning_a_type": ex.get("reasoning_a_type"),
            })
        return rows

    if benchmark in {"triviaqa", "math", "mmlu_pro"}:
        ds = _textbench_load_dataset_from_cfg(dataset_cfg)
        if int(dataset_cfg.get("max_samples", -1)) > 0:
            ds = ds.select(range(min(int(dataset_cfg["max_samples"]), len(ds))))
        rows = []
        for i in range(len(ds)):
            ex = ds[i]
            ex = {k: ex[k] for k in ds.column_names}
            question = _textbench_build_question_text(ex)
            choices = _textbench_extract_choices(ex)
            gold_raw = _textbench_extract_raw_gold(ex, benchmark)
            gold_answer = _textbench_extract_gold_answer(ex, benchmark)
            if benchmark == "mmlu_pro":
                gold_answer = _textbench_expand_gold_for_mcq(gold_answer, choices)
            example_id = ex.get("id") or ex.get("example_id") or ex.get("question_id") or ex.get("original_id") or ex.get("sample_idx") or i
            rows.append({
                "sample_idx": i,
                "id": str(example_id),
                "question": question,
                "prompt_text": textbench_build_prompt(ex, benchmark),
                "choices": choices,
                "gold_answer": gold_answer,
                "gold_answer_raw": gold_raw,
                "decoded_image": None,
                "metadata": {
                    "original_source": _textbench_normalize_text(next((ex.get(k) for k in ORIGINAL_SOURCE_CANDIDATES if ex.get(k) is not None), "")),
                    "dataset_name": dataset_cfg.get("dataset_name") or dataset_cfg.get("dataset_id"),
                    "dataset_split": dataset_cfg.get("split") or dataset_cfg.get("dataset_split"),
                    "data_path": dataset_cfg.get("data_path"),
                    "data_mode": dataset_cfg.get("data_mode", "hf"),
                },
            })
        return rows

    raise RuntimeError(f"Unknown benchmark: {benchmark}")


def build_generation_row(
    benchmark: str,
    ex: Dict[str, Any],
    strategy_name: str,
    final_model_name: str,
    final_response: str,
    usage_by_model: Dict[str, TokenUsage],
    trace: List[Dict[str, Any]],
    wall_time_sec: float,
) -> Dict[str, Any]:
    benchmark = canonicalize_benchmark_name(benchmark)
    usage_dict = {k: asdict(v) for k, v in usage_by_model.items()}
    if benchmark in {"mathvista", "mathverse"}:
        return {
            "benchmark": benchmark,
            "strategy_name": strategy_name,
            "final_model_name": final_model_name,
            "sample_idx": int(ex["sample_idx"]),
            "prompt_text": ex["prompt_text"],
            "question": ex.get("question"),
            "query": ex.get("query"),
            "choices": ex.get("choices"),
            "gold_answer": ex.get("answer"),
            "raw_response": final_response,
            "boxed_answer": extract_last_boxed(final_response),
            "judge_prompt": None,
            "judge_raw": None,
            "judge_label": None,
            "benchmark_correct": None,
            "usage_by_model": usage_dict,
            "trace": trace,
            "wall_time_sec": wall_time_sec,
        }
    if benchmark in {"triviaqa", "math", "mmlu_pro"}:
        return {
            "benchmark": benchmark,
            "strategy_name": strategy_name,
            "final_model_name": final_model_name,
            "sample_idx": int(ex["sample_idx"]),
            "id": ex.get("id"),
            "question": ex.get("question"),
            "prompt_text": ex["prompt_text"],
            "choices": ex.get("choices"),
            "gold_answer": ex.get("gold_answer"),
            "gold_answer_raw": ex.get("gold_answer_raw"),
            "raw_response": final_response,
            "boxed_answer": extract_last_boxed(final_response),
            "judge_prompt": None,
            "judge_raw": None,
            "judge_label": None,
            "benchmark_correct": None,
            "pred_ans_preview": None,
            "gold_ans_preview": None,
            "pred_parsed": None,
            "label_source": None,
            "metadata": ex.get("metadata"),
            "usage_by_model": usage_dict,
            "trace": trace,
            "wall_time_sec": wall_time_sec,
        }

    if benchmark == "screenspot_pro":
        return {
            "benchmark": benchmark,
            "strategy_name": strategy_name,
            "final_model_name": final_model_name,
            "sample_idx": int(ex["sample_idx"]),
            "id": ex.get("id"),
            "img_filename": ex.get("img_filename"),
            "img_path": ex.get("img_path"),
            "prompt_text": ex["prompt_text"],
            "prompt_to_evaluate": ex.get("prompt_to_evaluate"),
            "gt_type": ex.get("gt_type"),
            "bbox": ex.get("bbox"),
            "img_size": ex.get("img_size"),
            "platform": ex.get("platform"),
            "application": ex.get("application"),
            "group": ex.get("group"),
            "language": ex.get("language"),
            "instruction_style": ex.get("instruction_style"),
            "ui_type": ex.get("ui_type"),
            "task_filename": ex.get("task_filename"),
            "metadata": ex.get("metadata"),
            "raw_response": final_response,
            "parsed_result": None,
            "parsed_point": None,
            "correctness": None,
            "benchmark_correct": None,
            "benchmark_score": None,
            "usage_by_model": usage_dict,
            "trace": trace,
            "wall_time_sec": wall_time_sec,
        }
    if benchmark in {"simplevqa", "charxiv_reasoning"}:
        final_answer, final_source = simplevqa_extract_final_answer(final_response)
        row = {
            "benchmark": benchmark,
            "strategy_name": strategy_name,
            "final_model_name": final_model_name,
            "dataset_index": int(ex["dataset_index"]),
            "id": ex["id"],
            "question": ex["question"],
            "prompt_text": ex["prompt_text"],
            "gold_answer": ex["gold_answer"],
            "raw_response": final_response,
            "boxed_answer": extract_last_boxed(final_response),
            "final_answer": final_answer,
            "final_answer_source": final_source,
            "judge_prompt": None,
            "judge_raw": None,
            "judge_label": None,
            "simplevqa_label": None,
            "benchmark_correct": None,
            "usage_by_model": usage_dict,
            "trace": trace,
            "wall_time_sec": wall_time_sec,
        }
        if benchmark == "charxiv_reasoning":
            row["category"] = ex.get("category")
            row["year"] = ex.get("year")
        return row
    raise RuntimeError(f"Unknown benchmark: {benchmark}")


def prepare_evaluation_row(benchmark: str, row: Dict[str, Any], judge_runtime: Optional[VLLMChatRuntime], judge_sampling: Optional[SamplingConfig]) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
    benchmark = canonicalize_benchmark_name(benchmark)
    raw_response = str(row.get("raw_response") or "")
    boxed_answer = extract_last_boxed(raw_response)

    if benchmark in {"mathvista", "mathverse"}:
        if judge_runtime is None or judge_sampling is None:
            raise RuntimeError(f"{benchmark} evaluation requires a judge runtime")
        candidate_answer = boxed_answer if boxed_answer is not None and str(boxed_answer).strip() else _judge_tail_candidate_text(raw_response, judge_runtime)
        candidate_source = "boxed" if boxed_answer is not None and str(boxed_answer).strip() else f"tail_{EVAL_TAIL_FALLBACK_MAX_TOKENS}_tokens"
        prompt_builder = mathvista_build_judge_prompt if benchmark == "mathvista" else mathverse_build_judge_prompt
        prompt = prompt_builder({
            "question": row.get("question"),
            "query": row.get("query"),
            "choices": row.get("choices"),
            "answer": row.get("gold_answer"),
            "gold_answer": row.get("gold_answer"),
        }, candidate_answer)
        return {
            **row,
            "boxed_answer": boxed_answer,
            "judge_candidate_answer": candidate_answer,
            "judge_candidate_source": candidate_source,
            "boxed_missing_fallback_used": int(candidate_source != "boxed"),
            "boxed_fallback_tail_text": candidate_answer if candidate_source.startswith("tail_") else None,
            "judge_prompt": prompt,
        }, None

    if benchmark in {"triviaqa", "math", "mmlu_pro"}:
        correct, pred_preview, gold_preview, parsed, label_source = _direct_eval_batch(benchmark, [row], [raw_response])[0]
        use_direct = (
            benchmark == "triviaqa"
            and boxed_answer is not None
            and bool(str(boxed_answer).strip())
            and bool(parsed)
            and correct is not None
            and not is_refusal(raw_response)
        )
        if use_direct:
            judge_label = int(float(correct) >= 0.5)
            return {
                **row,
                "boxed_answer": boxed_answer,
                "judge_candidate_answer": pred_preview,
                "judge_candidate_source": "direct_parser",
                "boxed_missing_fallback_used": 0,
                "boxed_fallback_tail_text": None,
                "judge_prompt": None,
                "judge_raw": None,
                "judge_label": judge_label,
                "benchmark_correct": judge_label,
                "pred_ans_preview": pred_preview,
                "gold_ans_preview": gold_preview,
                "pred_parsed": bool(parsed),
                "label_source": label_source,
            }, None

        if judge_runtime is None or judge_sampling is None:
            judge_label = None if correct is None else int(float(correct) >= 0.5)
            return {
                **row,
                "boxed_answer": boxed_answer,
                "judge_candidate_answer": pred_preview,
                "judge_candidate_source": label_source,
                "boxed_missing_fallback_used": int(boxed_answer is None or not str(boxed_answer).strip()),
                "boxed_fallback_tail_text": None,
                "judge_prompt": None,
                "judge_raw": None,
                "judge_label": judge_label,
                "benchmark_correct": judge_label,
                "pred_ans_preview": pred_preview,
                "gold_ans_preview": gold_preview,
                "pred_parsed": bool(parsed),
                "label_source": label_source,
            }, None

        candidate_answer = boxed_answer if boxed_answer is not None and str(boxed_answer).strip() else _judge_tail_candidate_text(raw_response, judge_runtime)
        candidate_source = "boxed" if boxed_answer is not None and str(boxed_answer).strip() else f"tail_{EVAL_TAIL_FALLBACK_MAX_TOKENS}_tokens"
        prompt = _build_textbench_rm_judge_prompt(
            benchmark=benchmark,
            row=row,
            candidate_text=candidate_answer,
            candidate_source=candidate_source,
        )
        return {
            **row,
            "boxed_answer": boxed_answer,
            "judge_candidate_answer": candidate_answer,
            "judge_candidate_source": candidate_source,
            "boxed_missing_fallback_used": int(candidate_source != "boxed"),
            "boxed_fallback_tail_text": candidate_answer if candidate_source.startswith("tail_") else None,
            "judge_prompt": prompt,
            "pred_ans_preview": pred_preview,
            "gold_ans_preview": gold_preview,
            "pred_parsed": bool(parsed),
            "label_source": "rm_tail_fallback" if candidate_source.startswith("tail_") else "rm_boxed_check",
        }, None

    if benchmark == "screenspot_pro":
        extra = screenspot_eval_saved_row(row)
        if extra.get("parsed_result") != "wrong_format":
            return {
                **row,
                **extra,
                "judge_candidate_answer": None,
                "judge_candidate_source": None,
                "boxed_missing_fallback_used": 0,
                "boxed_fallback_tail_text": None,
                "judge_prompt": None,
                "judge_raw": None,
                "judge_label": int(extra.get("benchmark_correct", 0) or 0),
            }, None
        if judge_runtime is None or judge_sampling is None:
            return {**row, **extra, "judge_prompt": None}, None
        candidate_answer = _judge_tail_candidate_text(raw_response, judge_runtime)
        prompt = _build_screenspot_rm_judge_prompt(row, candidate_answer, f"tail_{EVAL_TAIL_FALLBACK_MAX_TOKENS}_tokens")
        image = None
        try:
            image = load_image_any(row.get("img_path"))
        except Exception:
            image = None
        return {
            **row,
            **extra,
            "judge_candidate_answer": candidate_answer,
            "judge_candidate_source": f"tail_{EVAL_TAIL_FALLBACK_MAX_TOKENS}_tokens",
            "boxed_missing_fallback_used": 1,
            "boxed_fallback_tail_text": candidate_answer,
            "judge_prompt": prompt,
        }, image

    if benchmark in {"simplevqa", "charxiv_reasoning"}:
        if judge_runtime is None or judge_sampling is None:
            raise RuntimeError(f"{benchmark} evaluation requires a judge runtime")
        final_answer, final_source = simplevqa_extract_final_answer(raw_response)
        candidate_answer = final_answer if final_answer is not None and str(final_answer).strip() else _judge_tail_candidate_text(raw_response, judge_runtime)
        candidate_source = final_source if final_answer is not None and str(final_answer).strip() else f"tail_{EVAL_TAIL_FALLBACK_MAX_TOKENS}_tokens"
        build_prompt = simplevqa_build_judge_prompt if benchmark == "simplevqa" else charxiv_build_judge_prompt
        prompt = build_prompt(row["question"], row["gold_answer"], candidate_answer if candidate_answer else None)
        return {
            **row,
            "boxed_answer": boxed_answer,
            "final_answer": final_answer,
            "final_answer_source": final_source,
            "judge_candidate_answer": candidate_answer,
            "judge_candidate_source": candidate_source,
            "boxed_missing_fallback_used": int(candidate_source.startswith("tail_")),
            "boxed_fallback_tail_text": candidate_answer if candidate_source.startswith("tail_") else None,
            "judge_prompt": prompt,
        }, None

    raise RuntimeError(f"Unknown benchmark: {benchmark}")

def finish_evaluation_row(benchmark: str, prepared: Dict[str, Any], gen: GenerationResult) -> Dict[str, Any]:
    row = dict(prepared)
    if benchmark in {"simplevqa", "charxiv_reasoning"}:
        row["simplevqa_label"] = simplevqa_parse_judge_label(gen.text)
        score = int(row["simplevqa_label"] == "correct")
    else:
        score = int(mathvista_parse_judge_label(gen.text))
    if benchmark == "screenspot_pro":
        row["correctness"] = "correct_by_rm" if score else "wrong_by_rm"
        row["benchmark_score"] = float(score)
    return {
        **row,
        "judge_raw": gen.text,
        "judge_label": score,
        "benchmark_correct": score,
        "judge_usage": {"prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens},
    }


def evaluate_saved_row(benchmark: str, row: Dict[str, Any], judge_runtime: Optional[VLLMChatRuntime], judge_sampling: Optional[SamplingConfig]) -> Dict[str, Any]:
    benchmark = canonicalize_benchmark_name(benchmark)
    prepared, image = prepare_evaluation_row(benchmark, row, judge_runtime, judge_sampling)
    if prepared.get("judge_prompt") is None:
        return prepared
    gen = judge_runtime.generate(
        messages=[{"role": "user", "content": prepared["judge_prompt"]}],
        image=image, sampling_cfg=judge_sampling,
    )
    return finish_evaluation_row(benchmark, prepared, gen)


def summarize_scored_rows(benchmark: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    benchmark = canonicalize_benchmark_name(benchmark)
    if benchmark in {"mathvista", "mathverse"}:
        total = len(rows)
        num_correct = sum(int(r.get("judge_label", 0)) for r in rows)
        return {"num_rows": total, "num_correct": num_correct, "accuracy": num_correct / max(total, 1)}
    if benchmark in {"triviaqa", "math", "mmlu_pro"}:
        total = len(rows)
        num_correct = sum(int(r.get("judge_label", 0)) for r in rows if r.get("judge_label") is not None)
        num_labeled = sum(1 for r in rows if r.get("judge_label") is not None)
        return {
            "num_rows": total,
            "num_labeled_rows": num_labeled,
            "num_correct": num_correct,
            "accuracy": num_correct / max(num_labeled, 1),
        }
    if benchmark == "screenspot_pro":
        return screenspot_summarize(rows)
    if benchmark in {"simplevqa", "charxiv_reasoning"}:
        return simplevqa_metrics([str(r.get("simplevqa_label", "not_attempted")) for r in rows])
    raise RuntimeError(f"Unknown benchmark: {benchmark}")


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


SELF_REPAIR_TEXT = (
    "Your previous answer may contain an error. Re-check the entire problem carefully, "
    "correct any mistakes, and then give a revised final answer."
)
HANDOFF_TEXT = (
    "Another model produced a draft answer, but it may be wrong. "
    "Please verify the problem independently, fix any mistakes, and provide the best final answer."
)


class MultiAgentOrchestrator:
    def __init__(
        self,
        benchmark: str,
        model_bundles: Dict[str, ModelBundle],
        debug_mode: bool = False,
        debug_max_chars: int = 220,
    ) -> None:
        self.benchmark = benchmark
        self.model_bundles = model_bundles
        self.debug_mode = bool(debug_mode)
        self.debug_max_chars = int(debug_max_chars)
        self._gens: Dict[str, VLLMChatRuntime] = {}
        self._aux: Dict[str, AuxHeadRuntime] = {}

    def get_generator(self, model_name: str) -> VLLMChatRuntime:
        # Strong safety rule: only one vLLM generator may stay resident at a time.
        # Aux heads may remain loaded, but other generator runtimes are unloaded before
        # instantiating or returning the requested generator.
        for other_name in list(self._gens.keys()):
            if other_name != model_name:
                self.unload_model(other_name, unload_generator=True, unload_aux=False, drop_processors=False)
        if model_name not in self._gens:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._gens[model_name] = VLLMChatRuntime(self.model_bundles[model_name].generator_cfg)
        return self._gens[model_name]

    def get_aux(self, model_name: str) -> Optional[AuxHeadRuntime]:
        bundle = self.model_bundles[model_name]
        if not bundle.aux_cfg.enabled or not bundle.aux_cfg.aux_head_ckpt:
            return None
        generator = self.get_generator(model_name)
        if model_name not in self._aux:
            aux = AuxHeadRuntime(bundle.aux_cfg, vllm_runtime=generator)
            self._aux[model_name] = aux
        self._aux[model_name].attach_vllm_runtime(generator)
        return self._aux[model_name]

    def unload_model(self, model_name: str, unload_generator: bool = True, unload_aux: bool = True, drop_processors: bool = False) -> None:
        if unload_generator and model_name in self._gens:
            runtime = self._gens.pop(model_name)
            runtime.unload(drop_processor=drop_processors)
        if unload_aux and model_name in self._aux:
            aux = self._aux.pop(model_name)
            aux.unload(drop_processor=drop_processors)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self, drop_processors: bool = False) -> None:
        for model_name in list(self._gens.keys()):
            self.unload_model(model_name, unload_generator=True, unload_aux=False, drop_processors=drop_processors)
        for model_name in list(self._aux.keys()):
            self.unload_model(model_name, unload_generator=False, unload_aux=True, drop_processors=drop_processors)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_messages_for_turn(self, ex: Dict[str, Any], handoff_payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        base = list(build_initial_messages(self.benchmark, ex))
        if not handoff_payload:
            return base
        mode = str(handoff_payload.get("mode", "handoff_fresh"))
        if mode == "handoff_fresh":
            return base
        if mode == "handoff_with_context":
            return base + [
                {"role": "assistant", "content": f"Draft answer from {handoff_payload.get('from_model', 'previous_model')}:\n\n{handoff_payload.get('draft_response', '')}"},
                {"role": "user", "content": HANDOFF_TEXT},
            ]
        return base

    def _score_messages(self, model_name: str, messages: List[Dict[str, Any]], image: Optional[Image.Image]) -> AuxScore:
        aux = self.get_aux(model_name)
        if aux is None:
            raise RuntimeError(f"No aux head configured for {model_name}")
        return aux.score_messages(messages=messages, image=image)

    def _score_response(self, ex: Dict[str, Any], model_name: str, response_text: str) -> AuxScore:
        image = get_example_image_for_benchmark(ex)
        messages = self._build_messages_for_turn(ex, None) + [{"role": "assistant", "content": response_text}]
        return self._score_messages(model_name, messages, image)

    def _new_usage_by_model(self) -> Dict[str, TokenUsage]:
        return {"model1": TokenUsage(), "model2": TokenUsage()}

    def _example_id(self, ex: Dict[str, Any]) -> Any:
        return ex.get("example_id", ex.get("pid", ex.get("question_id", ex.get("sample_idx", ex.get("dataset_index", "?")))))

    def _make_base_result_tuple(
        self,
        final_model_name: str,
        final_response: str,
        usage_by_model: Dict[str, TokenUsage],
        trace: List[Dict[str, Any]],
        wall_time_sec: float,
    ) -> Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]:
        return final_model_name, final_response, usage_by_model, trace, float(wall_time_sec)

    def _single_agent_batch_generate(self, examples: List[Dict[str, Any]], model_name: str) -> List[Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]]:
        if not examples:
            return []
        t0 = time.time()
        bundle = self.model_bundles[model_name]
        runtime = self.get_generator(model_name)
        messages_list = [self._build_messages_for_turn(ex, None) for ex in examples]
        images = [get_example_image_for_benchmark(ex) for ex in examples]
        gens = runtime.generate_batch(
            messages_list=messages_list,
            images=images,
            sampling_cfg=bundle.sampling_cfg,
            continue_final_messages=[False] * len(examples),
        )
        results = []
        for ex, gen in zip(examples, gens):
            usage_by_model = self._new_usage_by_model()
            usage_by_model[model_name].prompt_tokens += gen.prompt_tokens
            usage_by_model[model_name].completion_tokens += gen.completion_tokens
            usage_by_model[model_name].generation_calls += 1
            usage_by_model[model_name].generation_time_sec += float(gen.generation_time_sec)
            trace = [{"event": "full_generation", "model": model_name, "completion_tokens": gen.completion_tokens, "generation_time_sec": float(gen.generation_time_sec)}]
            results.append(self._make_base_result_tuple(model_name, gen.text, usage_by_model, trace, time.time() - t0))
        return results

    def _build_m1_cache_from_single_results(
        self,
        examples: List[Dict[str, Any]],
        single_results: List[Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]],
    ) -> List[Dict[str, Any]]:
        if len(examples) != len(single_results):
            raise RuntimeError(f"Cached single-agent results length mismatch: {len(examples)} vs {len(single_results)}")
        cache: List[Dict[str, Any]] = []
        for ex, result in zip(examples, single_results):
            final_model_name, final_response, usage_by_model, trace, wall_time_sec = result
            model1_usage = usage_by_model.get("model1", TokenUsage())
            cache.append({
                "example_id": self._example_id(ex),
                "response_text": final_response,
                "prompt_tokens": int(model1_usage.prompt_tokens),
                "completion_tokens": int(model1_usage.completion_tokens),
                "generation_calls": int(model1_usage.generation_calls),
                "generation_time_sec": float(model1_usage.generation_time_sec),
                "trace": list(trace),
                "wall_time_sec": float(wall_time_sec),
                "final_model_name": final_model_name,
            })
        return cache

    def _apply_cached_usage(self, usage_by_model: Dict[str, TokenUsage], cache_item: Dict[str, Any]) -> None:
        usage_by_model["model1"].prompt_tokens += int(cache_item.get("prompt_tokens", 0))
        usage_by_model["model1"].completion_tokens += int(cache_item.get("completion_tokens", 0))
        usage_by_model["model1"].generation_calls += int(cache_item.get("generation_calls", 0))
        usage_by_model["model1"].generation_time_sec += float(cache_item.get("generation_time_sec", 0.0) or 0.0)

    def _run_strategy_from_m1_cache(
        self,
        examples: List[Dict[str, Any]],
        strategy: StrategyConfig,
        cached_m1_first_pass: List[Dict[str, Any]],
    ) -> List[Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]]:
        if len(examples) != len(cached_m1_first_pass):
            raise RuntimeError(f"examples/cache length mismatch: {len(examples)} vs {len(cached_m1_first_pass)}")
        t0 = time.time()
        policy1 = strategy.model_policies.get("model1", AuxPolicy(enabled=False))
        policy2 = strategy.model_policies.get("model2", AuxPolicy(enabled=False))
        if policy1.enabled and str(policy1.trigger_mode) != "after_finish":
            raise RuntimeError(
                f"Strategy {strategy.name} uses trigger_mode={policy1.trigger_mode!r}, which is not supported by the cached shared orchestrator path. "
                "Use the generation driver that materializes routed results explicitly for prefix/chunk handoff strategies."
            )
        if not policy1.enabled:
            # nothing special; just return cached single-agent model1 results
            out = []
            for cache_item in cached_m1_first_pass:
                usage_by_model = self._new_usage_by_model()
                self._apply_cached_usage(usage_by_model, cache_item)
                out.append(self._make_base_result_tuple("model1", cache_item["response_text"], usage_by_model, list(cache_item.get("trace", [])), time.time() - t0))
            return out
        if self.get_aux("model1") is None:
            raise RuntimeError("Strategy requires model1 aux scoring, but model1 aux head is disabled")

        results: List[Optional[Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]]] = [None] * len(examples)
        handoff_indices: List[int] = []
        handoff_payloads: List[Dict[str, Any]] = []
        retry_indices: List[int] = []
        retry_messages: List[List[Dict[str, Any]]] = []
        retry_images: List[Optional[Image.Image]] = []
        repaired_indices: List[int] = []
        repaired_messages: List[List[Dict[str, Any]]] = []
        repaired_images: List[Optional[Image.Image]] = []

        for i, (ex, cache_item) in enumerate(zip(examples, cached_m1_first_pass)):
            candidate = str(cache_item["response_text"])
            usage_by_model = self._new_usage_by_model()
            self._apply_cached_usage(usage_by_model, cache_item)
            trace = list(cache_item.get("trace", []))
            trace.append({"event": "reused_model1_first_pass", "model": "model1", "strategy": strategy.name})
            score = self._score_messages(
                "model1",
                self._build_messages_for_turn(ex, None) + [{"role": "assistant", "content": candidate}],
                get_example_image_for_benchmark(ex),
            )
            usage_by_model["model1"].aux_calls += 1
            usage_by_model["model1"].aux_scored_tokens += int(cache_item.get("completion_tokens", 0))
            trace.append({"event": "aux_score", "model": "model1", "prob_correct": score.prob_correct, "threshold": policy1.threshold, "reused_cached_completion": True})

            if score.prob_correct >= policy1.threshold or policy1.action_below_threshold == "accept":
                results[i] = self._make_base_result_tuple("model1", candidate, usage_by_model, trace, time.time() - t0)
                continue

            if policy1.action_below_threshold == "retry":
                retry_indices.append(i)
                retry_messages.append(self._build_messages_for_turn(ex, None))
                retry_images.append(get_example_image_for_benchmark(ex))
                results[i] = self._make_base_result_tuple("model1", candidate, usage_by_model, trace, time.time() - t0)
                continue

            if policy1.action_below_threshold == "self_repair":
                repaired_indices.append(i)
                repaired_messages.append(self._build_messages_for_turn(ex, None) + [{"role": "assistant", "content": candidate}, {"role": "user", "content": SELF_REPAIR_TEXT}])
                repaired_images.append(get_example_image_for_benchmark(ex))
                results[i] = self._make_base_result_tuple("model1", candidate, usage_by_model, trace, time.time() - t0)
                continue

            if policy1.action_below_threshold in {"handoff_fresh", "handoff_with_context"}:
                handoff_indices.append(i)
                handoff_payloads.append({"from_model": "model1", "mode": policy1.action_below_threshold, "draft_response": candidate})
                results[i] = self._make_base_result_tuple("model1", candidate, usage_by_model, trace + [{"event": "handoff", "from_model": "model1", "to_model": str(policy1.next_model), "mode": policy1.action_below_threshold}], time.time() - t0)
                continue

            raise RuntimeError(f"Unsupported low-score action: {policy1.action_below_threshold}")

        if retry_indices:
            runtime = self.get_generator("model1")
            bundle = self.model_bundles["model1"]
            gens = runtime.generate_batch(
                messages_list=retry_messages,
                images=retry_images,
                sampling_cfg=bundle.sampling_cfg,
                continue_final_messages=[False] * len(retry_messages),
            )
            for idx, gen in zip(retry_indices, gens):
                ex = examples[idx]
                final_model_name, _, usage_by_model, trace, _ = results[idx]
                usage_by_model["model1"].prompt_tokens += gen.prompt_tokens
                usage_by_model["model1"].completion_tokens += gen.completion_tokens
                usage_by_model["model1"].generation_calls += 1
                usage_by_model["model1"].generation_time_sec += float(gen.generation_time_sec)
                trace.append({"event": "retry_generation", "model": "model1", "completion_tokens": gen.completion_tokens, "without_previous_attempt_context": True, "generation_time_sec": float(gen.generation_time_sec)})
                final_response = gen.text
                if self.get_aux("model1") is not None:
                    retry_score = self._score_messages(
                        "model1",
                        retry_messages[retry_indices.index(idx)] + [{"role": "assistant", "content": final_response}],
                        retry_images[retry_indices.index(idx)],
                    )
                    usage_by_model["model1"].aux_calls += 1
                    usage_by_model["model1"].aux_scored_tokens += max(0, gen.completion_tokens)
                    trace.append({"event": "retry_aux_score", "model": "model1", "prob_correct": retry_score.prob_correct, "threshold": policy1.threshold})
                results[idx] = self._make_base_result_tuple(final_model_name, final_response, usage_by_model, trace, time.time() - t0)

        if repaired_indices:
            runtime = self.get_generator("model1")
            bundle = self.model_bundles["model1"]
            gens = runtime.generate_batch(
                messages_list=repaired_messages,
                images=repaired_images,
                sampling_cfg=bundle.sampling_cfg,
                continue_final_messages=[False] * len(repaired_messages),
            )
            for idx, gen in zip(repaired_indices, gens):
                ex = examples[idx]
                final_model_name, _, usage_by_model, trace, _ = results[idx]
                usage_by_model["model1"].prompt_tokens += gen.prompt_tokens
                usage_by_model["model1"].completion_tokens += gen.completion_tokens
                usage_by_model["model1"].generation_calls += 1
                usage_by_model["model1"].generation_time_sec += float(gen.generation_time_sec)
                trace.append({"event": "self_repair_generation", "model": "model1", "completion_tokens": gen.completion_tokens, "generation_time_sec": float(gen.generation_time_sec)})
                final_response = gen.text
                if self.get_aux("model1") is not None:
                    repaired_score = self._score_messages(
                        "model1",
                        repaired_messages[repaired_indices.index(idx)] + [{"role": "assistant", "content": final_response}],
                        repaired_images[repaired_indices.index(idx)],
                    )
                    usage_by_model["model1"].aux_calls += 1
                    usage_by_model["model1"].aux_scored_tokens += max(0, gen.completion_tokens)
                    trace.append({"event": "self_repair_aux_score", "model": "model1", "prob_correct": repaired_score.prob_correct, "threshold": policy1.threshold})
                results[idx] = self._make_base_result_tuple(final_model_name, final_response, usage_by_model, trace, time.time() - t0)

        if handoff_indices:
            # Cached handoff strategies already have model1 generations.
            # Keep model1 aux loaded for scoring across all batches, but free any
            # model1 vLLM runtime before starting model2.
            self.unload_model("model1", unload_generator=True, unload_aux=False)
            runtime2 = self.get_generator("model2")
            bundle2 = self.model_bundles["model2"]
            handoff_messages = [self._build_messages_for_turn(examples[idx], payload) for idx, payload in zip(handoff_indices, handoff_payloads)]
            handoff_images = [get_example_image_for_benchmark(examples[idx]) for idx in handoff_indices]
            gens2 = runtime2.generate_batch(
                messages_list=handoff_messages,
                images=handoff_images,
                sampling_cfg=bundle2.sampling_cfg,
                continue_final_messages=[False] * len(handoff_messages),
            )

            model2_repair_indices: List[int] = []
            model2_repair_messages: List[List[Dict[str, Any]]] = []
            model2_repair_images: List[Optional[Image.Image]] = []
            for idx, gen2 in zip(handoff_indices, gens2):
                ex = examples[idx]
                _, _, usage_by_model, trace, _ = results[idx]
                usage_by_model["model2"].prompt_tokens += gen2.prompt_tokens
                usage_by_model["model2"].completion_tokens += gen2.completion_tokens
                usage_by_model["model2"].generation_calls += 1
                usage_by_model["model2"].generation_time_sec += float(gen2.generation_time_sec)
                trace.append({"event": "full_generation", "model": "model2", "completion_tokens": gen2.completion_tokens, "generation_time_sec": float(gen2.generation_time_sec)})
                final_response = gen2.text
                final_model_name = "model2"
                if policy2.enabled and self.get_aux("model2") is not None:
                    score2 = self._score_messages(
                        "model2",
                        handoff_messages[handoff_indices.index(idx)] + [{"role": "assistant", "content": final_response}],
                        handoff_images[handoff_indices.index(idx)],
                    )
                    usage_by_model["model2"].aux_calls += 1
                    usage_by_model["model2"].aux_scored_tokens += max(0, gen2.completion_tokens)
                    trace.append({"event": "aux_score", "model": "model2", "prob_correct": score2.prob_correct, "threshold": policy2.threshold})
                    if score2.prob_correct < policy2.threshold and policy2.action_below_threshold == "self_repair" and policy2.max_self_repairs > 0:
                        model2_repair_indices.append(idx)
                        model2_repair_messages.append(self._build_messages_for_turn(ex, handoff_payloads[handoff_indices.index(idx)]) + [{"role": "assistant", "content": final_response}, {"role": "user", "content": SELF_REPAIR_TEXT}])
                        model2_repair_images.append(get_example_image_for_benchmark(ex))
                results[idx] = self._make_base_result_tuple(final_model_name, final_response, usage_by_model, trace, time.time() - t0)

            if model2_repair_indices:
                gens2r = runtime2.generate_batch(
                    messages_list=model2_repair_messages,
                    images=model2_repair_images,
                    sampling_cfg=bundle2.sampling_cfg,
                    continue_final_messages=[False] * len(model2_repair_messages),
                )
                for idx, gen2r in zip(model2_repair_indices, gens2r):
                    ex = examples[idx]
                    final_model_name, _, usage_by_model, trace, _ = results[idx]
                    usage_by_model["model2"].prompt_tokens += gen2r.prompt_tokens
                    usage_by_model["model2"].completion_tokens += gen2r.completion_tokens
                    usage_by_model["model2"].generation_calls += 1
                    usage_by_model["model2"].generation_time_sec += float(gen2r.generation_time_sec)
                    trace.append({"event": "self_repair_generation", "model": "model2", "completion_tokens": gen2r.completion_tokens, "generation_time_sec": float(gen2r.generation_time_sec)})
                    final_response = gen2r.text
                    if self.get_aux("model2") is not None:
                        repaired_score2 = self._score_messages(
                            "model2",
                            model2_repair_messages[model2_repair_indices.index(idx)] + [{"role": "assistant", "content": final_response}],
                            model2_repair_images[model2_repair_indices.index(idx)],
                        )
                        usage_by_model["model2"].aux_calls += 1
                        usage_by_model["model2"].aux_scored_tokens += max(0, gen2r.completion_tokens)
                        trace.append({"event": "self_repair_aux_score", "model": "model2", "prob_correct": repaired_score2.prob_correct, "threshold": policy2.threshold})
                    results[idx] = self._make_base_result_tuple(final_model_name, final_response, usage_by_model, trace, time.time() - t0)

        return [r for r in results if r is not None]

    def run_examples_batched(
        self,
        examples: List[Dict[str, Any]],
        strategy: StrategyConfig,
        batch_size: Optional[int] = None,
        cached_model1_single_results: Optional[List[Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]]] = None,
    ) -> List[Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]]:
        if not examples:
            return []
        if strategy.name == "single_agent_model1":
            return self._single_agent_batch_generate(examples, "model1")
        if strategy.name == "single_agent_model2":
            return self._single_agent_batch_generate(examples, "model2")
        if strategy.entry_model == "model1":
            if cached_model1_single_results is None:
                cached_model1_single_results = self._single_agent_batch_generate(examples, "model1")
            cached = self._build_m1_cache_from_single_results(examples, cached_model1_single_results)
            return self._run_strategy_from_m1_cache(examples, strategy, cached)
        return [self.run_example(ex, strategy) for ex in examples]

    def run_example(self, ex: Dict[str, Any], strategy: StrategyConfig) -> Tuple[str, str, Dict[str, TokenUsage], List[Dict[str, Any]], float]:
        # Keep single-example path for compatibility and fallback.
        return self.run_examples_batched([ex], strategy, batch_size=1)[0]

# -----------------------------------------------------------------------------
# Convenience builders
# -----------------------------------------------------------------------------


def build_model_bundle(
    *,
    model_name_or_path: str,
    aux_head_ckpt: str,
    runtime_profile: Dict[str, Any],
    sampling_profiles: Mapping[str, Dict[str, Any]],
    aux_profile: Dict[str, Any],
    sampling_override: Optional[Dict[str, Any]] = None,
    model_family: str = "auto",
    thinking_mode: Any = "auto",
) -> ModelBundle:
    base_sampling = auto_sampling_from_model_name(
        model_name_or_path,
        sampling_profiles,
        model_family=model_family,
        thinking_mode=thinking_mode,
    )
    if sampling_override:
        merged = {
            "greedy": base_sampling.greedy,
            "temperature": base_sampling.temperature,
            "top_p": base_sampling.top_p,
            "top_k": base_sampling.top_k,
            "repetition_penalty": base_sampling.repetition_penalty,
            "presence_penalty": base_sampling.presence_penalty,
            "max_new_tokens": base_sampling.max_new_tokens,
        }
        merged.update(dict(sampling_override))
        sampling_cfg = SamplingConfig(
            greedy=bool(merged.get("greedy", False)),
            temperature=float(merged.get("temperature", 0.7)),
            top_p=float(merged.get("top_p", 0.8)),
            top_k=int(merged.get("top_k", -1)),
            repetition_penalty=float(merged.get("repetition_penalty", 1.0)),
            presence_penalty=float(merged.get("presence_penalty", 0.0)),
            max_new_tokens=int(merged.get("max_new_tokens", merged.get("out_seq_length", 15000))),
        )
    else:
        sampling_cfg = base_sampling
    aux_runtime_cfg = AuxHeadRuntimeConfig(
        enabled=bool(aux_head_ckpt),
        model_name_or_path=model_name_or_path,
        aux_head_ckpt=str(aux_head_ckpt or ""),
        revision=runtime_profile.get("revision"),
        trust_remote_code=bool(aux_profile["trust_remote_code"]),
        dtype=str(aux_profile["dtype"]),
        max_seq_len=int(aux_profile["max_seq_len"]),
        max_pixels=int(aux_profile["max_pixels"]),
        regression_threshold=float(aux_profile["regression_threshold"]),
        head_input_mode=aux_profile.get("head_input_mode"),
        hidden_layer_selection=aux_profile.get("hidden_layer_selection"),
        hidden_layer_index=aux_profile.get("hidden_layer_index"),
        hidden_layer_indices=aux_profile.get("hidden_layer_indices"),
        model_family=str(aux_profile.get("model_family", model_family)),
        thinking_mode=aux_profile.get("thinking_mode", thinking_mode),
        hidden_states_storage_path=aux_profile.get("hidden_states_storage_path"),
    )
    extraction_layer_ids = None
    if aux_runtime_cfg.enabled:
        extraction_layer_ids = AuxHeadRuntime(
            aux_runtime_cfg
        ).required_hidden_state_layer_ids()
    return ModelBundle(
        name=model_name_or_path,
        generator_cfg=VLLMRuntimeConfig(
            model_name_or_path=model_name_or_path,
            revision=runtime_profile.get("revision"),
            dtype=str(runtime_profile["dtype"]),
            max_model_len=int(runtime_profile["max_model_len"]),
            tensor_parallel_size=int(runtime_profile["tensor_parallel_size"]),
            gpu_memory_utilization=float(runtime_profile["gpu_memory_utilization"]),
            max_num_seqs=int(runtime_profile["max_num_seqs"]),
            enforce_eager=bool(runtime_profile["enforce_eager"]),
            trust_remote_code=bool(runtime_profile["trust_remote_code"]),
            limit_mm_images=int(runtime_profile["limit_mm_images"]),
            model_family=str(runtime_profile.get("model_family", model_family)),
            thinking_mode=runtime_profile.get("thinking_mode", thinking_mode),
            enable_hidden_state_extraction=bool(extraction_layer_ids),
            hidden_state_layer_ids=extraction_layer_ids,
            hidden_states_storage_path=aux_profile.get("hidden_states_storage_path"),
        ),
        sampling_cfg=sampling_cfg,
        aux_cfg=aux_runtime_cfg,
    )


def build_judge_runtime_and_sampling(
    judge_model_name_or_path: str,
    judge_runtime_profile: Dict[str, Any],
    judge_sampling_profiles: Mapping[str, Dict[str, Any]],
    judge_model_family: str = "auto",
    judge_thinking_mode: Any = "auto",
) -> Tuple[VLLMChatRuntime, SamplingConfig]:
    runtime = VLLMChatRuntime(VLLMRuntimeConfig(
        model_name_or_path=judge_model_name_or_path,
        dtype=str(judge_runtime_profile["dtype"]),
        max_model_len=int(judge_runtime_profile["max_model_len"]),
        tensor_parallel_size=int(judge_runtime_profile["tensor_parallel_size"]),
        gpu_memory_utilization=float(judge_runtime_profile["gpu_memory_utilization"]),
        max_num_seqs=int(judge_runtime_profile["max_num_seqs"]),
        enforce_eager=bool(judge_runtime_profile["enforce_eager"]),
        trust_remote_code=bool(judge_runtime_profile["trust_remote_code"]),
        limit_mm_images=0,
        model_family=str(judge_runtime_profile.get("model_family", judge_model_family)),
        thinking_mode=judge_runtime_profile.get("thinking_mode", judge_thinking_mode),
    ))
    sampling = auto_sampling_from_model_name(
        judge_model_name_or_path,
        judge_sampling_profiles,
        model_family=judge_model_family,
        thinking_mode=judge_thinking_mode,
    )
    return runtime, sampling
