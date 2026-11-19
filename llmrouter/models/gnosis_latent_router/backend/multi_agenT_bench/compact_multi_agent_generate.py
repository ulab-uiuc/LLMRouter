#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
import re
from tqdm.auto import tqdm

from llmrouter.models.gnosis_latent_router.backend.multi_agenT_bench.run_multi_agent_generate_then_eval import (
    _compute_results_root,
    _format_threshold_for_path,
    _parse_threshold_list,
    _runtime_mode_tag,
    _sanitize_name_for_path,
)

from llmrouter.models.gnosis_latent_router.backend.multi_agenT_bench.compact_multi_agent_shared_optimized_v4_textbench import (
    MultiAgentOrchestrator,
    SELF_REPAIR_TEXT,
    benchmark_needs_judge,
    build_default_two_model_suite,
    build_generation_row,
    build_judge_runtime_and_sampling,
    evaluate_saved_row,
    build_model_bundle,
    debug_print,
    filter_strategies,
    get_example_image_for_benchmark,
    json_dump,
    load_examples_for_benchmark,
    response_final_answer_status,
    set_seed,
    write_jsonl,
    load_jsonl,
    TokenUsage
)

from llmrouter.models.gnosis_latent_router.backend.aux_eval_metrics_utils import (
    compute_all_binary_metrics,
    metrics_report_text,
    save_probability_distribution_plots,
    save_threshold_sweep_json,
)

# =============================================================================
# HARD-CODED CONFIG BLOCK
# =============================================================================

BENCHMARK = "triviaqa"
CURRENT_BENCHMARK = BENCHMARK
STRATEGY_NAMES = (
    "single_agent_model1,"
    "single_agent_model2,"
    # "m1_after_finish_self_repair,"
    "m1_after_finish_retry,"
    "m1_after_finish_handoff_fresh_m2,"
    # "m1_after_finish_handoff_context_m2,"
    # "m1_after_1000tok_handoff_context_m2"
)

SEED = 42
OVERWRITE = False
RESUME_MODE = True
DEBUG_MODE = False
DEBUG_MAX_CHARS = 2000

# Sweep THESE thresholds for model1 aux.
MODEL1_AUX_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]

# Kept fixed here. Only relevant if you later enable a strategy that uses model2 aux.
MODEL2_AUX_THRESHOLD = 0.80

CHUNK_TOKENS = 128
PREFIX_HANDOFF_TOKENS = 1000

MODEL1_NAME_OR_PATH = "Qwen/Qwen3-VL-4B-Instruct"
MODEL2_NAME_OR_PATH = "Qwen/Qwen3-VL-32B-Thinking-FP8"

MODEL1_MODEL_FAMILY = "auto"
MODEL1_THINKING_MODE = "auto"
MODEL2_MODEL_FAMILY = "auto"
MODEL2_THINKING_MODE = "auto"

MODEL1_AUX_HEAD_CKPT = "trained_models/Qwen3VL-4B_instruct_120K/aux_head_final.pt"
MODEL2_AUX_HEAD_CKPT = ""

AUX_EVAL_JUDGE_MODEL_NAME_OR_PATH = "Qwen/Qwen3-VL-8B-Instruct"
AUX_EVAL_JUDGE_MODEL_FAMILY = "auto"
AUX_EVAL_JUDGE_THINKING_MODE = "auto"
AUX_EVAL_JUDGE_RUNTIME_PROFILE = {
    "dtype": "bfloat16",
    "max_model_len": 32000,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.60,
    "max_num_seqs": 128,
    "enforce_eager": False,
    "trust_remote_code": False,
}
AUX_EVAL_JUDGE_SAMPLING_PROFILES = {
    "default": {
        "greedy": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 16000,
    },
    "thinking": {
        "greedy": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 16000,
    },
    "instruct": {
        "greedy": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 16000,
    },
}

OUTPUT_BASE_ROOT = "eval_outputs/multi_agent_compact"
SHARED_SINGLE_AGENT_CACHE_DIRNAME = "_shared_single_model_cache"
SHARED_SINGLE_AGENT_CACHE_ENABLED = True

DATASET_CONFIGS = {
    "mathvista": {
        "dataset_name": "AI4Math/MathVista",
        "split": "testmini",
        "max_samples": -1,
    },
    "mathverse": {
        "dataset_name": "AI4Math/MathVerse",
        "dataset_config_name": "testmini",
        "split": "testmini",
        "max_samples": 1000,
    },
    "charxiv_reasoning": {
        "dataset_name": "princeton-nlp/CharXiv",
        "split": "validation",
        "max_samples": 1000,
    },
    "screenspot_pro": {
        "dataset_repo_id": "likaixin/ScreenSpot-Pro",
        "screenspot_root": "",
        "screenspot_test": "",
        "screenspot_imgs": "",
        "task": "all",
        "inst_style": "instruction",
        "language": "en",
        "gt_type": "positive",
        "max_samples": -1,
    },
    "simplevqa": {
        "dataset_name": "m-a-p/SimpleVQA",
        "split": "test",
        "max_samples": -1,
    },
    "triviaqa": {
        "data_mode": "hf",
        "dataset_name": "mandarjoshi/trivia_qa",
        "dataset_config_name": "rc",
        "split": "validation",
        "max_samples": 1000,
    },
    "math": {
        "data_mode": "csv",
        "data_path": "data/benchmarks/merged_math.csv",
        "max_samples": 1000,
    },
    "mmlu_pro": {
        "data_mode": "csv",
        "data_path": "data/benchmarks/test.csv",
        "max_samples": 1000,
    },
}

SAMPLING_PROFILES = {
    "default": {
        "greedy": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 15000,
    },
    "thinking": {
        "greedy": False,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "max_new_tokens": 16000,
    },
    "instruct": {
        "greedy": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 15000,
    },
}

MODEL_SAMPLING_OVERRIDES = {
    "model1": {},
    "model2": {},
}

AUX_PROFILE = {
    "trust_remote_code": True,
    "dtype": "bf16",
    "max_seq_len": 32768,
    "max_pixels": 200000,
    "regression_threshold": 0.6,
    # The checkpoint is authoritative for token masking and layer selection.
    "head_input_mode": None,
    "hidden_layer_selection": None,
    "hidden_layer_index": None,
    "hidden_layer_indices": None,
}

MODEL1_VLLM_BASE = {
    "dtype": "bfloat16",
    "max_model_len": 32768,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.70,
    "max_num_seqs": 128,
    "enforce_eager": False,
    "trust_remote_code": False,
    "limit_mm_images": 1,
}

MODEL2_VLLM_BASE = {
    "dtype": "bfloat16",
    "max_model_len": 32768,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 32,
    "enforce_eager": False,
    "trust_remote_code": False,
    "limit_mm_images": 1,
}

PRECOMPUTE_MODEL1_BATCH_SIZE = 128
PRECOMPUTE_MODEL1_GPU_UTIL = 0.70
PRECOMPUTE_MODEL1_MAX_NUM_SEQS = 128

STRATEGY_EXECUTION = {
    "single_agent_model1": {
        "batch_size": 128,
        "need_model1_vllm": True,
        "need_model2_vllm": False,
        "need_model1_aux": False,
        "need_model2_aux": False,
        "model1_gpu_memory_utilization": 0.70,
        "model1_max_num_seqs": 128,
        "model2_gpu_memory_utilization": None,
        "model2_max_num_seqs": None,
    },
    "single_agent_model2": {
        "batch_size": 64,
        "need_model1_vllm": False,
        "need_model2_vllm": True,
        "need_model1_aux": False,
        "need_model2_aux": False,
        "model1_gpu_memory_utilization": None,
        "model1_max_num_seqs": None,
        "model2_gpu_memory_utilization": 0.80,
        "model2_max_num_seqs": 64,
    },
    "m1_after_finish_self_repair": {
        "batch_size": 32,
        "need_model1_vllm": True,
        "need_model2_vllm": False,
        "need_model1_aux": True,
        "need_model2_aux": False,
        "model1_gpu_memory_utilization": 0.60,
        "model1_max_num_seqs": 32,
        "model2_gpu_memory_utilization": None,
        "model2_max_num_seqs": None,
    },
    "m1_after_finish_retry": {
        "batch_size": 32,
        "need_model1_vllm": True,
        "need_model2_vllm": False,
        "need_model1_aux": True,
        "need_model2_aux": False,
        "model1_gpu_memory_utilization": 0.60,
        "model1_max_num_seqs": 32,
        "model2_gpu_memory_utilization": None,
        "model2_max_num_seqs": None,
    },
    "m1_after_finish_handoff_fresh_m2": {
        "batch_size": 16,
        "need_model1_vllm": False,
        "need_model2_vllm": True,
        "need_model1_aux": True,
        "need_model2_aux": False,
        "model1_gpu_memory_utilization": None,
        "model1_max_num_seqs": None,
        "model2_gpu_memory_utilization": 0.70,
        "model2_max_num_seqs": 16,
    },
    "m1_after_finish_handoff_context_m2": {
        "batch_size": 16,
        "need_model1_vllm": False,
        "need_model2_vllm": True,
        "need_model1_aux": True,
        "need_model2_aux": False,
        "model1_gpu_memory_utilization": None,
        "model1_max_num_seqs": None,
        "model2_gpu_memory_utilization": 0.70,
        "model2_max_num_seqs": 16,
    },
    "m1_after_1000tok_handoff_context_m2": {
        "batch_size": 16,
        "need_model1_vllm": False,
        "need_model2_vllm": True,
        "need_model1_aux": True,
        "need_model2_aux": False,
        "model1_gpu_memory_utilization": None,
        "model1_max_num_seqs": None,
        "model2_gpu_memory_utilization": 0.80,
        "model2_max_num_seqs": 16,
    },
}

EXAMPLE_COMMANDS = """
python compact_multi_agent_generate.py --benchmark mathvista
python compact_multi_agent_generate.py --benchmark mathverse
python compact_multi_agent_generate.py --benchmark charxiv_reasoning
python compact_multi_agent_generate.py --benchmark screenspot_pro
python compact_multi_agent_generate.py --benchmark simplevqa
python compact_multi_agent_generate.py --benchmark triviaqa
python compact_multi_agent_generate.py --benchmark math
python compact_multi_agent_generate.py --benchmark mmlu_pro
python compact_multi_agent_generate.py --benchmark mathvista --debug --overwrite
"""

# =============================================================================
# HELPERS
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=str, default=BENCHMARK, choices=["mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa", "triviaqa", "math", "mmlu_pro"])
    ap.add_argument("--model1_name_or_path", default=MODEL1_NAME_OR_PATH)
    ap.add_argument("--model2_name_or_path", default=MODEL2_NAME_OR_PATH)
    ap.add_argument("--model1_revision")
    ap.add_argument("--model1_aux_head_ckpt", default=MODEL1_AUX_HEAD_CKPT)
    ap.add_argument("--model2_aux_threshold", type=float, default=MODEL2_AUX_THRESHOLD)
    ap.add_argument("--output_base_root", default=OUTPUT_BASE_ROOT)
    ap.add_argument("--judge_model_name_or_path", default=AUX_EVAL_JUDGE_MODEL_NAME_OR_PATH)
    ap.add_argument("--judge_model_family", default=AUX_EVAL_JUDGE_MODEL_FAMILY)
    ap.add_argument("--judge_thinking_mode", default=AUX_EVAL_JUDGE_THINKING_MODE)
    ap.add_argument("--strategy_names", type=str, default=STRATEGY_NAMES)
    ap.add_argument("--model1_model_family", type=str, default=MODEL1_MODEL_FAMILY, choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--model1_thinking_mode", type=str, default=MODEL1_THINKING_MODE, choices=["auto", "on", "off"])
    ap.add_argument("--model2_model_family", type=str, default=MODEL2_MODEL_FAMILY, choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--model2_thinking_mode", type=str, default=MODEL2_THINKING_MODE, choices=["auto", "on", "off"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional deterministic benchmark prefix for smoke tests.",
    )
    ap.add_argument("--no_shared_single_agent_cache", action="store_true")
    ap.add_argument(
        "--model1_aux_thresholds",
        type=str,
        default=",".join(f"{x:.4f}".rstrip("0").rstrip(".") for x in MODEL1_AUX_THRESHOLDS),
        help="Comma-separated model1 aux thresholds, e.g. 0.5,0.7,0.8",
    )
    return ap.parse_args()


def _benchmark_is_multimodal(benchmark: str) -> bool:
    return str(benchmark or "").strip().lower() in {
        "mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa"
    }


def _benchmark_is_reasoning_text(benchmark: str) -> bool:
    return str(benchmark or "").strip().lower() in {"math", "mmlu_pro"}


def _resolve_thinking_enabled_local(model_name_or_path: str, model_family: str, thinking_mode: str) -> bool:
    mode = str(thinking_mode or "auto").strip().lower()
    if mode in {"on", "true", "1", "yes"}:
        return True
    if mode in {"off", "false", "0", "no"}:
        return False
    lname = str(model_name_or_path or "").lower()
    family = str(model_family or "auto").lower()
    if family == "qwen3_5":
        if any(x in lname for x in ["qwen3.5-0.8b", "qwen3.5-2b"]):
            return False
        return True
    if family == "qwen3_vl":
        return "thinking" in lname
    if family == "qwen3":
        return "instruct" not in lname or "thinking" in lname
    if family == "gemma4":
        return False
    return "thinking" in lname


def _official_sampling_override_for_model(*, model_name_or_path: str, model_family: str, thinking_mode: str, benchmark: str) -> Dict[str, Any]:
    family = str(model_family or "auto").strip().lower()
    if family == "auto":
        lname = str(model_name_or_path or "").lower()
        if "qwen3.5" in lname:
            family = "qwen3_5"
        elif "qwen3-vl" in lname:
            family = "qwen3_vl"
        elif "gemma-4" in lname:
            family = "gemma4"
        elif "qwen3" in lname:
            family = "qwen3"
        else:
            family = "other"

    thinking_enabled = _resolve_thinking_enabled_local(model_name_or_path, family, thinking_mode)
    multimodal = _benchmark_is_multimodal(benchmark)
    text_reasoning = _benchmark_is_reasoning_text(benchmark)

    if family == "gemma4":
        return {
            "greedy": False,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "max_new_tokens": 16000,
        }

    if family == "qwen3_5":
        if multimodal:
            if thinking_enabled:
                return {
                    "greedy": False,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 1.5,
                    "max_new_tokens": 16000,
                }
            return {
                "greedy": False,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.5,
                "max_new_tokens": 16000,
            }
        if thinking_enabled:
            return {
                "greedy": False,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.5,
                "max_new_tokens": 16000,
            }
        if text_reasoning:
            return {
                "greedy": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 40,
                "repetition_penalty": 1.0,
                "presence_penalty": 2.0,
                "max_new_tokens": 16000,
            }
        return {
            "greedy": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "max_new_tokens": 16000,
        }

    if family in {"qwen3_vl", "qwen3"}:
        if thinking_enabled:
            return {
                "greedy": False,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "repetition_penalty": 1.0,
                "presence_penalty": 1.5,
                "max_new_tokens": 16000,
            }
        return {
            "greedy": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "max_new_tokens": 16000,
        }

    return {
        "greedy": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "max_new_tokens": 16000,
    }


def _shared_single_agent_strategy_name_for_slot(slot: str) -> str:
    if slot == "model1":
        return "single_agent_model1"
    if slot == "model2":
        return "single_agent_model2"
    raise RuntimeError(f"Unsupported slot for shared single-agent cache: {slot}")


def _normalize_example_id(x: Any) -> str:
    if x is None:
        return "__none__"
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    try:
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(x)


def _example_id_from_example(ex: Dict[str, Any]) -> str:
    return _normalize_example_id(ex.get("example_id", ex.get("pid", ex.get("question_id", ex.get("id", ex.get("sample_idx", ex.get("dataset_index", "?")))))))


def _example_id_from_row(row: Dict[str, Any]) -> str:
    return _normalize_example_id(row.get("example_id", row.get("pid", row.get("question_id", row.get("id", row.get("sample_idx", row.get("dataset_index", "?")))))))


def _sampling_cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    return {
        "greedy": bool(cfg.greedy),
        "temperature": float(cfg.temperature),
        "top_p": float(cfg.top_p),
        "top_k": int(cfg.top_k),
        "repetition_penalty": float(cfg.repetition_penalty),
        "presence_penalty": float(cfg.presence_penalty),
        "max_new_tokens": int(cfg.max_new_tokens),
    }


def _shared_single_agent_cache_meta(benchmark: str, slot: str, model_name_or_path: str, model_family: str, thinking_mode: str, sampling_cfg: Any) -> Dict[str, Any]:
    strategy_name = _shared_single_agent_strategy_name_for_slot(slot)
    return {
        "cache_version": 1,
        "revision": MODEL1_VLLM_BASE.get("revision") if slot == "model1" else None,
        "benchmark": str(benchmark),
        "slot": str(slot),
        "strategy_name": strategy_name,
        "model_name_or_path": str(model_name_or_path),
        "model_family": str(model_family),
        "thinking_mode": str(thinking_mode),
        "sampling_cfg": _sampling_cfg_to_dict(sampling_cfg),
        "dataset_config": copy.deepcopy(DATASET_CONFIGS.get(str(benchmark), {})),
    }


def _shared_single_agent_cache_dir(benchmark: str, slot: str, model_name_or_path: str, model_family: str, thinking_mode: str, sampling_cfg: Any) -> Path:
    strategy_name = _shared_single_agent_strategy_name_for_slot(slot)
    model_tag = _sanitize_name_for_path(model_name_or_path)
    mode_tag = _runtime_mode_tag(model_family, thinking_mode)
    meta = _shared_single_agent_cache_meta(benchmark, slot, model_name_or_path, model_family, thinking_mode, sampling_cfg)
    meta_blob = json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(meta_blob.encode("utf-8")).hexdigest()[:16]
    return Path(OUTPUT_BASE_ROOT) / SHARED_SINGLE_AGENT_CACHE_DIRNAME / f"{benchmark}_split" / strategy_name / f"{model_tag}__{mode_tag}__{digest}"


def _result_tuple_to_row(benchmark: str, ex: Dict[str, Any], strategy_name: str, result: Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]) -> Dict[str, Any]:
    return build_rows_from_results(benchmark, [ex], strategy_name, [result])[0]


def _rows_to_result_tuples(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = []
    for row in rows:
        usage_raw = row.get("usage_by_model", {}) or {}
        usage_by_model: Dict[str, Any] = {}
        for model_key in ["model1", "model2"]:
            payload = usage_raw.get(model_key, {}) or {}
            usage_by_model[model_key] = TokenUsage(
                prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
                completion_tokens=int(payload.get("completion_tokens", 0) or 0),
                aux_scored_tokens=int(payload.get("aux_scored_tokens", 0) or 0),
                aux_calls=int(payload.get("aux_calls", 0) or 0),
                generation_calls=int(payload.get("generation_calls", 0) or 0),
            )
        trace = row.get("trace", [])
        if not isinstance(trace, list):
            trace = []
        results.append((
            str(row.get("final_model_name", "")),
            str(row.get("raw_response", "")),
            usage_by_model,
            trace,
            float(row.get("wall_time_sec", 0.0) or 0.0),
        ))
    return results


def try_load_shared_single_agent_cache(
    benchmark: str,
    examples: List[Dict[str, Any]],
    slot: str,
    model_name_or_path: str,
    model_family: str,
    thinking_mode: str,
    sampling_cfg: Any,
    debug: bool,
) -> Optional[List[Dict[str, Any]]]:
    cache_dir = _shared_single_agent_cache_dir(
        benchmark, slot, model_name_or_path, model_family, thinking_mode, sampling_cfg
    )
    results_jsonl = cache_dir / "results.jsonl"
    summary_json = cache_dir / "generation_summary.json"
    meta_json = cache_dir / "cache_meta.json"

    debug_print(debug, f"[CACHE] lookup slot={slot} benchmark={benchmark}")
    debug_print(debug, f"[CACHE] expected cache_dir={cache_dir}")

    missing = [str(p.name) for p in (results_jsonl, summary_json, meta_json) if not p.exists()]
    if missing:
        debug_print(
            debug,
            f"[CACHE] cache miss: missing files in {cache_dir}: {missing}",
        )
        return None

    try:
        rows = load_jsonl(results_jsonl)
        summary = _read_json(summary_json)
        meta = _read_json(meta_json)

        if not isinstance(summary, dict):
            debug_print(debug, f"[CACHE] invalid summary json type: {type(summary)}")
            return None
        if not isinstance(meta, dict):
            debug_print(debug, f"[CACHE] invalid meta json type: {type(meta)}")
            return None

        expected_num = len(examples)
        summary_num = int(summary.get("num_rows", -1))
        rows_num = len(rows)
        meta_num = int(meta.get("num_rows", -1)) if "num_rows" in meta else None

        debug_print(
            debug,
            f"[CACHE] counts expected={expected_num} summary_num={summary_num} rows_num={rows_num} meta_num={meta_num}"
        )

        if summary_num != expected_num or rows_num != expected_num:
            debug_print(debug, f"[CACHE] shared single-agent cache row mismatch -> ignoring: {cache_dir}")
            return None

        expected_ids = [_example_id_from_example(ex) for ex in examples]
        row_ids = [_example_id_from_row(r) for r in rows]
        meta_ids = meta.get("example_ids", None)

        if isinstance(meta_ids, list):
            debug_print(
                debug,
                f"[CACHE] meta example_ids present: {len(meta_ids)} entries"
            )
            if len(meta_ids) != expected_num:
                debug_print(
                    debug,
                    f"[CACHE] meta example_ids length mismatch: expected={expected_num} meta={len(meta_ids)}"
                )

        if expected_ids != row_ids:
            first_bad = None
            max_check = min(len(expected_ids), len(row_ids))
            for i in range(max_check):
                if expected_ids[i] != row_ids[i]:
                    first_bad = i
                    break
            if first_bad is None and len(expected_ids) != len(row_ids):
                first_bad = max_check

            debug_print(debug, f"[CACHE] shared single-agent cache example-id mismatch -> ignoring: {cache_dir}")
            debug_print(debug, f"[CACHE] expected_ids_len={len(expected_ids)} row_ids_len={len(row_ids)}")

            if first_bad is not None:
                lo = max(0, first_bad - 3)
                hi = min(max(len(expected_ids), len(row_ids)), first_bad + 3)
                debug_print(debug, f"[CACHE] first mismatch index={first_bad}")
                for j in range(lo, hi):
                    exp_j = expected_ids[j] if j < len(expected_ids) else "<missing>"
                    row_j = row_ids[j] if j < len(row_ids) else "<missing>"
                    debug_print(debug, f"[CACHE] idx={j} expected={exp_j!r} row={row_j!r}")

            expected_set = set(expected_ids)
            row_set = set(row_ids)
            missing_from_rows = list(expected_set - row_set)[:10]
            extra_in_rows = list(row_set - expected_set)[:10]
            debug_print(debug, f"[CACHE] ids missing_from_rows(sample)={missing_from_rows}")
            debug_print(debug, f"[CACHE] ids extra_in_rows(sample)={extra_in_rows}")

            if isinstance(meta_ids, list):
                meta_ids = [str(x) for x in meta_ids]
                if expected_ids == meta_ids:
                    debug_print(debug, "[CACHE] NOTE: meta example_ids match expected_ids, but row_ids do not.")
                elif row_ids == meta_ids:
                    debug_print(debug, "[CACHE] NOTE: row_ids match meta example_ids, but expected_ids do not.")
                else:
                    meta_set = set(meta_ids)
                    debug_print(debug, f"[CACHE] meta_ids missing_from_expected(sample)={list(meta_set - expected_set)[:10]}")
                    debug_print(debug, f"[CACHE] expected_ids missing_from_meta(sample)={list(expected_set - meta_set)[:10]}")

            return None

        debug_print(debug, f"[CACHE] cache metadata summary: {json.dumps(meta, ensure_ascii=False)[:2000]}")
        tqdm.write(f"[GEN] Reusing shared cache for {_shared_single_agent_strategy_name_for_slot(slot)} from {cache_dir}")
        return rows

    except Exception as e:
        debug_print(debug, f"[CACHE] Failed to load shared single-agent cache {cache_dir}: {type(e).__name__}: {e}")
        import traceback
        debug_print(debug, traceback.format_exc())
        return None

def save_shared_single_agent_cache(benchmark: str, examples: List[Dict[str, Any]], slot: str, model_name_or_path: str, model_family: str, thinking_mode: str, sampling_cfg: Any, rows: List[Dict[str, Any]], debug: bool) -> Path:
    cache_dir = _shared_single_agent_cache_dir(benchmark, slot, model_name_or_path, model_family, thinking_mode, sampling_cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    strategy_name = _shared_single_agent_strategy_name_for_slot(slot)
    write_jsonl(cache_dir / "results.jsonl", rows)
    save_rows_to_parquet(cache_dir / "results.parquet", rows, debug=debug)
    json_dump(cache_dir / "generation_summary.json", make_summary(benchmark, strategy_name, rows))
    meta = _shared_single_agent_cache_meta(benchmark, slot, model_name_or_path, model_family, thinking_mode, sampling_cfg)
    meta["num_rows"] = len(rows)
    meta["example_ids"] = [_example_id_from_example(ex) for ex in examples]
    json_dump(cache_dir / "cache_meta.json", meta)
    tqdm.write(f"[GEN] Saved shared cache for {strategy_name}: {cache_dir}")
    return cache_dir


def build_output_base(benchmark: str, threshold_list: List[float]) -> Path:
    return _compute_results_root(
        output_base_root=OUTPUT_BASE_ROOT, benchmark=benchmark,
        model1_name_or_path=MODEL1_NAME_OR_PATH, model2_name_or_path=MODEL2_NAME_OR_PATH,
        model1_model_family=MODEL1_MODEL_FAMILY, model1_thinking_mode=MODEL1_THINKING_MODE,
        model2_model_family=MODEL2_MODEL_FAMILY, model2_thinking_mode=MODEL2_THINKING_MODE,
        model1_aux_thresholds=threshold_list, model2_aux_threshold=MODEL2_AUX_THRESHOLD,
    )


def threshold_output_root(benchmark: str, threshold_list: List[float], thr1: float) -> Path:
    base = build_output_base(benchmark, threshold_list)
    return base / f"thr1_{_format_threshold_for_path(thr1)}__thr2_{_format_threshold_for_path(MODEL2_AUX_THRESHOLD)}"


def prepare_output_root(output_root: Path, overwrite: bool, resume: bool, debug: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if overwrite:
            debug_print(debug, f"Removing existing output directory: {output_root}")
            shutil.rmtree(output_root)
        elif resume:
            debug_print(debug, f"Resume mode enabled; keeping existing output directory: {output_root}")
        else:
            raise RuntimeError(
                f"Output directory already exists and is not empty: {output_root}\n"
                f"Use --overwrite, set OVERWRITE=True, enable resume, or change OUTPUT_BASE_ROOT."
            )
    output_root.mkdir(parents=True, exist_ok=True)


def _parquet_safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Path):
        return str(value)
    return value


def save_rows_to_parquet(path: Path, rows: Iterable[Dict[str, Any]], debug: bool = False) -> bool:
    rows = list(rows)
    try:
        import pandas as pd
        flat_rows = [{k: _parquet_safe_value(v) for k, v in row.items()} for row in rows]
        df = pd.DataFrame(flat_rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        debug_print(debug, f"saved {path}")
        return True
    except Exception as e:
        err_path = path.with_suffix(path.suffix + ".error.txt")
        msg = (
            f"Failed to save parquet file: {path}\n"
            f"Error: {type(e).__name__}: {e}\n"
            f"Tip: install a parquet engine, usually with: pip install pyarrow\n"
        )
        err_path.write_text(msg, encoding="utf-8")
        debug_print(debug, msg.strip())
        return False


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_completed_strategy_summary(strategy_dir: Path, expected_num_rows: int, debug: bool = False) -> Tuple[bool, Optional[Dict[str, Any]]]:
    results_jsonl = strategy_dir / "results.jsonl"
    summary_json = strategy_dir / "generation_summary.json"
    if not results_jsonl.exists() or not summary_json.exists():
        return False, None
    summary = _read_json(summary_json)
    if not isinstance(summary, dict):
        return False, None
    summary_num_rows = int(summary.get("num_rows", -1))
    jsonl_num_rows = _count_jsonl_rows(results_jsonl)
    is_complete = (summary_num_rows == expected_num_rows) and (jsonl_num_rows == expected_num_rows)
    if is_complete:
        debug_print(debug, f"Resume: strategy already complete: {strategy_dir.name} ({expected_num_rows} rows)")
        return True, summary
    debug_print(
        debug,
        f"Resume: strategy incomplete: {strategy_dir.name} "
        f"(summary_num_rows={summary_num_rows}, jsonl_num_rows={jsonl_num_rows}, expected={expected_num_rows})",
    )
    return False, None


def _parse_prefix_handoff_strategy_name(strategy_name: str) -> Optional[Dict[str, Any]]:
    name = str(strategy_name or "").strip()
    if name == "m1_after_1000tok_handoff_context_m2":
        return {"prefix_tokens": 1000, "use_model2_aux": False, "canonical_name": name}
    m = re.fullmatch(r"m1_after_(\d+)tok_handoff_context_m2(_with_m2_aux)?", name)
    if m is None:
        return None
    return {
        "prefix_tokens": int(m.group(1)),
        "use_model2_aux": bool(m.group(2)),
        "canonical_name": name,
    }


def is_supported_generation_strategy_name(strategy_name: str) -> bool:
    return (
        strategy_name in {
            "single_agent_model1",
            "single_agent_model2",
            "m1_after_finish_self_repair",
            "m1_after_finish_retry",
            "m1_after_finish_handoff_fresh_m2",
            "m1_after_finish_handoff_context_m2",
            "m1_after_1000tok_handoff_context_m2",
        }
        or _parse_prefix_handoff_strategy_name(strategy_name) is not None
    )


def supported_generation_strategy_names() -> set[str]:
    return {
        "single_agent_model1",
        "single_agent_model2",
        "m1_after_finish_self_repair",
        "m1_after_finish_retry",
        "m1_after_finish_handoff_fresh_m2",
        "m1_after_finish_handoff_context_m2",
        "m1_after_1000tok_handoff_context_m2",
        "m1_after_<N>tok_handoff_context_m2",
        "m1_after_<N>tok_handoff_context_m2_with_m2_aux",
    }


def strategy_reuses_model1_first_pass(strategy_name: str) -> bool:
    return strategy_name in {
        "m1_after_finish_self_repair",
        "m1_after_finish_retry",
        "m1_after_finish_handoff_fresh_m2",
        "m1_after_finish_handoff_context_m2",
    } or _parse_prefix_handoff_strategy_name(strategy_name) is not None


def strategy_uses_full_m1_aux_threshold(strategy_name: str) -> bool:
    return strategy_name in {
        "m1_after_finish_self_repair",
        "m1_after_finish_retry",
        "m1_after_finish_handoff_fresh_m2",
        "m1_after_finish_handoff_context_m2",
    }


def strategy_uses_prefix_handoff(strategy_name: str) -> bool:
    return _parse_prefix_handoff_strategy_name(strategy_name) is not None


def strategy_reuses_model2_single_pass(strategy_name: str) -> bool:
    return strategy_name in {"m1_after_finish_handoff_fresh_m2"}


def get_strategy_exec_cfg(strategy_name: str) -> Dict[str, Any]:
    if strategy_name in STRATEGY_EXECUTION:
        return dict(STRATEGY_EXECUTION[strategy_name])
    prefix_info = _parse_prefix_handoff_strategy_name(strategy_name)
    if prefix_info is not None:
        return {
            "batch_size": 16,
            "need_model1_vllm": False,
            "need_model2_vllm": True,
            "need_model1_aux": True,
            "need_model2_aux": bool(prefix_info["use_model2_aux"] and MODEL2_AUX_HEAD_CKPT),
            "model1_gpu_memory_utilization": None,
            "model1_max_num_seqs": None,
            "model2_gpu_memory_utilization": 0.80,
            "model2_max_num_seqs": 16,
        }
    raise RuntimeError(f"Missing STRATEGY_EXECUTION config for: {strategy_name}")


def _apply_runtime_override(base: Dict[str, Any], gpu_mem: Optional[float], max_num_seqs: Optional[int]) -> Dict[str, Any]:
    prof = dict(base)
    if gpu_mem is not None:
        prof["gpu_memory_utilization"] = float(gpu_mem)
    if max_num_seqs is not None:
        prof["max_num_seqs"] = int(max_num_seqs)
    return prof


def build_strategy_model_bundles(strategy_name: str) -> Dict[str, Any]:
    cfg = get_strategy_exec_cfg(strategy_name)
    model_bundles: Dict[str, Any] = {}

    model1_override = dict(MODEL_SAMPLING_OVERRIDES.get("model1") or {})
    model1_override.update(
        _official_sampling_override_for_model(
            model_name_or_path=MODEL1_NAME_OR_PATH,
            model_family=MODEL1_MODEL_FAMILY,
            thinking_mode=MODEL1_THINKING_MODE,
            benchmark=CURRENT_BENCHMARK,
        )
    )
    model2_override = dict(MODEL_SAMPLING_OVERRIDES.get("model2") or {})
    model2_override.update(
        _official_sampling_override_for_model(
            model_name_or_path=MODEL2_NAME_OR_PATH,
            model_family=MODEL2_MODEL_FAMILY,
            thinking_mode=MODEL2_THINKING_MODE,
            benchmark=CURRENT_BENCHMARK,
        )
    )

    if cfg["need_model1_vllm"] or cfg["need_model1_aux"]:
        rp1 = _apply_runtime_override(
            MODEL1_VLLM_BASE,
            cfg["model1_gpu_memory_utilization"],
            cfg["model1_max_num_seqs"],
        )
        model_bundles["model1"] = build_model_bundle(
            model_name_or_path=MODEL1_NAME_OR_PATH,
            aux_head_ckpt=MODEL1_AUX_HEAD_CKPT,
            runtime_profile=rp1,
            sampling_profiles=SAMPLING_PROFILES,
            aux_profile=AUX_PROFILE,
            sampling_override=model1_override,
            model_family=MODEL1_MODEL_FAMILY,
            thinking_mode=MODEL1_THINKING_MODE,
        )

    if cfg["need_model2_vllm"] or cfg["need_model2_aux"]:
        rp2 = _apply_runtime_override(
            MODEL2_VLLM_BASE,
            cfg["model2_gpu_memory_utilization"],
            cfg["model2_max_num_seqs"],
        )
        model_bundles["model2"] = build_model_bundle(
            model_name_or_path=MODEL2_NAME_OR_PATH,
            aux_head_ckpt=MODEL2_AUX_HEAD_CKPT,
            runtime_profile=rp2,
            sampling_profiles=SAMPLING_PROFILES,
            aux_profile=AUX_PROFILE,
            sampling_override=model2_override,
            model_family=MODEL2_MODEL_FAMILY,
            thinking_mode=MODEL2_THINKING_MODE,
        )

    return model_bundles


def clone_result_tuple(result: Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]) -> Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]:
    final_model_name, final_response, usage_by_model, trace, wall_time_sec = result
    return (
        final_model_name,
        final_response,
        copy.deepcopy(usage_by_model),
        copy.deepcopy(trace),
        float(wall_time_sec),
    )


def make_summary(benchmark: str, strategy_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "benchmark": benchmark,
        "strategy_name": strategy_name,
        "num_rows": len(rows),
        "avg_wall_time_sec": sum(float(r.get("wall_time_sec", 0.0)) for r in rows) / max(len(rows), 1),
        "usage_totals_by_model": {
            m: {
                k: sum(int(r.get("usage_by_model", {}).get(m, {}).get(k, 0)) for r in rows)
                for k in ["prompt_tokens", "completion_tokens", "aux_scored_tokens", "aux_calls", "generation_calls"]
            }
            for m in ["model1", "model2"]
        },
        "note": "Generation-only run. Final benchmark metrics are produced by compact_multi_agent_evaluate.py",
    }


def save_strategy_outputs(strategy_dir: Path, benchmark: str, strategy_name: str, rows: List[Dict[str, Any]], suite_summary: Dict[str, Dict[str, Any]], debug: bool) -> None:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl_path = strategy_dir / "results.jsonl"
    write_jsonl(results_jsonl_path, rows)
    save_rows_to_parquet(strategy_dir / "results.parquet", rows, debug=debug)
    summary = make_summary(benchmark, strategy_name, rows)
    json_dump(strategy_dir / "generation_summary.json", summary)
    suite_summary[strategy_name] = summary
    tqdm.write(f"[GEN] Finished strategy: {strategy_name}")
    tqdm.write(f"[GEN] Saved: {results_jsonl_path}")
    tqdm.write(f"[GEN] Saved: {strategy_dir / 'results.parquet'}")
    tqdm.write(f"[GEN] Saved: {strategy_dir / 'generation_summary.json'}")


def threshold_strategy_dir(base_root: Path, strategy_name: str) -> Path:
    return base_root / strategy_name


def format_threshold_value(x: float) -> str:
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def resolve_requested_strategy_names(names_csv: str) -> List[str]:
    names = [x.strip() for x in str(names_csv).split(",") if x.strip()]
    if not names or names == ["all"]:
        return [
            "single_agent_model1",
            "single_agent_model2",
            "m1_after_finish_self_repair",
            "m1_after_finish_retry",
            "m1_after_finish_handoff_fresh_m2",
            "m1_after_finish_handoff_context_m2",
            "m1_after_1000tok_handoff_context_m2",
        ]
    bad = [name for name in names if not is_supported_generation_strategy_name(name)]
    if bad:
        raise RuntimeError(
            f"Unsupported strategy names: {bad}. "
            f"Supported strategies here are: {sorted(supported_generation_strategy_names())}"
        )
    return names


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def get_model1_first_pass_status(benchmark: str, response_text: str, aux_score: float, threshold: float) -> Dict[str, Any]:
    final_info = response_final_answer_status(benchmark, response_text)
    has_final_answer = bool(final_info.get("has_final_answer", False))
    low_aux = float(aux_score) < float(threshold)
    needs_branch = (not has_final_answer) or low_aux
    if (not has_final_answer) and low_aux:
        reason = "missing_final_answer_and_low_aux"
    elif not has_final_answer:
        reason = "missing_final_answer"
    elif low_aux:
        reason = "low_aux"
    else:
        reason = "accept"
    return {
        "has_final_answer": has_final_answer,
        "final_answer_reason": str(final_info.get("reason", "unknown")),
        "needs_branch": bool(needs_branch),
        "reason": reason,
    }


def should_branch_model1_first_pass(benchmark: str, response_text: str, aux_score: float, threshold: float) -> bool:
    return bool(get_model1_first_pass_status(benchmark, response_text, aux_score, threshold).get("needs_branch", False))


def _routing_trace_payload(benchmark: str, response_text: str, aux_score: float, threshold: float) -> Dict[str, Any]:
    info = get_model1_first_pass_status(benchmark, response_text, aux_score, threshold)
    return {
        "prob_correct": float(aux_score),
        "threshold": float(threshold),
        "has_final_answer": bool(info["has_final_answer"]),
        "final_answer_reason": str(info["final_answer_reason"]),
        "routing_reason": str(info["reason"]),
    }


def get_model1_prefix_status(benchmark: str, response_text: str, completion_tokens: int, aux_score: float, threshold: float, prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> Dict[str, Any]:
    final_info = response_final_answer_status(benchmark, response_text)
    has_final_answer = bool(final_info.get("has_final_answer", False))
    ended_before_prefix = int(completion_tokens) < int(prefix_tokens)
    if ended_before_prefix and not has_final_answer:
        needs_branch = True
        reason = "ended_before_prefix_without_final_answer"
    elif float(aux_score) < float(threshold):
        needs_branch = True
        reason = "low_aux_at_prefix"
    else:
        needs_branch = False
        reason = "accept"
    return {
        "has_final_answer": has_final_answer,
        "final_answer_reason": str(final_info.get("reason", "unknown")),
        "ended_before_prefix": bool(ended_before_prefix),
        "prefix_tokens": int(prefix_tokens),
        "observed_completion_tokens": int(completion_tokens),
        "needs_branch": bool(needs_branch),
        "reason": reason,
    }


def should_branch_model1_prefix(response_text: str, completion_tokens: int, aux_score: float, threshold: float, benchmark: str, prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> bool:
    return bool(get_model1_prefix_status(benchmark, response_text, completion_tokens, aux_score, threshold, prefix_tokens).get("needs_branch", False))


def _prefix_routing_trace_payload(benchmark: str, response_text: str, completion_tokens: int, aux_score: float, threshold: float, prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> Dict[str, Any]:
    info = get_model1_prefix_status(benchmark, response_text, completion_tokens, aux_score, threshold, prefix_tokens)
    return {
        "routing_stage": f"prefix_{int(prefix_tokens)}",
        "prob_correct": float(aux_score),
        "threshold": float(threshold),
        "has_final_answer": bool(info["has_final_answer"]),
        "final_answer_reason": str(info["final_answer_reason"]),
        "ended_before_prefix": bool(info["ended_before_prefix"]),
        "observed_completion_tokens": int(info["observed_completion_tokens"]),
        "prefix_tokens": int(info["prefix_tokens"]),
        "routing_reason": str(info["reason"]),
    }


def precompute_model1_first_pass(benchmark: str, examples: List[Dict[str, Any]], debug: bool) -> List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    precompute_model_bundles = {
        "model1": build_model_bundle(
            model_name_or_path=MODEL1_NAME_OR_PATH,
            aux_head_ckpt=MODEL1_AUX_HEAD_CKPT,
            runtime_profile=_apply_runtime_override(MODEL1_VLLM_BASE, PRECOMPUTE_MODEL1_GPU_UTIL, PRECOMPUTE_MODEL1_MAX_NUM_SEQS),
            sampling_profiles=SAMPLING_PROFILES,
            aux_profile=AUX_PROFILE,
            sampling_override=MODEL_SAMPLING_OVERRIDES.get("model1"),
            model_family=MODEL1_MODEL_FAMILY,
            thinking_mode=MODEL1_THINKING_MODE,
        )
    }
    precompute_orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=precompute_model_bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        tqdm.write(f"[GEN] Precomputing reusable model1 first-pass completions once | batch_size={PRECOMPUTE_MODEL1_BATCH_SIZE}")
        single_agent_m1_strategy = next(
            s for s in build_default_two_model_suite(
                threshold1=MODEL1_AUX_THRESHOLDS[0],
                threshold2=MODEL2_AUX_THRESHOLD,
                chunk_tokens=CHUNK_TOKENS,
                enable_model2_aux=bool(MODEL2_AUX_HEAD_CKPT),
            )
            if s.name == "single_agent_model1"
        )
        reusable_m1_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = []
        pre_bar = tqdm(total=len(examples), desc="precompute_m1", unit="example", dynamic_ncols=True, leave=False)
        for batch_start in range(0, len(examples), PRECOMPUTE_MODEL1_BATCH_SIZE):
            batch_examples = examples[batch_start: batch_start + PRECOMPUTE_MODEL1_BATCH_SIZE]
            batch_results = precompute_orchestrator.run_examples_batched(batch_examples, single_agent_m1_strategy, batch_size=len(batch_examples))
            reusable_m1_results.extend(batch_results)
            pre_bar.update(len(batch_examples))
        pre_bar.close()
        return reusable_m1_results
    finally:
        precompute_orchestrator.unload_all(drop_processors=False)
        del precompute_orchestrator


def precompute_model2_single_pass(benchmark: str, examples: List[Dict[str, Any]], debug: bool) -> List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    precompute_model_bundles = {
        "model2": build_model_bundle(
            model_name_or_path=MODEL2_NAME_OR_PATH,
            aux_head_ckpt=MODEL2_AUX_HEAD_CKPT,
            runtime_profile=_apply_runtime_override(MODEL2_VLLM_BASE, get_strategy_exec_cfg("single_agent_model2")["model2_gpu_memory_utilization"], get_strategy_exec_cfg("single_agent_model2")["model2_max_num_seqs"]),
            sampling_profiles=SAMPLING_PROFILES,
            aux_profile=AUX_PROFILE,
            sampling_override=MODEL_SAMPLING_OVERRIDES.get("model2"),
            model_family=MODEL2_MODEL_FAMILY,
            thinking_mode=MODEL2_THINKING_MODE,
        )
    }
    precompute_orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=precompute_model_bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        batch_size = int(get_strategy_exec_cfg("single_agent_model2")["batch_size"])
        tqdm.write(f"[GEN] Precomputing reusable model2 single-agent completions once | batch_size={batch_size}")
        single_agent_m2_strategy = next(
            s for s in build_default_two_model_suite(
                threshold1=MODEL1_AUX_THRESHOLDS[0],
                threshold2=MODEL2_AUX_THRESHOLD,
                chunk_tokens=CHUNK_TOKENS,
                enable_model2_aux=bool(MODEL2_AUX_HEAD_CKPT),
            )
            if s.name == "single_agent_model2"
        )
        reusable_m2_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = []
        pre_bar = tqdm(total=len(examples), desc="precompute_m2", unit="example", dynamic_ncols=True, leave=False)
        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start: batch_start + batch_size]
            batch_results = precompute_orchestrator.run_examples_batched(batch_examples, single_agent_m2_strategy, batch_size=len(batch_examples))
            reusable_m2_results.extend(batch_results)
            pre_bar.update(len(batch_examples))
        pre_bar.close()
        return reusable_m2_results
    finally:
        precompute_orchestrator.unload_all(drop_processors=False)
        del precompute_orchestrator


def precompute_model1_prefix_pass(benchmark: str, examples: List[Dict[str, Any]], debug: bool, prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    precompute_model_bundles = {
        "model1": build_model_bundle(
            model_name_or_path=MODEL1_NAME_OR_PATH,
            aux_head_ckpt=MODEL1_AUX_HEAD_CKPT,
            runtime_profile=_apply_runtime_override(MODEL1_VLLM_BASE, PRECOMPUTE_MODEL1_GPU_UTIL, PRECOMPUTE_MODEL1_MAX_NUM_SEQS),
            sampling_profiles=SAMPLING_PROFILES,
            aux_profile=AUX_PROFILE,
            sampling_override={**MODEL_SAMPLING_OVERRIDES.get("model1", {}), "max_new_tokens": int(prefix_tokens)},
            model_family=MODEL1_MODEL_FAMILY,
            thinking_mode=MODEL1_THINKING_MODE,
        )
    }
    precompute_orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=precompute_model_bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        t0 = time.time()
        runtime1 = precompute_orchestrator.get_generator("model1")
        bundle1 = precompute_model_bundles["model1"]
        reusable_prefix_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = []
        pre_bar = tqdm(total=len(examples), desc=f"precompute_m1_prefix_{int(prefix_tokens)}", unit="example", dynamic_ncols=True, leave=False)
        for batch_start in range(0, len(examples), PRECOMPUTE_MODEL1_BATCH_SIZE):
            batch_examples = examples[batch_start: batch_start + PRECOMPUTE_MODEL1_BATCH_SIZE]
            messages_list = [precompute_orchestrator._build_messages_for_turn(ex, None) for ex in batch_examples]
            images = [get_example_image_for_benchmark(ex) for ex in batch_examples]
            gens = runtime1.generate_batch(messages_list=messages_list, images=images, sampling_cfg=bundle1.sampling_cfg, continue_final_messages=[False] * len(batch_examples))
            for gen in gens:
                usage_by_model = precompute_orchestrator._new_usage_by_model()
                usage_by_model["model1"].prompt_tokens += int(gen.prompt_tokens)
                usage_by_model["model1"].completion_tokens += int(gen.completion_tokens)
                usage_by_model["model1"].generation_calls += 1
                trace = [{"event": "prefix_generation", "model": "model1", "completion_tokens": int(gen.completion_tokens), "prefix_tokens": int(prefix_tokens)}]
                reusable_prefix_results.append(("model1", gen.text, usage_by_model, trace, time.time() - t0))
            pre_bar.update(len(batch_examples))
        pre_bar.close()
        return reusable_prefix_results
    finally:
        precompute_orchestrator.unload_all(drop_processors=False)
        del precompute_orchestrator


def precompute_model1_prefix_aux_scores(benchmark: str, examples: List[Dict[str, Any]], reusable_prefix_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], debug: bool, prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> List[float]:
    score_bundles = {
        "model1": build_model_bundle(
            model_name_or_path=MODEL1_NAME_OR_PATH,
            aux_head_ckpt=MODEL1_AUX_HEAD_CKPT,
            runtime_profile=_apply_runtime_override(MODEL1_VLLM_BASE, None, None),
            sampling_profiles=SAMPLING_PROFILES,
            aux_profile=AUX_PROFILE,
            sampling_override=MODEL_SAMPLING_OVERRIDES.get("model1"),
            model_family=MODEL1_MODEL_FAMILY,
            thinking_mode=MODEL1_THINKING_MODE,
        )
    }
    orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=score_bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        aux = orchestrator.get_aux("model1")
        if aux is None:
            raise RuntimeError("MODEL1_AUX_HEAD_CKPT is required for prefix handoff strategies")
        aux.load()
        scores: List[float] = []
        bar = tqdm(total=len(examples), desc=f"score_m1_prefix_{int(prefix_tokens)}_aux_once", unit="example", dynamic_ncols=True, leave=False)
        for ex, result in zip(examples, reusable_prefix_results):
            response_text = str(result[1])
            score = orchestrator._score_response(ex, "model1", response_text)
            scores.append(float(score.prob_correct))
            bar.update(1)
        bar.close()
        return scores
    finally:
        orchestrator.unload_all(drop_processors=False)
        del orchestrator


def precompute_model1_aux_scores_and_answer_flags(benchmark: str, examples: List[Dict[str, Any]], reusable_m1_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], debug: bool) -> Tuple[List[float], List[bool], List[str]]:
    score_bundles = {
        "model1": build_model_bundle(
            model_name_or_path=MODEL1_NAME_OR_PATH,
            aux_head_ckpt=MODEL1_AUX_HEAD_CKPT,
            runtime_profile=_apply_runtime_override(MODEL1_VLLM_BASE, None, None),
            sampling_profiles=SAMPLING_PROFILES,
            aux_profile=AUX_PROFILE,
            sampling_override=MODEL_SAMPLING_OVERRIDES.get("model1"),
            model_family=MODEL1_MODEL_FAMILY,
            thinking_mode=MODEL1_THINKING_MODE,
        )
    }
    orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=score_bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        aux = orchestrator.get_aux("model1")
        if aux is None:
            raise RuntimeError("MODEL1_AUX_HEAD_CKPT is required for threshold sweep strategies")
        aux.load()
        scores: List[float] = []
        has_final_answers: List[bool] = []
        final_answer_reasons: List[str] = []
        bar = tqdm(total=len(examples), desc="score_m1_aux_once", unit="example", dynamic_ncols=True, leave=False)
        for ex, result in zip(examples, reusable_m1_results):
            response_text = str(result[1])
            score = orchestrator._score_response(ex, "model1", response_text)
            final_info = response_final_answer_status(benchmark, response_text)
            scores.append(float(score.prob_correct))
            has_final_answers.append(bool(final_info.get("has_final_answer", False)))
            final_answer_reasons.append(str(final_info.get("reason", "unknown")))
            bar.update(1)
        bar.close()
        return scores, has_final_answers, final_answer_reasons
    finally:
        orchestrator.unload_all(drop_processors=False)
        del orchestrator


def build_rows_from_results(benchmark: str, examples: List[Dict[str, Any]], strategy_name: str, results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ex, result in zip(examples, results):
        final_model_name, final_response, usage_by_model, trace, wall_time_sec = result
        rows.append(build_generation_row(
            benchmark=benchmark,
            ex=ex,
            strategy_name=strategy_name,
            final_model_name=final_model_name,
            final_response=final_response,
            usage_by_model=usage_by_model,
            trace=trace,
            wall_time_sec=float(wall_time_sec),
        ))
    return rows


def run_single_agent_strategy_once(benchmark: str, examples: List[Dict[str, Any]], strategy_name: str, debug: bool) -> List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    strategy = next(
        s for s in build_default_two_model_suite(
            threshold1=MODEL1_AUX_THRESHOLDS[0],
            threshold2=MODEL2_AUX_THRESHOLD,
            chunk_tokens=CHUNK_TOKENS,
            enable_model2_aux=bool(MODEL2_AUX_HEAD_CKPT),
        ) if s.name == strategy_name
    )
    cfg = get_strategy_exec_cfg(strategy_name)
    bundles = build_strategy_model_bundles(strategy_name)
    orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        batch_size = int(cfg["batch_size"])
        out: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = []
        bar = tqdm(total=len(examples), desc=strategy_name, unit="example", dynamic_ncols=True, leave=False)
        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start: batch_start + batch_size]
            batch_results = orchestrator.run_examples_batched(batch_examples, strategy, batch_size=len(batch_examples))
            out.extend(batch_results)
            bar.update(len(batch_examples))
        bar.close()
        return out
    finally:
        orchestrator.unload_all(drop_processors=False)
        del orchestrator


def make_threshold_accept_result(benchmark: str, base_result: Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float], aux_score: float, threshold: float) -> Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]:
    final_model_name, final_response, usage_by_model, trace, wall_time_sec = clone_result_tuple(base_result)
    model1_completion_tokens = int(usage_by_model["model1"].completion_tokens)
    usage_by_model["model1"].aux_calls += 1
    usage_by_model["model1"].aux_scored_tokens += max(0, model1_completion_tokens)
    payload = _routing_trace_payload(benchmark, final_response, aux_score, threshold)
    trace.append({"event": "cached_aux_score", "model": "model1", **payload, "decision": "accept"})
    return final_model_name, final_response, usage_by_model, trace, wall_time_sec


def compute_new_self_repair_results(benchmark: str, examples: List[Dict[str, Any]], reusable_m1_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], model1_scores: List[float], indices: List[int], threshold: float, debug: bool) -> Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    if not indices:
        return {}
    bundles = build_strategy_model_bundles("m1_after_finish_self_repair")
    orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        t_branch = time.time()
        runtime1 = orchestrator.get_generator("model1")
        bundle1 = bundles["model1"]
        out: Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = {}
        batch_size = int(get_strategy_exec_cfg("m1_after_finish_self_repair")["batch_size"])
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start: start + batch_size]
            messages_list = []
            images = []
            for idx in batch_indices:
                ex = examples[idx]
                draft = reusable_m1_results[idx][1]
                base = orchestrator._build_messages_for_turn(ex, None)
                msgs = base + [
                    {"role": "assistant", "content": str(draft)},
                    {"role": "user", "content": SELF_REPAIR_TEXT},
                ]
                messages_list.append(msgs)
                images.append(get_example_image_for_benchmark(ex))
            gens = runtime1.generate_batch(messages_list=messages_list, images=images, sampling_cfg=bundle1.sampling_cfg, continue_final_messages=[False] * len(batch_indices))
            for idx, gen in zip(batch_indices, gens):
                _, _, usage_by_model, trace, wall_time_sec = clone_result_tuple(reusable_m1_results[idx])
                usage_by_model["model1"].aux_calls += 1
                usage_by_model["model1"].aux_scored_tokens += int(reusable_m1_results[idx][2]["model1"].completion_tokens)
                payload = _routing_trace_payload(benchmark, reusable_m1_results[idx][1], model1_scores[idx], threshold)
                trace.append({"event": "cached_aux_score", "model": "model1", **payload, "decision": "self_repair"})
                usage_by_model["model1"].prompt_tokens += int(gen.prompt_tokens)
                usage_by_model["model1"].completion_tokens += int(gen.completion_tokens)
                usage_by_model["model1"].generation_calls += 1
                trace.append({"event": "self_repair_generation", "model": "model1", "completion_tokens": int(gen.completion_tokens)})
                out[idx] = ("model1", gen.text, usage_by_model, trace, float(wall_time_sec) + (time.time() - t_branch))
        return out
    finally:
        orchestrator.unload_all(drop_processors=False)
        del orchestrator


def compute_new_handoff_results(benchmark: str, examples: List[Dict[str, Any]], reusable_m1_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], model1_scores: List[float], indices: List[int], threshold: float, strategy_name: str, debug: bool, reusable_m2_results: Optional[List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]] = None) -> Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    if not indices:
        return {}
    if strategy_name not in {"m1_after_finish_handoff_fresh_m2", "m1_after_finish_handoff_context_m2"}:
        raise RuntimeError(f"Unsupported handoff strategy: {strategy_name}")

    out: Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = {}
    if strategy_name == "m1_after_finish_handoff_fresh_m2":
        if reusable_m2_results is None:
            raise RuntimeError("Fresh handoff reuse requires reusable model2 single-agent results")
        for idx in indices:
            _, _, usage_by_model, trace, wall_time_sec = clone_result_tuple(reusable_m1_results[idx])
            usage_by_model["model1"].aux_calls += 1
            usage_by_model["model1"].aux_scored_tokens += int(reusable_m1_results[idx][2]["model1"].completion_tokens)
            payload = _routing_trace_payload(benchmark, reusable_m1_results[idx][1], model1_scores[idx], threshold)
            trace.append({
                "event": "cached_aux_score",
                "model": "model1",
                **payload,
                "decision": "handoff",
                "handoff_mode": "fresh",
                "reused_existing_model2_completion": True,
            })
            _, model2_response, m2_usage_by_model, m2_trace, m2_wall_time_sec = clone_result_tuple(reusable_m2_results[idx])
            usage_by_model["model2"].prompt_tokens += int(m2_usage_by_model["model2"].prompt_tokens)
            usage_by_model["model2"].completion_tokens += int(m2_usage_by_model["model2"].completion_tokens)
            usage_by_model["model2"].generation_calls += int(m2_usage_by_model["model2"].generation_calls)
            usage_by_model["model2"].aux_scored_tokens += int(m2_usage_by_model["model2"].aux_scored_tokens)
            usage_by_model["model2"].aux_calls += int(m2_usage_by_model["model2"].aux_calls)
            trace.append({
                "event": "reused_single_agent_model2_completion",
                "model": "model2",
                "completion_tokens": int(m2_usage_by_model["model2"].completion_tokens),
                "prompt_tokens": int(m2_usage_by_model["model2"].prompt_tokens),
            })
            out[idx] = ("model2", model2_response, usage_by_model, trace + list(m2_trace), float(wall_time_sec) + float(m2_wall_time_sec))
        return out

    bundles = build_strategy_model_bundles(strategy_name)
    orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        t_branch = time.time()
        runtime2 = orchestrator.get_generator("model2")
        bundle2 = bundles["model2"]
        batch_size = int(get_strategy_exec_cfg(strategy_name)["batch_size"])
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start: start + batch_size]
            messages_list = []
            images = []
            for idx in batch_indices:
                ex = examples[idx]
                payload = {"mode": "handoff_with_context", "from_model": "model1", "draft_response": reusable_m1_results[idx][1]}
                msgs = orchestrator._build_messages_for_turn(ex, payload)
                messages_list.append(msgs)
                images.append(get_example_image_for_benchmark(ex))
            gens = runtime2.generate_batch(messages_list=messages_list, images=images, sampling_cfg=bundle2.sampling_cfg, continue_final_messages=[False] * len(batch_indices))
            for idx, gen in zip(batch_indices, gens):
                _, _, usage_by_model, trace, wall_time_sec = clone_result_tuple(reusable_m1_results[idx])
                usage_by_model["model1"].aux_calls += 1
                usage_by_model["model1"].aux_scored_tokens += int(reusable_m1_results[idx][2]["model1"].completion_tokens)
                payload = _routing_trace_payload(benchmark, reusable_m1_results[idx][1], model1_scores[idx], threshold)
                trace.append({
                    "event": "cached_aux_score",
                    "model": "model1",
                    **payload,
                    "decision": "handoff",
                    "handoff_mode": "with_context",
                })
                usage_by_model["model2"].prompt_tokens += int(gen.prompt_tokens)
                usage_by_model["model2"].completion_tokens += int(gen.completion_tokens)
                usage_by_model["model2"].generation_calls += 1
                trace.append({"event": "handoff_generation", "model": "model2", "completion_tokens": int(gen.completion_tokens)})
                out[idx] = ("model2", gen.text, usage_by_model, trace, float(wall_time_sec) + (time.time() - t_branch))
        return out
    finally:
        orchestrator.unload_all(drop_processors=False)
        del orchestrator


def make_prefix_threshold_accept_result(benchmark: str, base_result: Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float], prefix_result: Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float], aux_score: float, threshold: float, prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]:
    final_model_name, final_response, usage_by_model, trace, wall_time_sec = clone_result_tuple(base_result)
    prefix_completion_tokens = int(prefix_result[2]["model1"].completion_tokens)
    payload = _prefix_routing_trace_payload(benchmark, prefix_result[1], prefix_completion_tokens, aux_score, threshold, prefix_tokens)
    usage_by_model["model1"].aux_calls += 1
    usage_by_model["model1"].aux_scored_tokens += max(0, prefix_completion_tokens)
    trace.append({"event": "cached_aux_score", "model": "model1", **payload, "decision": "accept"})
    return final_model_name, final_response, usage_by_model, trace, wall_time_sec


def compute_new_prefix_handoff_results(benchmark: str, examples: List[Dict[str, Any]], reusable_prefix_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], prefix_scores: List[float], indices: List[int], threshold: float, strategy_name: str, debug: bool, prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    if not indices:
        return {}
    bundles = build_strategy_model_bundles(strategy_name)
    orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        t_branch = time.time()
        runtime2 = orchestrator.get_generator("model2")
        bundle2 = bundles["model2"]
        out: Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = {}
        batch_size = int(get_strategy_exec_cfg(strategy_name)["batch_size"])
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start: start + batch_size]
            messages_list = []
            images = []
            for idx in batch_indices:
                ex = examples[idx]
                prefix_text = reusable_prefix_results[idx][1]
                payload = {"mode": "handoff_with_context", "from_model": "model1", "draft_response": prefix_text}
                messages_list.append(orchestrator._build_messages_for_turn(ex, payload))
                images.append(get_example_image_for_benchmark(ex))
            gens = runtime2.generate_batch(messages_list=messages_list, images=images, sampling_cfg=bundle2.sampling_cfg, continue_final_messages=[False] * len(batch_indices))
            for idx, gen in zip(batch_indices, gens):
                _, prefix_text, usage_by_model, trace, wall_time_sec = clone_result_tuple(reusable_prefix_results[idx])
                prefix_completion_tokens = int(reusable_prefix_results[idx][2]["model1"].completion_tokens)
                usage_by_model["model1"].aux_calls += 1
                usage_by_model["model1"].aux_scored_tokens += max(0, prefix_completion_tokens)
                payload = _prefix_routing_trace_payload(benchmark, prefix_text, prefix_completion_tokens, prefix_scores[idx], threshold, prefix_tokens)
                trace.append({"event": "cached_aux_score", "model": "model1", **payload, "decision": "handoff", "handoff_mode": "with_context"})
                usage_by_model["model2"].prompt_tokens += int(gen.prompt_tokens)
                usage_by_model["model2"].completion_tokens += int(gen.completion_tokens)
                usage_by_model["model2"].generation_calls += 1
                trace.append({"event": "handoff_generation", "model": "model2", "completion_tokens": int(gen.completion_tokens), "routing_stage": f"prefix_{int(prefix_tokens)}"})
                out[idx] = ("model2", gen.text, usage_by_model, trace, float(wall_time_sec) + (time.time() - t_branch))
        return out
    finally:
        orchestrator.unload_all(drop_processors=False)
        del orchestrator


def materialize_threshold_results_prefix(benchmark: str, strategy_name: str, threshold: float, reusable_m1_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], reusable_prefix_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], prefix_scores: List[float], branch_cache: Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], prefix_tokens: int = PREFIX_HANDOFF_TOKENS) -> List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = []
    for idx, base_result in enumerate(reusable_m1_results):
        prefix_result = reusable_prefix_results[idx]
        prefix_completion_tokens = int(prefix_result[2]["model1"].completion_tokens)
        score = float(prefix_scores[idx])
        if should_branch_model1_prefix(prefix_result[1], prefix_completion_tokens, score, threshold, benchmark, prefix_tokens):
            if idx not in branch_cache:
                raise RuntimeError(f"Missing cached branch result for idx={idx}, strategy={strategy_name}, threshold={threshold}")
            results.append(clone_result_tuple(branch_cache[idx]))
        else:
            results.append(make_prefix_threshold_accept_result(benchmark, base_result, prefix_result, score, threshold, prefix_tokens))
    return results


def materialize_threshold_results(benchmark: str, strategy_name: str, threshold: float, reusable_m1_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], model1_scores: List[float], branch_cache: Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]) -> List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = []
    for idx, base_result in enumerate(reusable_m1_results):
        score = float(model1_scores[idx])
        response_text = str(base_result[1])
        if should_branch_model1_first_pass(benchmark, response_text, score, threshold):
            if idx not in branch_cache:
                raise RuntimeError(f"Missing cached branch result for idx={idx}, strategy={strategy_name}, threshold={threshold}")
            results.append(clone_result_tuple(branch_cache[idx]))
        else:
            results.append(make_threshold_accept_result(benchmark, base_result, score, threshold))
    return results


def ensure_suite_summary_written(base_root: Path, suite_summary: Dict[str, Dict[str, Any]], debug: bool) -> None:
    json_dump(base_root / "suite_summary_generation.json", suite_summary)
    save_rows_to_parquet(base_root / "suite_summary_generation.parquet", [{"strategy_name": k, **v} for k, v in suite_summary.items()], debug=debug)



def compute_new_retry_results(benchmark: str, examples: List[Dict[str, Any]], reusable_m1_results: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]], model1_scores: List[float], indices: List[int], threshold: float, debug: bool) -> Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]:
    if not indices:
        return {}
    bundles = build_strategy_model_bundles("m1_after_finish_retry")
    orchestrator = MultiAgentOrchestrator(benchmark=benchmark, model_bundles=bundles, debug_mode=debug, debug_max_chars=DEBUG_MAX_CHARS)
    try:
        t_branch = time.time()
        runtime1 = orchestrator.get_generator("model1")
        bundle1 = bundles["model1"]
        out: Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = {}
        batch_size = int(get_strategy_exec_cfg("m1_after_finish_retry")["batch_size"])
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start: start + batch_size]
            messages_list = [orchestrator._build_messages_for_turn(examples[idx], None) for idx in batch_indices]
            images = [get_example_image_for_benchmark(examples[idx]) for idx in batch_indices]
            gens = runtime1.generate_batch(messages_list=messages_list, images=images, sampling_cfg=bundle1.sampling_cfg, continue_final_messages=[False] * len(batch_indices))
            for idx, gen in zip(batch_indices, gens):
                _, _, usage_by_model, trace, wall_time_sec = clone_result_tuple(reusable_m1_results[idx])
                usage_by_model["model1"].aux_calls += 1
                usage_by_model["model1"].aux_scored_tokens += int(reusable_m1_results[idx][2]["model1"].completion_tokens)
                payload = _routing_trace_payload(benchmark, reusable_m1_results[idx][1], model1_scores[idx], threshold)
                trace.append({"event": "cached_aux_score", "model": "model1", **payload, "decision": "retry"})
                usage_by_model["model1"].prompt_tokens += int(gen.prompt_tokens)
                usage_by_model["model1"].completion_tokens += int(gen.completion_tokens)
                usage_by_model["model1"].generation_calls += 1
                trace.append({"event": "retry_generation", "model": "model1", "completion_tokens": int(gen.completion_tokens), "with_previous_attempt_context": False})
                out[idx] = ("model1", gen.text, usage_by_model, trace, float(wall_time_sec) + (time.time() - t_branch))
        return out
    finally:
        orchestrator.unload_all(drop_processors=False)
        del orchestrator


def run_aux_eval_for_single_agent_model1(run_base: Path, benchmark: str, rows: List[Dict[str, Any]], model1_scores: List[float], debug: bool) -> None:
    if len(rows) != len(model1_scores):
        raise RuntimeError(f"single_agent_model1 rows/model1_scores length mismatch: {len(rows)} vs {len(model1_scores)}")

    release_cuda_memory()
    judge_runtime = None
    judge_sampling = None
    if benchmark_needs_judge(benchmark):
        judge_runtime, judge_sampling = build_judge_runtime_and_sampling(
            judge_model_name_or_path=AUX_EVAL_JUDGE_MODEL_NAME_OR_PATH,
            judge_runtime_profile=AUX_EVAL_JUDGE_RUNTIME_PROFILE,
            judge_sampling_profiles=AUX_EVAL_JUDGE_SAMPLING_PROFILES,
            judge_model_family=AUX_EVAL_JUDGE_MODEL_FAMILY,
            judge_thinking_mode=AUX_EVAL_JUDGE_THINKING_MODE,
        )

    try:
        aux_dir = run_base / "single_agent_model1_aux_eval"
        aux_dir.mkdir(parents=True, exist_ok=True)

        scored_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        unlabeled_rows: List[Dict[str, Any]] = []

        bar = tqdm(total=len(rows), desc="single_agent_model1_aux_eval", unit="example", dynamic_ncols=True, leave=False)
        for idx, (row, score) in enumerate(zip(rows, model1_scores), start=1):
            try:
                scored = evaluate_saved_row(benchmark, row, judge_runtime, judge_sampling)
                scored["model1_aux_prob_correct"] = float(score)
                scored_rows.append(scored)

                if scored.get("benchmark_correct") is None:
                    unlabeled_rows.append({
                        "row_index": idx - 1,
                        "example_id": row.get("example_id", row.get("pid", row.get("question_id", row.get("id", idx - 1)))),
                        "reason": "benchmark_correct is None",
                        "pred_parsed": scored.get("pred_parsed"),
                        "pred_ans_preview": scored.get("pred_ans_preview"),
                        "gold_ans_preview": scored.get("gold_ans_preview"),
                        "label_source": scored.get("label_source"),
                        "raw_response": row.get("raw_response"),
                    })
            except Exception as e:
                skipped_rows.append({
                    "row_index": idx - 1,
                    "example_id": row.get("example_id", row.get("pid", row.get("question_id", row.get("id", idx - 1)))),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "raw_response": row.get("raw_response"),
                })
                if debug:
                    debug_print(True, f"[single_agent_model1_aux_eval] skipping row {idx - 1}: {type(e).__name__}: {e}")
            bar.update(1)
        bar.close()

        write_jsonl(aux_dir / "model1_aux_scored_rows.jsonl", scored_rows)
        save_rows_to_parquet(aux_dir / "model1_aux_scored_rows.parquet", scored_rows, debug=debug)
        json_dump(aux_dir / "model1_aux_skipped_rows.json", skipped_rows)
        json_dump(aux_dir / "model1_aux_unlabeled_rows.json", unlabeled_rows)

        labeled_rows = [r for r in scored_rows if r.get("benchmark_correct") is not None]
        if not labeled_rows:
            json_dump(aux_dir / "model1_aux_metrics.json", {
                "num_rows": len(scored_rows),
                "num_labeled": 0,
                "num_unlabeled": len(unlabeled_rows),
                "num_skipped": len(skipped_rows),
                "error": "No labeled rows available for aux metrics.",
            })
            (aux_dir / "model1_aux_metrics.txt").write_text(
                "num_rows: {rows}\nnum_labeled: 0\nnum_unlabeled: {unlabeled}\nnum_skipped: {skipped}\nNo labeled rows available for aux metrics.\n".format(
                    rows=len(scored_rows), unlabeled=len(unlabeled_rows), skipped=len(skipped_rows)
                ),
                encoding="utf-8",
            )
            return

        y_true = [int(r["benchmark_correct"]) for r in labeled_rows]
        y_prob = [float(r["model1_aux_prob_correct"]) for r in labeled_rows]

        metrics = compute_all_binary_metrics(y_true, y_prob)
        metrics["num_rows"] = len(scored_rows)
        metrics["num_labeled"] = len(labeled_rows)
        metrics["num_unlabeled"] = len(unlabeled_rows)
        metrics["num_skipped"] = len(skipped_rows)

        plot_paths = save_probability_distribution_plots(aux_dir, y_true, y_prob, prefix="model1_aux")
        save_threshold_sweep_json(aux_dir / "model1_aux_threshold_sweep.json", y_true, y_prob)

        json_dump(aux_dir / "model1_aux_metrics.json", metrics)
        (aux_dir / "model1_aux_metrics.txt").write_text(metrics_report_text(metrics), encoding="utf-8")
        json_dump(aux_dir / "model1_aux_plot_paths.json", plot_paths)
    finally:
        if judge_runtime is not None:
            judge_runtime.unload(drop_processor=False)
        release_cuda_memory()

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    global MODEL1_MODEL_FAMILY, MODEL1_THINKING_MODE, MODEL2_MODEL_FAMILY, MODEL2_THINKING_MODE
    global MODEL1_NAME_OR_PATH, MODEL2_NAME_OR_PATH, MODEL1_AUX_HEAD_CKPT, MODEL2_AUX_THRESHOLD, OUTPUT_BASE_ROOT
    global AUX_EVAL_JUDGE_MODEL_NAME_OR_PATH, AUX_EVAL_JUDGE_MODEL_FAMILY, AUX_EVAL_JUDGE_THINKING_MODE
    args = parse_args()
    MODEL1_NAME_OR_PATH = args.model1_name_or_path
    MODEL2_NAME_OR_PATH = args.model2_name_or_path
    MODEL1_AUX_HEAD_CKPT = args.model1_aux_head_ckpt
    MODEL2_AUX_THRESHOLD = args.model2_aux_threshold
    OUTPUT_BASE_ROOT = args.output_base_root
    MODEL1_VLLM_BASE["revision"] = args.model1_revision
    AUX_EVAL_JUDGE_MODEL_NAME_OR_PATH = args.judge_model_name_or_path
    AUX_EVAL_JUDGE_MODEL_FAMILY = args.judge_model_family
    AUX_EVAL_JUDGE_THINKING_MODE = args.judge_thinking_mode
    MODEL1_MODEL_FAMILY = str(args.model1_model_family)
    MODEL1_THINKING_MODE = str(args.model1_thinking_mode)
    MODEL2_MODEL_FAMILY = str(args.model2_model_family)
    MODEL2_THINKING_MODE = str(args.model2_thinking_mode)
    global CURRENT_BENCHMARK
    benchmark = args.benchmark
    CURRENT_BENCHMARK = benchmark
    overwrite = bool(args.overwrite or OVERWRITE)
    resume = bool(RESUME_MODE and not args.no_resume and not overwrite)
    debug = bool(args.debug or DEBUG_MODE)
    shared_single_agent_cache_enabled = bool(
        SHARED_SINGLE_AGENT_CACHE_ENABLED and (not args.no_shared_single_agent_cache)
    )
    threshold_list = _parse_threshold_list(args.model1_aux_thresholds)

    set_seed(SEED)
    examples = load_examples_for_benchmark(benchmark, DATASET_CONFIGS[benchmark])
    if args.max_examples is not None:
        if int(args.max_examples) <= 0:
            raise ValueError("--max_examples must be positive")
        examples = examples[: int(args.max_examples)]

    requested_strategy_names = resolve_requested_strategy_names(args.strategy_names)

    threshold_roots = [threshold_output_root(benchmark, threshold_list, thr) for thr in threshold_list]
    for root in threshold_roots:
        prepare_output_root(root, overwrite=overwrite, resume=resume, debug=debug)

    run_base = build_output_base(benchmark, threshold_list)
    run_base.mkdir(parents=True, exist_ok=True)

    json_dump(run_base / "generation_run_config.json", {
        "benchmark": benchmark,
        "example_commands": [line.strip() for line in EXAMPLE_COMMANDS.strip().splitlines() if line.strip()],
        "seed": SEED,
        "debug_mode": debug,
        "debug_max_chars": DEBUG_MAX_CHARS,
        "resume_mode": resume,
        "strategy_names": args.strategy_names,
        "resolved_strategy_names": requested_strategy_names,
        "model1_aux_thresholds": threshold_list,
        "model2_aux_threshold": MODEL2_AUX_THRESHOLD,
        "chunk_tokens": CHUNK_TOKENS,
        "prefix_handoff_tokens": PREFIX_HANDOFF_TOKENS,
        "model1_name_or_path": MODEL1_NAME_OR_PATH,
        "model2_name_or_path": MODEL2_NAME_OR_PATH,
        "model1_model_family": MODEL1_MODEL_FAMILY,
        "model1_thinking_mode": MODEL1_THINKING_MODE,
        "model2_model_family": MODEL2_MODEL_FAMILY,
        "model2_thinking_mode": MODEL2_THINKING_MODE,
        "model1_aux_head_ckpt": MODEL1_AUX_HEAD_CKPT,
        "model2_aux_head_ckpt": MODEL2_AUX_HEAD_CKPT,
        "aux_eval_judge_model_name_or_path": AUX_EVAL_JUDGE_MODEL_NAME_OR_PATH,
        "aux_eval_judge_model_family": AUX_EVAL_JUDGE_MODEL_FAMILY,
        "aux_eval_judge_thinking_mode": AUX_EVAL_JUDGE_THINKING_MODE,
        "aux_eval_judge_runtime_profile": AUX_EVAL_JUDGE_RUNTIME_PROFILE,
        "aux_eval_judge_sampling_profiles": AUX_EVAL_JUDGE_SAMPLING_PROFILES,
        "dataset_cfg": DATASET_CONFIGS[benchmark],
        "model1_vllm_base": MODEL1_VLLM_BASE,
        "model2_vllm_base": MODEL2_VLLM_BASE,
        "aux_profile": AUX_PROFILE,
        "sampling_profiles": SAMPLING_PROFILES,
        "model_sampling_overrides": MODEL_SAMPLING_OVERRIDES,
        "strategy_execution": STRATEGY_EXECUTION,
    })

    per_threshold_suite_summary: Dict[float, Dict[str, Dict[str, Any]]] = {thr: {} for thr in threshold_list}
    pending_by_threshold: Dict[float, List[str]] = {thr: [] for thr in threshold_list}
    for thr, root in zip(threshold_list, threshold_roots):
        for strategy_name in requested_strategy_names:
            strategy_dir = threshold_strategy_dir(root, strategy_name)
            if resume:
                is_complete, existing_summary = get_completed_strategy_summary(strategy_dir, len(examples), debug=debug)
                if is_complete and existing_summary is not None:
                    per_threshold_suite_summary[thr][strategy_name] = existing_summary
                    tqdm.write(f"[GEN][RESUME] Skipping completed strategy: thr1={format_threshold_value(thr)} | {strategy_name}")
                    continue
            pending_by_threshold[thr].append(strategy_name)

    all_pending = sorted(set(name for names in pending_by_threshold.values() for name in names))
    if not all_pending:
        for thr, root in zip(threshold_list, threshold_roots):
            ensure_suite_summary_written(root, per_threshold_suite_summary[thr], debug=debug)
        print(json.dumps({format_threshold_value(k): v for k, v in per_threshold_suite_summary.items()}, ensure_ascii=False, indent=2))
        return

    reusable_m1_results: Optional[List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]] = None
    reusable_m2_results: Optional[List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]] = None
    reusable_m1_prefix_results_by_tokens: Dict[int, List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]]] = {}
    model1_scores: Optional[List[float]] = None
    model1_has_final_answers: Optional[List[bool]] = None
    model1_final_answer_reasons: Optional[List[str]] = None
    model1_prefix_scores_by_tokens: Dict[int, List[float]] = {}

    needs_full_m1 = ("single_agent_model1" in all_pending) or any(strategy_reuses_model1_first_pass(name) for name in all_pending)
    needs_full_m1_aux = any(strategy_uses_full_m1_aux_threshold(name) for name in all_pending)
    needs_full_m2 = ("single_agent_model2" in all_pending) or any(strategy_reuses_model2_single_pass(name) for name in all_pending)
    requested_prefix_token_values = sorted({
        int(info["prefix_tokens"])
        for name in all_pending
        for info in [(_parse_prefix_handoff_strategy_name(name) or {})]
        if info
    })
    needs_prefix_handoff = bool(requested_prefix_token_values)

    cached_m1_rows: Optional[List[Dict[str, Any]]] = None
    cached_m2_rows: Optional[List[Dict[str, Any]]] = None

    if needs_full_m1:
        if shared_single_agent_cache_enabled:
            precompute_model1_bundle = build_model_bundle(
                model_name_or_path=MODEL1_NAME_OR_PATH,
                aux_head_ckpt=MODEL1_AUX_HEAD_CKPT,
                runtime_profile=_apply_runtime_override(MODEL1_VLLM_BASE, PRECOMPUTE_MODEL1_GPU_UTIL, PRECOMPUTE_MODEL1_MAX_NUM_SEQS),
                sampling_profiles=SAMPLING_PROFILES,
                aux_profile=AUX_PROFILE,
                sampling_override=MODEL_SAMPLING_OVERRIDES.get("model1"),
                model_family=MODEL1_MODEL_FAMILY,
                thinking_mode=MODEL1_THINKING_MODE,
            )
            cached_m1_rows = try_load_shared_single_agent_cache(
                benchmark=benchmark,
                examples=examples,
                slot="model1",
                model_name_or_path=MODEL1_NAME_OR_PATH,
                model_family=MODEL1_MODEL_FAMILY,
                thinking_mode=MODEL1_THINKING_MODE,
                sampling_cfg=precompute_model1_bundle.sampling_cfg,
                debug=debug,
            )
            if cached_m1_rows is not None:
                reusable_m1_results = _rows_to_result_tuples(cached_m1_rows)
        if reusable_m1_results is None:
            reusable_m1_results = precompute_model1_first_pass(benchmark, examples, debug=debug)
            if shared_single_agent_cache_enabled:
                if cached_m1_rows is None:
                    cached_m1_rows = build_rows_from_results(benchmark, examples, "single_agent_model1", reusable_m1_results)
                save_shared_single_agent_cache(
                    benchmark=benchmark,
                    examples=examples,
                    slot="model1",
                    model_name_or_path=MODEL1_NAME_OR_PATH,
                    model_family=MODEL1_MODEL_FAMILY,
                    thinking_mode=MODEL1_THINKING_MODE,
                    sampling_cfg=precompute_model1_bundle.sampling_cfg,
                    rows=cached_m1_rows,
                    debug=debug,
                )
    if needs_full_m1_aux:
        if reusable_m1_results is None:
            raise RuntimeError("Internal error: reusable_m1_results missing before full-threshold aux scoring")
        model1_scores, model1_has_final_answers, model1_final_answer_reasons = precompute_model1_aux_scores_and_answer_flags(benchmark, examples, reusable_m1_results, debug=debug)
    if needs_full_m2:
        if shared_single_agent_cache_enabled:
            precompute_model2_bundle = build_model_bundle(
                model_name_or_path=MODEL2_NAME_OR_PATH,
                aux_head_ckpt=MODEL2_AUX_HEAD_CKPT,
                runtime_profile=_apply_runtime_override(MODEL2_VLLM_BASE, get_strategy_exec_cfg("single_agent_model2")["model2_gpu_memory_utilization"], get_strategy_exec_cfg("single_agent_model2")["model2_max_num_seqs"]),
                sampling_profiles=SAMPLING_PROFILES,
                aux_profile=AUX_PROFILE,
                sampling_override=MODEL_SAMPLING_OVERRIDES.get("model2"),
                model_family=MODEL2_MODEL_FAMILY,
                thinking_mode=MODEL2_THINKING_MODE,
            )
            cached_m2_rows = try_load_shared_single_agent_cache(
                benchmark=benchmark,
                examples=examples,
                slot="model2",
                model_name_or_path=MODEL2_NAME_OR_PATH,
                model_family=MODEL2_MODEL_FAMILY,
                thinking_mode=MODEL2_THINKING_MODE,
                sampling_cfg=precompute_model2_bundle.sampling_cfg,
                debug=debug,
            )
            if cached_m2_rows is not None:
                reusable_m2_results = _rows_to_result_tuples(cached_m2_rows)
        if reusable_m2_results is None:
            reusable_m2_results = precompute_model2_single_pass(benchmark, examples, debug=debug)
            if shared_single_agent_cache_enabled:
                if cached_m2_rows is None:
                    cached_m2_rows = build_rows_from_results(benchmark, examples, "single_agent_model2", reusable_m2_results)
                save_shared_single_agent_cache(
                    benchmark=benchmark,
                    examples=examples,
                    slot="model2",
                    model_name_or_path=MODEL2_NAME_OR_PATH,
                    model_family=MODEL2_MODEL_FAMILY,
                    thinking_mode=MODEL2_THINKING_MODE,
                    sampling_cfg=precompute_model2_bundle.sampling_cfg,
                    rows=cached_m2_rows,
                    debug=debug,
                )
    if needs_prefix_handoff:
        for prefix_tokens in requested_prefix_token_values:
            reusable_prefix = precompute_model1_prefix_pass(benchmark, examples, debug=debug, prefix_tokens=prefix_tokens)
            reusable_m1_prefix_results_by_tokens[int(prefix_tokens)] = reusable_prefix
            model1_prefix_scores_by_tokens[int(prefix_tokens)] = precompute_model1_prefix_aux_scores(
                benchmark,
                examples,
                reusable_prefix,
                debug=debug,
                prefix_tokens=prefix_tokens,
            )

    if "single_agent_model1" in all_pending:
        if reusable_m1_results is not None:
            m1_single_results = [clone_result_tuple(x) for x in reusable_m1_results]
        else:
            m1_single_results = run_single_agent_strategy_once(benchmark, examples, "single_agent_model1", debug=debug)

        if MODEL1_AUX_HEAD_CKPT and model1_scores is None:
            model1_scores, model1_has_final_answers, model1_final_answer_reasons = precompute_model1_aux_scores_and_answer_flags(benchmark, examples, m1_single_results, debug=debug)

        if cached_m1_rows is not None and len(cached_m1_rows) == len(examples):
            m1_rows = copy.deepcopy(cached_m1_rows)
        else:
            m1_rows = build_rows_from_results(benchmark, examples, "single_agent_model1", m1_single_results)
        for thr, root in zip(threshold_list, threshold_roots):
            if "single_agent_model1" in pending_by_threshold[thr]:
                save_strategy_outputs(threshold_strategy_dir(root, "single_agent_model1"), benchmark, "single_agent_model1", m1_rows, per_threshold_suite_summary[thr], debug=debug)

        if MODEL1_AUX_HEAD_CKPT and model1_scores is not None:
            run_aux_eval_for_single_agent_model1(run_base, benchmark, m1_rows, model1_scores, debug=debug)

    if "single_agent_model2" in all_pending:
        if reusable_m2_results is not None:
            m2_single_results = [clone_result_tuple(x) for x in reusable_m2_results]
        else:
            m2_single_results = run_single_agent_strategy_once(benchmark, examples, "single_agent_model2", debug=debug)
        if cached_m2_rows is not None and len(cached_m2_rows) == len(examples):
            m2_rows = copy.deepcopy(cached_m2_rows)
        else:
            m2_rows = build_rows_from_results(benchmark, examples, "single_agent_model2", m2_single_results)
        for thr, root in zip(threshold_list, threshold_roots):
            if "single_agent_model2" in pending_by_threshold[thr]:
                save_strategy_outputs(threshold_strategy_dir(root, "single_agent_model2"), benchmark, "single_agent_model2", m2_rows, per_threshold_suite_summary[thr], debug=debug)

    dependent_strategy_names = [name for name in requested_strategy_names if strategy_reuses_model1_first_pass(name)]
    if dependent_strategy_names:
        sorted_thresholds = list(sorted(threshold_list))
        for strategy_name in dependent_strategy_names:
            if not any(strategy_name in pending_by_threshold[thr] for thr in threshold_list):
                continue

            tqdm.write(f"[GEN] Sweeping thresholds for strategy: {strategy_name}")
            branch_cache: Dict[int, Tuple[str, str, Dict[str, Any], List[Dict[str, Any]], float]] = {}
            already_materialized: set[int] = set()

            if strategy_uses_prefix_handoff(strategy_name):
                prefix_info = _parse_prefix_handoff_strategy_name(strategy_name)
                if reusable_m1_results is None or prefix_info is None:
                    raise RuntimeError("Internal error: missing reusable model1 full results for prefix handoff strategy")
                prefix_tokens = int(prefix_info["prefix_tokens"])
                reusable_m1_prefix_results = reusable_m1_prefix_results_by_tokens.get(prefix_tokens)
                model1_prefix_scores = model1_prefix_scores_by_tokens.get(prefix_tokens)
                if reusable_m1_prefix_results is None or model1_prefix_scores is None:
                    raise RuntimeError(f"Internal error: missing reusable model1 prefix results/scores for prefix_tokens={prefix_tokens}")
                for thr in sorted_thresholds:
                    needed_indices = [
                        idx
                        for idx, (prefix_result, score) in enumerate(zip(reusable_m1_prefix_results, model1_prefix_scores))
                        if should_branch_model1_prefix(prefix_result[1], int(prefix_result[2]["model1"].completion_tokens), float(score), float(thr), benchmark, prefix_tokens)
                    ]
                    new_indices = [idx for idx in needed_indices if idx not in already_materialized]
                    if new_indices:
                        tqdm.write(f"[GEN] Strategy={strategy_name} | thr1={format_threshold_value(thr)} | prefix_tokens={prefix_tokens} | computing new routed examples={len(new_indices)}")
                        branch_cache.update(compute_new_prefix_handoff_results(benchmark, examples, reusable_m1_prefix_results, model1_prefix_scores, new_indices, thr, strategy_name, debug=debug, prefix_tokens=prefix_tokens))
                        already_materialized.update(new_indices)
                    if strategy_name not in pending_by_threshold[thr]:
                        continue
                    threshold_results = materialize_threshold_results_prefix(benchmark, strategy_name, thr, reusable_m1_results, reusable_m1_prefix_results, model1_prefix_scores, branch_cache, prefix_tokens=prefix_tokens)
                    threshold_rows = build_rows_from_results(benchmark, examples, strategy_name, threshold_results)
                    root = threshold_output_root(benchmark, threshold_list, thr)
                    save_strategy_outputs(threshold_strategy_dir(root, strategy_name), benchmark, strategy_name, threshold_rows, per_threshold_suite_summary[thr], debug=debug)
                continue

            if reusable_m1_results is None or model1_scores is None:
                raise RuntimeError("Internal error: missing reusable_m1_results/model1_scores for threshold-dependent strategies")

            for thr in sorted_thresholds:
                needed_indices = [
                    idx
                    for idx, score in enumerate(model1_scores)
                    if should_branch_model1_first_pass(benchmark, reusable_m1_results[idx][1], float(score), float(thr))
                ]
                new_indices = [idx for idx in needed_indices if idx not in already_materialized]

                if new_indices:
                    tqdm.write(f"[GEN] Strategy={strategy_name} | thr1={format_threshold_value(thr)} | computing new routed examples={len(new_indices)}")
                    if strategy_name == "m1_after_finish_self_repair":
                        branch_cache.update(compute_new_self_repair_results(benchmark, examples, reusable_m1_results, model1_scores, new_indices, thr, debug=debug))
                    elif strategy_name == "m1_after_finish_retry":
                        branch_cache.update(compute_new_retry_results(benchmark, examples, reusable_m1_results, model1_scores, new_indices, thr, debug=debug))
                    else:
                        branch_cache.update(compute_new_handoff_results(benchmark, examples, reusable_m1_results, model1_scores, new_indices, thr, strategy_name, debug=debug, reusable_m2_results=reusable_m2_results))
                    already_materialized.update(new_indices)

                if strategy_name not in pending_by_threshold[thr]:
                    continue

                threshold_results = materialize_threshold_results(benchmark, strategy_name, thr, reusable_m1_results, model1_scores, branch_cache)
                threshold_rows = build_rows_from_results(benchmark, examples, strategy_name, threshold_results)
                root = threshold_output_root(benchmark, threshold_list, thr)
                save_strategy_outputs(threshold_strategy_dir(root, strategy_name), benchmark, strategy_name, threshold_rows, per_threshold_suite_summary[thr], debug=debug)

    for thr, root in zip(threshold_list, threshold_roots):
        ensure_suite_summary_written(root, per_threshold_suite_summary[thr], debug=debug)

    print(json.dumps({format_threshold_value(k): v for k, v in per_threshold_suite_summary.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
