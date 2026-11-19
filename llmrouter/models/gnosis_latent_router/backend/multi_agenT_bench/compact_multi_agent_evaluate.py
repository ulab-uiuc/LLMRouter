#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

"""
Recursive batched evaluation runner for compact multi-agent benchmark outputs.

Usage examples
--------------
python compact_multi_agent_evaluate_auto_hardcoded_batched.py
python compact_multi_agent_evaluate_auto_hardcoded_batched.py --debug
python compact_multi_agent_evaluate_auto_hardcoded_batched.py --strategy_names single_agent_model1,m1_after_finish_retry

What it does
------------
- Takes a single dataset results root.
- Recursively discovers every strategy directory that contains results.jsonl.
- Evaluates all discovered runs, including different threshold subfolders / modes.
- Uses batched judge generation for judge-based benchmarks to speed up evaluation.
- Saves per-strategy scored JSONL/parquet, skipped rows, and summaries.
- Saves one suite summary per parent folder, plus one root-level master summary.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from llmrouter.models.gnosis_latent_router.backend.multi_agenT_bench.compact_multi_agent_shared_optimized_v4_textbench import (
    benchmark_needs_judge,
    build_judge_runtime_and_sampling,
    debug_print,
    finish_evaluation_row,
    prepare_evaluation_row,
    json_dump,
    load_jsonl,
    summarize_scored_rows,
    response_has_usable_final_answer,
    write_jsonl,
)


# =============================================================================
# HARD-CODED CONFIG BLOCK
# =============================================================================

DEFAULT_RESULTS_ROOT = "eval_outputs/multi_agent_compact/qwen__qwen3_vl_2b_thinking__qwen__qwen3_vl_32b_thinking_fp8__thr1s_0p50_0p60_0p70_0p80_0p90__thr2_0p80/triviaqa_split"
BENCHMARK = "triviaqa"
STRATEGY_NAMES = ""
JUDGE_MODEL_NAME_OR_PATH = "Qwen/Qwen3-VL-8B-Instruct"
JUDGE_MODEL_FAMILY = "auto"
JUDGE_THINKING_MODE = "auto"
DEBUG_MODE = False

# Batched judge settings.
JUDGE_EVAL_BATCH_SIZE = 32

JUDGE_RUNTIME_PROFILE = {
    "dtype": "bfloat16",
    "max_model_len": 8192,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.40,
    "max_num_seqs": 32,
    "enforce_eager": False,
    "trust_remote_code": True,
}

JUDGE_DEFAULT_SAMPLING_PROFILE = {
    "temperature": 0.6,
    "top_p": 1.0,
    "max_new_tokens": 3000,
}
JUDGE_THINKING_SAMPLING_PROFILE = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 2000,
}
JUDGE_INSTRUCT_SAMPLING_PROFILE = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
}

KNOWN_BENCHMARKS = {"mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa", "triviaqa", "math", "mmlu_pro"}

# =============================================================================
# END CONFIG BLOCK
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help="Path to a dataset results root (for example .../simplevqa_split). The script will recurse automatically.",
    )
    ap.add_argument(
        "--benchmark",
        type=str,
        default=BENCHMARK,
        choices=["auto", "mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa", "triviaqa", "math", "mmlu_pro"],
    )
    ap.add_argument(
        "--strategy_names",
        type=str,
        default=STRATEGY_NAMES,
        help="Optional comma-separated strategy folder names to evaluate. Empty means all discovered strategy dirs.",
    )
    ap.add_argument(
        "--judge_batch_size",
        type=int,
        default=JUDGE_EVAL_BATCH_SIZE,
        help="Number of judge prompts to send per batched vLLM call.",
    )
    ap.add_argument("--judge_model_name_or_path", type=str, default=JUDGE_MODEL_NAME_OR_PATH)
    ap.add_argument("--judge_model_family", type=str, default=JUDGE_MODEL_FAMILY, choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--judge_thinking_mode", type=str, default=JUDGE_THINKING_MODE, choices=["auto", "on", "off"])
    ap.add_argument("--model1_params", type=float, default=0.0, help="Model1 parameter count for FLOPs estimation. 0 means infer automatically when possible.")
    ap.add_argument("--model2_params", type=float, default=0.0, help="Model2 parameter count for FLOPs estimation. 0 means infer automatically when possible.")
    ap.add_argument("--skip_auto_summary", action="store_true", help="Skip writing the post-run requested summary tables.")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


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


def parse_csv_names(csv: str) -> Optional[set[str]]:
    items = [x.strip() for x in str(csv or "").split(",") if x.strip()]
    return set(items) if items else None


def discover_strategy_dirs(results_root: Path) -> List[Path]:
    if not results_root.exists():
        raise RuntimeError(f"results_root does not exist: {results_root}")
    if results_root.is_file():
        raise RuntimeError(f"results_root must be a directory, got file: {results_root}")

    dirs = sorted({p.parent.resolve() for p in results_root.rglob("results.jsonl")})
    if not dirs:
        raise RuntimeError(f"No results.jsonl files found under: {results_root}")
    return dirs


def infer_benchmark_from_rows(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    for row in rows:
        value = row.get("benchmark")
        if isinstance(value, str) and value.strip() in KNOWN_BENCHMARKS:
            return value.strip()
    return None


def infer_benchmark(results_root: Path, strategy_dirs: Sequence[Path], benchmark_arg: str) -> str:
    if benchmark_arg != "auto":
        return benchmark_arg

    for part in [results_root.name, *reversed(results_root.parts)]:
        p = str(part)
        if p.endswith("_split"):
            cand = p[: -len("_split")]
            if cand in KNOWN_BENCHMARKS:
                return cand
        if p in KNOWN_BENCHMARKS:
            return p

    for strategy_dir in strategy_dirs:
        rows = load_jsonl(strategy_dir / "results.jsonl")
        benchmark = infer_benchmark_from_rows(rows)
        if benchmark is not None:
            return benchmark

    raise RuntimeError("Could not infer benchmark automatically. Pass --benchmark explicitly.")


def group_strategy_dirs_by_suite_root(strategy_dirs: Sequence[Path]) -> List[Tuple[Path, List[Path]]]:
    grouped: Dict[Path, List[Path]] = defaultdict(list)
    for strategy_dir in strategy_dirs:
        grouped[strategy_dir.parent].append(strategy_dir)
    return sorted((root, sorted(dirs)) for root, dirs in grouped.items())


def safe_summarize_scored_rows(
    benchmark: str,
    strategy_name: str,
    scored_rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if scored_rows:
        summary = summarize_scored_rows(benchmark, scored_rows)
        if not isinstance(summary, dict):
            raise RuntimeError(f"summarize_scored_rows returned non-dict: {type(summary)}")
    else:
        summary = {
            "benchmark": benchmark,
            "strategy_name": strategy_name,
            "num_rows": 0,
            "error": "All rows failed judge/parsing and were skipped.",
        }
    summary["benchmark"] = benchmark
    summary["strategy_name"] = strategy_name
    summary["num_scored"] = len(scored_rows)
    summary["num_skipped"] = len(skipped_rows)
    return summary


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _usage_value(row: Dict[str, Any], model_name: str, key: str) -> int:
    try:
        return int(row.get("usage_by_model", {}).get(model_name, {}).get(key, 0) or 0)
    except Exception:
        return 0


def _judge_usage_value(row: Dict[str, Any], key: str) -> int:
    try:
        return int(row.get("judge_usage", {}).get(key, 0) or 0)
    except Exception:
        return 0


def _trace_events(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    trace = row.get("trace", [])
    return trace if isinstance(trace, list) else []


def _row_has_trace(row: Dict[str, Any], *, events: Optional[set[str]] = None, decisions: Optional[set[str]] = None) -> bool:
    for item in _trace_events(row):
        if not isinstance(item, dict):
            continue
        if events is not None and str(item.get("event")) in events:
            return True
        if decisions is not None and str(item.get("decision")) in decisions:
            return True
    return False


def _row_has_routing_reason(row: Dict[str, Any], substr: str) -> bool:
    needle = str(substr)
    for item in _trace_events(row):
        if not isinstance(item, dict):
            continue
        value = str(item.get("routing_reason", ""))
        if needle in value:
            return True
    return False


def benchmark_primary_metric_name(benchmark: str) -> str:
    if benchmark in {"mathvista", "mathverse", "triviaqa", "math", "mmlu_pro"}:
        return "accuracy"
    if benchmark == "screenspot_pro":
        return "action_acc"
    if benchmark in {"simplevqa", "charxiv_reasoning"}:
        return "is_correct"
    return "benchmark_correct"


def benchmark_primary_metric_value(benchmark: str, summary: Dict[str, Any]) -> float:
    if benchmark in {"mathvista", "mathverse", "triviaqa", "math", "mmlu_pro"}:
        return float(summary.get("accuracy", 0.0) or 0.0)
    if benchmark == "screenspot_pro":
        return float(summary.get("overall", {}).get("action_acc", 0.0) or 0.0)
    if benchmark in {"simplevqa", "charxiv_reasoning"}:
        return float(summary.get("is_correct", 0.0) or 0.0)
    return float(summary.get("benchmark_correct", 0.0) or 0.0)


def _dataset_tag_for_filename(benchmark: str) -> str:
    tag = str(benchmark or "dataset").strip().lower()
    tag = re.sub(r"[^a-z0-9_]+", "_", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag or "dataset"


def _strategy_summary_json_path(strategy_dir: Path, benchmark: str) -> Path:
    return strategy_dir / f"{_dataset_tag_for_filename(benchmark)}_summary_scored.json"


def _suite_summary_json_path(suite_root: Path, benchmark: str) -> Path:
    return suite_root / f"{_dataset_tag_for_filename(benchmark)}_suite_summary_scored.json"


def _suite_comparison_json_path(suite_root: Path, benchmark: str) -> Path:
    return suite_root / f"{_dataset_tag_for_filename(benchmark)}_suite_comparison_scored.json"


def _suite_summary_parquet_path(suite_root: Path, benchmark: str) -> Path:
    return suite_root / f"{_dataset_tag_for_filename(benchmark)}_suite_summary_scored.parquet"


def _suite_comparison_parquet_path(suite_root: Path, benchmark: str) -> Path:
    return suite_root / f"{_dataset_tag_for_filename(benchmark)}_suite_comparison_scored.parquet"


def _all_suite_summaries_json_path(results_root: Path, benchmark: str) -> Path:
    return results_root / f"{_dataset_tag_for_filename(benchmark)}_all_suite_summaries_scored.json"


def _all_suite_summaries_parquet_path(results_root: Path, benchmark: str) -> Path:
    return results_root / f"{_dataset_tag_for_filename(benchmark)}_all_suite_summaries_scored.parquet"


def _all_suite_comparisons_parquet_path(results_root: Path, benchmark: str) -> Path:
    return results_root / f"{_dataset_tag_for_filename(benchmark)}_all_suite_comparisons_scored.parquet"


def parse_threshold_tags(suite_root_name: str) -> Dict[str, Any]:
    m = re.match(r"^thr1_([^_]+(?:__[^_]+)?)__thr2_([^_]+(?:__[^_]+)?)$", str(suite_root_name))
    if not m:
        return {"suite_root_name": str(suite_root_name), "thr1_tag": None, "thr2_tag": None, "thr1": None, "thr2": None}
    thr1_tag = m.group(1)
    thr2_tag = m.group(2)
    def _parse(tag: str):
        try:
            return float(str(tag).replace("p", "."))
        except Exception:
            return None
    return {
        "suite_root_name": str(suite_root_name),
        "thr1_tag": thr1_tag,
        "thr2_tag": thr2_tag,
        "thr1": _parse(thr1_tag),
        "thr2": _parse(thr2_tag),
    }


def summarize_strategy_cost_and_routing(benchmark: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    wall_total = sum(float(r.get("wall_time_sec", 0.0) or 0.0) for r in rows)

    model_usage_totals: Dict[str, Dict[str, int]] = {}
    for model_name in ("model1", "model2"):
        prompt_tokens = sum(_usage_value(r, model_name, "prompt_tokens") for r in rows)
        completion_tokens = sum(_usage_value(r, model_name, "completion_tokens") for r in rows)
        aux_scored_tokens = sum(_usage_value(r, model_name, "aux_scored_tokens") for r in rows)
        generation_calls = sum(_usage_value(r, model_name, "generation_calls") for r in rows)
        aux_calls = sum(_usage_value(r, model_name, "aux_calls") for r in rows)
        generation_time_sec = sum(float(r.get("usage_by_model", {}).get(model_name, {}).get("generation_time_sec", 0.0) or 0.0) for r in rows)
        model_usage_totals[model_name] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_tokens": prompt_tokens + completion_tokens,
            "aux_scored_tokens": aux_scored_tokens,
            "generation_calls": generation_calls,
            "aux_calls": aux_calls,
            "generation_time_sec": generation_time_sec,
            "total_token_touches": prompt_tokens + completion_tokens + aux_scored_tokens,
        }

    judge_prompt_tokens = sum(_judge_usage_value(r, "prompt_tokens") for r in rows)
    judge_completion_tokens = sum(_judge_usage_value(r, "completion_tokens") for r in rows)
    judge_calls = sum(1 for r in rows if isinstance(r.get("judge_usage"), dict))

    generation_token_total = sum(v["generation_tokens"] for v in model_usage_totals.values())
    aux_scored_token_total = sum(v["aux_scored_tokens"] for v in model_usage_totals.values())
    judge_token_total = judge_prompt_tokens + judge_completion_tokens
    total_compute_tokens = generation_token_total + aux_scored_token_total + judge_token_total
    benchmark_correct_total = sum(int(r.get("benchmark_correct", 0) or 0) for r in rows)

    final_model_counts = {
        "model1": sum(1 for r in rows if str(r.get("final_model_name")) == "model1"),
        "model2": sum(1 for r in rows if str(r.get("final_model_name")) == "model2"),
    }

    retry_count = sum(1 for r in rows if _row_has_trace(r, events={"retry_generation"}, decisions={"retry"}))
    self_repair_count = sum(1 for r in rows if _row_has_trace(r, events={"self_repair_generation"}, decisions={"self_repair"}))
    handoff_count = sum(1 for r in rows if _row_has_trace(r, events={"handoff_generation", "handoff"}, decisions={"handoff"}))
    any_branch_count = sum(1 for r in rows if _row_has_trace(r, events={"retry_generation", "self_repair_generation", "handoff_generation", "handoff"}, decisions={"retry", "self_repair", "handoff"}))
    accepted_first_pass_count = sum(1 for r in rows if not _row_has_trace(r, events={"retry_generation", "self_repair_generation", "handoff_generation", "handoff"}, decisions={"retry", "self_repair", "handoff"}))

    low_aux_trigger_count = sum(1 for r in rows if _row_has_routing_reason(r, "low_aux") and _row_has_trace(r, decisions={"retry", "self_repair", "handoff"}))
    missing_final_trigger_count = sum(1 for r in rows if _row_has_routing_reason(r, "missing_final_answer") and _row_has_trace(r, decisions={"retry", "self_repair", "handoff"}))
    no_final_answer_count = sum(1 for r in rows if not response_has_usable_final_answer(benchmark, r.get("raw_response", "")))

    averages_by_model = {
        model_name: {
            k.replace("tokens", "tokens_per_row").replace("calls", "calls_per_row"): _safe_div(v, total)
            if k != "generation_time_sec" else _safe_div(v, total)
            for k, v in model_usage_totals[model_name].items()
        }
        for model_name in ("model1", "model2")
    }
    for model_name in ("model1", "model2"):
        averages_by_model[model_name]["generation_time_sec_per_row"] = _safe_div(model_usage_totals[model_name].get("generation_time_sec", 0.0), total)

    return {
        "total_wall_time_sec": wall_total,
        "avg_wall_time_sec": _safe_div(wall_total, total),
        "usage_totals_by_model": model_usage_totals,
        "usage_averages_by_model": averages_by_model,
        "judge_usage_totals": {
            "prompt_tokens": judge_prompt_tokens,
            "completion_tokens": judge_completion_tokens,
            "judge_tokens": judge_token_total,
            "judge_calls": judge_calls,
        },
        "judge_usage_averages": {
            "prompt_tokens_per_row": _safe_div(judge_prompt_tokens, total),
            "completion_tokens_per_row": _safe_div(judge_completion_tokens, total),
            "judge_tokens_per_row": _safe_div(judge_token_total, total),
            "judge_calls_per_row": _safe_div(judge_calls, total),
        },
        "token_totals": {
            "generation_tokens": generation_token_total,
            "aux_scored_tokens": aux_scored_token_total,
            "judge_tokens": judge_token_total,
            "total_compute_tokens": total_compute_tokens,
        },
        "token_averages": {
            "generation_tokens_per_row": _safe_div(generation_token_total, total),
            "aux_scored_tokens_per_row": _safe_div(aux_scored_token_total, total),
            "judge_tokens_per_row": _safe_div(judge_token_total, total),
            "total_compute_tokens_per_row": _safe_div(total_compute_tokens, total),
        },
        "routing_counts": {
            "retry": retry_count,
            "self_repair": self_repair_count,
            "handoff": handoff_count,
            "any_branch": any_branch_count,
            "accepted_first_pass": accepted_first_pass_count,
            "low_aux_trigger": low_aux_trigger_count,
            "missing_final_answer_trigger": missing_final_trigger_count,
        },
        "routing_rates": {
            "retry_rate": _safe_div(retry_count, total),
            "self_repair_rate": _safe_div(self_repair_count, total),
            "handoff_rate": _safe_div(handoff_count, total),
            "any_branch_rate": _safe_div(any_branch_count, total),
            "accepted_first_pass_rate": _safe_div(accepted_first_pass_count, total),
            "low_aux_trigger_rate": _safe_div(low_aux_trigger_count, total),
            "missing_final_answer_trigger_rate": _safe_div(missing_final_trigger_count, total),
        },
        "final_model_counts": final_model_counts,
        "final_model_rates": {k + "_rate": _safe_div(v, total) for k, v in final_model_counts.items()},
        "output_quality": {
            "num_with_final_answer": total - no_final_answer_count,
            "num_without_final_answer": no_final_answer_count,
            "with_final_answer_rate": _safe_div(total - no_final_answer_count, total),
            "without_final_answer_rate": _safe_div(no_final_answer_count, total),
        },
        "paper_efficiency": {
            "num_correct": benchmark_correct_total,
            "correct_per_1k_generation_tokens": 1000.0 * _safe_div(benchmark_correct_total, generation_token_total),
            "correct_per_1k_total_compute_tokens": 1000.0 * _safe_div(benchmark_correct_total, total_compute_tokens),
            "correct_per_second": _safe_div(benchmark_correct_total, wall_total),
        },
        "avg_generation_tokens_per_row": _safe_div(generation_token_total, total),
        "avg_total_compute_tokens_per_row": _safe_div(total_compute_tokens, total),
        "judge_tokens_total": judge_token_total,
        "generation_tokens_total": generation_token_total,
        "total_compute_tokens_total": total_compute_tokens,
        "model2_final_rate": _safe_div(final_model_counts["model2"], total),
        "no_final_answer_rate": _safe_div(no_final_answer_count, total),
    }


def enrich_strategy_summary(benchmark: str, suite_root: Path, strategy_name: str, scored_rows: List[Dict[str, Any]], skipped_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = safe_summarize_scored_rows(benchmark, strategy_name, scored_rows, skipped_rows)
    summary.update(parse_threshold_tags(suite_root.name))
    summary["suite_root"] = str(suite_root)
    summary["results_root_name"] = str(suite_root.parent.name)
    summary["primary_metric_name"] = benchmark_primary_metric_name(benchmark)
    summary["primary_metric_value"] = benchmark_primary_metric_value(benchmark, summary)
    summary.update(summarize_strategy_cost_and_routing(benchmark, scored_rows))
    return summary


def build_strategy_comparison(current: Dict[str, Any], baseline: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    cur_primary = benchmark_primary_metric_value(benchmark, current)
    base_primary = benchmark_primary_metric_value(benchmark, baseline)
    cur_wall = float(current.get("avg_wall_time_sec", 0.0) or 0.0)
    base_wall = float(baseline.get("avg_wall_time_sec", 0.0) or 0.0)
    cur_gen = float(current.get("avg_generation_tokens_per_row", 0.0) or 0.0)
    base_gen = float(baseline.get("avg_generation_tokens_per_row", 0.0) or 0.0)
    cur_comp = float(current.get("avg_total_compute_tokens_per_row", 0.0) or 0.0)
    base_comp = float(baseline.get("avg_total_compute_tokens_per_row", 0.0) or 0.0)
    cur_nf = float(current.get("no_final_answer_rate", 0.0) or 0.0)
    base_nf = float(baseline.get("no_final_answer_rate", 0.0) or 0.0)
    return {
        "baseline_strategy_name": str(baseline.get("strategy_name")),
        "primary_metric_name": benchmark_primary_metric_name(benchmark),
        "delta_primary_metric": cur_primary - base_primary,
        "relative_primary_metric_change_pct": 100.0 * _safe_div(cur_primary - base_primary, abs(base_primary)) if base_primary != 0 else None,
        "delta_avg_wall_time_sec": cur_wall - base_wall,
        "speedup_vs_baseline": _safe_div(base_wall, cur_wall) if cur_wall > 0 else None,
        "delta_avg_generation_tokens_per_row": cur_gen - base_gen,
        "generation_token_savings_vs_baseline_pct": 100.0 * _safe_div(base_gen - cur_gen, base_gen) if base_gen > 0 else None,
        "delta_avg_total_compute_tokens_per_row": cur_comp - base_comp,
        "total_compute_token_savings_vs_baseline_pct": 100.0 * _safe_div(base_comp - cur_comp, base_comp) if base_comp > 0 else None,
        "delta_no_final_answer_rate": cur_nf - base_nf,
        "better_or_equal_primary_and_lower_or_equal_compute": (cur_primary >= base_primary) and (cur_comp <= base_comp),
        "better_primary_and_lower_wall_time": (cur_primary > base_primary) and (cur_wall < base_wall),
    }


def build_suite_comparison_summary(benchmark: str, suite_summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Dict[str, Any]] = {}
    base_m1 = suite_summary.get("single_agent_model1")
    base_m2 = suite_summary.get("single_agent_model2")
    for strategy_name, summary in suite_summary.items():
        comp: Dict[str, Any] = {}
        if base_m1 is not None and strategy_name != "single_agent_model1":
            comp["vs_single_agent_model1"] = build_strategy_comparison(summary, base_m1, benchmark)
        if base_m2 is not None and strategy_name != "single_agent_model2":
            comp["vs_single_agent_model2"] = build_strategy_comparison(summary, base_m2, benchmark)
        out[strategy_name] = comp
    return out


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_strategy_sort_key(name: str) -> Tuple[int, str]:
    order = {
        "single_agent_model1": 0,
        "single_agent_model2": 1,
        "m1_after_finish_retry": 2,
        "m1_after_finish_self_repair": 3,
        "m1_after_finish_handoff_fresh_m2": 4,
        "m1_after_finish_handoff_context_m2": 5,
    }
    if name in order:
        return (order[name], name)
    m = re.fullmatch(r"m1_after_(\d+)tok_handoff_context_m2(?:_with_m2_aux)?", str(name))
    if m:
        return (100 + int(m.group(1)), str(name))
    return (999, str(name))


def _summary_parse_thr_dir_name(name: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.fullmatch(r"thr1_([0-9p]+)__thr2_([0-9p]+)", str(name))
    if not m:
        return None, None
    return float(m.group(1).replace("p", ".")), float(m.group(2).replace("p", "."))


def _has_threshold_dirs(path: Path) -> bool:
    return path.is_dir() and any(p.is_dir() and p.name.startswith("thr1_") for p in path.iterdir())


def _discover_dataset_dirs_for_summary(results_root: Path) -> List[Path]:
    if _has_threshold_dirs(results_root):
        return [results_root]
    dataset_dirs = [p for p in sorted(results_root.iterdir(), key=lambda x: x.name) if _has_threshold_dirs(p)]
    if not dataset_dirs:
        raise RuntimeError(f"Could not find dataset dirs under {results_root}")
    return dataset_dirs


def _benchmark_name_from_dir(dataset_dir: Path) -> str:
    return dataset_dir.name[:-6] if dataset_dir.name.endswith("_split") else dataset_dir.name


def _get_score(summary: Dict[str, Any]) -> Optional[float]:
    for key in ["primary_metric_value", "is_correct", "accuracy", "overall_accuracy", "benchmark_correct"]:
        if key in summary and summary[key] is not None:
            return float(summary[key])
    return None


def _example_key(row: Dict[str, Any]) -> Tuple[str, Any]:
    for key in ["id", "example_id", "question_id", "original_id"]:
        if key in row and row[key] is not None:
            return key, row[key]
    for key in ["dataset_index", "sample_idx", "pid"]:
        if key in row and row[key] is not None:
            return key, row[key]
    raise RuntimeError(f"Could not find stable example key in row keys: {sorted(row.keys())}")


def _first_trace_decision(row: Dict[str, Any]) -> Optional[str]:
    trace = row.get("trace", [])
    if not isinstance(trace, list):
        return None
    for item in trace:
        if isinstance(item, dict) and item.get("decision") is not None:
            return str(item.get("decision"))
    return None


def _has_trace_decision(row: Dict[str, Any], decision: str) -> bool:
    trace = row.get("trace", [])
    return isinstance(trace, list) and any(isinstance(item, dict) and str(item.get("decision")) == str(decision) for item in trace)


def _find_cached_accept_trace(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    trace = row.get("trace", [])
    if not isinstance(trace, list):
        return None
    for item in trace:
        if isinstance(item, dict) and str(item.get("event")) == "cached_aux_score" and str(item.get("model")) == "model1" and str(item.get("decision")) == "accept":
            return item
    return None


def _response_has_usable_answer(benchmark: str, row: Dict[str, Any]) -> bool:
    return response_has_usable_final_answer(benchmark, row.get("raw_response", ""))


def _get_model_usage_from_summary(summary: Dict[str, Any], model_name: str) -> Dict[str, float]:
    usage = summary.get("usage_totals_by_model", {}).get(model_name, {})
    avg = summary.get("usage_averages_by_model", {}).get(model_name, {})
    total_token_touches_per_row = avg.get("total_token_touches_per_row", avg.get("total_token_touches", 0))
    return {
        "prompt_tokens": float(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": float(usage.get("completion_tokens", 0) or 0),
        "generation_tokens": float(usage.get("generation_tokens", 0) or 0),
        "aux_scored_tokens": float(usage.get("aux_scored_tokens", 0) or 0),
        "generation_calls": float(usage.get("generation_calls", 0) or 0),
        "aux_calls": float(usage.get("aux_calls", 0) or 0),
        "generation_time_sec": float(usage.get("generation_time_sec", 0.0) or 0.0),
        "total_token_touches": float(usage.get("total_token_touches", 0) or 0),
        "prompt_tokens_per_row": float(avg.get("prompt_tokens_per_row", 0) or 0),
        "completion_tokens_per_row": float(avg.get("completion_tokens_per_row", 0) or 0),
        "generation_tokens_per_row": float(avg.get("generation_tokens_per_row", 0) or 0),
        "aux_scored_tokens_per_row": float(avg.get("aux_scored_tokens_per_row", 0) or 0),
        "generation_calls_per_row": float(avg.get("generation_calls_per_row", 0) or 0),
        "aux_calls_per_row": float(avg.get("aux_calls_per_row", 0) or 0),
        "generation_time_sec_per_row": float(avg.get("generation_time_sec_per_row", 0.0) or 0.0),
        "total_token_touches_per_row": float(total_token_touches_per_row or 0),
    }


def _human_flops(x: float) -> str:
    x = float(x)
    if x >= 1e15:
        return f"{x / 1e15:.3f} PF"
    if x >= 1e12:
        return f"{x / 1e12:.3f} TF"
    if x >= 1e9:
        return f"{x / 1e9:.3f} GF"
    return f"{x:.3f}"


def _infer_param_count_from_model_name(name: str) -> Optional[float]:
    s = str(name or "").lower()
    for pat in [r"gemma-4-e(\d+(?:\.\d+)?)b", r"qwen3\.5-(\d+(?:\.\d+)?)b", r"qwen3-vl-(\d+(?:\.\d+)?)b", r"qwen3-(\d+(?:\.\d+)?)b"]:
        m = re.search(pat, s)
        if m:
            return float(m.group(1)) * 1e9
    m = re.search(r"(?:^|[_/\-])(\d+(?:\.\d+)?)b(?:$|[_/\-])", s)
    if m:
        return float(m.group(1)) * 1e9
    return None


def _infer_model_params(results_root: Path, override_m1: float, override_m2: float) -> Tuple[float, float]:
    if float(override_m1) > 0 and float(override_m2) > 0:
        return float(override_m1), float(override_m2)
    run_manifest_path = results_root / "run_manifest.json"
    model1_name = ""
    model2_name = ""
    if run_manifest_path.exists():
        try:
            blob = load_json(run_manifest_path)
            model1_name = str(blob.get("model1_name_or_path", ""))
            model2_name = str(blob.get("model2_name_or_path", ""))
        except Exception:
            pass
    if not model1_name or not model2_name:
        parts = results_root.parent.name if results_root.name.endswith("_split") else results_root.name
        segs = str(parts).split("__")
        joined = " ".join(segs)
        guesses = re.findall(r"(qwen[^\s]+|google/gemma[^\s]+|gemma[^\s]+)", joined, flags=re.IGNORECASE)
        if len(guesses) >= 2:
            model1_name, model2_name = guesses[0], guesses[1]
    m1 = float(override_m1) if float(override_m1) > 0 else (_infer_param_count_from_model_name(model1_name) or 2e9)
    m2 = float(override_m2) if float(override_m2) > 0 else (_infer_param_count_from_model_name(model2_name) or 32e9)
    return m1, m2


def build_requested_summary_table(results_root: Path, model1_params: float, model2_params: float, handoff_token_cap: int = 200) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    dataset_dirs = _discover_dataset_dirs_for_summary(results_root)
    for dataset_dir in dataset_dirs:
        benchmark = _benchmark_name_from_dir(dataset_dir)
        thr_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("thr1_")], key=lambda p: p.name)
        for thr_dir in thr_dirs:
            thr1, thr2 = _summary_parse_thr_dir_name(thr_dir.name)
            strategy_dirs = sorted([p for p in thr_dir.iterdir() if p.is_dir()], key=lambda p: _summary_strategy_sort_key(p.name))
            model2_scored_path = thr_dir / "single_agent_model2" / "results_scored.jsonl"
            model2_map: Dict[Tuple[str, Any], Dict[str, Any]] = {}
            if model2_scored_path.exists():
                try:
                    model2_rows = load_jsonl(model2_scored_path)
                    model2_map = {_example_key(r): r for r in model2_rows}
                except Exception:
                    model2_map = {}
            for strategy_dir in strategy_dirs:
                summary_path = _strategy_summary_json_path(strategy_dir, benchmark)
                if not summary_path.exists():
                    summary_path = strategy_dir / "summary_scored.json"
                scored_path = strategy_dir / "results_scored.jsonl"
                if not summary_path.exists() or not scored_path.exists():
                    continue
                summary = load_json(summary_path)
                rows = load_jsonl(scored_path)
                strategy_name = strategy_dir.name
                score = _get_score(summary)
                m1 = _get_model_usage_from_summary(summary, "model1")
                m2 = _get_model_usage_from_summary(summary, "model2")
                m1_flops = 2.0 * model1_params * float(m1["generation_tokens"])
                m2_flops = 2.0 * model2_params * float(m2["generation_tokens"])
                strategy_flops = m1_flops + m2_flops
                handoff_rows = [r for r in rows if _has_trace_decision(r, "handoff")]
                adjusted_m1_generation_tokens = 0
                wrong_high_score_no_handoff = 0
                wrong_high_score_no_handoff_rescuable_by_model2 = 0
                for r in rows:
                    m1_prompt = int(r.get("usage_by_model", {}).get("model1", {}).get("prompt_tokens", 0) or 0)
                    m1_completion = int(r.get("usage_by_model", {}).get("model1", {}).get("completion_tokens", 0) or 0)
                    if _has_trace_decision(r, "handoff"):
                        adjusted_m1_generation_tokens += m1_prompt + min(m1_completion, int(handoff_token_cap))
                    else:
                        adjusted_m1_generation_tokens += m1_prompt + m1_completion
                    accept_item = _find_cached_accept_trace(r)
                    if accept_item is not None and int(r.get("benchmark_correct", 0) or 0) == 0:
                        wrong_high_score_no_handoff += 1
                        try:
                            m2_row = model2_map.get(_example_key(r))
                        except Exception:
                            m2_row = None
                        if m2_row is not None and int(m2_row.get("benchmark_correct", 0) or 0) == 1:
                            wrong_high_score_no_handoff_rescuable_by_model2 += 1
                total_rows = max(int(summary.get("num_rows", len(rows)) or len(rows)), 1)
                handoff_model2_answered = sum(1 for r in handoff_rows if str(r.get("final_model_name", "")) == "model2" and _response_has_usable_answer(benchmark, r))
                handoff_model2_correct = sum(1 for r in handoff_rows if str(r.get("final_model_name", "")) == "model2" and int(r.get("benchmark_correct", 0) or 0) == 1)
                total_generation_time_sec = float(m1.get("generation_time_sec", 0.0)) + float(m2.get("generation_time_sec", 0.0))
                out_rows.append({
                    "benchmark": str(summary.get("benchmark") or benchmark),
                    "results_root_name": str(summary.get("results_root_name") or dataset_dir.name),
                    "thr1": thr1,
                    "thr2": thr2,
                    "strategy": strategy_name,
                    "score": score,
                    "num_rows": int(summary.get("num_rows", len(rows)) or len(rows)),
                    "num_correct": int(summary.get("num_correct", 0) or 0),
                    "model1_overall_tokens": int(m1["generation_tokens"]),
                    "model2_overall_tokens": int(m2["generation_tokens"]),
                    "model1_average_tokens": float(m1["generation_tokens_per_row"]),
                    "model2_average_tokens": float(m2["generation_tokens_per_row"]),
                    "model1_calls": int(m1["generation_calls"]),
                    "model2_calls": int(m2["generation_calls"]),
                    "model1_flops": m1_flops,
                    "model1_flops_human": _human_flops(m1_flops),
                    "model2_flops": m2_flops,
                    "model2_flops_human": _human_flops(m2_flops),
                    "strategy_flops": strategy_flops,
                    "strategy_flops_human": _human_flops(strategy_flops),
                    "model1_overall_tokens_handoff200": int(adjusted_m1_generation_tokens),
                    "model1_average_tokens_handoff200": float(adjusted_m1_generation_tokens) / total_rows,
                    "handoff_cases": int(len(handoff_rows)),
                    "handoff_cases_model2_answered": int(handoff_model2_answered),
                    "handoff_cases_model2_correct": int(handoff_model2_correct),
                    "wrong_high_headscore_no_handoff": int(wrong_high_score_no_handoff),
                    "wrong_high_headscore_no_handoff_model2_could_answer": int(wrong_high_score_no_handoff_rescuable_by_model2),
                    "avg_generation_time_sec": total_generation_time_sec / total_rows,
                    "model1_avg_generation_time_sec": float(m1.get("generation_time_sec_per_row", 0.0) or 0.0),
                    "model2_avg_generation_time_sec": float(m2.get("generation_time_sec_per_row", 0.0) or 0.0),
                    "summary_path": str(summary_path),
                    "scored_path": str(scored_path),
                })
    out_rows.sort(key=lambda r: (str(r.get("benchmark")), float(r.get("thr1") or -1), float(r.get("thr2") or -1), _summary_strategy_sort_key(str(r.get("strategy")))))
    return out_rows


def write_requested_summary_reports(results_root: Path, model1_params: float, model2_params: float, debug: bool = False) -> Dict[str, str]:
    rows = build_requested_summary_table(results_root, model1_params=model1_params, model2_params=model2_params)
    if not rows:
        raise RuntimeError(f"No requested summary rows built under {results_root}")
    dataset_tag = _dataset_tag_for_filename(rows[0].get("benchmark", "dataset"))
    json_path = results_root / f"{dataset_tag}_requested_summary_report.json"
    csv_path = results_root / f"{dataset_tag}_requested_summary_report.csv"
    md_path = results_root / f"{dataset_tag}_requested_summary_report.md"
    json_dump(json_path, rows)
    # CSV and markdown
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        compact_cols = [
            "benchmark", "thr1", "strategy", "score",
            "model1_overall_tokens", "model2_overall_tokens",
            "model1_average_tokens", "model2_average_tokens",
            "model1_flops_human", "model2_flops_human", "strategy_flops_human",
            "model1_overall_tokens_handoff200",
            "model1_calls", "model2_calls",
            "handoff_cases", "handoff_cases_model2_answered", "handoff_cases_model2_correct",
            "wrong_high_headscore_no_handoff", "wrong_high_headscore_no_handoff_model2_could_answer",
            "avg_generation_time_sec",
        ]
        df[compact_cols].to_markdown(md_path, index=False)
    except Exception as e:
        with open(csv_path, "w", encoding="utf-8") as f:
            headers = list(rows[0].keys())
            f.write(",".join(headers) + "\n")
            for row in rows:
                vals = [json.dumps(row.get(h, ""), ensure_ascii=False) if isinstance(row.get(h), (dict, list)) else str(row.get(h, "")) for h in headers]
                f.write(",".join(vals) + "\n")
        md_path.write_text(f"Markdown export skipped: {type(e).__name__}: {e}\n", encoding="utf-8")
    debug_print(debug, f"[SUMMARY] Saved requested summary JSON: {json_path}")
    debug_print(debug, f"[SUMMARY] Saved requested summary CSV: {csv_path}")
    debug_print(debug, f"[SUMMARY] Saved requested summary MD: {md_path}")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def _make_skip_row(row_idx: int, row: Dict[str, Any], err: Exception) -> Dict[str, Any]:
    ex_id = row.get("example_id", row.get("pid", row.get("question_id", row.get("id", row_idx + 1))))
    return {
        "row_index": row_idx,
        "example_id": ex_id,
        "error_type": type(err).__name__,
        "error": str(err),
        "raw_response": row.get("raw_response"),
    }


def evaluate_saved_rows_batched(
    benchmark: str,
    rows: List[Dict[str, Any]],
    judge_runtime,
    judge_sampling,
    judge_batch_size: int,
    debug: bool,
    progress_desc: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    outputs = []
    skipped_rows = []
    if judge_batch_size < 1:
        raise ValueError("judge_batch_size must be positive")

    with tqdm(total=len(rows), desc=progress_desc, unit="example", dynamic_ncols=True, leave=False) as bar:
        for start in range(0, len(rows), judge_batch_size):
            chunk = rows[start:start + judge_batch_size]
            scored = [None] * len(chunk)
            pending = []
            for index, row in enumerate(chunk):
                try:
                    prepared, image = prepare_evaluation_row(benchmark, row, judge_runtime, judge_sampling)
                    if prepared.get("judge_prompt") is None:
                        scored[index] = prepared
                    else:
                        pending.append((index, prepared, image))
                except Exception as exc:
                    skipped_rows.append(_make_skip_row(start + index, row, exc))
            if pending:
                messages = [[{"role": "user", "content": row["judge_prompt"]}] for _, row, _ in pending]
                images = [image for _, _, image in pending]
                try:
                    generations = judge_runtime.generate_batch(messages_list=messages, images=images, sampling_cfg=judge_sampling)
                    if len(generations) != len(pending):
                        raise RuntimeError("Judge output count differs from its input batch")
                except Exception as exc:
                    debug_print(debug, f"Retrying failed judge batch by row: {exc}")
                    generations = None
                for offset, (index, prepared, image) in enumerate(pending):
                    try:
                        gen = generations[offset] if generations is not None else judge_runtime.generate(
                            messages=messages[offset], image=image, sampling_cfg=judge_sampling,
                        )
                        scored[index] = finish_evaluation_row(benchmark, prepared, gen)
                    except Exception as exc:
                        skipped_rows.append(_make_skip_row(start + index, chunk[index], exc))
            outputs.extend(row for row in scored if row is not None)
            bar.update(len(chunk))
    return outputs, skipped_rows


def evaluate_one_strategy_dir(
    benchmark: str,
    strategy_dir: Path,
    judge_runtime,
    judge_sampling,
    judge_batch_size: int,
    debug: bool,
) -> Dict[str, Any]:
    strategy_name = strategy_dir.name
    results_path = strategy_dir / "results.jsonl"
    if not results_path.exists():
        raise RuntimeError(f"Missing generation results: {results_path}")

    rows = load_jsonl(results_path)
    scored_rows, skipped_rows = evaluate_saved_rows_batched(
        benchmark=benchmark,
        rows=rows,
        judge_runtime=judge_runtime,
        judge_sampling=judge_sampling,
        judge_batch_size=judge_batch_size,
        debug=debug,
        progress_desc=strategy_name,
    )

    results_jsonl_path = strategy_dir / "results_scored.jsonl"
    skipped_json_path = strategy_dir / "results_scored_skipped.json"
    summary_json_path = _strategy_summary_json_path(strategy_dir, benchmark)

    write_jsonl(results_jsonl_path, scored_rows)
    save_rows_to_parquet(strategy_dir / "results_scored.parquet", scored_rows, debug=debug)
    json_dump(skipped_json_path, skipped_rows)

    summary = enrich_strategy_summary(benchmark, strategy_dir.parent, strategy_name, scored_rows, skipped_rows)
    json_dump(summary_json_path, summary)

    tqdm.write(f"[EVAL] Finished strategy: {strategy_name}")
    tqdm.write(f"[EVAL] Saved: {results_jsonl_path}")
    tqdm.write(f"[EVAL] Saved: {strategy_dir / 'results_scored.parquet'}")
    tqdm.write(f"[EVAL] Saved: {skipped_json_path}")
    tqdm.write(f"[EVAL] Saved: {summary_json_path}")
    return summary


def main() -> None:
    args = parse_args()
    debug = bool(args.debug or DEBUG_MODE)
    results_root = Path(args.results_root).expanduser().resolve()
    requested_strategy_names = parse_csv_names(args.strategy_names)
    judge_batch_size = max(1, int(args.judge_batch_size))

    strategy_dirs = discover_strategy_dirs(results_root)
    if requested_strategy_names is not None:
        strategy_dirs = [p for p in strategy_dirs if p.name in requested_strategy_names]
        if not strategy_dirs:
            raise RuntimeError(
                f"No discovered strategy dirs matched --strategy_names={sorted(requested_strategy_names)} under {results_root}"
            )

    benchmark = infer_benchmark(results_root, strategy_dirs, args.benchmark)
    grouped_suite_roots = group_strategy_dirs_by_suite_root(strategy_dirs)

    run_config = {
        "results_root": str(results_root),
        "benchmark": benchmark,
        "strategy_names_filter": args.strategy_names,
        "debug_mode": debug,
        "judge_batch_size": judge_batch_size,
        "judge_model_name_or_path": JUDGE_MODEL_NAME_OR_PATH,
        "judge_runtime_profile": JUDGE_RUNTIME_PROFILE,
        "judge_sampling_profiles": {
            "default": JUDGE_DEFAULT_SAMPLING_PROFILE,
            "thinking": JUDGE_THINKING_SAMPLING_PROFILE,
            "instruct": JUDGE_INSTRUCT_SAMPLING_PROFILE,
        },
        "discovered_strategy_dirs": [str(p) for p in strategy_dirs],
        "discovered_suite_roots": [str(root) for root, _ in grouped_suite_roots],
    }

    judge_runtime = None
    judge_sampling = None
    if benchmark_needs_judge(benchmark):
        judge_runtime, judge_sampling = build_judge_runtime_and_sampling(
            judge_model_name_or_path=str(args.judge_model_name_or_path),
            judge_runtime_profile=JUDGE_RUNTIME_PROFILE,
            judge_sampling_profiles={
                "default": JUDGE_DEFAULT_SAMPLING_PROFILE,
                "thinking": JUDGE_THINKING_SAMPLING_PROFILE,
                "instruct": JUDGE_INSTRUCT_SAMPLING_PROFILE,
            },
            judge_model_family=str(args.judge_model_family),
            judge_thinking_mode=str(args.judge_thinking_mode),
        )

    master_summary: Dict[str, Dict[str, Any]] = {}
    try:
        for suite_root, suite_strategy_dirs in grouped_suite_roots:
            json_dump(suite_root / f"{_dataset_tag_for_filename(benchmark)}_evaluation_run_config.json", run_config)
            suite_summary: Dict[str, Dict[str, Any]] = {}
            strategy_bar = tqdm(
                list(enumerate(suite_strategy_dirs, start=1)),
                total=len(suite_strategy_dirs),
                desc=f"Strategies[{suite_root.name}]",
                unit="strategy",
                dynamic_ncols=True,
            )

            for strategy_idx, strategy_dir in strategy_bar:
                strategy_name = strategy_dir.name
                strategy_bar.set_description(f"Strategy {strategy_idx}/{len(suite_strategy_dirs)}")
                strategy_bar.set_postfix_str(f"{suite_root.name}:{strategy_name}")
                tqdm.write(f"[EVAL] Running strategy {strategy_idx}/{len(suite_strategy_dirs)}: {strategy_name} | root={suite_root.name}")
                suite_summary[strategy_name] = evaluate_one_strategy_dir(
                    benchmark=benchmark,
                    strategy_dir=strategy_dir,
                    judge_runtime=judge_runtime,
                    judge_sampling=judge_sampling,
                    judge_batch_size=judge_batch_size,
                    debug=debug,
                )

            suite_comparison = build_suite_comparison_summary(benchmark, suite_summary)
            json_dump(_suite_summary_json_path(suite_root, benchmark), suite_summary)
            json_dump(_suite_comparison_json_path(suite_root, benchmark), suite_comparison)
            save_rows_to_parquet(
                _suite_summary_parquet_path(suite_root, benchmark),
                [{"strategy_name": k, **v} for k, v in suite_summary.items()],
                debug=debug,
            )
            save_rows_to_parquet(
                _suite_comparison_parquet_path(suite_root, benchmark),
                [
                    {"strategy_name": strategy_name, "comparison_key": comp_key, **comp_val}
                    for strategy_name, comps in suite_comparison.items()
                    for comp_key, comp_val in comps.items()
                ],
                debug=debug,
            )
            master_summary[str(suite_root)] = {
                "suite_summary": suite_summary,
                "suite_comparison": suite_comparison,
            }

        json_dump(_all_suite_summaries_json_path(results_root, benchmark), master_summary)
        save_rows_to_parquet(
            _all_suite_summaries_parquet_path(results_root, benchmark),
            [
                {"suite_root": suite_root, "strategy_name": strategy_name, **summary}
                for suite_root, suite_payload in master_summary.items()
                for strategy_name, summary in suite_payload.get("suite_summary", {}).items()
            ],
            debug=debug,
        )
        save_rows_to_parquet(
            _all_suite_comparisons_parquet_path(results_root, benchmark),
            [
                {"suite_root": suite_root, "strategy_name": strategy_name, "comparison_key": comp_key, **comp_val}
                for suite_root, suite_payload in master_summary.items()
                for strategy_name, comps in suite_payload.get("suite_comparison", {}).items()
                for comp_key, comp_val in comps.items()
            ],
            debug=debug,
        )
        if not bool(args.skip_auto_summary):
            m1_params, m2_params = _infer_model_params(results_root, float(args.model1_params), float(args.model2_params))
            summary_outputs = write_requested_summary_reports(results_root, model1_params=m1_params, model2_params=m2_params, debug=debug)
            print(json.dumps({"requested_summary_outputs": summary_outputs, "model1_params": m1_params, "model2_params": m2_params}, ensure_ascii=False, indent=2))
        print(json.dumps(master_summary, ensure_ascii=False, indent=2))
    finally:
        if judge_runtime is not None:
            judge_runtime.unload(drop_processor=False)


if __name__ == "__main__":
    main()
