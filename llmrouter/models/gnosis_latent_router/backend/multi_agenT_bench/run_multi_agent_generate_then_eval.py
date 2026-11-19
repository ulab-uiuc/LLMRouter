#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List


def _sanitize_name_for_path(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("/", "__").replace("\\", "__")
    s = s.replace(" ", "_").replace(".", "p").replace("-", "_")
    while "___" in s:
        s = s.replace("___", "__")
    return s


def _format_threshold_for_path(x: float) -> str:
    # Preserve established two-decimal paths and distinguish nearby thresholds.
    value = f"{x:.2f}" if round(x, 2) == x else str(x)
    return value.replace(".", "p")


def _runtime_mode_tag(model_family: str, thinking_mode: str) -> str:
    fam = _sanitize_name_for_path(model_family or "auto")
    think = _sanitize_name_for_path(thinking_mode or "auto")
    return f"fam_{fam}__think_{think}"


def _parse_threshold_list(csv: str) -> List[float]:
    vals: List[float] = []
    for part in str(csv).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise RuntimeError("No thresholds provided")
    if not all(0 <= value <= 1 for value in vals):
        raise ValueError("Capability thresholds must be between 0 and 1")
    return sorted(set(vals))


def _compute_results_root(
    *,
    output_base_root: str,
    benchmark: str,
    model1_name_or_path: str,
    model2_name_or_path: str,
    model1_model_family: str,
    model1_thinking_mode: str,
    model2_model_family: str,
    model2_thinking_mode: str,
    model1_aux_thresholds: List[float],
    model2_aux_threshold: float,
) -> Path:
    model1_tag = _sanitize_name_for_path(model1_name_or_path)
    model2_tag = _sanitize_name_for_path(model2_name_or_path)
    model1_mode_tag = _runtime_mode_tag(model1_model_family, model1_thinking_mode)
    model2_mode_tag = _runtime_mode_tag(model2_model_family, model2_thinking_mode)
    thr_list_tag = "_".join(_format_threshold_for_path(x) for x in model1_aux_thresholds)
    thr2_tag = _format_threshold_for_path(model2_aux_threshold)
    run_tag = f"{model1_tag}__{model1_mode_tag}__{model2_tag}__{model2_mode_tag}__thr1s_{thr_list_tag}__thr2_{thr2_tag}_lite"
    return Path(output_base_root) / run_tag / f"{benchmark}_split"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Run compact multi-agent generation, then evaluate the results end to end.")
    ap.add_argument("--benchmark", type=str, default="simplevqa", choices=["mathvista", "mathverse", "charxiv_reasoning", "screenspot_pro", "simplevqa", "triviaqa", "math", "mmlu_pro"])
    ap.add_argument("--model1_name_or_path", type=str, required=True)
    ap.add_argument("--model1_revision", help="Target backbone revision used to train the head")
    ap.add_argument("--model2_name_or_path", type=str, required=True)
    ap.add_argument("--model1_aux_head_ckpt", type=str, required=True)
    ap.add_argument("--model1_model_family", type=str, default="auto", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--model1_thinking_mode", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--model2_model_family", type=str, default="auto", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    ap.add_argument("--model2_thinking_mode", type=str, default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--model1_params", type=float, default=0.0)
    ap.add_argument("--model2_params", type=float, default=0.0)
    ap.add_argument("--strategy_names", type=str, required=True, help="Comma-separated strategy names")
    ap.add_argument("--model1_aux_thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--model2_aux_threshold", type=float, default=0.80)
    ap.add_argument(
        "--output_base_root",
        type=str,
        default="eval_outputs/multi_agent_compact",
    )
    ap.add_argument(
        "--generate_script",
        type=str,
        default=str(here / "compact_multi_agent_generate.py"),
    )
    ap.add_argument(
        "--evaluate_script",
        type=str,
        default=str(here / "compact_multi_agent_evaluate.py"),
    )
    ap.add_argument("--judge_batch_size", type=int, default=32)
    ap.add_argument("--judge_model_name_or_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--judge_model_family", type=str, default="auto")
    ap.add_argument("--judge_thinking_mode", type=str, default="auto")
    ap.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional deterministic benchmark prefix for smoke tests.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def _run(cmd: List[str], *, cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()

    launch_cwd = Path.cwd().resolve()
    generate_script = Path(args.generate_script).expanduser().resolve()
    evaluate_script = Path(args.evaluate_script).expanduser().resolve()
    output_base_root = str(Path(args.output_base_root).expanduser().resolve())
    model1_aux_head_ckpt = str(Path(args.model1_aux_head_ckpt).expanduser().resolve())
    if not generate_script.exists():
        raise RuntimeError(f"Generate script not found: {generate_script}")
    if not evaluate_script.exists():
        raise RuntimeError(f"Evaluate script not found: {evaluate_script}")

    thresholds = _parse_threshold_list(args.model1_aux_thresholds)
    results_root = _compute_results_root(
        output_base_root=output_base_root,
        benchmark=args.benchmark,
        model1_name_or_path=args.model1_name_or_path,
        model2_name_or_path=args.model2_name_or_path,
        model1_model_family=args.model1_model_family,
        model1_thinking_mode=args.model1_thinking_mode,
        model2_model_family=args.model2_model_family,
        model2_thinking_mode=args.model2_thinking_mode,
        model1_aux_thresholds=thresholds,
        model2_aux_threshold=float(args.model2_aux_threshold),
    )

    run_manifest = {
        "benchmark": args.benchmark,
        "model1_name_or_path": args.model1_name_or_path,
        "model2_name_or_path": args.model2_name_or_path,
        "model1_aux_head_ckpt": model1_aux_head_ckpt,
        "checkpoint_sha256": hashlib.sha256(Path(model1_aux_head_ckpt).read_bytes()).hexdigest(),
        "model1_model_family": args.model1_model_family,
        "model1_thinking_mode": args.model1_thinking_mode,
        "model2_model_family": args.model2_model_family,
        "model2_thinking_mode": args.model2_thinking_mode,
        "model1_params": float(args.model1_params),
        "model2_params": float(args.model2_params),
        "strategy_names": args.strategy_names,
        "model1_aux_thresholds": thresholds,
        "model2_aux_threshold": float(args.model2_aux_threshold),
        "output_base_root": output_base_root,
        "results_root": str(results_root),
        "generate_script": str(generate_script),
        "evaluate_script": str(evaluate_script),
        "model1_revision": args.model1_revision,
        "judge_batch_size": int(args.judge_batch_size),
        "judge_model_name_or_path": args.judge_model_name_or_path,
        "judge_model_family": args.judge_model_family,
        "judge_thinking_mode": args.judge_thinking_mode,
        "max_examples": args.max_examples,
        "overwrite": bool(args.overwrite),
        "resume_mode": not bool(args.no_resume),
        "debug": bool(args.debug),
    }

    results_root.mkdir(parents=True, exist_ok=True)
    manifest_path = results_root / "run_manifest.json"
    if manifest_path.is_file() and not args.overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("checkpoint_sha256", "model1_revision", "judge_model_name_or_path",
                      "judge_model_family", "judge_thinking_mode"):
            if previous.get(field) != run_manifest[field]:
                raise ValueError(f"Benchmark {field} changed. Use a new --output_base_root or --overwrite.")
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    gen_cmd = [
        sys.executable,
        str(generate_script),
        "--model1_name_or_path", args.model1_name_or_path,
        "--model2_name_or_path", args.model2_name_or_path,
        "--model1_aux_head_ckpt", model1_aux_head_ckpt,
        "--model2_aux_threshold", str(args.model2_aux_threshold),
        "--output_base_root", output_base_root,
        "--judge_model_name_or_path", args.judge_model_name_or_path,
        "--judge_model_family", args.judge_model_family,
        "--judge_thinking_mode", args.judge_thinking_mode,
        "--benchmark",
        args.benchmark,
        "--strategy_names",
        args.strategy_names,
        "--model1_aux_thresholds",
        args.model1_aux_thresholds,
        "--model1_model_family",
        args.model1_model_family,
        "--model1_thinking_mode",
        args.model1_thinking_mode,
        "--model2_model_family",
        args.model2_model_family,
        "--model2_thinking_mode",
        args.model2_thinking_mode,
    ]
    if args.model1_revision:
        gen_cmd.extend(["--model1_revision", args.model1_revision])
    if args.overwrite:
        gen_cmd.append("--overwrite")
    if args.no_resume:
        gen_cmd.append("--no_resume")
    if args.debug:
        gen_cmd.append("--debug")
    if args.max_examples is not None:
        gen_cmd.extend(["--max_examples", str(args.max_examples)])
    _run(gen_cmd, cwd=launch_cwd)

    eval_cmd = [
        sys.executable,
        str(evaluate_script),
        "--results_root",
        str(results_root),
        "--benchmark",
        args.benchmark,
        "--strategy_names",
        args.strategy_names,
        "--judge_batch_size",
        str(args.judge_batch_size),
        "--model1_params",
        str(args.model1_params),
        "--model2_params",
        str(args.model2_params),
        "--judge_model_name_or_path",
        args.judge_model_name_or_path,
        "--judge_model_family",
        args.judge_model_family,
        "--judge_thinking_mode",
        args.judge_thinking_mode,
    ]
    if args.debug:
        eval_cmd.append("--debug")
    _run(eval_cmd, cwd=launch_cwd)

    print(f"\n[DONE] Results root: {results_root}", flush=True)
    print(f"[DONE] Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
