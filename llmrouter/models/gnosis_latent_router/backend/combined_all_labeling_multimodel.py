#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified correctness labeling for the mixed generated dataset from combined_all_datagen.py.

Supported subsets:
- FineVision subsets including aokvqa, ai2d_merged, infographic_vqa, groundui, aguvis-stage-1
- MM-OpenR1
- DAPO-Math
- TriviaQA
- smolagents/aguvis-stage-2 (android_control)
- Salesforce/APIGen-MT-5k

Design:
- Reads raw parquet shards from one mixed raw directory.
- Uses fast structural / exact-match checks when possible.
- Uses a judge model for open-ended or unresolved cases.
- Saves a unified verified schema with a superset of all useful fields.

Scoring highlights:
- General QA: exact/heuristic first, judge fallback.
- GroundUI: bounding-box structural score.
- AGUVIS-stage-1: action structural score.
- AGUVIS-stage-2: final-action-only judge after stripping any thinking block.
- APIGen-MT-5k: lenient multi-aspect next-turn grading.
"""

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import io
import re
import gc
import json
import base64
import unicodedata
import ast
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import pyarrow.parquet as pq
from PIL import Image
from datasets import load_dataset, Dataset, Features, Sequence, Value, Image as HFImage
from filelock import FileLock
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from qwen_vl_utils import process_vision_info

try:
    from math_verify import LatexExtractionConfig, parse, verify
    HAVE_MATH_VERIFY = True
except ImportError:
    HAVE_MATH_VERIFY = False

# =============================================================================
# CONFIG
# =============================================================================
# data/train/Qwen3VL/Qwen3_VL_4B_Instruct_hard_Mixed_Sources_120k/raw
DEFAULT_DATA_ROOT = Path("data/train")
RUN_NAME = "Qwen3-VL-2B-Instruct_hard_Mixed_Sources_120k"
SAVE_ROOT = DEFAULT_DATA_ROOT / "Qwen3VL" / RUN_NAME
RAW_SAVE_DIR = SAVE_ROOT / "raw"
VERIFIED_SAVE_DIR = SAVE_ROOT / "verified"
VERIFICATION_STATS_PATH = SAVE_ROOT / "verification_stats.json"

RAW_SHARD_GLOB = "shard-*.parquet"
OVERWRITE_VERIFIED_OUTPUT = False
RESUME_IF_VERIFIED_EXISTS = True

# JUDGE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
JUDGE_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
JUDGE_GPU_MEMORY_UTILIZATION = 0.90
JUDGE_TENSOR_PARALLEL_SIZE = 1
JUDGE_MAX_MODEL_LEN = 40000
JUDGE_MAX_TOKENS = 20000
JUDGE_BATCH_SIZE = 64
JUDGE_TEMPERATURE = 1
JUDGE_TOP_P = 0.95

DEBUG_MODE = True
DEBUG_PREVIEW_ROWS = 5
DEBUG_TEXT_PREVIEW_CHARS = 240

GROUNDUI_CENTER_TOL = 0.05
GROUNDUI_SIZE_TOL = 0.10
AGUVIS_POINT_TOL = 0.05
AGUVIS_DRAG_TOL = 0.05
AGUVIS2_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.IGNORECASE | re.DOTALL)
AGUVIS2_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", flags=re.IGNORECASE | re.DOTALL)

GENERAL_JUDGE_SYSTEM_PROMPT = (
    "You are a careful grader. Grade the student's final answer against the reference answer. "
    "Be semantically fair and allow minor wording differences if the answer is still correct. "
    "For multiple-choice questions, accept either the correct letter or the correct answer text when they clearly match. "
    "Return only JSON with exactly one key: total_score, where the value is a number between 0 and 1."
)

AGUVIS_ACTION_RM_SYSTEM_PROMPT = (
    "You are grading a GUI-agent action prediction. Compare the student's action against the reference action for the given task and screenshot. "
    "Be strict about action semantics, but ignore harmless formatting differences and tiny coordinate noise. "
    "Return only JSON with exactly one key: total_score, where the value is a number between 0 and 1."
)

AGUVIS2_JUDGE_SYSTEM_PROMPT = (
    "You are grading the student's final GUI action for the next move. Any hidden reasoning or thinking has already been stripped away. "
    "Use the task, screenshot, previous actions, and reference action. Be strict about action semantics but ignore harmless formatting differences and tiny coordinate noise. "
    "Return only JSON with exactly one key: total_score, where the value is a number between 0 and 1."
)

APIGEN_JUDGE_SYSTEM_PROMPT = (
    "You are grading the next assistant turn in a multi-turn tool-use conversation. "
    "Be semantically lenient: do not require exact wording when the student's response is equally correct, natural, and useful. "
    "For tool calls, ignore harmless formatting differences, key ordering, single-vs-double quotes, and equivalent Python-dict versus JSON formatting. "
    "Give partial credit when the response is on the right track but incomplete. "
    "Return only JSON with exactly these keys: relevance_score, correctness_subscore, policy_score, format_score, total_score. "
    "All scores must be numbers between 0 and 1, and total_score should be the average of the four aspect scores."
)

RAW_REQUIRED_COLUMNS = [
    "images", "subset_name", "macro_category", "source", "modality", "row_index", "qa_index", "turn_index",
    "question", "original_answer", "system_prompt", "human_prompt", "task_text", "previous_actions",
    "gt_reasoning", "gt_action", "tools", "context_text", "prefix_json", "last_user_or_observation",
    "target_role", "num_prior_turns", "completion", "generation_index", "completion_length", "two_step_applied",
    "qa_relevance", "qa_visual_dependency", "qa_image_correspondence", "qa_formatting",
]

VERIFIED_FEATURES = Features({
    "images": Sequence(HFImage()),
    "subset_name": Value("string"),
    "macro_category": Value("string"),
    "source": Value("string"),
    "modality": Value("string"),
    "row_index": Value("int32"),
    "qa_index": Value("int32"),
    "turn_index": Value("int32"),
    "question": Value("string"),
    "original_answer": Value("string"),
    "system_prompt": Value("string"),
    "human_prompt": Value("string"),
    "task_text": Value("string"),
    "previous_actions": Value("string"),
    "gt_reasoning": Value("string"),
    "gt_action": Value("string"),
    "tools": Value("string"),
    "context_text": Value("string"),
    "prefix_json": Value("string"),
    "last_user_or_observation": Value("string"),
    "target_role": Value("string"),
    "num_prior_turns": Value("int32"),
    "completion": Value("string"),
    "generation_index": Value("int32"),
    "completion_length": Value("int32"),
    "two_step_applied": Value("bool"),
    "qa_relevance": Value("int32"),
    "qa_visual_dependency": Value("int32"),
    "qa_image_correspondence": Value("int32"),
    "qa_formatting": Value("int32"),
    "final_response": Value("string"),
    "pred_summary": Value("string"),
    "pred_action": Value("string"),
    "normalized_pred": Value("string"),
    "normalized_gt": Value("string"),
    "exact_match_score": Value("float32"),
    "summary_score": Value("float32"),
    "action_score": Value("float32"),
    "relevance_score": Value("float32"),
    "correctness_subscore": Value("float32"),
    "policy_score": Value("float32"),
    "format_score": Value("float32"),
    "rule_score": Value("float32"),
    "rm_score": Value("float32"),
    "correctness_score": Value("float32"),
    "correctness_label": Value("int32"),
    "judge_used": Value("bool"),
    "judge_raw": Value("string"),
    "json_parsed": Value("bool"),
    "action_exact_match": Value("bool"),
    "eval_method": Value("string"),
})

# =============================================================================
# HELPERS
# =============================================================================

def _debug(msg: str):
    if DEBUG_MODE:
        print(msg, flush=True)


def _preview_text(x: Any, limit: int = DEBUG_TEXT_PREVIEW_CHARS) -> str:
    s = str(x).replace("\n", " ").replace("\r", " ").strip()
    s = " ".join(s.split())
    if len(s) > limit:
        return s[:limit] + " ..."
    return s



def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--judge-model-id", default=None)
    ap.add_argument("--judge-revision", help="Optional Hugging Face revision for the judge and its processor.")
    ap.add_argument("--judge-trust-remote-code", action="store_true")
    ap.add_argument("--judge-batch-size", type=int, default=None)
    ap.add_argument("--judge-gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--judge-tensor-parallel-size", type=int, default=None)
    ap.add_argument("--judge-max-model-len", type=int, default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=None)
    ap.add_argument("--judge-enforce-eager", action="store_true", help="Disable CUDA graph capture for smaller runs.")
    return ap.parse_args()


def _remove_gemma_thought_block(text: str) -> str:
    text = text or ""
    text = re.sub(r"^\s*<\|channel\>thought\s*.*?<channel\|>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".json.tmp")
    with pending.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    pending.replace(path)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _label_from_score(score: float, threshold: float = 0.5) -> int:
    return int(float(score) >= threshold)


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _decode_bytes_maybe(b: Any) -> bytes:
    if isinstance(b, (bytes, bytearray)):
        return bytes(b)
    if isinstance(b, str):
        return base64.b64decode(b)
    raise TypeError(f"Unsupported bytes type: {type(b)}")


def _to_pil(val: Any) -> Image.Image:
    if isinstance(val, Image.Image):
        return val.convert("RGB")
    if isinstance(val, dict):
        if val.get("bytes") is not None:
            return Image.open(io.BytesIO(_decode_bytes_maybe(val["bytes"]))).convert("RGB")
        if val.get("path"):
            return Image.open(val["path"]).convert("RGB")
    if isinstance(val, str) and os.path.exists(val):
        return Image.open(val).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(val)}")


def extract_final_response(completion: str) -> str:
    text = (completion or "").strip()

    # Qwen / DeepSeek-style thought channel.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Gemma 4 thought channel.
    text = _remove_gemma_thought_block(text)

    # AGUVIS-stage-2 final action block.
    if "<code>" in text and "</code>" in text:
        blocks = re.findall(r"<code>\s*(.*?)\s*</code>", text, flags=re.DOTALL)
        if blocks:
            text = blocks[-1].strip()
    return text


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_answer_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.lower().strip()
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = text.replace("final answer", "answer")
    text = re.sub(r"^(?:the\s+)?answer\s*[:=\-]\s*", "", text)
    text = re.sub(r"^(?:the\s+)?correct\s+answer\s*[:=\-]\s*", "", text)
    text = re.sub(r"^answer\s+is\s+", "", text)
    text = re.sub(r"^option\s+([a-z])\b", r"\1", text)
    text = re.sub(r"^choice\s+([a-z])\b", r"\1", text)
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"(?:option|choice) [a-z]", text):
        text = text.split()[-1]
    if re.fullmatch(r"answer [a-z]", text):
        text = text.split()[-1]
    return text


def _sequence_similarity(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    brace_matches = re.findall(r"\{.*\}", stripped, flags=re.DOTALL)
    candidates.extend(brace_matches)
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _canonicalize_jsonish(text: str) -> str:
    obj = _extract_json_object(text or "")
    if obj is None:
        return ""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return ""


def _extract_last_float_list(text: str, expected: int) -> Optional[List[float]]:
    if not text:
        return None
    matches = re.findall(r"\[\s*([0-9eE+\-.,\s]+?)\s*\]", text)
    candidates = [text] + matches[::-1]
    for cand in candidates:
        vals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cand)
        if vals is not None and len(vals) >= expected:
            return [float(x) for x in vals[:expected]]
    return None


def _is_valid_unit_value(x: float, eps: float = 1e-3) -> bool:
    return (-eps) <= x <= (1.0 + eps)


def _parse_groundui_bbox(text: str) -> Optional[List[float]]:
    vals = _extract_last_float_list(text, 4)
    if vals is None:
        return None
    cx, cy, w, h = vals[:4]
    if not all(_is_valid_unit_value(v) for v in [cx, cy, w, h]):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return [min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)]


def _bbox_center_to_xyxy(box: List[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = box
    x1 = min(max(cx - w / 2.0, 0.0), 1.0)
    y1 = min(max(cy - h / 2.0, 0.0), 1.0)
    x2 = min(max(cx + w / 2.0, 0.0), 1.0)
    y2 = min(max(cy + h / 2.0, 0.0), 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_center_to_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_center_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _score_from_abs_delta(delta: float, tol: float) -> float:
    if tol <= 0:
        return 0.0
    return _clamp01(1.0 - abs(float(delta)) / float(tol))


def _normalize_action_call(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("'", '"')
    return s.lower()


def _extract_action_call(text: str) -> Tuple[Optional[str], Optional[str]]:
    candidates = [text or ""]
    m = re.search(r"action\s*:\s*(.*)$", text or "", flags=re.IGNORECASE | re.DOTALL)
    if m:
        candidates.insert(0, m.group(1).strip())
    for src in candidates:
        m = re.search(r"([a-zA-Z0-9_.]+)\s*\((.*)\)", src, flags=re.DOTALL)
        if m:
            return m.group(1), m.group(2)
    return None, None


def _normalize_action_name(name: str) -> str:
    return (name or "").strip().lower()


def _extract_named_number(args: str, key: str) -> Optional[float]:
    m = re.search(rf"\b{re.escape(key)}\s*=\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", args or "")
    return float(m.group(1)) if m else None


def _extract_named_point(args: str, key: str) -> Optional[Tuple[float, float]]:
    m = re.search(rf"\b{re.escape(key)}\s*=\s*\[\s*([-+]?(?:\d*\.\d+|\d+))\s*,\s*([-+]?(?:\d*\.\d+|\d+))\s*\]", args or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _valid_point(pt: Tuple[float, float]) -> bool:
    return _is_valid_unit_value(pt[0]) and _is_valid_unit_value(pt[1])


def _parse_gui_action(text: str) -> Optional[Dict[str, Any]]:
    name, args = _extract_action_call(text)
    if not name:
        return None
    action = _normalize_action_name(name)
    item: Dict[str, Any] = {"action": action, "point": None, "from": None, "to": None, "text": None, "keys": None, "direction": None, "app_name": None, "seconds": None, "status": None}

    if action in {"click", "double_click", "right_click", "hover", "long_press"}:
        x = _extract_named_number(args, "x")
        y = _extract_named_number(args, "y")
        if x is None or y is None:
            pt = _extract_named_point(args, "coord") or _extract_named_point(args, "coords")
            if pt is not None:
                x, y = pt
        if x is None or y is None:
            return None
        item["point"] = (float(x), float(y))
        if not _valid_point(item["point"]):
            return None
        return item

    if action.endswith("swipe") or action == "drag":
        p_from = _extract_named_point(args, "from_coord") or _extract_named_point(args, "start_coord")
        p_to = _extract_named_point(args, "to_coord") or _extract_named_point(args, "end_coord")
        if p_from is None or p_to is None:
            return None
        if not _valid_point(p_from) or not _valid_point(p_to):
            return None
        item["from"] = p_from
        item["to"] = p_to
        return item

    if action in {"open_app"}:
        m = re.search(r"\bapp_name\s*=\s*([\"'])(.*?)\1", args or "", flags=re.DOTALL)
        item["app_name"] = _normalize_whitespace(m.group(2)) if m else None
        return item

    if action in {"type"}:
        m = re.search(r"\btext\s*=\s*([\"'])(.*?)\1", args or "", flags=re.DOTALL)
        item["text"] = _normalize_whitespace(m.group(2)) if m else None
        return item

    if action in {"press"}:
        item["keys"] = _normalize_whitespace(args or "")
        return item

    if action in {"scroll"}:
        m = re.search(r"\bdirection\s*=\s*([\"'])(.*?)\1", args or "", flags=re.DOTALL)
        item["direction"] = _normalize_whitespace(m.group(2)) if m else None
        return item

    if action in {"wait"}:
        m = re.search(r"\bseconds\s*=\s*([-+]?(?:\d*\.\d+|\d+))", args or "")
        item["seconds"] = float(m.group(1)) if m else None
        return item

    if action in {"mobile.home", "mobile.back", "mobile.terminate", "home", "back", "terminate", "navigate_back", "final_answer"}:
        m = re.search(r"\bstatus\s*=\s*([\"'])(.*?)\1", args or "", flags=re.DOTALL)
        item["status"] = _normalize_whitespace(m.group(2)) if m else None
        return item

    return {"action": action, "raw_args": _normalize_whitespace(args or "")}


def _point_score(a: Tuple[float, float], b: Tuple[float, float], tol: float) -> float:
    return (_score_from_abs_delta(a[0] - b[0], tol) + _score_from_abs_delta(a[1] - b[1], tol)) / 2.0


def _score_gui_action(pred_text: str, gt_text: str) -> float:
    norm_pred = _normalize_action_call(pred_text)
    norm_gt = _normalize_action_call(gt_text)
    if norm_pred and norm_pred == norm_gt:
        return 1.0
    pred = _parse_gui_action(pred_text)
    gt = _parse_gui_action(gt_text)
    if pred is None or gt is None:
        return 0.0
    if pred.get("action") != gt.get("action"):
        return 0.0
    if gt.get("point") is not None:
        if pred.get("point") is None:
            return 0.0
        return _point_score(pred["point"], gt["point"], AGUVIS_POINT_TOL)
    if gt.get("from") is not None and gt.get("to") is not None:
        if pred.get("from") is None or pred.get("to") is None:
            return 0.0
        return (_point_score(pred["from"], gt["from"], AGUVIS_DRAG_TOL) + _point_score(pred["to"], gt["to"], AGUVIS_DRAG_TOL)) / 2.0
    for key in ["app_name", "text", "keys", "direction", "status", "raw_args"]:
        gt_val = gt.get(key)
        if gt_val not in {None, ""}:
            pred_val = pred.get(key)
            if pred_val in {None, ""}:
                return 0.0
            a = _normalize_whitespace(str(pred_val)).lower()
            b = _normalize_whitespace(str(gt_val)).lower()
            if a == b:
                return 1.0
            return _sequence_similarity(a, b)
    if gt.get("seconds") is not None:
        pred_val = pred.get("seconds")
        if pred_val is None:
            return 0.0
        return _score_from_abs_delta(float(pred_val) - float(gt["seconds"]), 1.0)
    return 1.0



def _normalize_aguvis2_explanation_text(text: str) -> str:
    s = _normalize_text(text)
    s = re.sub(r"^<think>|</think>$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:summary|explanation|reasoning|rationale|thought|action)\s*:\s*", "", s, flags=re.IGNORECASE)
    return _normalize_whitespace(s)


def _aguvis2_summary_rule_score(pred_summary: str, gt_summary: str) -> float:
    a = _normalize_aguvis2_explanation_text(pred_summary)
    b = _normalize_aguvis2_explanation_text(gt_summary)
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return _sequence_similarity(a, b)

def parse_aguvis2_prediction_fields(final_response: str) -> Tuple[str, str]:
    text = final_response or ""
    summary = ""
    action = ""

    m = AGUVIS2_THINK_RE.search(text)
    if m:
        summary = _normalize_aguvis2_explanation_text(m.group(1))

    m = AGUVIS2_CODE_RE.search(text)
    if m:
        action = _normalize_text(m.group(1))

    if not summary:
        m = re.search(r"(?:summary|explanation|reasoning|thought)\s*:\s*(.*?)(?:\n\s*(?:action|api(?:\s*|_)call)\s*:|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            summary = _normalize_aguvis2_explanation_text(m.group(1))

    if not action:
        m = re.search(r"(?:action|api(?:\s*|_)call)\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            action = _normalize_text(m.group(1))

    if not action:
        code_fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if code_fence:
            action = _normalize_text(code_fence.group(1))

    if not action:
        name, args = _extract_action_call(text)
        if name is not None:
            action = f"{name}({args})" if args is not None else name

    if not summary:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith("<code>") or ln.startswith("</code>"):
                continue
            if re.match(r"^(?:action|api(?:\s*|_)call)\s*:", ln, flags=re.IGNORECASE):
                continue
            if "(" in ln and ")" in ln:
                continue
            summary = _normalize_aguvis2_explanation_text(ln)
            if summary:
                break

    return summary, action


def _normalize_response(text: str, target_role: str) -> str:
    text = (text or "").strip()
    if target_role == "function_call":
        j = _canonicalize_jsonish(text)
        if j:
            return j
    return _normalize_whitespace(text)


def _boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.findall(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
    if m:
        return _normalize_answer_text(m[-1])
    return None


def _math_score(pred: str, gt: str) -> Optional[float]:
    if not HAVE_MATH_VERIFY:
        return None
    try:
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", pred.strip()):
            pred_parsed = parse(pred)
        else:
            pred_parsed = parse(pred, extraction_config=[LatexExtractionConfig()])
        # DAPO's reference is normally a bare number; model answers use LaTeX.
        gt_parsed = parse(gt)
        if not pred_parsed or not gt_parsed:
            return None
        ok = verify(gt_parsed, pred_parsed)
        return 1.0 if ok else 0.0
    except Exception:
        return None


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm: Dict[str, Any] = {}
    if image_inputs is not None:
        mm["image"] = image_inputs
    if video_inputs is not None:
        mm["video"] = video_inputs
    if not mm:
        return {"prompt": text}
    return {"prompt": text, "multi_modal_data": mm, "mm_processor_kwargs": video_kwargs}


def _build_messages_with_optional_images(system_prompt: str, user_text: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
    if images:
        user_content: List[Dict[str, Any]] = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def build_general_judge_messages(row: Dict[str, Any], final_response: str) -> List[Dict[str, Any]]:
    question = str(row.get("question") or "").strip()
    reference = str(row.get("original_answer") or "").strip()
    user_text = (
        f"Question / task:\n{question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Student final response:\n{final_response}\n\n"
        "Grade how correct the student's final response is on a scale from 0 to 1. "
        "Reward semantically correct answers even if wording differs."
    )
    images = [_to_pil(x) for x in (row.get("images") or [])]
    return _build_messages_with_optional_images(GENERAL_JUDGE_SYSTEM_PROMPT, user_text, images)


def build_aguvis_action_judge_messages(row: Dict[str, Any], final_response: str, pred_action: str, gt_action: str) -> List[Dict[str, Any]]:
    task_text = str(row.get("task_text") or row.get("question") or "").strip()
    system_prompt = str(row.get("system_prompt") or "").strip()
    user_text = (
        f"Task:\n{task_text}\n\n"
        f"System prompt:\n{system_prompt or '[missing]'}\n\n"
        f"Ground-truth action:\n{gt_action}\n\n"
        f"Student full response:\n{final_response}\n\n"
        f"Student parsed action:\n{pred_action or '[missing]'}"
    )
    images = [_to_pil(x) for x in (row.get("images") or [])]
    return _build_messages_with_optional_images(AGUVIS_ACTION_RM_SYSTEM_PROMPT, user_text, images)


def build_aguvis2_judge_messages(row: Dict[str, Any], final_response: str, pred_summary: str, pred_action: str) -> List[Dict[str, Any]]:
    gt_action = str(row.get("gt_action") or row.get("original_answer") or "").strip()
    task_text = str(row.get("task_text") or row.get("question") or "").strip()
    prev_actions = str(row.get("previous_actions") or "").strip()
    system_prompt = str(row.get("system_prompt") or "").strip()
    user_text = (
        f"Task:\n{task_text}\n\n"
        f"Previous actions:\n{prev_actions or 'None'}\n\n"
        f"System prompt:\n{system_prompt}\n\n"
        f"Ground-truth API action:\n{gt_action}\n\n"
        f"Student final action:\n{final_response or pred_action or '[missing]'}"
    )
    images = [_to_pil(x) for x in (row.get("images") or [])]
    return _build_messages_with_optional_images(AGUVIS2_JUDGE_SYSTEM_PROMPT, user_text, images)


def build_apigen_judge_messages(row: Dict[str, Any], final_response: str, normalized_pred: str, normalized_gt: str) -> List[Dict[str, Any]]:
    target_role = str(row.get("target_role") or "gpt").strip()
    system_prompt = str(row.get("system_prompt") or "").strip()
    tools = str(row.get("tools") or "").strip()
    context = str(row.get("context_text") or row.get("question") or "").strip()
    gt = str(row.get("original_answer") or "").strip()
    user_text = (
        f"Task: grade the student's next turn in context.\n\n"
        f"Conversation prefix:\n{context}\n\n"
        f"System prompt:\n{system_prompt}\n\n"
        # f"Available tools:\n{tools}\n\n"
        # f"Expected next turn type:\n{target_role}\n\n"
        f"Ground-truth next turn:\n{gt}\n\n"
        f"Student next turn:\n{final_response}\n\n"
        # f"Normalized ground truth:\n{normalized_gt or '[empty]'}\n\n"
        # f"Normalized student turn:\n{normalized_pred or '[empty]'}\n\n"
    )
    return [
        {"role": "system", "content": APIGEN_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _judge_score(obj: Dict[str, Any], key: str) -> float:
    value = obj.get(key)
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"Judge must return a numeric {key} between 0 and 1, received {value!r}")
    return float(value)


def _judge_sampling_params(kind: str) -> SamplingParams:
    keys = ["total_score"]
    if kind == "apigen":
        keys = ["relevance_score", "correctness_subscore", "policy_score", "format_score", "total_score"]
    schema = {
        "type": "object",
        "properties": {key: {"type": "number", "minimum": 0, "maximum": 1} for key in keys},
        "required": keys,
        "additionalProperties": False,
    }
    return SamplingParams(
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
        max_tokens=JUDGE_MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=schema),
    )


def load_rows(path: Path) -> Dataset:
    """Load one raw shard without materializing every row as Python objects."""
    ds = load_dataset("parquet", data_files=str(path), split="train")
    for col in RAW_REQUIRED_COLUMNS:
        if col not in ds.column_names:
            raise ValueError(f"Missing required column '{col}' in {path}")
    # Keep encoded bytes in the shard buffer and decode only the active judge batch.
    return ds.cast_column("images", Sequence(HFImage(decode=False)))


def _default_verified_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: raw.get(k) for k in RAW_REQUIRED_COLUMNS}
    row.update({
        "final_response": "",
        "pred_summary": "",
        "pred_action": "",
        "normalized_pred": "",
        "normalized_gt": "",
        "exact_match_score": 0.0,
        "summary_score": 0.0,
        "action_score": 0.0,
        "relevance_score": 0.0,
        "correctness_subscore": 0.0,
        "policy_score": 0.0,
        "format_score": 0.0,
        "rule_score": 0.0,
        "rm_score": 0.0,
        "correctness_score": 0.0,
        "correctness_label": 0,
        "judge_used": False,
        "judge_raw": "",
        "json_parsed": False,
        "action_exact_match": False,
        "eval_method": "",
    })
    return row


def clean_verified_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key in VERIFIED_FEATURES.keys():
        value = row.get(key)
        if key == "images":
            out[key] = value if isinstance(value, list) else []
        elif key in {
            "row_index", "qa_index", "turn_index", "num_prior_turns", "generation_index", "completion_length",
            "qa_relevance", "qa_visual_dependency", "qa_image_correspondence", "qa_formatting", "correctness_label"
        }:
            out[key] = int(value or 0)
        elif key in {
            "two_step_applied", "judge_used", "json_parsed", "action_exact_match"
        }:
            out[key] = bool(value)
        elif key in {
            "exact_match_score", "summary_score", "action_score", "relevance_score", "correctness_subscore",
            "policy_score", "format_score", "rule_score", "rm_score", "correctness_score"
        }:
            out[key] = float(value or 0.0)
        else:
            out[key] = "" if value is None else str(value)
    return out


def save_verified_shard(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list([clean_verified_row(r) for r in rows], features=VERIFIED_FEATURES)
    pending = out_path.with_suffix(".parquet.tmp")
    ds.to_parquet(str(pending))
    pending.replace(out_path)


def _judge_outputs(
    judge_llm: LLM,
    judge_processor: AutoProcessor,
    jobs: List[Dict[str, Any]],
) -> List[str]:
    if not jobs:
        return []
    try:
        requests = [
            prepare_inputs_for_vllm(job["messages"], judge_processor)
            for job in jobs
        ]
        sampling = [_judge_sampling_params(job["kind"]) for job in jobs]
        outputs = judge_llm.generate(requests, sampling, use_tqdm=False)
        texts = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            texts.append(extract_final_response(text))
        torch.cuda.empty_cache()
        gc.collect()
        return texts
    except Exception as exc:
        if "maximum context length" in str(exc).lower():
            raise ValueError("Judge prompt exceeds its context. Increase --judge-max-model-len and rerun labeling.") from exc
        raise


def _apply_judge_batch(
    prepared_rows: List[Dict[str, Any]],
    judge_jobs: List[Dict[str, Any]],
    raw_outputs: List[str],
    global_stats: Dict[str, Any],
) -> None:
    """Apply one bounded judge batch and release its multimodal requests."""
    if len(raw_outputs) != len(judge_jobs):
        raise RuntimeError(f"Expected {len(judge_jobs)} judge responses, received {len(raw_outputs)}")
    global_stats["num_judge_calls"] += len(judge_jobs)
    for job, judge_raw in zip(judge_jobs, raw_outputs):
        row = prepared_rows[job["row_idx"]]
        row["judge_used"] = True
        row["judge_raw"] = judge_raw
        kind = job["kind"]
        obj = _extract_json_object(judge_raw)
        if obj is None:
            raise ValueError(
                f"Judge returned incomplete JSON for row {job['row_idx']}; expected total_score. "
                f"Check --judge-max-tokens. Received {judge_raw[:160]!r}"
            )
        score = _judge_score(obj, "total_score")
        row["json_parsed"] = True
        row["rm_score"] = score

        if kind == "general":
            row["correctness_score"] = score
            row["correctness_label"] = _label_from_score(score)
        elif kind == "aguvis_action":
            rule_score = float(job.get("rule_score", 0.0))
            final_score = max(score, rule_score)
            row["rule_score"] = rule_score
            row["action_score"] = final_score
            row["correctness_score"] = final_score
            row["correctness_label"] = _label_from_score(final_score)
        elif kind == "aguvis2":
            action_rule_score = float(job.get("action_rule_score", 0.0))
            final_score = max(score, action_rule_score)
            row["rule_score"] = action_rule_score
            row["summary_score"] = -1.0
            row["action_score"] = final_score
            row["correctness_score"] = final_score
            row["correctness_label"] = _label_from_score(final_score)
        elif kind == "apigen":
            rule_score = float(job.get("rule_score", 0.0))
            final_total = max(score, rule_score)
            row["rule_score"] = rule_score
            for key in ("relevance_score", "correctness_subscore", "policy_score", "format_score"):
                row[key] = _judge_score(obj, key)
            row["correctness_score"] = final_total
            row["correctness_label"] = _label_from_score(final_total)
        else:
            raise ValueError(f"Unknown judge job kind: {kind}")


# =============================================================================
# EVALUATION PREP
# =============================================================================

def _prepare_row_for_eval(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    out = _default_verified_row(row)
    subset = str(row.get("subset_name") or "")
    final_response = extract_final_response(str(row.get("completion") or ""))
    gt = str(row.get("original_answer") or "")
    out["final_response"] = final_response

    if subset == "groundui":
        pred_box = _parse_groundui_bbox(final_response)
        gt_box = _parse_groundui_bbox(gt)
        if pred_box is not None and gt_box is not None:
            iou = _bbox_iou(pred_box, gt_box)
            center_score = (
                _score_from_abs_delta(pred_box[0] - gt_box[0], GROUNDUI_CENTER_TOL) +
                _score_from_abs_delta(pred_box[1] - gt_box[1], GROUNDUI_CENTER_TOL)
            ) / 2.0
            size_score = (
                _score_from_abs_delta(pred_box[2] - gt_box[2], GROUNDUI_SIZE_TOL) +
                _score_from_abs_delta(pred_box[3] - gt_box[3], GROUNDUI_SIZE_TOL)
            ) / 2.0
            score = _clamp01((iou + center_score + size_score) / 3.0)
            out["correctness_score"] = score
            out["correctness_label"] = _label_from_score(score)
            out["eval_method"] = "groundui_structural"
            return out, None
        job = {"kind": "general", "messages": build_general_judge_messages(row, final_response)}
        out["eval_method"] = "groundui_judge"
        return out, job

    if subset == "aguvis-stage-1":
        pred_action = final_response
        action_exact = _normalize_action_call(pred_action) == _normalize_action_call(gt) and bool(pred_action)
        action_rule_score = 1.0 if action_exact else _score_gui_action(pred_action, gt)
        out["pred_action"] = pred_action
        out["action_exact_match"] = action_exact
        out["action_score"] = action_rule_score
        out["rule_score"] = action_rule_score
        job = {
            "kind": "aguvis_action",
            "messages": build_aguvis_action_judge_messages(row, final_response, pred_action, gt),
            "rule_score": action_rule_score,
        }
        out["eval_method"] = "aguvis_stage1_rm"
        return out, job

    if subset == "aguvis-stage-2":
        pred_summary, pred_action = parse_aguvis2_prediction_fields(final_response)
        gt_action = str(row.get("gt_action") or gt)
        out["pred_summary"] = ""
        out["pred_action"] = pred_action or final_response
        action_exact = bool(pred_action or final_response) and (_normalize_action_call(pred_action or final_response) == _normalize_action_call(gt_action))
        action_rule_score = 1.0 if action_exact else _score_gui_action(pred_action or final_response, gt_action)
        out["action_exact_match"] = action_exact
        out["summary_score"] = -1.0
        out["action_score"] = action_rule_score
        out["rule_score"] = action_rule_score
        job = {
            "kind": "aguvis2",
            "messages": build_aguvis2_judge_messages(row, final_response, pred_summary, pred_action or final_response),
            "action_rule_score": action_rule_score,
        }
        out["eval_method"] = "aguvis_stage2_final_answer_rm"
        return out, job

    if subset == "apigen-mt-5k":
        target_role = str(row.get("target_role") or "gpt")
        normalized_pred = _normalize_response(final_response, target_role)
        normalized_gt = _normalize_response(gt, target_role)
        exact_match_score = 1.0 if normalized_pred and normalized_pred == normalized_gt else 0.0
        sim = _sequence_similarity(normalized_pred, normalized_gt)
        rule_score = max(exact_match_score, sim)
        out["normalized_pred"] = normalized_pred
        out["normalized_gt"] = normalized_gt
        out["exact_match_score"] = exact_match_score
        out["rule_score"] = rule_score
        job = {
            "kind": "apigen",
            "messages": build_apigen_judge_messages(row, final_response, normalized_pred, normalized_gt),
            "rule_score": rule_score,
        }
        out["eval_method"] = "apigen_rm"
        return out, job

    if subset == "dapo":
        fast_score = _math_score(final_response, gt)
        if fast_score is not None:
            out["exact_match_score"] = fast_score
            out["correctness_score"] = fast_score
            out["correctness_label"] = _label_from_score(fast_score)
            out["eval_method"] = "math_verify"
            return out, None
        pred_box = _boxed_answer(final_response)
        gt_box = _boxed_answer(gt)
        if pred_box and gt_box and pred_box == gt_box:
            out["exact_match_score"] = 1.0
            out["correctness_score"] = 1.0
            out["correctness_label"] = 1
            out["eval_method"] = "boxed_exact"
            return out, None
        job = {"kind": "general", "messages": build_general_judge_messages(row, final_response)}
        out["eval_method"] = "math_judge"
        return out, job

    if subset == "triviaqa":
        norm_pred = _normalize_answer_text(final_response)
        norm_gt = _normalize_answer_text(gt)
        out["normalized_pred"] = norm_pred
        out["normalized_gt"] = norm_gt
        if norm_pred and norm_pred == norm_gt:
            out["exact_match_score"] = 1.0
            out["correctness_score"] = 1.0
            out["correctness_label"] = 1
            out["eval_method"] = "trivia_normalized_exact"
            return out, None
        job = {"kind": "general", "messages": build_general_judge_messages(row, final_response)}
        out["eval_method"] = "trivia_judge"
        return out, job

    # Generic path for FineVision QA-style subsets (including A-OKVQA, AI2D_merged, InfographicVQA), standard VQA-style subsets, and MM-OpenR1
    norm_pred = _normalize_answer_text(final_response)
    norm_gt = _normalize_answer_text(gt)
    out["normalized_pred"] = norm_pred
    out["normalized_gt"] = norm_gt
    if norm_pred and norm_gt and norm_pred == norm_gt:
        out["exact_match_score"] = 1.0
        out["correctness_score"] = 1.0
        out["correctness_label"] = 1
        out["eval_method"] = "normalized_exact"
        return out, None
    pred_box = _boxed_answer(final_response)
    gt_box = _boxed_answer(gt)
    if pred_box and gt_box and pred_box == gt_box:
        out["exact_match_score"] = 1.0
        out["correctness_score"] = 1.0
        out["correctness_label"] = 1
        out["eval_method"] = "boxed_exact"
        return out, None
    job = {"kind": "general", "messages": build_general_judge_messages(row, final_response)}
    out["eval_method"] = "general_judge"
    return out, job


# =============================================================================
# MAIN
# =============================================================================

def main():
    global RUN_NAME, SAVE_ROOT, RAW_SAVE_DIR, VERIFIED_SAVE_DIR, VERIFICATION_STATS_PATH
    global JUDGE_MODEL_ID, JUDGE_BATCH_SIZE, JUDGE_GPU_MEMORY_UTILIZATION
    global JUDGE_TENSOR_PARALLEL_SIZE, JUDGE_MAX_MODEL_LEN, JUDGE_MAX_TOKENS

    args = _parse_args()
    if args.run_root is not None:
        SAVE_ROOT = Path(args.run_root)
    elif args.run_name is not None:
        RUN_NAME = args.run_name
        SAVE_ROOT = DEFAULT_DATA_ROOT / RUN_NAME

    RAW_SAVE_DIR = SAVE_ROOT / "raw"
    VERIFIED_SAVE_DIR = SAVE_ROOT / "verified"
    VERIFICATION_STATS_PATH = SAVE_ROOT / "verification_stats.json"

    if args.judge_model_id is not None:
        JUDGE_MODEL_ID = args.judge_model_id
    if args.judge_batch_size is not None:
        JUDGE_BATCH_SIZE = int(args.judge_batch_size)
    if args.judge_gpu_memory_utilization is not None:
        JUDGE_GPU_MEMORY_UTILIZATION = float(args.judge_gpu_memory_utilization)
    if args.judge_tensor_parallel_size is not None:
        JUDGE_TENSOR_PARALLEL_SIZE = int(args.judge_tensor_parallel_size)
    if args.judge_max_model_len is not None:
        JUDGE_MAX_MODEL_LEN = int(args.judge_max_model_len)
    if args.judge_max_tokens is not None:
        JUDGE_MAX_TOKENS = int(args.judge_max_tokens)
    if min(JUDGE_BATCH_SIZE, JUDGE_MAX_TOKENS, JUDGE_MAX_MODEL_LEN, JUDGE_TENSOR_PARALLEL_SIZE) < 1:
        raise ValueError("Judge batch size, token limits and tensor parallel size must be positive")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(SAVE_ROOT.parent / f".{SAVE_ROOT.name}.data.lock"), timeout=0):
        _label(args)


def _label(args):
    print(f"[runtime] raw_dir={RAW_SAVE_DIR}")
    print(f"[runtime] verified_dir={VERIFIED_SAVE_DIR}")
    print(f"[runtime] judge_model_id={JUDGE_MODEL_ID}")

    if OVERWRITE_VERIFIED_OUTPUT and VERIFIED_SAVE_DIR.exists():
        import shutil
        shutil.rmtree(VERIFIED_SAVE_DIR)

    VERIFIED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(Path(RAW_SAVE_DIR).glob(RAW_SHARD_GLOB))
    if not raw_paths:
        raise FileNotFoundError(f"No raw shards found under {RAW_SAVE_DIR}")

    if VERIFICATION_STATS_PATH.exists() and any(VERIFIED_SAVE_DIR.glob(RAW_SHARD_GLOB)):
        previous = json.loads(VERIFICATION_STATS_PATH.read_text(encoding="utf-8"))
        if previous["model_id"] != JUDGE_MODEL_ID or previous.get("model_revision") != args.judge_revision:
            raise ValueError("Existing verified shards used a different judge. Use a separate run root.")
    judge_processor = None
    judge_llm = None

    global_stats = {
        "model_id": JUDGE_MODEL_ID,
        "model_revision": args.judge_revision,
        "num_raw_shards": len(raw_paths),
        "processed_shards": 0,
        "skipped_shards": 0,
        "num_rows": 0,
        "num_judge_calls": 0,
        "per_subset": {},
    }
    _save_json(VERIFICATION_STATS_PATH, global_stats)

    for raw_path in raw_paths:
        out_path = VERIFIED_SAVE_DIR / raw_path.name
        if RESUME_IF_VERIFIED_EXISTS and out_path.exists() and not OVERWRITE_VERIFIED_OUTPUT:
            table = pq.read_table(out_path, columns=["subset_name", "judge_used"])
            if table.num_rows != pq.ParquetFile(raw_path).metadata.num_rows:
                raise ValueError(f"Verified and raw row counts differ for {raw_path.name}")
            for subset, used in zip(table.column(0).to_pylist(), table.column(1).to_pylist()):
                subset_stats = global_stats["per_subset"].setdefault(subset, {"rows": 0, "judge_rows": 0})
                subset_stats["rows"] += 1
                subset_stats["judge_rows"] += int(used)
                global_stats["num_judge_calls"] += int(used)
            global_stats["num_rows"] += table.num_rows
            print(f"[skip] verified shard exists: {out_path}")
            global_stats["skipped_shards"] += 1
            continue

        print(f"[load] {raw_path}")
        rows = load_rows(raw_path)
        prepared_rows: List[Dict[str, Any]] = []
        judge_jobs: List[Dict[str, Any]] = []
        shard_judge_rows = 0
        shard_judge_completed = 0

        for i, row in enumerate(rows):
            prepared, job = _prepare_row_for_eval(row)
            prepared_rows.append(prepared)
            subset = prepared["subset_name"]
            subset_stats = global_stats["per_subset"].setdefault(subset, {"rows": 0, "judge_rows": 0})
            subset_stats["rows"] += 1
            global_stats["num_rows"] += 1
            if job is not None:
                job["row_idx"] = i
                judge_jobs.append(job)
                subset_stats["judge_rows"] += 1
                shard_judge_rows += 1

            # Keep decoded image messages bounded instead of constructing
            # requests for the entire 4,000-row shard before inference.
            if judge_jobs and (len(judge_jobs) >= JUDGE_BATCH_SIZE or i + 1 == len(rows)):
                if judge_llm is None:
                    judge_processor = AutoProcessor.from_pretrained(
                        JUDGE_MODEL_ID, revision=args.judge_revision, trust_remote_code=args.judge_trust_remote_code
                    )
                    judge_llm = LLM(
                        model=JUDGE_MODEL_ID,
                        revision=args.judge_revision,
                        trust_remote_code=args.judge_trust_remote_code,
                        gpu_memory_utilization=JUDGE_GPU_MEMORY_UTILIZATION,
                        enforce_eager=args.judge_enforce_eager,
                        max_num_seqs=JUDGE_BATCH_SIZE,
                        tensor_parallel_size=JUDGE_TENSOR_PARALLEL_SIZE,
                        max_model_len=JUDGE_MAX_MODEL_LEN,
                        structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
                    )
                raw_outputs = _judge_outputs(judge_llm, judge_processor, judge_jobs)
                _apply_judge_batch(prepared_rows, judge_jobs, raw_outputs, global_stats)
                shard_judge_completed += len(judge_jobs)
                print(
                    f"[judge] {raw_path.name} completed={shard_judge_completed}/"
                    f"{len(rows)} rows batch={len(judge_jobs)}",
                    flush=True,
                )
                judge_jobs.clear()
                gc.collect()

        save_verified_shard(prepared_rows, out_path)
        print(f"[save] {out_path} rows={len(prepared_rows)} judge_rows={shard_judge_rows}")
        global_stats["processed_shards"] += 1
        _save_json(VERIFICATION_STATS_PATH, global_stats)
        del rows, prepared_rows
        gc.collect()
        torch.cuda.empty_cache()

    _save_json(VERIFICATION_STATS_PATH, global_stats)
    print(f"[done] verification complete -> {VERIFIED_SAVE_DIR}")


if __name__ == "__main__":
    main()
