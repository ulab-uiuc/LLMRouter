#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified raw generation across all current datasets:
- FineVision subsets: vqav2, scienceqa, chartqa, docvqa, screenqa, groundui, aguvis-stage-1, aokvqa, ai2d_merged, infographic_vqa
- MM-OpenR1
- DAPO-Math
- TriviaQA
- xlangai/aguvis-stage2
- Salesforce/APIGen-MT-5k

Design goals:
- One script for all sources together.
- Keeps dataset-specific preprocessing while sharing one generation loop.
- Resume support from existing raw parquet shards.
- Saves a superset schema so the labeler can handle every dataset from the same directory.
- Uses vLLM LLM.chat(...) directly for all model families.
- Uses model-family-correct prompting: Qwen3/Qwen3.5 via enable_thinking, Gemma 4 via <|think|> in the system prompt.
- Saves the whole raw response exactly as generated, with no manual </think> injection.

Notes:
- Edit CONFIG only; no argparse on purpose, matching your earlier scripts.
- For AGUVIS-stage2, the prompt asks the model to return both a short summary and one exact API call.
- For APIGen-MT-5k, each conversation row is expanded into multiple training targets: one per assistant/function_call turn.
"""

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import io
import gc
import re
import json
import hashlib
import base64
import shutil
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict, defaultdict

import torch
from PIL import Image
import pyarrow.parquet as pq
from datasets import load_dataset, Dataset, Features, Sequence, Value, Image as HFImage
from filelock import FileLock
from vllm import LLM, SamplingParams

# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_DATA_ROOT = Path("data/train")

RUN_NAME = "Qwen3_5_2B_All_Mixed_Sources_120k"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_FAMILY = "auto"      # auto | qwen3_5 | qwen3 | qwen3_vl | gemma4
THINKING_MODE = "auto"     # auto | on | off

SAVE_ROOT = DEFAULT_DATA_ROOT / "Qwen3.5" / RUN_NAME
RAW_SAVE_DIR = SAVE_ROOT / "raw"
SELECTION_MANIFEST_PATH = SAVE_ROOT / "selection_manifest.json"
GENERATION_STATS_PATH = SAVE_ROOT / "generation_stats.json"

RESUME_IF_OUTPUT_EXISTS = True
OVERWRITE_OUTPUT = False

TOTAL_QA_PAIRS = 120_000
NUM_GENERATIONS = 1
MAX_MODEL_LEN = 32768

ACTIVE_MODEL_FAMILY = None
ACTIVE_THINKING_ENABLED = False

SOURCE_PORTIONS: "OrderedDict[str, float]" = OrderedDict([
    ("vqav2", 2),
    ("scienceqa", 0.5),
    ("chartqa", 0.5),
    ("docvqa", 0.5),
    ("screenqa", 0.5),
    ("aokvqa", 2),
    ("ai2d_merged", 2),
    ("infographic_vqa", 2),
    ("groundui", 0.5),
    ("aguvis-stage-1", 1),
    ("aguvis-stage-2", 2),
    ("mm-openr1", 2),
    ("dapo", 2),
    ("triviaqa", 2),
    ("apigen-mt-5k", 4),
])

SOURCE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vqav2": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "vqav2",
        "split": "train",
        "macro_category": "General VQA",
        "system_prompt": "Answer the visual question accurately and return the final answer.",
        "modality": "vision",
    },
    "scienceqa": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "scienceqa",
        "split": "train",
        "macro_category": "Science",
        "system_prompt": "Choose the best answer based on the question and image, and return the final answer.",
        "modality": "vision",
    },
    "chartqa": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "chartqa",
        "split": "train",
        "macro_category": "Chart & Table",
        "system_prompt": "Read the chart carefully and return the final answer.",
        "modality": "vision",
    },
    "docvqa": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "docvqa",
        "split": "train",
        "macro_category": "OCR QA",
        "system_prompt": "Read the document carefully and return the answer text.",
        "modality": "vision",
    },
    "screenqa": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "screenqa",
        "split": "train",
        "macro_category": "Screen QA",
        "system_prompt": "Answer the question using the screen content and return the answer.",
        "modality": "vision",
    },
    "aokvqa": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "aokvqa",
        "split": "train",
        "macro_category": "Knowledge-intensive VQA",
        "system_prompt": "Answer the question using the image and commonsense or world knowledge when needed. Return only the final answer.",
        "modality": "vision",
    },
    "ai2d_merged": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "ai2d_merged",
        "split": "train",
        "macro_category": "Diagram Reasoning",
        "system_prompt": "Read the diagram carefully, reason about it, and return only the final answer.",
        "modality": "vision",
    },
    "infographic_vqa": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "infographic_vqa",
        "split": "train",
        "macro_category": "Infographic QA",
        "system_prompt": "Read the infographic carefully and return only the final answer.",
        "modality": "vision",
    },
    "groundui": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "groundui",
        "split": "train",
        "macro_category": "Grounding",
        "system_prompt": (
            "Identify the correct UI element and return exactly one normalized bounding box "
            "in this format: Normalized bounding box (center_x, center_y, width, height): [x, y, w, h]."
        ),
        "modality": "vision",
    },
    "aguvis-stage-1": {
        "kind": "finevision",
        "dataset_id": "HuggingFaceM4/FineVision",
        "dataset_config": "aguvis-stage-1",
        "split": "train",
        "macro_category": "GUI / Agentic",
        "system_prompt": (
            "Return exactly one grounded GUI action using normalized coordinates in the range [0,1]. "
            "Return only the action. Follow the action style used in the dataset. "
            "Examples include click(x=0.4174, y=0.4735), double_click(x=0.4358, y=0.8844), "
            "and drag(from_coord=[0.2, 0.3174], to_coord=[0.3219, 0.3174])."
        ),
        "modality": "vision",
        "streaming": True,
    },
    "aguvis-stage-2": {
        "kind": "aguvis_stage2",
        "dataset_id": "smolagents/aguvis-stage-2",
        "dataset_config": "android_control",
        "split": "train",
        "macro_category": "GUI / Agentic",
        "modality": "vision",
        "streaming": False,
    },
    "mm-openr1": {
        "kind": "mm_openr1",
        "dataset_id": "lmms-lab/multimodal-open-r1-8k-verified",
        "split": "train",
        "macro_category": "Math / Reasoning",
        "system_prompt": "Please reason carefully and put your final answer inside \\boxed{}.",
        "modality": "vision",
    },
    "dapo": {
        "kind": "text_qa",
        "dataset_id": "open-r1/DAPO-Math-17k-Processed",
        "dataset_config": "en",
        "split": "train",
        "macro_category": "Text Math",
        "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.",
        "modality": "text",
    },
    "triviaqa": {
        "kind": "text_qa",
        "dataset_id": "mandarjoshi/trivia_qa",
        "dataset_config": "rc",
        "split": "train",
        "macro_category": "Text Trivia",
        "system_prompt": "This is a trivia question. Put your final answer within \\boxed{}.",
        "modality": "text",
    },
    "apigen-mt-5k": {
        "kind": "apigen_mt_5k",
        "dataset_id": "Salesforce/APIGen-MT-5k",
        "split": "train",
        "macro_category": "Tool Use / Multi-turn",
        "modality": "text",
    },
}

USE_EACH_QA_PAIR_IN_TEXTS = True
QA_MIN_RELEVANCE = 4
QA_MIN_VISUAL_DEPENDENCY = 1
QA_MIN_IMAGE_CORRESPONDENCE = 1
QA_MIN_FORMATTING = 3
MAX_ROWS_SCAN_PER_FINEVISION_SUBSET = None

STREAMING_SHUFFLE_BUFFER_SIZE = 10_000
STREAMING_MAX_ROWS_PER_SOURCE = {
    "aguvis-stage-1": None,
    "aguvis-stage-2": None,
}
STREAMING_STOP_WHEN_TARGET_REACHED = True

SEED = 1337
GEN_CHUNK_SIZE = 128
RAW_SHARD_SIZE = 4000
GPU_MEMORY_UTILIZATION = 0.90
TENSOR_PARALLEL_SIZE = 1
LIMIT_MM_PER_PROMPT = {"image": 8}

VISION_MAX_TOKENS = 16000
VISION_TEMPERATURE = 0.6
VISION_TOP_P = 0.95
VISION_TOP_K = 20
VISION_MIN_P = 0.0
VISION_REPETITION_PENALTY = 1.0
VISION_PRESENCE_PENALTY = 0.0

TEXT_MAX_TOKENS = 16000
TEXT_TEMPERATURE = 1.0
TEXT_TOP_P = 0.95
TEXT_TOP_K = 20
TEXT_MIN_P = 0.0
TEXT_REPETITION_PENALTY = 1.0
TEXT_PRESENCE_PENALTY = 1.5

DEBUG_MODE = False
DEBUG_PROMPT_PREVIEWS = 2
DEBUG_COMPLETION_PREVIEWS = 3
DEBUG_TEXT_PREVIEW_CHARS = 220
DEBUG_PRINT_REQUEST_DETAILS_EVERY_CHUNK = False
DEBUG_PRINT_COMPLETION_DETAILS_EVERY_CHUNK = False

TEXT_PROMPT_CANDIDATES = ["prompt", "question", "query", "input", "instruction"]
TEXT_ANSWER_CANDIDATES = ["answer", "final_answer", "target", "label", "answers"]
MM_OPENR1_IMAGE_CANDS = ["image", "images", "img", "picture"]
QUESTION_MARK_RE = re.compile(r"\bQuestion\s*:\s*", flags=re.IGNORECASE)

AGUVIS2_CONVERSATION_COLUMN_CANDIDATES = ["conversations", "conversation", "messages", "dialog"]
AGUVIS2_TEXTS_COLUMN_CANDIDATES = ["texts", "text", "records"]
AGUVIS2_IMAGE_COLUMN_CANDIDATES = ["images", "image", "screenshot", "screenshots"]
AGUVIS2_ALLOWED_SOURCES = {"android_control"}
AGUVIS2_TASK_RE = re.compile(r"Instruction\s*:\s*(.*?)(?:\n\nPrevious actions\s*:|$)", flags=re.IGNORECASE | re.DOTALL)
AGUVIS2_PREV_ACTIONS_RE = re.compile(r"Previous actions\s*:\s*(.*)$", flags=re.IGNORECASE | re.DOTALL)
AGUVIS2_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.IGNORECASE | re.DOTALL)
AGUVIS2_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", flags=re.IGNORECASE | re.DOTALL)
AGUVIS2_FORMAT_INSTRUCTION = (
    "\n\nReturn exactly one <think>...</think> block followed by one <code>...</code> block and nothing else.\n"
    "- In <think>, write one short sentence describing the next move.\n"
    "- In <code>, write exactly one API call in the dataset style.\n"
    "- Do not use markdown fences. Do not add any extra prose outside the tags."
)

APIGEN_NEXT_TURN_INSTRUCTION = (
    "Continue the conversation by writing the next assistant turn only.\n"
    "The next turn may be either:\n"
    "- a normal assistant reply, or\n"
    "- a single function-call JSON object\n"
    "Choose whichever is appropriate from the conversation state.\n"
    "Do not add role labels. Do not explain what you are doing outside the next turn itself."
)
APIGEN_TURN_ROLE_CANONICAL = {
    "assistant": "gpt",
    "model": "gpt",
    "tool": "observation",
    "function": "observation",
    "api": "observation",
    "user": "human",
}

RAW_FEATURES = Features({
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



def _infer_model_family(model_id: str, requested_family: str = "auto") -> str:
    requested_family = (requested_family or "auto").strip().lower()
    if requested_family != "auto":
        return requested_family
    mid = (model_id or "").strip().lower()
    if "qwen3.5" in mid:
        return "qwen3_5"
    if "qwen3-vl" in mid:
        return "qwen3_vl"
    if "gemma-4" in mid:
        return "gemma4"
    if "qwen3" in mid:
        return "qwen3"
    return "other"


def _resolve_thinking_enabled(model_id: str, model_family: str, thinking_mode: str) -> bool:
    mode = (thinking_mode or "auto").strip().lower()
    if mode == "on":
        return True
    if mode == "off":
        return False

    mid = (model_id or "").strip().lower()
    if model_family == "qwen3_5":
        # Official Qwen3.5 model cards:
        # - 2B defaults to non-thinking
        # - 4B defaults to thinking
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


def _default_family_output_dir(model_family: str) -> str:
    return {
        "qwen3_5": "Qwen3.5",
        "qwen3_vl": "Qwen3VL",
        "qwen3": "Qwen3",
        "gemma4": "Gemma4",
    }.get(model_family, "misc")


def _sanitize_model_id_for_name(model_id: str) -> str:
    s = (model_id or "").strip()
    s = s.replace("/", "__").replace(".", "_").replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _default_run_name(model_id: str, thinking_enabled: bool) -> str:
    think_tag = "think_on" if thinking_enabled else "think_off"
    return f"{_sanitize_model_id_for_name(model_id)}_{think_tag}_hard_Mixed_Sources_120k"


def _remove_gemma_think_prefix(text: str) -> str:
    text = text or ""
    return re.sub(r"^\s*<\|think\|>\s*\n?", "", text, count=1)


def _apply_runtime_prompting(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global ACTIVE_MODEL_FAMILY, ACTIVE_THINKING_ENABLED
    family = ACTIVE_MODEL_FAMILY or _infer_model_family(MODEL_ID, MODEL_FAMILY)

    # Shallow copy top-level list / dicts; preserve PIL images and multimodal payloads.
    patched: List[Dict[str, Any]] = []
    for msg in messages:
        cloned = dict(msg)
        if isinstance(cloned.get("content"), list):
            cloned["content"] = list(cloned["content"])
        patched.append(cloned)

    if family != "gemma4":
        return patched

    if not patched or patched[0].get("role") != "system":
        if ACTIVE_THINKING_ENABLED:
            patched.insert(0, {"role": "system", "content": "<|think|>"})
        return patched

    system_content = patched[0].get("content", "")
    if not isinstance(system_content, str):
        return patched

    system_content = _remove_gemma_think_prefix(system_content)
    if ACTIVE_THINKING_ENABLED:
        system_content = "<|think|>\n" + system_content if system_content else "<|think|>"
    patched[0]["content"] = system_content
    return patched


def _chat_template_kwargs_for_runtime() -> Optional[Dict[str, Any]]:
    global ACTIVE_MODEL_FAMILY, ACTIVE_THINKING_ENABLED
    family = ACTIVE_MODEL_FAMILY or _infer_model_family(MODEL_ID, MODEL_FAMILY)

    if family in {"qwen3_5", "qwen3"}:
        return {"enable_thinking": bool(ACTIVE_THINKING_ENABLED)}
    return None


def _apply_family_generation_defaults(model_family: str):
    global VISION_TEMPERATURE, VISION_TOP_P, VISION_TOP_K, VISION_MIN_P
    global VISION_REPETITION_PENALTY, VISION_PRESENCE_PENALTY
    global TEXT_TEMPERATURE, TEXT_TOP_P, TEXT_TOP_K, TEXT_MIN_P
    global TEXT_REPETITION_PENALTY, TEXT_PRESENCE_PENALTY

    if model_family == "gemma4":
        # Recommended Gemma 4 sampling behavior.
        VISION_TEMPERATURE = 1.0
        VISION_TOP_P = 0.95
        VISION_TOP_K = 64
        VISION_MIN_P = 0.0
        VISION_REPETITION_PENALTY = 1.0
        VISION_PRESENCE_PENALTY = 0.0

        TEXT_TEMPERATURE = 1.0
        TEXT_TOP_P = 0.95
        TEXT_TOP_K = 64
        TEXT_MIN_P = 0.0
        TEXT_REPETITION_PENALTY = 1.0
        TEXT_PRESENCE_PENALTY = 0.0


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--model-revision", help="Backbone revision used by the training recipe.")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--model-family", default=None, choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4"])
    ap.add_argument("--thinking-mode", default=None, choices=["auto", "on", "off"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--total-qa-pairs", type=int, default=None)
    ap.add_argument("--source", choices=SOURCE_CONFIGS, help="Generate only this source from the mixture.")
    ap.add_argument("--source-file", type=Path, help="Local Parquet file with the selected source's original columns.")
    ap.add_argument("--max-new-tokens", type=int, help="Limit completion length for both text and vision.")
    ap.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph capture for smaller runs.")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gen-chunk-size", type=int, default=None)
    ap.add_argument("--raw-shard-size", type=int, default=None)
    return ap.parse_args()

def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\x00", " ").strip()


def _clean_mm_openr1_question(q: Any) -> str:
    t = _normalize_text(q)
    m = QUESTION_MARK_RE.search(t)
    if not m:
        return t
    return t[m.end():].strip()


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


def _normalize_portions(portions: "OrderedDict[str, float]") -> OrderedDict:
    total = sum(float(v) for v in portions.values())
    if total <= 0:
        raise ValueError(f"Total SOURCE_PORTIONS must be > 0, got {total}")
    out = OrderedDict()
    for name, value in portions.items():
        value = float(value)
        if value < 0:
            raise ValueError(f"Portion for {name} must be >= 0, got {value}")
        out[name] = value / total
    return out


def _allocate_counts(total: int, portions: "OrderedDict[str, float]") -> OrderedDict:
    ratios = _normalize_portions(portions)
    floors = []
    used = 0
    for name, ratio in ratios.items():
        exact = total * ratio
        base = floor(exact)
        frac = exact - base
        floors.append((name, base, frac))
        used += base
    remainder = total - used
    floors.sort(key=lambda x: (-x[2], x[0]))
    extras = {name: 0 for name in ratios.keys()}
    for i in range(remainder):
        extras[floors[i][0]] += 1
    out = OrderedDict()
    base_map = {n: b for n, b, _ in floors}
    for name in ratios.keys():
        out[name] = base_map[name] + extras[name]
    return out


def _messages_from_example(images: List[Image.Image], question: str, system_prompt: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if images:
        content: List[Dict[str, Any]] = []
        for img in images:
            content.append({"type": "image_pil", "image_pil": img})
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": question})
    return messages


def _save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".json.tmp")
    with pending.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    pending.replace(path)


def _save_shard(rows: List[Dict[str, Any]], save_dir: Path, shard_idx: int) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    shard_path = save_dir / f"shard-{shard_idx:06d}.parquet"
    ds = Dataset.from_list(rows, features=RAW_FEATURES)
    pending = shard_path.with_suffix(".parquet.tmp")
    ds.to_parquet(str(pending))
    pending.replace(shard_path)
    return shard_path


def _should_resume() -> bool:
    return RESUME_IF_OUTPUT_EXISTS and SAVE_ROOT.exists()


def _prepare_output_dirs() -> bool:
    if RESUME_IF_OUTPUT_EXISTS and OVERWRITE_OUTPUT:
        raise ValueError("Set only one of RESUME_IF_OUTPUT_EXISTS or OVERWRITE_OUTPUT, not both.")
    resume_mode = _should_resume()
    if resume_mode:
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        RAW_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        return True
    if OVERWRITE_OUTPUT and SAVE_ROOT.exists():
        shutil.rmtree(SAVE_ROOT)
    RAW_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    return False


def _list_existing_raw_shards() -> List[Path]:
    if not RAW_SAVE_DIR.exists():
        return []
    return sorted(RAW_SAVE_DIR.glob("shard-*.parquet"))


def _next_shard_index_from_disk() -> int:
    shards = _list_existing_raw_shards()
    if not shards:
        return 1
    max_idx = 0
    for path in shards:
        m = re.match(r"shard-(\d{6})\.parquet$", path.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _spec_identity(spec: Dict[str, Any]) -> Tuple[str, int, int, int]:
    return (
        str(spec.get("subset_name", "")),
        int(spec.get("row_index", -1)),
        int(spec.get("qa_index", -1)),
        int(spec.get("turn_index", -1)),
    )


def _load_completed_generations() -> Tuple[Dict[Tuple[str, int, int, int], int], set]:
    completed_counts: Dict[Tuple[str, int, int, int], int] = defaultdict(int)
    completed_generation_keys = set()
    for shard_path in _list_existing_raw_shards():
        table = pq.read_table(
            shard_path,
            columns=["subset_name", "row_index", "qa_index", "turn_index", "generation_index"],
        )
        subset_names = table.column("subset_name").to_pylist()
        row_indices = table.column("row_index").to_pylist()
        qa_indices = table.column("qa_index").to_pylist()
        turn_indices = table.column("turn_index").to_pylist()
        generation_indices = table.column("generation_index").to_pylist()
        for subset_name, row_index, qa_index, turn_index, generation_index in zip(
            subset_names, row_indices, qa_indices, turn_indices, generation_indices
        ):
            spec_key = (str(subset_name), int(row_index), int(qa_index), int(turn_index))
            gen_key = spec_key + (int(generation_index),)
            if gen_key not in completed_generation_keys:
                completed_generation_keys.add(gen_key)
                completed_counts[spec_key] += 1
    return dict(completed_counts), completed_generation_keys


def _load_source(source_name: str, source_seed: int, source_file: Optional[Path] = None):
    cfg = SOURCE_CONFIGS[source_name]
    dataset_id = cfg["dataset_id"]
    dataset_config = cfg.get("dataset_config")
    split = cfg["split"]
    use_streaming = cfg.get("streaming", False)

    if source_file:
        ds = load_dataset("parquet", data_files=str(source_file), split="train", streaming=use_streaming)
    else:
        ds = load_dataset(dataset_id, dataset_config, split=split, streaming=use_streaming)
    if use_streaming:
        if cfg["kind"] == "finevision" and ds.features is not None:
            ds = ds.cast_column("images", Sequence(HFImage(decode=False)))
        return ds.shuffle(seed=source_seed, buffer_size=STREAMING_SHUFFLE_BUFFER_SIZE)

    if cfg["kind"] == "finevision" and MAX_ROWS_SCAN_PER_FINEVISION_SUBSET is not None and MAX_ROWS_SCAN_PER_FINEVISION_SUBSET < len(ds):
        ds = ds.select(range(MAX_ROWS_SCAN_PER_FINEVISION_SUBSET))
    if len(ds) > 0:
        ds = ds.shuffle(seed=source_seed)
    return ds


def _resolve_mm_openr1_image_col(ds) -> str:
    for c in MM_OPENR1_IMAGE_CANDS:
        if c in ds.column_names:
            return c
    raise ValueError(f"Image column not found for MM-OpenR1. Tried {MM_OPENR1_IMAGE_CANDS}. Available: {ds.column_names}")


def _get_optional_int_rating(row: Dict[str, Any], field_name: str, qa_index: int):
    candidate_names = [
        field_name,
        f"{field_name}_ratings",
        f"{field_name}_rating",
        f"{field_name}s",
    ]
    for candidate in candidate_names:
        if candidate not in row:
            continue
        values = row[candidate]
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            if qa_index >= len(values):
                continue
            try:
                return int(values[qa_index])
            except Exception:
                continue
        try:
            return int(values)
        except Exception:
            continue

    min_candidate_names = [
        f"{field_name}_min",
        f"min_{field_name}",
        f"{field_name}_minimum",
    ]
    for candidate in min_candidate_names:
        if candidate not in row:
            continue
        value = row[candidate]
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _qa_is_eligible(row: Dict[str, Any], qa_index: int, question: str, answer: str) -> Tuple[bool, Dict[str, int]]:
    relevance = _get_optional_int_rating(row, "relevance", qa_index)
    visual_dependency = _get_optional_int_rating(row, "visual_dependency", qa_index)
    image_correspondence = _get_optional_int_rating(row, "image_correspondence", qa_index)
    formatting = _get_optional_int_rating(row, "formatting", qa_index)

    if not question or not answer:
        return False, {
            "qa_relevance": int(relevance or -1),
            "qa_visual_dependency": int(visual_dependency or -1),
            "qa_image_correspondence": int(image_correspondence or -1),
            "qa_formatting": int(formatting or -1),
        }

    if relevance is not None and relevance < QA_MIN_RELEVANCE:
        return False, {
            "qa_relevance": int(relevance),
            "qa_visual_dependency": int(visual_dependency or -1),
            "qa_image_correspondence": int(image_correspondence or -1),
            "qa_formatting": int(formatting or -1),
        }
    if visual_dependency is not None and visual_dependency < QA_MIN_VISUAL_DEPENDENCY:
        return False, {
            "qa_relevance": int(relevance or -1),
            "qa_visual_dependency": int(visual_dependency),
            "qa_image_correspondence": int(image_correspondence or -1),
            "qa_formatting": int(formatting or -1),
        }
    if image_correspondence is not None and image_correspondence < QA_MIN_IMAGE_CORRESPONDENCE:
        return False, {
            "qa_relevance": int(relevance or -1),
            "qa_visual_dependency": int(visual_dependency or -1),
            "qa_image_correspondence": int(image_correspondence),
            "qa_formatting": int(formatting or -1),
        }
    if formatting is not None and formatting < QA_MIN_FORMATTING:
        return False, {
            "qa_relevance": int(relevance or -1),
            "qa_visual_dependency": int(visual_dependency or -1),
            "qa_image_correspondence": int(image_correspondence or -1),
            "qa_formatting": int(formatting),
        }

    return True, {
        "qa_relevance": int(relevance or -1),
        "qa_visual_dependency": int(visual_dependency or -1),
        "qa_image_correspondence": int(image_correspondence or -1),
        "qa_formatting": int(formatting or -1),
    }


def _resolve_text_col(ds, candidates: List[str], required: bool) -> Optional[str]:
    for c in candidates:
        if c in ds.column_names:
            return c
    if required:
        raise ValueError(f"Could not find required text column among {candidates}. Available: {ds.column_names}")
    return None


def _question_with_options(question: str, row: Dict[str, Any]) -> str:
    options = row.get("choices") or row.get("options") or row.get("candidates")
    if not options:
        return question
    if isinstance(options, list):
        option_lines = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx, opt in enumerate(options):
            letter = letters[idx] if idx < len(letters) else str(idx + 1)
            option_lines.append(f"({letter}) {_normalize_text(opt)}")
        return question + "\nOptions:\n" + "\n".join(option_lines)
    return question


def _sample_finevision_regular(ds, source_name: str, target_count: int):
    selected: List[Dict[str, Any]] = []
    rows_scanned = 0
    rows_with_images = 0
    qa_pairs_scanned = 0
    eligible_qa_pairs = 0
    cfg = SOURCE_CONFIGS[source_name]

    for row_index in range(len(ds)):
        row = ds[row_index]
        rows_scanned += 1
        images = row.get("images") or []
        texts = row.get("texts") or []
        if len(images) == 0:
            continue
        rows_with_images += 1
        if not USE_EACH_QA_PAIR_IN_TEXTS and len(texts) != 1:
            continue
        for qa_index, qa in enumerate(texts):
            qa_pairs_scanned += 1
            question = _normalize_text(qa.get("user", ""))
            answer = _normalize_text(qa.get("assistant", ""))
            ok, metrics = _qa_is_eligible(row, qa_index, question, answer)
            if not ok:
                continue
            eligible_qa_pairs += 1
            selected.append({
                # This Dataset remains random-accessible. Reload images only
                # for the active generation chunk instead of retaining every
                # decoded image in the selected-spec list.
                "images": None,
                "subset_name": source_name,
                "macro_category": cfg["macro_category"],
                "source": cfg["dataset_id"],
                "modality": cfg["modality"],
                "row_index": int(row_index),
                "qa_index": int(qa_index),
                "turn_index": -1,
                "question": question,
                "original_answer": answer,
                "system_prompt": cfg["system_prompt"],
                "human_prompt": "",
                "task_text": "",
                "previous_actions": "",
                "gt_reasoning": "",
                "gt_action": "",
                "tools": "",
                "context_text": "",
                "prefix_json": "",
                "last_user_or_observation": "",
                "target_role": "",
                "num_prior_turns": 0,
                **metrics,
            })
            if len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break

    stats = {
        "rows_scanned": rows_scanned,
        "rows_with_images": rows_with_images,
        "qa_pairs_scanned": qa_pairs_scanned,
        "eligible_qa_pairs": eligible_qa_pairs,
        "selected": len(selected),
        "streaming_mode": False,
    }
    return selected, stats


def _sample_finevision_streaming(ds_stream, source_name: str, target_count: int):
    selected: List[Dict[str, Any]] = []
    eligible_seen = 0
    rows_scanned = 0
    rows_with_images = 0
    qa_pairs_scanned = 0
    max_rows = STREAMING_MAX_ROWS_PER_SOURCE.get(source_name, None)
    cfg = SOURCE_CONFIGS[source_name]

    for row_index, row in enumerate(ds_stream):
        if max_rows is not None and rows_scanned >= max_rows:
            break
        rows_scanned += 1
        images = row.get("images") or []
        texts = row.get("texts") or []
        if len(images) == 0:
            continue
        rows_with_images += 1
        if not USE_EACH_QA_PAIR_IN_TEXTS and len(texts) != 1:
            continue
        for qa_index, qa in enumerate(texts):
            qa_pairs_scanned += 1
            question = _normalize_text(qa.get("user", ""))
            answer = _normalize_text(qa.get("assistant", ""))
            ok, metrics = _qa_is_eligible(row, qa_index, question, answer)
            if not ok:
                continue
            eligible_seen += 1
            selected.append({
                "images": images,
                "subset_name": source_name,
                "macro_category": cfg["macro_category"],
                "source": cfg["dataset_id"],
                "modality": cfg["modality"],
                "row_index": int(row_index),
                "qa_index": int(qa_index),
                "turn_index": -1,
                "question": question,
                "original_answer": answer,
                "system_prompt": cfg["system_prompt"],
                "human_prompt": "",
                "task_text": "",
                "previous_actions": "",
                "gt_reasoning": "",
                "gt_action": "",
                "tools": "",
                "context_text": "",
                "prefix_json": "",
                "last_user_or_observation": "",
                "target_role": "",
                "num_prior_turns": 0,
                **metrics,
            })
            if STREAMING_STOP_WHEN_TARGET_REACHED and len(selected) >= target_count:
                break
        if STREAMING_STOP_WHEN_TARGET_REACHED and len(selected) >= target_count:
            break

    stats = {
        "rows_scanned": rows_scanned,
        "rows_with_images": rows_with_images,
        "qa_pairs_scanned": qa_pairs_scanned,
        "eligible_qa_pairs": eligible_seen,
        "selected": len(selected),
        "streaming_mode": True,
    }
    return selected, stats


def _sample_mm_openr1(ds, target_count: int):
    selected: List[Dict[str, Any]] = []
    image_col = _resolve_mm_openr1_image_col(ds)
    q_col = _resolve_text_col(ds, ["question", "problem", "prompt"], required=True)
    a_col = _resolve_text_col(ds, ["answer", "solution", "final_answer"], required=True)

    for row_index in range(len(ds)):
        row = ds[row_index]
        question = _clean_mm_openr1_question(row.get(q_col, ""))
        answer = _normalize_text(row.get(a_col, ""))
        image_val = row.get(image_col)
        if isinstance(image_val, list):
            images = image_val
        else:
            images = [image_val] if image_val is not None else []
        if not images or not question or not answer:
            continue
        selected.append({
            # MM-OpenR1 is random-accessible; defer image decoding until the
            # active generation chunk is built.
            "images": None,
            "subset_name": "mm-openr1",
            "macro_category": SOURCE_CONFIGS["mm-openr1"]["macro_category"],
            "source": SOURCE_CONFIGS["mm-openr1"]["dataset_id"],
            "modality": "vision",
            "row_index": int(row_index),
            "qa_index": 0,
            "turn_index": -1,
            "question": question,
            "original_answer": answer,
            "system_prompt": SOURCE_CONFIGS["mm-openr1"]["system_prompt"],
            "human_prompt": "",
            "task_text": "",
            "previous_actions": "",
            "gt_reasoning": "",
            "gt_action": "",
            "tools": "",
            "context_text": "",
            "prefix_json": "",
            "last_user_or_observation": "",
            "target_role": "",
            "num_prior_turns": 0,
            "qa_relevance": -1,
            "qa_visual_dependency": -1,
            "qa_image_correspondence": -1,
            "qa_formatting": -1,
        })
        if len(selected) >= target_count:
            break

    stats = {
        "rows_scanned": len(ds),
        "eligible_qa_pairs": len(selected),
        "selected": len(selected),
        "streaming_mode": False,
    }
    return selected, stats


def _sample_text_qa(ds, source_name: str, target_count: int):
    cfg = SOURCE_CONFIGS[source_name]
    if source_name == "dapo":
        q_col = "prompt"
        a_col = "solution"
    else:
        q_col = _resolve_text_col(ds, TEXT_PROMPT_CANDIDATES, required=True)
        a_col = _resolve_text_col(ds, TEXT_ANSWER_CANDIDATES, required=True)
    selected: List[Dict[str, Any]] = []
    rows_scanned = 0
    for row_index in range(len(ds)):
        rows_scanned += 1
        row = ds[row_index]
        question = _normalize_text(row.get(q_col, ""))
        answer = row.get(a_col)
        if source_name == "triviaqa" and isinstance(answer, dict):
            answer = answer.get("value") or answer.get("aliases")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = _normalize_text(answer)
        if not question or not answer:
            continue
        question = _question_with_options(question, row)
        selected.append({
            "images": [],
            "subset_name": source_name,
            "macro_category": cfg["macro_category"],
            "source": cfg["dataset_id"],
            "modality": "text",
            "row_index": int(row_index),
            "qa_index": 0,
            "turn_index": -1,
            "question": question,
            "original_answer": answer,
            "system_prompt": cfg["system_prompt"],
            "human_prompt": "",
            "task_text": "",
            "previous_actions": "",
            "gt_reasoning": "",
            "gt_action": "",
            "tools": "",
            "context_text": "",
            "prefix_json": "",
            "last_user_or_observation": "",
            "target_role": "",
            "num_prior_turns": 0,
            "qa_relevance": -1,
            "qa_visual_dependency": -1,
            "qa_image_correspondence": -1,
            "qa_formatting": -1,
        })
        if len(selected) >= target_count:
            break

    stats = {
        "rows_scanned": rows_scanned,
        "eligible_qa_pairs": len(selected),
        "selected": len(selected),
        "streaming_mode": False,
    }
    return selected, stats


def _resolve_column(column_names: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
    for name in candidates:
        if name in column_names:
            return name
    if required:
        raise ValueError(f"Could not resolve required column from {candidates}. Available: {column_names}")
    return None


def _ensure_image_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _clean_human_text(text: str) -> str:
    return _normalize_text(text).replace("<image>", "").strip()


def _clean_aguvis2_explanation(text: str) -> str:
    s = _normalize_text(text)
    s = re.sub(r"^<think>|</think>$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:summary|explanation|reasoning|rationale|thought|action)\s*:\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def _parse_aguvis2_assistant_text(text: str) -> Tuple[str, str]:
    s = _normalize_text(text)
    reasoning = ""
    action = ""
    m = AGUVIS2_THINK_RE.search(s)
    if m:
        reasoning = _clean_aguvis2_explanation(m.group(1))
    m = AGUVIS2_CODE_RE.search(s)
    if m:
        action = _normalize_text(m.group(1))
    if not action:
        code_fence = re.search(r"```(?:python)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
        if code_fence:
            action = _normalize_text(code_fence.group(1))
    if not action:
        m = re.search(r"(?:action|api(?:\s*|_)call)\s*:\s*(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            action = _normalize_text(m.group(1))
    if not reasoning:
        m = re.search(r"(?:summary|explanation|reasoning|thought)\s*:\s*(.*?)(?:\n\s*(?:action|api(?:\s*|_)call)\s*:|$)", s, flags=re.IGNORECASE | re.DOTALL)
        if m:
            reasoning = _clean_aguvis2_explanation(m.group(1))
    if not reasoning:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith("<code>") or re.match(r"^(?:action|api(?:\s*|_)call)\s*:", ln, flags=re.IGNORECASE):
                continue
            if "(" in ln and ")" in ln:
                continue
            reasoning = _clean_aguvis2_explanation(ln)
            if reasoning:
                break
    return reasoning, action


def _extract_task_and_history(human_prompt: str) -> Tuple[str, str]:
    task_text = ""
    previous_actions = ""
    m = AGUVIS2_TASK_RE.search(human_prompt)
    if m:
        task_text = _normalize_text(m.group(1))
    m = AGUVIS2_PREV_ACTIONS_RE.search(human_prompt)
    if m:
        previous_actions = _normalize_text(m.group(1))
    if not task_text:
        task_text = _normalize_text(human_prompt)
    return task_text, previous_actions


def _extract_aguvis_stage2_fields_from_conversations(row: Dict[str, Any], conv_col: str, image_col: str, row_index: int) -> Optional[Dict[str, Any]]:
    conversations = row.get(conv_col)
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None
    system_prompt = ""
    human_prompt = ""
    gt_reasoning = ""
    gt_action = ""
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = _normalize_text(msg.get("from", "")).lower()
        recipient = _normalize_text(msg.get("recipient", "")).lower()
        value = _normalize_text(msg.get("value", ""))
        if role == "system" and not system_prompt:
            system_prompt = value
        elif role == "human" and not human_prompt:
            human_prompt = _clean_human_text(value)
        elif role == "gpt" and recipient == "all" and not gt_reasoning:
            gt_reasoning = _clean_aguvis2_explanation(value)
        elif role == "gpt" and recipient == "os":
            gt_action = _normalize_text(value)
    if not human_prompt or not gt_action:
        return None
    raw_images = _ensure_image_list(row.get(image_col))
    if not raw_images:
        return None
    task_text, previous_actions = _extract_task_and_history(human_prompt)
    assistant_target = f"<think>\n{gt_reasoning}\n</think>\n<code>\n{gt_action}\n</code>".strip()
    return {
        # AGUVIS stage 2 is random-accessible. Do not retain thousands of
        # decoded screenshots across a dataset boundary.
        "images": None,
        "subset_name": "aguvis-stage-2",
        "macro_category": SOURCE_CONFIGS["aguvis-stage-2"]["macro_category"],
        "source": _normalize_text(row.get("source", "")) or SOURCE_CONFIGS["aguvis-stage-2"]["dataset_id"],
        "modality": "vision",
        "row_index": int(row_index),
        "qa_index": 0,
        "turn_index": -1,
        "question": human_prompt,
        "original_answer": assistant_target,
        "system_prompt": system_prompt,
        "human_prompt": human_prompt,
        "task_text": task_text,
        "previous_actions": previous_actions,
        "gt_reasoning": gt_reasoning,
        "gt_action": gt_action,
        "tools": "",
        "context_text": "",
        "prefix_json": "",
        "last_user_or_observation": "",
        "target_role": "",
        "num_prior_turns": 0,
        "qa_relevance": -1,
        "qa_visual_dependency": -1,
        "qa_image_correspondence": -1,
        "qa_formatting": -1,
    }


def _extract_aguvis_stage2_fields_from_texts(row: Dict[str, Any], texts_col: str, image_col: str, row_index: int) -> Optional[Dict[str, Any]]:
    row_source = _normalize_text(row.get("source", "")).strip().lower()
    if AGUVIS2_ALLOWED_SOURCES and row_source and row_source not in AGUVIS2_ALLOWED_SOURCES:
        return None
    texts = row.get(texts_col)
    if not isinstance(texts, list) or not texts:
        return None
    record = texts[0]
    if not isinstance(record, dict):
        return None
    system_prompt = _normalize_text(record.get("system", ""))
    human_prompt = _clean_human_text(record.get("user", ""))
    assistant_text = _normalize_text(record.get("assistant", ""))
    gt_reasoning, gt_action = _parse_aguvis2_assistant_text(assistant_text)
    if not human_prompt or not gt_action:
        return None
    raw_images = _ensure_image_list(row.get(image_col))
    if not raw_images:
        return None
    task_text, previous_actions = _extract_task_and_history(human_prompt)
    assistant_target = assistant_text or f"<think>\n{gt_reasoning}\n</think>\n<code>\n{gt_action}\n</code>".strip()
    return {
        # Reload the screenshot for the active generation chunk only.
        "images": None,
        "subset_name": "aguvis-stage-2",
        "macro_category": SOURCE_CONFIGS["aguvis-stage-2"]["macro_category"],
        "source": _normalize_text(row.get("source", "")) or SOURCE_CONFIGS["aguvis-stage-2"]["dataset_id"],
        "modality": "vision",
        "row_index": int(row_index),
        "qa_index": 0,
        "turn_index": -1,
        "question": human_prompt,
        "original_answer": assistant_target,
        "system_prompt": system_prompt,
        "human_prompt": human_prompt,
        "task_text": task_text,
        "previous_actions": previous_actions,
        "gt_reasoning": gt_reasoning,
        "gt_action": gt_action,
        "tools": "",
        "context_text": "",
        "prefix_json": "",
        "last_user_or_observation": "",
        "target_role": "",
        "num_prior_turns": 0,
        "qa_relevance": -1,
        "qa_visual_dependency": -1,
        "qa_image_correspondence": -1,
        "qa_formatting": -1,
    }


def _extract_aguvis_stage2_fields(row: Dict[str, Any], column_names: List[str], row_index: int) -> Optional[Dict[str, Any]]:
    image_col = _resolve_column(column_names, AGUVIS2_IMAGE_COLUMN_CANDIDATES, required=True)
    texts_col = _resolve_column(column_names, AGUVIS2_TEXTS_COLUMN_CANDIDATES, required=False)
    if texts_col:
        spec = _extract_aguvis_stage2_fields_from_texts(row, texts_col, image_col, row_index)
        if spec is not None:
            return spec
    conv_col = _resolve_column(column_names, AGUVIS2_CONVERSATION_COLUMN_CANDIDATES, required=False)
    if conv_col:
        return _extract_aguvis_stage2_fields_from_conversations(row, conv_col, image_col, row_index)
    return None


def _sample_aguvis_stage2(ds_stream, target_count: int):
    specs: List[Dict[str, Any]] = []
    rows_scanned = 0
    eligible_rows = 0

    it = iter(ds_stream)
    try:
        first_row = next(it)
    except StopIteration:
        return [], {"rows_scanned": 0, "eligible_rows": 0, "selected": 0, "streaming_mode": False}

    column_names = list(first_row.keys())
    image_col = _resolve_column(column_names, AGUVIS2_IMAGE_COLUMN_CANDIDATES, required=True)
    texts_col = _resolve_column(column_names, AGUVIS2_TEXTS_COLUMN_CANDIDATES, required=False)
    conv_col = _resolve_column(column_names, AGUVIS2_CONVERSATION_COLUMN_CANDIDATES, required=False)

    def handle_row(row: Dict[str, Any], row_index: int):
        nonlocal eligible_rows
        spec = _extract_aguvis_stage2_fields(row, column_names, row_index)
        if spec is None:
            return
        eligible_rows += 1
        specs.append(spec)

    rows_scanned += 1
    handle_row(first_row, 0)
    for row_index, row in enumerate(it, start=1):
        if len(specs) >= target_count:
            break
        rows_scanned += 1
        handle_row(row, row_index)

    stats = {
        "rows_scanned": rows_scanned,
        "eligible_rows": eligible_rows,
        "selected": len(specs),
        "streaming_mode": False,
        "texts_col": texts_col,
        "conv_col": conv_col,
        "image_col": image_col,
    }
    return specs, stats

def _canonical_turn_role(role: Any) -> str:
    key = _normalize_text(role).lower()
    return APIGEN_TURN_ROLE_CANONICAL.get(key, key)


def _parse_tools_text(raw_tools: Any) -> str:
    if raw_tools is None:
        return "[]"
    if isinstance(raw_tools, str):
        text = raw_tools.strip()
        try:
            obj = json.loads(text)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return text
    try:
        return json.dumps(raw_tools, ensure_ascii=False, indent=2)
    except Exception:
        return str(raw_tools)


def _turn_to_transcript_line(turn: Dict[str, Any]) -> str:
    role = _canonical_turn_role(turn.get("from", ""))
    value = _normalize_text(turn.get("value", ""))
    if role == "human":
        return f"User: {value}"
    if role == "gpt":
        return f"Assistant: {value}"
    if role == "function_call":
        return f"Assistant Function Call: {value}"
    if role == "observation":
        return f"Tool Observation: {value}"
    return f"{role}: {value}"


def _serialize_prefix(prefix_turns: List[Dict[str, Any]]) -> str:
    if not prefix_turns:
        return "[Conversation start]"
    return "\n".join(_turn_to_transcript_line(t) for t in prefix_turns)


def _last_userish_turn_text(prefix_turns: List[Dict[str, Any]]) -> str:
    for turn in reversed(prefix_turns):
        role = _canonical_turn_role(turn.get("from", ""))
        if role in {"human", "observation"}:
            return _normalize_text(turn.get("value", ""))
    return ""


def _extract_apigen_turn_specs(row: Dict[str, Any], row_index: int) -> List[Dict[str, Any]]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return []
    system_prompt = _normalize_text(row.get("system"))
    tools_text = _parse_tools_text(row.get("tools"))
    specs: List[Dict[str, Any]] = []
    prefix_turns: List[Dict[str, Any]] = []
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            continue
        role = _canonical_turn_role(turn.get("from", ""))
        value = _normalize_text(turn.get("value", ""))
        if not value:
            prefix_turns.append(turn)
            continue
        if role in {"gpt", "function_call"}:
            context_text = _serialize_prefix(prefix_turns)
            last_user_or_observation = _last_userish_turn_text(prefix_turns)
            specs.append({
                "images": [],
                "subset_name": "apigen-mt-5k",
                "macro_category": SOURCE_CONFIGS["apigen-mt-5k"]["macro_category"],
                "source": SOURCE_CONFIGS["apigen-mt-5k"]["dataset_id"],
                "modality": "text",
                "row_index": int(row_index),
                "qa_index": -1,
                "turn_index": int(turn_index),
                "question": context_text,
                "original_answer": value,
                "system_prompt": system_prompt,
                "human_prompt": "",
                "task_text": "",
                "previous_actions": "",
                "gt_reasoning": "",
                "gt_action": "",
                "tools": tools_text,
                "context_text": context_text,
                "prefix_json": json.dumps(prefix_turns, ensure_ascii=False),
                "last_user_or_observation": last_user_or_observation,
                "target_role": role,
                "num_prior_turns": int(len(prefix_turns)),
                "qa_relevance": -1,
                "qa_visual_dependency": -1,
                "qa_image_correspondence": -1,
                "qa_formatting": -1,
            })
        prefix_turns.append(turn)
    return specs


def _sample_apigen_mt_5k(ds, target_count: int):
    specs: List[Dict[str, Any]] = []
    rows_scanned = 0
    total_model_turns = 0
    gpt_turns = 0
    function_call_turns = 0
    for row_index in range(len(ds)):
        row = ds[row_index]
        rows_scanned += 1
        row_specs = _extract_apigen_turn_specs(row, row_index)
        for spec in row_specs:
            total_model_turns += 1
            if spec["target_role"] == "gpt":
                gpt_turns += 1
            elif spec["target_role"] == "function_call":
                function_call_turns += 1
            specs.append(spec)
            if len(specs) >= target_count:
                break
        if len(specs) >= target_count:
            break
    stats = {
        "rows_scanned": rows_scanned,
        "total_model_turns_found": total_model_turns,
        "gpt_turns": gpt_turns,
        "function_call_turns": function_call_turns,
        "selected": len(specs),
        "streaming_mode": False,
    }
    return specs, stats


def _sample_source(ds, source_name: str, target_count: int):
    kind = SOURCE_CONFIGS[source_name]["kind"]
    if kind == "finevision":
        if SOURCE_CONFIGS[source_name].get("streaming", False):
            return _sample_finevision_streaming(ds, source_name, target_count)
        return _sample_finevision_regular(ds, source_name, target_count)
    if kind == "mm_openr1":
        return _sample_mm_openr1(ds, target_count)
    if kind == "text_qa":
        return _sample_text_qa(ds, source_name, target_count)
    if kind == "aguvis_stage2":
        return _sample_aguvis_stage2(ds, target_count)
    if kind == "apigen_mt_5k":
        return _sample_apigen_mt_5k(ds, target_count)
    raise ValueError(f"Unsupported source kind: {kind}")


def _build_aguvis2_messages(spec: Dict[str, Any], images: List[Image.Image]) -> List[Dict[str, Any]]:
    user_text = spec["human_prompt"].rstrip()
    if AGUVIS2_FORMAT_INSTRUCTION not in user_text:
        user_text += AGUVIS2_FORMAT_INSTRUCTION
    content: List[Dict[str, Any]] = []
    for img in images:
        content.append({"type": "image_pil", "image_pil": img})
    content.append({"type": "text", "text": user_text})
    messages: List[Dict[str, Any]] = []
    if spec["system_prompt"]:
        messages.append({"role": "system", "content": spec["system_prompt"]})
    messages.append({"role": "user", "content": content})
    return messages


def _build_apigen_messages(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    combined_system = (spec["system_prompt"].strip() + "\n\nAvailable tools:\n" + spec["tools"].strip()).strip()
    user_text = f"Conversation so far:\n{spec['context_text']}\n\n{APIGEN_NEXT_TURN_INSTRUCTION}"
    return [
        {"role": "system", "content": combined_system},
        {"role": "user", "content": user_text},
    ]


def _build_conversation(spec: Dict[str, Any], ds):
    kind = SOURCE_CONFIGS[spec["subset_name"]]["kind"]
    if kind == "aguvis_stage2":
        image_col = _resolve_column(
            list(ds.column_names),
            AGUVIS2_IMAGE_COLUMN_CANDIDATES,
            required=True,
        )
        raw_images = _ensure_image_list(ds[spec["row_index"]].get(image_col))
        pil_images = [_to_pil(x) for x in raw_images]
        messages = _build_aguvis2_messages(spec, pil_images)
        return _apply_runtime_prompting(messages), raw_images
    if kind == "apigen_mt_5k":
        messages = _build_apigen_messages(spec)
        return _apply_runtime_prompting(messages), []

    if kind == "mm_openr1":
        image_col = _resolve_mm_openr1_image_col(ds)
        raw_images = _ensure_image_list(ds[spec["row_index"]].get(image_col))
    elif "images" in spec and spec["images"] is not None:
        raw_images = spec["images"]
    else:
        raw_images = ds[spec["row_index"]].get("images", []) if ds is not None else []
    pil_images = [_to_pil(x) for x in raw_images] if raw_images else []
    messages = _messages_from_example(
        images=pil_images,
        question=spec["question"],
        system_prompt=spec["system_prompt"],
    )
    return _apply_runtime_prompting(messages), raw_images


def _flush_save_if_needed(shard_rows: List[Dict[str, Any]], shard_idx: int):
    if len(shard_rows) < RAW_SHARD_SIZE:
        return shard_rows, shard_idx
    shard_path = _save_shard(shard_rows[:RAW_SHARD_SIZE], RAW_SAVE_DIR, shard_idx)
    print(f"[save] wrote raw shard {shard_idx:06d} with {RAW_SHARD_SIZE} rows -> {shard_path}")
    return shard_rows[RAW_SHARD_SIZE:], shard_idx + 1


def _flush_source_boundary(
    shard_rows: List[Dict[str, Any]],
    shard_idx: int,
    source_name: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Persist partial image shards before loading the next dataset."""
    if not shard_rows:
        return shard_rows, shard_idx
    shard_path = _save_shard(shard_rows, RAW_SAVE_DIR, shard_idx)
    print(
        f"[save] wrote source-boundary raw shard {shard_idx:06d} "
        f"with {len(shard_rows)} rows for {source_name} -> {shard_path}"
    )
    return [], shard_idx + 1


def _process_memory_summary() -> str:
    values: Dict[str, str] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM", "VmSwap"}:
                    values[key] = value.strip()
    except OSError:
        return "memory=unavailable"
    return " ".join(
        f"{key}={values.get(key, 'unknown')}"
        for key in ("VmRSS", "VmHWM", "VmSwap")
    )


def _sampling_bundle_for_modality(modality: str) -> Dict[str, Any]:
    if modality == "vision":
        return {
            "params": SamplingParams(
                temperature=VISION_TEMPERATURE,
                top_p=VISION_TOP_P,
                top_k=VISION_TOP_K,
                min_p=VISION_MIN_P,
                max_tokens=VISION_MAX_TOKENS,
                n=NUM_GENERATIONS,
                repetition_penalty=VISION_REPETITION_PENALTY,
                presence_penalty=VISION_PRESENCE_PENALTY,
            ),
        }
    if modality == "text":
        return {
            "params": SamplingParams(
                temperature=TEXT_TEMPERATURE,
                top_p=TEXT_TOP_P,
                top_k=TEXT_TOP_K,
                min_p=TEXT_MIN_P,
                max_tokens=TEXT_MAX_TOKENS,
                n=NUM_GENERATIONS,
                repetition_penalty=TEXT_REPETITION_PENALTY,
                presence_penalty=TEXT_PRESENCE_PENALTY,
            ),
        }
    raise ValueError(f"Unknown modality: {modality}")


def _default_raw_row() -> Dict[str, Any]:
    return {
        "images": [],
        "subset_name": "",
        "macro_category": "",
        "source": "",
        "modality": "",
        "row_index": -1,
        "qa_index": -1,
        "turn_index": -1,
        "question": "",
        "original_answer": "",
        "system_prompt": "",
        "human_prompt": "",
        "task_text": "",
        "previous_actions": "",
        "gt_reasoning": "",
        "gt_action": "",
        "tools": "",
        "context_text": "",
        "prefix_json": "",
        "last_user_or_observation": "",
        "target_role": "",
        "num_prior_turns": 0,
        "completion": "",
        "generation_index": 0,
        "completion_length": 0,
        "two_step_applied": False,
        "qa_relevance": -1,
        "qa_visual_dependency": -1,
        "qa_image_correspondence": -1,
        "qa_formatting": -1,
    }


def _generate_source(
    llm: LLM,
    ds,
    source_name: str,
    pending_specs: List[Dict[str, Any]],
    sampling_bundle: Dict[str, Any],
    shard_rows: List[Dict[str, Any]],
    shard_idx: int,
    completed_counts: Dict[Tuple[str, int, int, int], int],
    total_requests: int,
    total_outputs: int,
):
    conversations = []
    chunk_specs = []
    chunk_saved_images = []
    source_requests_done = 0
    first_flush = True
    modality = SOURCE_CONFIGS[source_name]["modality"]

    def flush_chunk(start_row_marker: int):
        nonlocal conversations, chunk_specs, chunk_saved_images
        nonlocal shard_rows, shard_idx, total_requests, total_outputs, first_flush
        if not conversations:
            return
        if DEBUG_MODE and (first_flush or DEBUG_PRINT_REQUEST_DETAILS_EVERY_CHUNK):
            preview_n = min(DEBUG_PROMPT_PREVIEWS, len(chunk_specs))
            _debug(f"[debug][generate] previewing {preview_n} {modality} request(s) for source={source_name} near row {start_row_marker}")
            for i in range(preview_n):
                spec = chunk_specs[i]
                _debug(
                    f"  request#{i+1} subset={spec['subset_name']} row_index={spec['row_index']} qa_index={spec['qa_index']} turn_index={spec['turn_index']} q={_preview_text(spec.get('question', ''))}"
                )

        chat_kwargs = _chat_template_kwargs_for_runtime()
        llm_chat_kwargs = {
            "sampling_params": sampling_bundle["params"],
            "use_tqdm": False,
        }
        if chat_kwargs:
            llm_chat_kwargs["chat_template_kwargs"] = chat_kwargs
        outputs = llm.chat(conversations, **llm_chat_kwargs)
        if len(outputs) != len(chunk_specs):
            raise RuntimeError(f"Expected {len(chunk_specs)} generation results, received {len(outputs)}")
        total_requests += len(conversations)

        debug_completion_printed = 0
        for spec, saved_images, out in zip(chunk_specs, chunk_saved_images, outputs):
            if len(out.outputs) != NUM_GENERATIONS:
                raise RuntimeError(f"Missing completion for {_spec_identity(spec)}")
            for gen_idx, candidate in enumerate(out.outputs):
                completion = candidate.text or ""
                raw = _default_raw_row()
                raw.update(spec)
                raw.update({
                    "images": saved_images,
                    "completion": completion,
                    "generation_index": int(gen_idx),
                    "completion_length": len(completion),
                    "two_step_applied": False,
                })
                shard_rows.append(raw)
                total_outputs += 1
                spec_key = _spec_identity(spec)
                completed_counts[spec_key] = completed_counts.get(spec_key, 0) + 1
                shard_rows, shard_idx = _flush_save_if_needed(shard_rows, shard_idx)
                if DEBUG_MODE and (first_flush or DEBUG_PRINT_COMPLETION_DETAILS_EVERY_CHUNK) and debug_completion_printed < DEBUG_COMPLETION_PREVIEWS:
                    debug_completion_printed += 1
                    _debug(
                        f"[debug][completion] subset={spec['subset_name']} row_index={spec['row_index']} qa_index={spec['qa_index']} turn_index={spec['turn_index']} model={_preview_text(completion)}"
                    )

        conversations.clear()
        chunk_specs.clear()
        chunk_saved_images.clear()
        first_flush = False
        torch.cuda.empty_cache()
        gc.collect()

    for spec in pending_specs:
        conversation, raw_images = _build_conversation(spec, ds)
        conversations.append(conversation)
        chunk_specs.append(spec)
        chunk_saved_images.append(raw_images)
        source_requests_done += 1
        if len(conversations) >= GEN_CHUNK_SIZE:
            flush_chunk(int(spec["row_index"]))
            print(f"[generate] source={source_name} {source_requests_done}/{len(pending_specs)} pending requests done")

    flush_chunk(-1)
    print(f"[generate] source={source_name} finished with {source_requests_done}/{len(pending_specs)} pending requests done")
    return shard_rows, shard_idx, total_requests, total_outputs


def main():
    global MODEL_ID, MODEL_FAMILY, THINKING_MODE
    global RUN_NAME, SAVE_ROOT, RAW_SAVE_DIR, SELECTION_MANIFEST_PATH, GENERATION_STATS_PATH
    global TOTAL_QA_PAIRS, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, TENSOR_PARALLEL_SIZE
    global VISION_MAX_TOKENS, TEXT_MAX_TOKENS
    global SEED, GEN_CHUNK_SIZE, RAW_SHARD_SIZE
    global ACTIVE_MODEL_FAMILY, ACTIVE_THINKING_ENABLED

    args = _parse_args()
    local_source = None
    if args.source_file:
        if not args.source:
            raise ValueError("Set --source when using --source-file so its schema is known.")
        digest = hashlib.sha256()
        with args.source_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        local_source = {"path": str(args.source_file.resolve()), "sha256": digest.hexdigest()}
    if args.max_new_tokens is not None:
        if args.max_new_tokens < 1:
            raise ValueError("max-new-tokens must be positive")
        VISION_MAX_TOKENS = TEXT_MAX_TOKENS = args.max_new_tokens
    if args.model_id is not None:
        MODEL_ID = args.model_id
    if args.model_family is not None:
        MODEL_FAMILY = args.model_family
    if args.thinking_mode is not None:
        THINKING_MODE = args.thinking_mode
    if args.total_qa_pairs is not None:
        TOTAL_QA_PAIRS = int(args.total_qa_pairs)
    if args.max_model_len is not None:
        MAX_MODEL_LEN = int(args.max_model_len)
    if args.gpu_memory_utilization is not None:
        GPU_MEMORY_UTILIZATION = float(args.gpu_memory_utilization)
    if args.tensor_parallel_size is not None:
        TENSOR_PARALLEL_SIZE = int(args.tensor_parallel_size)
    if args.seed is not None:
        SEED = int(args.seed)
    if args.gen_chunk_size is not None:
        GEN_CHUNK_SIZE = int(args.gen_chunk_size)
    if args.raw_shard_size is not None:
        RAW_SHARD_SIZE = int(args.raw_shard_size)
    if min(TOTAL_QA_PAIRS, GEN_CHUNK_SIZE, RAW_SHARD_SIZE) < 1:
        raise ValueError("total-qa-pairs, gen-chunk-size and raw-shard-size must be positive")

    ACTIVE_MODEL_FAMILY = _infer_model_family(MODEL_ID, MODEL_FAMILY)
    ACTIVE_THINKING_ENABLED = _resolve_thinking_enabled(MODEL_ID, ACTIVE_MODEL_FAMILY, THINKING_MODE)
    _apply_family_generation_defaults(ACTIVE_MODEL_FAMILY)

    if args.run_name is not None:
        RUN_NAME = args.run_name
    else:
        RUN_NAME = _default_run_name(MODEL_ID, ACTIVE_THINKING_ENABLED)

    if args.save_root is not None:
        SAVE_ROOT = Path(args.save_root)
    else:
        SAVE_ROOT = DEFAULT_DATA_ROOT / _default_family_output_dir(ACTIVE_MODEL_FAMILY) / RUN_NAME

    RAW_SAVE_DIR = SAVE_ROOT / "raw"
    SELECTION_MANIFEST_PATH = SAVE_ROOT / "selection_manifest.json"
    GENERATION_STATS_PATH = SAVE_ROOT / "generation_stats.json"

    SAVE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(SAVE_ROOT.parent / f".{SAVE_ROOT.name}.data.lock"), timeout=0):
        _generate(args, local_source)


def _generate(args, local_source):
    if RESUME_IF_OUTPUT_EXISTS and NUM_GENERATIONS != 1:
        raise ValueError("Resume logic currently assumes NUM_GENERATIONS == 1.")

    print(
        f"[runtime] model_id={MODEL_ID} family={ACTIVE_MODEL_FAMILY} "
        f"thinking_mode={THINKING_MODE} thinking_enabled={ACTIVE_THINKING_ENABLED}"
    )
    print(f"[runtime] save_root={SAVE_ROOT}")

    resume_mode = _prepare_output_dirs()
    portions = {args.source: 1} if args.source else SOURCE_PORTIONS
    normalized_source_ratios = _normalize_portions(portions)
    requested_counts = _allocate_counts(TOTAL_QA_PAIRS, portions)
    completed_counts, completed_generation_keys = _load_completed_generations() if resume_mode else ({}, set())

    initial_completed_specs = len(completed_counts)
    initial_completed_outputs = len(completed_generation_keys)
    shard_idx = _next_shard_index_from_disk() if resume_mode else 1

    manifest: Dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": args.model_revision,
        "seed": SEED,
        "local_source": local_source,
        "model_family": ACTIVE_MODEL_FAMILY,
        "thinking_mode_requested": THINKING_MODE,
        "thinking_enabled_resolved": ACTIVE_THINKING_ENABLED,
        "total_examples_requested": TOTAL_QA_PAIRS,
        "source_portions": dict(portions),
        "normalized_source_ratios": dict(normalized_source_ratios),
        "requested_source_counts": dict(requested_counts),
        "source_configs": {
            name: {
                "dataset_id": cfg["dataset_id"],
                "dataset_config": cfg.get("dataset_config"),
                "split": cfg["split"],
                "kind": cfg["kind"],
                "modality": cfg["modality"],
            }
            for name, cfg in SOURCE_CONFIGS.items() if name in portions
        },
        "resume_mode": resume_mode,
        "initial_completed_specs": initial_completed_specs,
        "initial_completed_outputs": initial_completed_outputs,
        "starting_shard_index": shard_idx,
        "qa_quality_thresholds": {
            "relevance": QA_MIN_RELEVANCE,
            "visual_dependency": QA_MIN_VISUAL_DEPENDENCY,
            "image_correspondence": QA_MIN_IMAGE_CORRESPONDENCE,
            "formatting": QA_MIN_FORMATTING,
        },
        "generation_hparams": {
            "max_model_len": MAX_MODEL_LEN,
            "num_generations": NUM_GENERATIONS,
            "vision": {
                "temperature": VISION_TEMPERATURE,
                "top_p": VISION_TOP_P,
                "top_k": VISION_TOP_K,
                "repetition_penalty": VISION_REPETITION_PENALTY,
                "presence_penalty": VISION_PRESENCE_PENALTY,
                "max_tokens": VISION_MAX_TOKENS,
                "min_p": VISION_MIN_P,
            },
            "text": {
                "temperature": TEXT_TEMPERATURE,
                "top_p": TEXT_TOP_P,
                "top_k": TEXT_TOP_K,
                "repetition_penalty": TEXT_REPETITION_PENALTY,
                "presence_penalty": TEXT_PRESENCE_PENALTY,
                "max_tokens": TEXT_MAX_TOKENS,
                "min_p": TEXT_MIN_P,
            },
        },
        "per_source_stats": {},
    }

    if initial_completed_outputs:
        if not SELECTION_MANIFEST_PATH.is_file():
            raise ValueError("Existing raw shards have no selection manifest. Use a new --save-root.")
        previous = json.loads(SELECTION_MANIFEST_PATH.read_text(encoding="utf-8"))
        keys = ("model_id", "model_revision", "seed", "local_source", "thinking_enabled_resolved", "model_family",
                "requested_source_counts", "source_configs", "qa_quality_thresholds", "generation_hparams")
        mismatched = [key for key in keys if previous.get(key) != manifest[key]]
        if mismatched:
            raise ValueError(f"Generation settings differ from existing raw shards ({', '.join(mismatched)}). Use a new --save-root.")
    _save_json(SELECTION_MANIFEST_PATH, manifest)

    llm = None
    sampling_bundles = {
        "vision": _sampling_bundle_for_modality("vision"),
        "text": _sampling_bundle_for_modality("text"),
    }

    total_requests = len(completed_generation_keys)
    total_outputs = len(completed_generation_keys)
    total_selected_examples = 0
    shard_rows: List[Dict[str, Any]] = []

    _debug(f"[debug] requested source target counts: {dict(requested_counts)}")
    _debug(f"[debug] resume_mode={resume_mode} initial_completed_specs={initial_completed_specs} initial_completed_outputs={initial_completed_outputs}")

    for source_order, (source_name, target_count) in enumerate(requested_counts.items()):
        source_seed = SEED + 1009 * (source_order + 1)
        print(
            f"[source] loading source={source_name} target_count={target_count} "
            f"{_process_memory_summary()}"
        )
        if int(target_count) <= 0:
            manifest["per_source_stats"][source_name] = {
                "requested_examples": 0,
                "selected_examples": 0,
                "completed_specs_before_run": 0,
                "pending_specs_before_run": 0,
                "completed_specs_after_run": 0,
            }
            print(f"[select] {source_name}: requested=0 selected=0 (skipped)")
            continue
        ds = _load_source(source_name, source_seed, args.source_file)
        selected, stats = _sample_source(ds, source_name, target_count)
        total_selected_examples += len(selected)

        completed_before = sum(1 for spec in selected if completed_counts.get(_spec_identity(spec), 0) >= NUM_GENERATIONS)
        pending_specs = [spec for spec in selected if completed_counts.get(_spec_identity(spec), 0) < NUM_GENERATIONS]

        manifest["per_source_stats"][source_name] = {
            "requested_examples": target_count,
            **stats,
            "completed_specs_before_run": completed_before,
            "completed_specs_after_run": completed_before,
            "pending_specs_before_run": len(pending_specs),
        }

        print(
            f"[select] {source_name}: requested={target_count} selected={len(selected)} "
            f"completed_before_run={completed_before} pending_before_run={len(pending_specs)}"
        )

        if not pending_specs:
            del pending_specs, selected, ds
            gc.collect()
            continue

        if llm is None:
            has_images = any(SOURCE_CONFIGS[name]["modality"] == "vision" for name in portions)
            llm = LLM(
                model=MODEL_ID,
                revision=args.model_revision,
                trust_remote_code=args.trust_remote_code,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                enforce_eager=args.enforce_eager,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                seed=SEED,
                max_model_len=MAX_MODEL_LEN,
                limit_mm_per_prompt=LIMIT_MM_PER_PROMPT if has_images else {"image": 0, "video": 0},
                max_num_seqs=GEN_CHUNK_SIZE,
            )
        modality = SOURCE_CONFIGS[source_name]["modality"]
        shard_rows, shard_idx, total_requests, total_outputs = _generate_source(
            llm=llm,
            ds=ds,
            source_name=source_name,
            pending_specs=pending_specs,
            sampling_bundle=sampling_bundles[modality],
            shard_rows=shard_rows,
            shard_idx=shard_idx,
            completed_counts=completed_counts,
            total_requests=total_requests,
            total_outputs=total_outputs,
        )

        manifest["per_source_stats"][source_name]["completed_specs_after_run"] = sum(
            1 for spec in selected if completed_counts.get(_spec_identity(spec), 0) >= NUM_GENERATIONS
        )

        shard_rows, shard_idx = _flush_source_boundary(
            shard_rows,
            shard_idx,
            source_name,
        )

        _save_json(SELECTION_MANIFEST_PATH, manifest)
        _save_json(
            GENERATION_STATS_PATH,
            {
                "model_id": MODEL_ID,
                "resume_mode": resume_mode,
                "total_selected_examples": total_selected_examples,
                "total_requests": total_requests,
                "total_outputs": total_outputs,
                "next_shard_index": shard_idx,
                "per_source_stats": manifest["per_source_stats"],
            },
        )
        del pending_specs, selected, ds
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[source] released source={source_name} {_process_memory_summary()}")

    if shard_rows:
        shard_path = _save_shard(shard_rows, RAW_SAVE_DIR, shard_idx)
        print(f"[save] wrote final raw shard {shard_idx:06d} with {len(shard_rows)} rows -> {shard_path}")
        shard_idx += 1

    _save_json(SELECTION_MANIFEST_PATH, manifest)
    _save_json(
        GENERATION_STATS_PATH,
        {
            "model_id": MODEL_ID,
            "resume_mode": resume_mode,
            "total_selected_examples": total_selected_examples,
            "total_requests": total_requests,
            "total_outputs": total_outputs,
            "final_next_shard_index": shard_idx,
            "per_source_stats": manifest["per_source_stats"],
        },
    )
    print(f"[done] generation complete. total_outputs={total_outputs} raw_dir={RAW_SAVE_DIR}")


if __name__ == "__main__":
    main()
