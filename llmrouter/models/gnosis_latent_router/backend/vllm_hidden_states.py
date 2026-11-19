"""Shared vLLM hidden-state extraction for Gnosis training and inference."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

FEATURE_BACKEND = "vllm_extract_hidden_states"
FEATURE_BACKEND_VERSION = 2


def normalize_vllm_dtype(value: Any) -> str:
    value = str(value or "auto").strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
    }
    return aliases.get(value, value)


@dataclass
class ExtractedHiddenStates:
    """One prompt's token IDs and selected hidden states."""

    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    layer_ids: list[int]


def model_dimensions(
    model_name_or_path: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
) -> tuple[int, int]:
    """Return text hidden size and transformer-layer count without loading weights."""

    cfg = AutoConfig.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    text_cfg = getattr(cfg, "text_config", None) or cfg
    return int(text_cfg.hidden_size), int(text_cfg.num_hidden_layers)


def _temporary_storage(configured_path: str | None):
    """Give each engine its own private connector files on shared storage."""
    raw = configured_path or os.environ.get("GNOSIS_HIDDEN_STATES_DIR")
    if raw:
        parent = Path(raw).expanduser()
        parent.mkdir(parents=True, exist_ok=True)
    else:
        parent = Path("/dev/shm")
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            parent = Path(tempfile.gettempdir())
    return tempfile.TemporaryDirectory(
        dir=parent.resolve(), prefix="gnosis-hidden-states-"
    )


class VLLMHiddenStateExtractor:
    """Configure a vLLM engine and read hidden states produced by that engine."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        storage_path: str | None = None,
    ) -> None:
        normalized: list[int] = []
        for layer_id in layer_ids:
            layer_id = int(layer_id)
            if layer_id < 0:
                raise ValueError(
                    f"vLLM hidden-state layer IDs must be non-negative, got {layer_id}"
                )
            if layer_id not in normalized:
                normalized.append(layer_id)
        if not normalized:
            raise ValueError("At least one vLLM hidden-state layer ID is required")
        if normalized != sorted(normalized):
            raise ValueError(
                "vLLM returns hidden states in ascending layer order; list layer IDs in that order"
            )
        self.layer_ids = normalized
        self._storage = _temporary_storage(storage_path)
        self.storage_path = self._storage.name

    def llm_kwargs(self) -> dict[str, Any]:
        """Arguments that turn a normal generation engine into an extractor too."""

        from vllm.config import KVTransferConfig

        return {
            "speculative_config": {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": list(self.layer_ids),
                    }
                },
            },
            "kv_transfer_config": KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={
                    "shared_storage_path": self.storage_path,
                    "use_synchronization_lock": True,
                },
            ),
        }

    @staticmethod
    def _request(
        prompt: str | list[int], image: Any, mm_processor_kwargs: dict | None
    ) -> Any:
        request = (
            {"prompt": prompt}
            if isinstance(prompt, str)
            else {"prompt_token_ids": prompt}
        )
        if image is not None:
            request["multi_modal_data"] = {"image": image}
            if mm_processor_kwargs:
                request["mm_processor_kwargs"] = mm_processor_kwargs
        return request

    def extract(
        self,
        llm: Any,
        *,
        prompts: Sequence[str | list[int]],
        images: Sequence[Any] | None = None,
        mm_processor_kwargs: dict | None = None,
        use_tqdm: bool = False,
    ) -> list[ExtractedHiddenStates]:
        """Prefill complete prompt/answer sequences and return their hidden states.

        A one-token decode is requested only to make vLLM complete the request. The
        connector saves prompt-token states, so every supplied answer token is
        included and the extra sampled token is excluded.
        """

        from vllm import SamplingParams
        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            cleanup_hidden_states,
            load_hidden_states,
        )

        if images is None:
            images = [None] * len(prompts)
        if len(prompts) != len(images):
            raise ValueError(
                f"prompts/images length mismatch: {len(prompts)} vs {len(images)}"
            )
        if not prompts:
            return []

        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            n=1,
            extra_args={"kv_transfer_params": {"include_output_tokens": False}},
        )
        requests = [
            self._request(prompt, image, mm_processor_kwargs)
            for prompt, image in zip(prompts, images)
        ]
        outputs = llm.generate(requests, sampling, use_tqdm=use_tqdm)
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"Invalid vLLM extraction output count: expected {len(requests)}, got {len(outputs)}"
            )

        extracted: list[ExtractedHiddenStates] = []
        for output in outputs:
            params = getattr(output, "kv_transfer_params", None) or {}
            path = params.get("hidden_states_path")
            if not path:
                raise RuntimeError(
                    "vLLM did not return hidden_states_path. Ensure this engine was "
                    "created with VLLMHiddenStateExtractor.llm_kwargs()."
                )
            try:
                saved = load_hidden_states(str(path))
                token_ids = saved["token_ids"].detach().cpu().long().clone()
                hidden = saved["hidden_states"].detach().cpu().clone()
            finally:
                cleanup_hidden_states(str(path))

            if hidden.ndim != 3:
                raise RuntimeError(
                    f"Expected vLLM hidden states shaped [tokens, layers, hidden], got {tuple(hidden.shape)}"
                )
            if int(hidden.shape[0]) != int(token_ids.numel()):
                raise RuntimeError(
                    "vLLM hidden-state/token length mismatch: "
                    f"{hidden.shape[0]} hidden rows vs {token_ids.numel()} token IDs"
                )
            if int(hidden.shape[1]) != len(self.layer_ids):
                raise RuntimeError(
                    "vLLM extracted an unexpected layer count: "
                    f"{hidden.shape[1]} vs requested {len(self.layer_ids)}"
                )
            extracted.append(
                ExtractedHiddenStates(
                    token_ids=token_ids,
                    hidden_states=hidden,
                    layer_ids=list(self.layer_ids),
                )
            )
        return extracted

    @staticmethod
    def discard_outputs(outputs: Sequence[Any]) -> None:
        """Wait for and remove connector files from ordinary generation calls."""

        from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
            cleanup_hidden_states,
            load_hidden_states,
        )

        for output in outputs:
            params = getattr(output, "kv_transfer_params", None) or {}
            path = params.get("hidden_states_path")
            if not path:
                continue
            try:
                # Loading acquires the connector lock and therefore cannot race
                # the asynchronous writer.
                load_hidden_states(str(path))
            finally:
                cleanup_hidden_states(str(path))


def align_extracted_batch(
    *,
    extracted: Sequence[ExtractedHiddenStates],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    head_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Validate tokenizer agreement and pad extracted states to one head batch."""

    if len(extracted) != int(input_ids.shape[0]):
        raise RuntimeError(
            f"Extracted batch size mismatch: {len(extracted)} vs tokenizer batch {input_ids.shape[0]}"
        )
    if not extracted:
        raise RuntimeError("Cannot align an empty hidden-state batch")

    layer_ids = list(extracted[0].layer_ids)
    hidden_size = int(extracted[0].hidden_states.shape[-1])
    max_len = max(int(item.hidden_states.shape[0]) for item in extracted)
    hidden_batch = extracted[0].hidden_states.new_zeros(
        (len(extracted), max_len, len(layer_ids), hidden_size)
    )
    mask_batch = torch.zeros((len(extracted), max_len), dtype=torch.long)

    for row, item in enumerate(extracted):
        if list(item.layer_ids) != layer_ids:
            raise RuntimeError(
                "All extracted examples must use the same hidden-state layer IDs"
            )
        valid = attention_mask[row].bool()
        expected_ids = input_ids[row][valid].detach().cpu().long()
        expected_head_mask = head_token_mask[row][valid].detach().cpu().long()
        actual_ids = item.token_ids
        if not torch.equal(expected_ids, actual_ids):
            mismatch = next(
                (
                    i
                    for i, (expected, actual) in enumerate(
                        zip(expected_ids.tolist(), actual_ids.tolist())
                    )
                    if expected != actual
                ),
                min(expected_ids.numel(), actual_ids.numel()),
            )
            raise RuntimeError(
                "The Hugging Face processor and vLLM tokenized the completed chat "
                f"differently at example {row}, token {mismatch} "
                f"(HF length={expected_ids.numel()}, vLLM length={actual_ids.numel()})."
            )
        length = int(actual_ids.numel())
        hidden_batch[row, :length] = item.hidden_states
        mask_batch[row, :length] = expected_head_mask
    return hidden_batch, mask_batch, layer_ids
