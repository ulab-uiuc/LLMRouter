import io
import json
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import datasets
import torch
from PIL import Image
from torch import nn

from llmrouter.models.gnosis_latent_router.backend.feature_extractors import (
    CorrectnessHeadLite,
    HiddenFeatureExtractorLite,
    StrongContentOnlyHiddenEncoder,
    StrongMultiLayerHiddenTrajectoryEncoder,
)
from llmrouter.models.gnosis_latent_router.backend.vllm_hidden_states import (
    FEATURE_BACKEND_VERSION,
)

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def dtype_from_str(s: str) -> torch.dtype | None:
    s = str(s or "bf16").lower()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "half"}:
        return torch.float16
    if s in {"fp32", "float32", "float"}:
        return torch.float32
    if s == "auto":
        return None
    raise ValueError(f"Unsupported head dtype: {s}")


def load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(p.read_text()) or {}
    if p.suffix == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"Training recipe must be a YAML or JSON file: {p}")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_dataset_auto(path: str):
    p = Path(path)
    if p.is_dir() and any(
        (p / marker).is_file() for marker in ("state.json", "dataset_dict.json")
    ):
        return datasets.load_from_disk(str(p))
    if p.is_dir():
        pq = sorted(str(shard) for shard in p.rglob("*.parquet"))
        if pq:
            return datasets.load_dataset("parquet", data_files={"train": pq})
        js = sorted(
            str(shard) for shard in p.rglob("*") if shard.suffix in {".json", ".jsonl"}
        )
        if js:
            return datasets.load_dataset("json", data_files={"train": js})
    if p.is_file() and p.suffix == ".parquet":
        return datasets.load_dataset("parquet", data_files={"train": str(p)})
    if p.is_file() and p.suffix in {".json", ".jsonl"}:
        return datasets.load_dataset("json", data_files={"train": str(p)})
    raise FileNotFoundError(f"Could not load dataset from {path}")


def filter_dataset_by_subset_name(ds, subset_name: str | None):
    if not subset_name:
        return ds
    subset_name = str(subset_name).strip().lower()
    if "subset_name" not in ds.column_names:
        raise ValueError("Filtering by subset_name requires a subset_name column")
    idx = [
        i
        for i, v in enumerate(ds["subset_name"])
        if str(v).strip().lower() == subset_name
    ]
    return ds.select(idx)


def load_image_any(x: Any) -> Image.Image | None:
    if x is None:
        return None
    if isinstance(x, dict):
        x = x.get("bytes") or x.get("path")
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        p = Path(x.removeprefix("file://"))
        with Image.open(p) as img:
            return img.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        with Image.open(io.BytesIO(x)) as img:
            return img.convert("RGB")
    raise ValueError(f"Unsupported training image representation: {type(x).__name__}")


ReplayImages = Image.Image | list[Image.Image] | None


def build_messages_from_prompt_completion(
    prompt: str,
    completion: str,
    has_image: int,
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    user_content: Any = [{"type": "image"} for _ in range(has_image)]
    if prompt:
        if has_image:
            user_content.append({"type": "text", "text": prompt})
        else:
            user_content = prompt
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": completion},
        ]
    )
    return messages


def _build_training_messages_for_subset(
    *,
    subset_name: str,
    system_prompt: str,
    prompt: str,
    completion: str,
    has_image: int,
    tools: str = "",
    context_text: str = "",
    human_prompt: str = "",
) -> list[dict[str, Any]]:
    subset = str(subset_name or "").strip().lower()

    if subset == "aguvis-stage-2":
        user_prompt = str(human_prompt or prompt or "").rstrip()
        if AGUVIS2_FORMAT_INSTRUCTION not in user_prompt:
            user_prompt += AGUVIS2_FORMAT_INSTRUCTION
        return build_messages_from_prompt_completion(
            prompt=user_prompt,
            completion=completion,
            has_image=has_image,
            system_prompt=system_prompt,
        )

    if subset == "apigen-mt-5k":
        combined_system = (
            str(system_prompt).strip() + "\n\nAvailable tools:\n" + str(tools).strip()
        ).strip()
        user_prompt = f"Conversation so far:\n{context_text or prompt or ''!s}\n\n{APIGEN_NEXT_TURN_INSTRUCTION}"
        return build_messages_from_prompt_completion(
            prompt=user_prompt,
            completion=completion,
            has_image=False,
            system_prompt=combined_system,
        )

    return build_messages_from_prompt_completion(
        prompt=prompt,
        completion=completion,
        has_image=has_image,
        system_prompt=system_prompt,
    )


def build_messages_and_image(
    ex: dict[str, Any],
) -> tuple[list[dict[str, Any]], ReplayImages]:
    raw = ex.get("images")
    if raw is None:
        raw = ex.get("image")
    raw = raw if isinstance(raw, (list, tuple)) else [raw]
    decoded = [load_image_any(item) for item in raw if item is not None]
    image = decoded if len(decoded) > 1 else next(iter(decoded), None)

    if ex.get("messages") is not None:
        return ex["messages"], image

    system_prompt = str(ex.get("system_prompt") or "")
    prompt = str(ex.get("prompt") or ex.get("question") or ex.get("text") or "")
    completion = str(ex.get("completion") or ex.get("answer") or "")
    subset_name = str(ex.get("subset_name") or "")
    tools = str(ex.get("tools") or "")
    context_text = str(ex.get("context_text") or "")
    human_prompt = str(ex.get("human_prompt") or "")

    messages = _build_training_messages_for_subset(
        subset_name=subset_name,
        system_prompt=system_prompt,
        prompt=prompt,
        completion=completion,
        has_image=len(decoded),
        tools=tools,
        context_text=context_text,
        human_prompt=human_prompt,
    )
    return messages, image


def set_max_pixels(processor, max_pixels: int) -> None:
    """Apply the pixel budget used to build multimodal head inputs."""
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if getattr(size, "shortest_edge", None) and getattr(size, "longest_edge", None):
        size.longest_edge = int(max_pixels)
    elif isinstance(size, dict) and "shortest_edge" in size and "longest_edge" in size:
        size["longest_edge"] = int(max_pixels)
    elif hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = int(max_pixels)


class ChatBatchBuilder:
    def __init__(
        self,
        processor: Any,
        max_seq_len: int,
        head_input_mode: str,
        feature_backend_version: int = FEATURE_BACKEND_VERSION,
    ):
        self.feature_backend_version = int(feature_backend_version)
        if self.feature_backend_version not in (1, FEATURE_BACKEND_VERSION):
            raise ValueError(
                f"Unsupported checkpoint feature version: {feature_backend_version}"
            )
        self.processor = processor
        self.max_seq_len = int(max_seq_len)
        if self.max_seq_len < 2:
            raise ValueError(
                "max_seq_len must leave room for a prompt and a vLLM decode token"
            )
        self.head_input_mode = str(head_input_mode)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.vision_token_ids = self._get_token_ids(
            [
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
                "<|image_start|>",
                "<|image_end|>",
                "<image>",
                "</image>",
            ]
        )
        self.control_token_ids = set(
            getattr(self.tokenizer, "all_special_ids", []) or []
        )
        self.control_token_ids.update(
            self._get_token_ids(
                [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|assistant|>",
                    "<|user|>",
                    "<|system|>",
                ]
            )
        )
        if self.feature_backend_version >= 2:
            self.control_token_ids.update(self._get_token_ids(["<think>", "</think>"]))

    def _token_id(self, token: str) -> int | None:
        try:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            if self.feature_backend_version >= 2 and tid == getattr(
                self.tokenizer, "unk_token_id", None
            ):
                return None
            return int(tid) if isinstance(tid, int) and tid >= 0 else None
        except (TypeError, ValueError, KeyError):
            return None

    def _get_token_ids(self, tokens: Sequence[str]) -> list[int]:
        ids = []
        for t in tokens:
            tid = self._token_id(t)
            if tid is not None and tid not in ids:
                ids.append(tid)
        return ids

    def _encode_prefix(self, text: str, image: ReplayImages) -> list[int]:
        kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": self.max_seq_len,
        }
        if image is not None:
            kwargs["images"] = image if isinstance(image, list) else [image]
        enc = self.processor(**kwargs)
        return enc["input_ids"][0][enc["attention_mask"][0].bool()].tolist()

    def _completion_edges(self, messages: list, text: str) -> tuple[str, str]:
        if messages[-1]["role"] != "assistant":
            raise ValueError("A completed chat must end with an assistant message")
        if self.feature_backend_version == 1:
            prefix = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=False
            )
            return prefix, ""

        # A rendered placeholder reveals the template's role header and closing
        # tokens, including synthetic empty reasoning blocks in Qwen templates.
        marker = "MHLC_COMPLETION_BOUNDARY"
        probe = self.processor.apply_chat_template(
            [*messages[:-1], {"role": "assistant", "content": marker}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix, found, suffix = probe.rpartition(marker)
        if not found or not text.endswith(suffix):
            raise ValueError("Cannot locate assistant completion in this chat template")
        # When a real reasoning block replaces an empty one, retain its content.
        return os.path.commonprefix([prefix, text]), suffix

    @staticmethod
    def _matching_tokens(first, second):
        return next(
            (
                index
                for index, (left, right) in enumerate(zip(first, second))
                if left != right
            ),
            min(len(first), len(second)),
        )

    def _drop_ids(
        self, input_ids: torch.Tensor, token_ids: Sequence[int]
    ) -> torch.Tensor:
        drop = torch.zeros_like(input_ids, dtype=torch.bool)
        for tid in token_ids:
            drop |= input_ids.eq(int(tid))
        return (~drop).long()

    def build_from_messages(
        self,
        messages_batch: list[list[dict[str, Any]]],
        images: list[ReplayImages],
        labels: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        texts, completion_edges = [], []
        for messages in messages_batch:
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            if self.head_input_mode in {
                "completion_only",
                "completion_text_only",
                "completion_first_200",
            }:
                completion_edges.append(self._completion_edges(messages, texts[-1]))
            elif self.head_input_mode not in {"all_tokens", "text_only"}:
                raise ValueError(f"Unsupported head_input_mode: {self.head_input_mode}")

        kwargs = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
        }
        if any(img is not None for img in images):
            kwargs["images"] = [
                item
                for image in images
                if image is not None
                for item in (image if isinstance(image, list) else [image])
            ]

        encoded = self.processor(**kwargs)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].long()
        # Pixel tensors are used by the processor to expand the vision tokens.
        # vLLM consumes the original image directly.
        batch = {"input_ids": input_ids}
        del encoded

        if self.head_input_mode == "all_tokens":
            head_mask = attention_mask.clone()

        elif self.head_input_mode == "text_only":
            head_mask = attention_mask * self._drop_ids(
                input_ids, self.vision_token_ids
            )

        else:
            head_mask = torch.zeros_like(attention_mask)
            for i, (prefix, suffix) in enumerate(completion_edges):
                positions = attention_mask[i].nonzero(as_tuple=True)[0]
                ids = input_ids[i, positions].tolist()
                prefix_ids = self._encode_prefix(prefix, images[i])
                if self.feature_backend_version == 1:
                    start = min(len(prefix_ids), len(ids))
                    end = len(ids)
                else:
                    start = self._matching_tokens(prefix_ids, ids)
                    suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                    end = len(ids) - self._matching_tokens(ids[::-1], suffix_ids[::-1])
                if self.head_input_mode == "completion_first_200":
                    end = min(end, start + 200)
                head_mask[i, positions[start:end]] = 1

            if self.head_input_mode == "completion_text_only":
                drop_ids = sorted(
                    set(self.vision_token_ids).union(self.control_token_ids)
                )
                head_mask = head_mask * self._drop_ids(input_ids, drop_ids)

        # The extraction engine needs one decode token after the replayed
        # prompt. Right-truncate both views of the same sequence together.
        replay_prompts = list(texts)
        for i, length in enumerate(attention_mask.sum(dim=1).tolist()):
            limit = self.max_seq_len - 1
            if length <= limit:
                continue
            positions = attention_mask[i].nonzero(as_tuple=True)[0]
            attention_mask[i, positions[limit:]] = 0
            if images[i] is None:
                replay_prompts[i] = input_ids[i, positions[:limit]].tolist()
            else:
                # vLLM expects one unexpanded image placeholder. The processor
                # has already expanded it, so drop the same trailing text tokens.
                raw_ids = self.tokenizer.encode(texts[i])
                keep = len(raw_ids) - (length - limit)
                if keep <= 0 or set(raw_ids[keep:]).intersection(self.vision_token_ids):
                    raise ValueError(
                        "max_seq_len truncates an image; increase the context length"
                    )
                replay_prompts[i] = raw_ids[:keep]
        batch["attention_mask"] = attention_mask
        head_mask = head_mask * attention_mask
        if not head_mask.bool().any(dim=1).all():
            raise ValueError(
                "No scoreable tokens in a completion. Check empty answers and max_seq_len."
            )
        batch["head_token_mask"] = head_mask
        # vLLM replays the same completed chat and image to obtain token states.
        batch["gnosis_prompt_texts"] = replay_prompts
        batch["gnosis_images"] = images
        image_processor = getattr(self.processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        mm_kwargs = None
        if self.feature_backend_version >= 2 and any(
            image is not None for image in images
        ):
            if (
                size is not None
                and hasattr(size, "get")
                and size.get("shortest_edge")
                and size.get("longest_edge")
            ):
                mm_kwargs = {
                    "size": {
                        "shortest_edge": size.get("shortest_edge"),
                        "longest_edge": size.get("longest_edge"),
                    }
                }
            elif hasattr(image_processor, "max_pixels"):
                mm_kwargs = {"max_pixels": image_processor.max_pixels}
        batch["gnosis_mm_processor_kwargs"] = mm_kwargs
        if labels is not None:
            batch["aux_labels"] = torch.tensor(
                [float(x) for x in labels], dtype=torch.float32
            )
        return batch


class VlmCollator(ChatBatchBuilder):
    def __init__(
        self,
        processor: Any,
        aux_label_column: str,
        max_seq_len: int,
        head_input_mode: str,
    ):
        super().__init__(processor, max_seq_len, head_input_mode)
        self.aux_label_column = aux_label_column

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        messages_batch, images, labels = [], [], []
        for ex in examples:
            messages, image = build_messages_and_image(ex)
            messages_batch.append(messages)
            images.append(image)
            labels.append(float(ex[self.aux_label_column]))
        return self.build_from_messages(messages_batch, images, labels)


class AuxHeadModule(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        hidden_encoder_type: str = "strong_single",
        num_labels: int = 2,
        selected_hidden_layer_indices: Sequence[int] | None = None,
    ):
        super().__init__()
        self.hidden_encoder_type = str(hidden_encoder_type)
        if selected_hidden_layer_indices:
            selected = []
            for layer_id in selected_hidden_layer_indices:
                layer_id = int(layer_id)
                if layer_id < 0:
                    layer_id = int(num_hidden_layers) + 1 + layer_id
                if not 0 <= layer_id <= int(num_hidden_layers):
                    raise ValueError(
                        f"Raw hidden-state layer ID {layer_id} is outside "
                        f"0..{num_hidden_layers}"
                    )
                if layer_id not in selected:
                    selected.append(layer_id)
            self.selected_hidden_layer_indices = selected
        else:
            self.selected_hidden_layer_indices = None

        if self.hidden_encoder_type == "lite":
            self.hid_extractor = HiddenFeatureExtractorLite(D_model=hidden_size)
        elif self.hidden_encoder_type == "strong_single":
            self.hid_extractor = StrongContentOnlyHiddenEncoder(D_model=hidden_size)
        elif self.hidden_encoder_type == "strong_multi":
            self.hid_extractor = StrongMultiLayerHiddenTrajectoryEncoder(
                D_model=hidden_size,
                D_HID=256,
                max_model_layers=num_hidden_layers,
            )
        else:
            raise ValueError(f"Unknown hidden encoder type: {self.hidden_encoder_type}")

        self.requires_all_hidden_states = self.hidden_encoder_type == "strong_multi"

        self.head = CorrectnessHeadLite(
            D_ATT=256,
            D_CONF=128,
            D_HID=256,
            use_attn=False,
            use_conf=False,
            use_hid=True,
            num_labels=int(num_labels),
        )

    def _hid_dtype(self) -> torch.dtype:
        return next(self.hid_extractor.parameters()).dtype

    def _head_dtype(self) -> torch.dtype:
        return next(self.head.parameters()).dtype

    def _encode_single_layer(
        self, last_hidden: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        target_dtype = self._hid_dtype()
        seqs = [
            last_hidden[i][token_mask[i].bool()].to(target_dtype)
            for i in range(last_hidden.shape[0])
        ]

        if self.hidden_encoder_type == "lite":
            z_hid = torch.cat([self.hid_extractor(x.unsqueeze(0)) for x in seqs], dim=0)
        else:
            z_hid = self.hid_extractor(seqs)

        return z_hid.to(self._head_dtype())

    def _encode_multi_layer(
        self, hidden_states: Sequence[torch.Tensor], token_mask: torch.Tensor
    ) -> torch.Tensor:
        target_dtype = self._hid_dtype()
        layer_ids = self.selected_hidden_layer_indices or [len(hidden_states) - 1]
        layer_hidden = [hidden_states[i].to(target_dtype) for i in layer_ids]
        stacked = torch.stack(layer_hidden, dim=1)
        seqs = [stacked[b][:, token_mask[b].bool(), :] for b in range(stacked.shape[0])]
        transformer_layer_ids = [max(0, int(i) - 1) for i in layer_ids]
        z_hid = self.hid_extractor(seqs, layer_ids=transformer_layer_ids)
        return z_hid.to(self._head_dtype())

    def forward(
        self,
        *,
        last_hidden: torch.Tensor,
        hidden_states: Sequence[torch.Tensor] | None,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_encoder_type == "strong_multi":
            z_hid = self._encode_multi_layer(hidden_states, token_mask)
        else:
            z_hid = self._encode_single_layer(last_hidden, token_mask)
        return self.head(z_att=None, z_conf=None, z_hid=z_hid, return_penultimate=False)

    def forward_extracted(
        self,
        *,
        extracted_hidden_states: torch.Tensor,
        extracted_layer_ids: Sequence[int],
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the head on vLLM states shaped [batch, tokens, layers, hidden]."""

        if extracted_hidden_states.ndim != 4:
            raise ValueError(
                "extracted_hidden_states must have shape [batch, tokens, layers, hidden], "
                f"got {tuple(extracted_hidden_states.shape)}"
            )
        if extracted_hidden_states.shape[2] != len(extracted_layer_ids):
            raise ValueError(
                f"Layer dimension {extracted_hidden_states.shape[2]} does not match "
                f"{len(extracted_layer_ids)} extracted layer IDs"
            )

        if self.hidden_encoder_type != "strong_multi":
            return self.forward(
                last_hidden=extracted_hidden_states[:, :, -1, :],
                hidden_states=None,
                token_mask=token_mask,
            )

        target_dtype = self._hid_dtype()
        stacked = extracted_hidden_states.permute(0, 2, 1, 3).to(target_dtype)
        seqs = [stacked[b][:, token_mask[b].bool(), :] for b in range(stacked.shape[0])]
        transformer_layer_ids = [
            max(0, int(layer_id) - 1) for layer_id in extracted_layer_ids
        ]
        z_hid = self.hid_extractor(seqs, layer_ids=transformer_layer_ids)
        z_hid = z_hid.to(self._head_dtype())
        return self.head(z_att=None, z_conf=None, z_hid=z_hid, return_penultimate=False)


def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def compute_metrics_from_confusion(conf: torch.Tensor) -> dict[str, float]:
    conf = conf.float()
    tp = torch.diag(conf)
    row = conf.sum(dim=1)
    col = conf.sum(dim=0)
    prec = tp / torch.clamp(col, min=1.0)
    rec = tp / torch.clamp(row, min=1.0)
    f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-8)
    return {
        "acc": (tp.sum() / torch.clamp(conf.sum(), min=1.0)).item(),
        "macro_precision": prec.mean().item(),
        "macro_recall": rec.mean().item(),
        "macro_f1": f1.mean().item(),
    }
