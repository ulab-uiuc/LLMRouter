"""Multi-Head Latent Control (MHLC) capability-head routing."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, List

import torch.nn as nn

from llmrouter.models.gnosis_latent_router.adapter import (
    load_gnosis_runtime_module,
    require_checkpoint,
    resolve_workspace,
)
from llmrouter.models.gnosis_latent_router.config import load_gnosis_config
from llmrouter.models.meta_router import MetaRouter


class GnosisLatentRouter(MetaRouter):
    """Generate, score, and route through Gnosis's original runtime classes."""

    def __init__(self, yaml_path: str, workspace: str | None = None):
        config = load_gnosis_config(yaml_path)
        super().__init__(model=nn.Identity(), yaml_path=yaml_path, data_root=config.get("_data_root"))
        self.gnosis_config = config
        self.workspace_root = resolve_workspace(config, workspace)
        self.threshold = float(config["threshold"])
        self.target_model = str(config["target_model"])
        self.target_model_name = str(config["target_model_name"])
        self.fallback_model = str(config["fallback_model"])
        self._runtime_module = None
        self._generator = None
        self._scorer = None
        self._sampling = None

    def load_router(self, path: str) -> None:
        """Select a trained capability head for subsequent queries."""
        checkpoint = Path(path).expanduser().resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError(f"Expected a checkpoint file at {checkpoint}")
        self.unload()
        self.gnosis_config["inference"]["checkpoint"] = str(checkpoint)

    def save_router(self, path: str) -> None:
        """Export the trained capability head, including its feature metadata."""
        checkpoint = self._checkpoint_path()
        target = Path(path).expanduser().resolve()
        if target != checkpoint:
            shutil.copyfile(checkpoint, target)

    def _checkpoint_path(self):
        return require_checkpoint(
            self.gnosis_config,
            workspace=self.workspace_root,
        )

    def _ensure_runtimes(self) -> None:
        if self._generator is not None and self._scorer is not None:
            return
        checkpoint = self._checkpoint_path()
        module = load_gnosis_runtime_module(self.gnosis_config)
        inference = self.gnosis_config["inference"]
        generator_config = dict(inference.get("generator") or {})
        auxiliary_config = dict(inference.get("auxiliary") or {})
        sampling_config = dict(inference.get("sampling") or {})

        generator_config.setdefault("model_name_or_path", self.target_model)
        # Both passes use the same resident backbone and chat template.
        for key in ("model_name_or_path", "revision", "model_family", "thinking_mode"):
            if key in generator_config:
                if auxiliary_config.get(key, generator_config[key]) != generator_config[key]:
                    raise ValueError(f"inference.generator and inference.auxiliary disagree on {key}")
                auxiliary_config[key] = generator_config[key]
            elif key in auxiliary_config:
                generator_config[key] = auxiliary_config[key]
        auxiliary_config["enabled"] = True
        auxiliary_config["aux_head_ckpt"] = str(checkpoint)

        scorer = module.AuxHeadRuntime(
            module.AuxHeadRuntimeConfig(**auxiliary_config)
        )
        generator_config["enable_hidden_state_extraction"] = True
        generator_config["hidden_state_layer_ids"] = (
            scorer.required_hidden_state_layer_ids()
        )
        if auxiliary_config.get("hidden_states_storage_path"):
            generator_config["hidden_states_storage_path"] = auxiliary_config[
                "hidden_states_storage_path"
            ]
        generator = module.VLLMChatRuntime(
            module.VLLMRuntimeConfig(**generator_config)
        )
        scorer.attach_vllm_runtime(generator)
        self._sampling = module.SamplingConfig(**sampling_config)
        self._runtime_module = module
        self._generator = generator
        self._scorer = scorer

    @staticmethod
    def _messages(
        prompt: str,
        system_prompt: str,
        *,
        has_image: bool,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_content: Any = prompt
        if has_image:
            user_content = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_content})
        return messages

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(query_input.get("query") or "")
        if not prompt:
            raise ValueError("GnosisLatentRouter requires a non-empty 'query'")
        self._ensure_runtimes()

        system_prompt = str(query_input.get("system_prompt") or "")
        image = query_input.get("image")
        if image is not None and hasattr(self._runtime_module, "load_image_any"):
            image = self._runtime_module.load_image_any(image)
        messages = self._messages(
            prompt,
            system_prompt,
            has_image=image is not None,
        )

        generated = self._generator.generate(
            messages=messages,
            image=image,
            sampling_cfg=self._sampling,
        )
        completion = str(generated.text)
        gnosis_score = None
        if completion.strip():
            score = self._scorer.score_messages(
                messages=[*messages, {"role": "assistant", "content": completion}],
                image=image,
            )
            gnosis_score = float(score.prob_correct)
        accepted = gnosis_score is not None and gnosis_score >= self.threshold
        selected_model = self.target_model_name if accepted else self.fallback_model

        result = dict(query_input)
        result.pop("response", None)
        result.update(
            {
                "model_name": selected_model,
                "predicted_llm": selected_model,
                "gnosis_score": gnosis_score,
                "threshold": self.threshold,
                "accepted": accepted,
                "candidate_model": self.target_model_name,
                "candidate_response": completion,
                "method": "gnosis_latent_router",
                "prompt_tokens": int(generated.prompt_tokens),
                "completion_tokens": int(generated.completion_tokens),
            }
        )
        if accepted:
            result["response"] = completion
        return result

    def route_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.route_single(record) for record in batch]

    def unload(self) -> None:
        if self._generator is not None:
            self._generator.unload(drop_processor=True)
        if self._scorer is not None:
            self._scorer.unload(drop_processor=True)
        self._generator = None
        self._scorer = None
        self._sampling = None
        self._runtime_module = None
