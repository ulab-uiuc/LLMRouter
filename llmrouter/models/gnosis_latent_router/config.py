"""Configuration for the package-integrated Gnosis Latent Router."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULTS: Dict[str, Any] = {
    "threshold": 0.5,
    "pipeline": {
        "workspace": None,
        "python": None,
        "run_stages": ["generate", "label", "train", "benchmark"],
        "environment": {},
        "stages": {},
    },
    "inference": {
        "runtime_module": (
            "multi_agenT_bench/"
            "compact_multi_agent_shared_optimized_v4_textbench.py"
        ),
        "checkpoint": None,
        "generator": {},
        "sampling": {},
        "auxiliary": {},
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_gnosis_config(yaml_path: str) -> Dict[str, Any]:
    """Load Gnosis configuration without rewriting pipeline arguments."""
    path = Path(yaml_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle) or {}
    if not isinstance(root, dict):
        raise ValueError("Router configuration must be a YAML mapping")
    section = root.get("gnosis_latent_router", root)
    if not isinstance(section, dict):
        raise ValueError("'gnosis_latent_router' configuration must be a mapping")

    config = _expand_environment(_merge(DEFAULTS, section))
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    # An installed wheel may be used with examples from a separate checkout.
    catalog = (root.get("data_path") or {}).get("llm_data")
    if catalog and not Path(catalog).is_absolute():
        roots = [path.parent]
        if path.parent.parent.name == "configs":
            roots.append(path.parent.parent.parent)
        for directory in roots:
            if (directory / catalog).is_file():
                config["_data_root"] = str(directory)
                break

    for key in ("target_model", "target_model_name", "fallback_model"):
        if not config.get(key):
            raise ValueError(f"gnosis_latent_router.{key} is required")
    threshold = float(config["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("gnosis_latent_router.threshold must be in [0, 1]")
    config["threshold"] = threshold

    stages = config["pipeline"].get("stages")
    if not isinstance(stages, dict):
        raise ValueError(
            "gnosis_latent_router.pipeline.stages must be a mapping"
        )
    for name, stage in stages.items():
        if not isinstance(stage, dict) or not isinstance(stage.get("command"), list):
            raise ValueError(
                f"Gnosis pipeline stage {name!r} must define a command list"
            )
        if not stage["command"] or not all(
            isinstance(token, (str, int, float)) for token in stage["command"]
        ):
            raise ValueError(
                f"Gnosis pipeline stage {name!r} has an invalid command list"
            )

    return config


def config_relative_path(config: Dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_config_dir"]) / path).resolve()
