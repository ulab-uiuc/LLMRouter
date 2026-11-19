"""Pass-through orchestration for the integrated Gnosis implementation."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping

from llmrouter.models.gnosis_latent_router.config import config_relative_path


PLACEHOLDER = re.compile(r"\{([A-Za-z_]\w*)\}")
STAGE_NAMES = ("generate", "label", "train", "benchmark")


def gnosis_backend_root() -> Path:
    """Return the integrated Gnosis implementation shipped with LLMRouter."""
    return Path(__file__).resolve().parent / "backend"


def resolve_workspace(
    config: Dict[str, Any], override: str | Path | None = None
) -> Path:
    """Resolve the writable data, checkpoint, and evaluation workspace."""
    override = override or os.environ.get("GNOSIS_WORKSPACE")
    if override:
        workspace = Path(override).expanduser().resolve()
    else:
        configured = config["pipeline"].get("workspace")
        workspace = config_relative_path(config, configured) if configured else Path.cwd()
    if not workspace.is_dir():
        raise FileNotFoundError(
            f"Gnosis workspace directory does not exist: {workspace}"
        )
    return workspace


def _template_values(
    config: Dict[str, Any],
    *,
    workspace: Path,
    stage: Mapping[str, Any] | None = None,
) -> Dict[str, str]:
    pipeline = config["pipeline"]
    stage = stage or {}
    python = stage.get("python") or pipeline.get("python") or sys.executable
    values = {
        "python": str(python),
        "backend_root": str(gnosis_backend_root()),
        "workspace_root": str(workspace),
        "config_dir": str(config["_config_dir"]),
    }
    for source in (pipeline.get("variables", {}), stage.get("variables", {})):
        values.update({str(key): str(value) for key, value in source.items()})
    values.update(
        {
            "target_model": str(config["target_model"]),
            "target_model_name": str(config["target_model_name"]),
            "fallback_model": str(config["fallback_model"]),
        }
    )
    def expand(key: str, parents: tuple[str, ...]) -> str:
        if key in parents:
            raise ValueError("Cyclic pipeline variable: " + " -> ".join((*parents, key)))
        return PLACEHOLDER.sub(
            lambda m: expand(m[1], (*parents, key)) if m[1] in values else m[0],
            values[key],
        )

    return {key: expand(key, ()) for key in values}


def render_value(value: str, values: Mapping[str, str]) -> str:
    """Preserve literal braces outside of declared pipeline placeholders."""
    return PLACEHOLDER.sub(lambda match: values.get(match[1], match[0]), str(value))


def resolve_checkpoint_path(
    config: Dict[str, Any],
    *,
    workspace: str | Path | None = None,
) -> Path:
    checkpoint = config.get("inference", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("gnosis_latent_router.inference.checkpoint is required")
    root = resolve_workspace(config, workspace)
    values = _template_values(config, workspace=root)
    path = Path(render_value(str(checkpoint), values)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def require_checkpoint(
    config: Dict[str, Any],
    *,
    workspace: str | Path | None = None,
) -> Path:
    checkpoint = resolve_checkpoint_path(config, workspace=workspace)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Gnosis checkpoint not found at "
            f"{checkpoint}. Match pipeline.variables.checkpoint to the training "
            "recipe's output_dir."
        )
    return checkpoint


def command_for_stage(
    config: Dict[str, Any],
    stage_name: str,
    *,
    workspace: str | Path | None = None,
) -> tuple[list[str], Path, Dict[str, str]]:
    if stage_name not in STAGE_NAMES:
        raise ValueError(f"Unknown Gnosis pipeline stage: {stage_name}")
    stage = config["pipeline"]["stages"].get(stage_name)
    if stage is None:
        raise ValueError(
            f"Gnosis pipeline stage {stage_name!r} is not configured"
        )
    root = resolve_workspace(config, workspace)
    values = _template_values(config, workspace=root, stage=stage)
    command = [render_value(str(token), values) for token in stage["command"]]
    cwd_value = render_value(
        str(stage.get("cwd", "{workspace_root}")), values
    )
    cwd = Path(cwd_value).expanduser()
    if not cwd.is_absolute():
        cwd = root / cwd
    cwd = cwd.resolve()

    environment = dict(os.environ)
    for source in (
        config["pipeline"].get("environment", {}),
        stage.get("environment", {}),
    ):
        environment.update(
            {
                str(key): render_value(str(value), values)
                for key, value in source.items()
            }
        )
    return command, cwd, environment


def _run_logged_subprocess(
    command: list[str],
    *,
    cwd: Path,
    environment: Dict[str, str],
    log_path: Path,
) -> None:
    """Stream a stage to the terminal and a durable append-only log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_environment = dict(environment)
    child_environment.setdefault("PYTHONUNBUFFERED", "1")
    started = datetime.now(timezone.utc).isoformat()
    header = (
        f"\n===== Gnosis stage start {started} =====\n"
        f"cwd: {cwd}\n"
        f"command: {shlex.join(command)}\n"
    ).encode("utf-8", errors="replace")

    with log_path.open("ab", buffering=0) as log_handle:
        log_handle.write(header)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("Could not capture Gnosis stage output")
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                log_handle.write(chunk)
                terminal = getattr(sys.stdout, "buffer", None)
                if terminal is not None:
                    terminal.write(chunk)
                    terminal.flush()
                else:
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
        return_code = process.wait()
        ended = datetime.now(timezone.utc).isoformat()
        footer = (
            f"\n===== Gnosis stage end {ended} return_code={return_code} =====\n"
        ).encode("utf-8", errors="replace")
        log_handle.write(footer)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_gnosis_stage(
    config: Dict[str, Any],
    stage_name: str,
    *,
    workspace: str | Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute one original pipeline command without argument rewriting."""
    root = resolve_workspace(config, workspace)
    command, cwd, environment = command_for_stage(config, stage_name, workspace=root)
    pipeline = config["pipeline"]
    values = _template_values(config, workspace=root, stage=pipeline["stages"][stage_name])
    log_dir = Path(render_value(pipeline.get("log_dir", "logs/gnosis"), values)).expanduser()
    log_path = root / log_dir / f"{stage_name}.log"
    result = {
        "stage": stage_name, "command": command, "cwd": str(cwd),
        "dry_run": dry_run, "log_path": str(log_path),
    }
    if dry_run:
        return result

    marker = pipeline.get("downloaded_release_marker")
    if stage_name in {"generate", "label"} and marker:
        marker_path = root / Path(render_value(marker, values)).expanduser()
        if marker_path.is_file():
            raise RuntimeError(
                f"Found a downloaded verified release at {marker_path}. "
                "Choose a fresh run_root for generation and labeling."
            )
    if stage_name == "benchmark":
        require_checkpoint(config, workspace=root)
    _run_logged_subprocess(command, cwd=cwd, environment=environment, log_path=log_path)
    if stage_name == "train":
        result["checkpoint"] = str(
            require_checkpoint(config, workspace=workspace)
        )
    return result


def run_gnosis_stages(
    config: Dict[str, Any],
    stages: Iterable[str],
    *,
    workspace: str | Path | None = None,
    dry_run: bool = False,
) -> list[Dict[str, Any]]:
    return [
        run_gnosis_stage(
            config,
            stage,
            workspace=workspace,
            dry_run=dry_run,
        )
        for stage in stages
    ]


def load_gnosis_runtime_module(config: Dict[str, Any]) -> ModuleType:
    """Load Gnosis generation and latent-scoring runtime in-process."""
    runtime = Path(config["inference"]["runtime_module"])
    if runtime.suffix != ".py" or not all(
        part.isidentifier() for part in runtime.with_suffix("").parts
    ):
        raise ValueError("inference.runtime_module must be a Python module inside the backend")
    name = ".".join(runtime.with_suffix("").parts)
    return import_module(f"{__package__}.backend.{name}")
