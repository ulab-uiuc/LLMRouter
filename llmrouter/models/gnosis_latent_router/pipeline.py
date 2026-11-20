"""LLMRouter orchestration around the integrated Gnosis pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

from llmrouter.models.gnosis_latent_router.adapter import (
    STAGE_NAMES,
    run_gnosis_stage,
    run_gnosis_stages,
)
from llmrouter.models.gnosis_latent_router.config import load_gnosis_config
from llmrouter.models.gnosis_latent_router.router import GnosisLatentRouter


def run_gnosis_pipeline(
    *,
    config_path: str,
    action: str,
    query: Optional[str] = None,
    workspace: Optional[str] = None,
    dry_run: bool = False,
    route_only: bool = False,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run original Gnosis stages or one routed LLMRouter inference."""
    valid_actions = {*STAGE_NAMES, "infer", "run"}
    if action not in valid_actions:
        raise ValueError(f"Unknown Gnosis Latent Router action: {action}")

    config = load_gnosis_config(config_path)
    if action == "run":
        configured = config["pipeline"].get("run_stages") or list(STAGE_NAMES)
        invalid = [stage for stage in configured if stage not in STAGE_NAMES]
        if invalid:
            raise ValueError(f"Invalid pipeline.run_stages: {invalid}")
        stages = run_gnosis_stages(
            config,
            configured,
            workspace=workspace,
            dry_run=dry_run,
        )
        result: Dict[str, Any] = {"action": "run", "stages": stages}
        if query and not dry_run:
            result["inference"] = _run_inference(
                config_path=config_path,
                query=query,
                workspace=workspace,
                route_only=route_only,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return result

    if action in STAGE_NAMES:
        return {
            "action": action,
            "stage": run_gnosis_stage(
                config,
                action,
                workspace=workspace,
                dry_run=dry_run,
            ),
        }

    if not query:
        raise ValueError("`infer` requires --query")
    if dry_run:
        raise ValueError(
            "--dry-run applies to Gnosis subprocess stages, not inference"
        )
    return {
        "action": "infer",
        "inference": _run_inference(
            config_path=config_path,
            query=query,
            workspace=workspace,
            route_only=route_only,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    }


def _run_inference(
    *,
    config_path: str,
    query: str,
    workspace: Optional[str],
    route_only: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    router = GnosisLatentRouter(config_path, workspace=workspace)
    try:
        if route_only:
            from llmrouter.cli.router_inference import route_query

            return route_query(query, router, "gnosis_latent_router")
        from llmrouter.cli.router_inference import infer_query

        return infer_query(
            query,
            router,
            "gnosis_latent_router",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    finally:
        router.unload()
