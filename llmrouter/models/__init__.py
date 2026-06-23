"""Router and trainer registry.

Names are resolved **lazily** (PEP 562 ``__getattr__``) so that
``import llmrouter.models`` does not eagerly import torch / torch-geometric /
transformers / peft for every router. Importing a specific name — e.g.
``from llmrouter.models import KNNRouter`` — loads only that router's submodule
(and hence only its dependencies). Optional routers whose heavy dependencies are
unavailable resolve to ``None``, preserving the previous behavior.
"""
import importlib

# public attribute name -> (submodule, is_optional)
# optional entries resolve to None when their (heavy/GPU) dependencies are missing.
_REGISTRY = {
    "MetaRouter": ("meta_router", False),
    "BaseTrainer": ("base_trainer", False),
    "SmallestLLM": ("smallest_llm", False),
    "LargestLLM": ("largest_llm", False),
    "KNNRouter": ("knnrouter", False),
    "KNNRouterTrainer": ("knnrouter", False),
    "SVMRouter": ("svmrouter", False),
    "SVMRouterTrainer": ("svmrouter", False),
    "MLPRouter": ("mlprouter", False),
    "MLPTrainer": ("mlprouter", False),
    "MFRouter": ("mfrouter", False),
    "MFRouterTrainer": ("mfrouter", False),
    "EloRouter": ("elorouter", False),
    "EloRouterTrainer": ("elorouter", False),
    "AutomixRouter": ("automix", False),
    "AutomixRouterTrainer": ("automix", False),
    "DCRouter": ("routerdc", False),
    "DCTrainer": ("routerdc", False),
    "HybridLLMRouter": ("hybrid_llm", False),
    "HybridLLMTrainer": ("hybrid_llm", False),
    "GraphRouter": ("graphrouter", True),
    "GraphTrainer": ("graphrouter", True),
    "CausalLMRouter": ("causallm_router", True),
    "CausalLMTrainer": ("causallm_router", True),
    "RouterR1": ("router_r1", True),
    "GMTRouter": ("gmtrouter", True),
    "GMTRouterTrainer": ("gmtrouter", True),
    "PersonalizedRouter": ("personalizedrouter", True),
    "PersonalizedRouterTrainer": ("personalizedrouter", True),
}

__all__ = list(_REGISTRY)


def __getattr__(name):
    """Lazily import and cache a registered router/trainer on first access."""
    try:
        submodule, optional = _REGISTRY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    try:
        module = importlib.import_module(f".{submodule}", __name__)
        value = getattr(module, name)
    except Exception:
        if optional:
            value = None  # optional router with missing deps -> None (legacy behavior)
        else:
            raise
    globals()[name] = value  # cache so later lookups bypass __getattr__
    return value


def __dir__():
    return sorted(__all__)
