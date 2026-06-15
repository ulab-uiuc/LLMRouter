"""fusion_gate — route-vs-fuse meta-router plugin for LLMRouter.

Auto-discovered from ./custom_routers/ . See router.py for the entry point.

``FusionGateRouter`` is imported LAZILY (PEP 562 ``__getattr__``) rather than
eagerly: ``router.py`` pulls in torch (MetaRouter subclasses ``nn.Module``), and
an eager import here would force torch to load whenever this package is merely
*resolved* — which pytest does for every test module under ``tests/`` while
walking the package hierarchy. That made the four torch-free test modules
uncollectable under the standard ``pytest custom_routers/fusion_gate/tests/``
invocation (ModuleNotFoundError: No module named 'torch'). Deferring the import
to first attribute access keeps package resolution torch-free while still
exposing ``FusionGateRouter`` as a top-level name when it is actually used.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import for type-checkers only; not executed at runtime
    from .router import FusionGateRouter

__all__ = ["FusionGateRouter"]


def __getattr__(name: str) -> Any:
    """Lazily import ``FusionGateRouter`` on first access (PEP 562).

    torch (a transitive dependency of ``router.py``) is loaded only when the
    router is actually requested, not at package-collection time.
    """
    if name == "FusionGateRouter":
        from .router import FusionGateRouter

        return FusionGateRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
