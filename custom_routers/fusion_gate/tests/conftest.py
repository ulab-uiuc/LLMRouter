"""pytest bootstrap for the fusion_gate test suite.

Makes ``pytest custom_routers/fusion_gate/tests/`` work out of the box, without
relying on the standalone ``python test_gate.py`` runner as a workaround.

Two things are guaranteed here:

  1. The repo root is on ``sys.path`` so ``custom_routers`` resolves as a package
     (the torch-dependent ``test_router`` imports ``custom_routers.fusion_gate.router``
     directly).
  2. ``--import-mode=importlib`` is enabled so pytest does not rewrite ``sys.path``
     in ways that re-trigger package ``__init__`` collection. Combined with the
     lazy ``__getattr__`` in ``fusion_gate/__init__.py``, the four torch-free test
     modules (gate / executor / capability / fusion_log / eval_harness) collect and
     run with no torch installed.
"""

from __future__ import annotations

import os
import sys

# Repo root = three levels up from this file (tests/ -> fusion_gate/ ->
# custom_routers/ -> repo root).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_configure(config) -> None:
    """Force importlib import mode so package collection stays torch-free.

    Setting it here (rather than only in pytest.ini) keeps the behavior local to
    this plugin's tests and avoids editing any repo-level config outside
    custom_routers/fusion_gate/.
    """
    config.option.importmode = "importlib"
