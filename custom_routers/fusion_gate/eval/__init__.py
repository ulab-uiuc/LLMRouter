"""fusion_gate.eval — offline eval + retrain harness for FusionGateRouter.

This package contains the route-vs-fuse evaluation harness (UMB-122/124) and the
scripted retrain loop (UMB-126). Both run fully offline in ``--mock`` mode against
the bundled fixtures under ``fixtures/`` and spend nothing; a keyed live-run path
is documented in ``results.md`` and in each module's docstring.

Design constraint: nothing here imports torch or pandas. The harness composes the
torch-free seams of the plugin directly — :class:`RouteGate`, :class:`CapabilityScorer`,
:class:`FusionExecutor` / a deterministic mock stub, and ``fusion_log`` — mirroring
what :class:`FusionGateRouter` wires internally, so the harness is importable and
testable with only the standard library.
"""
