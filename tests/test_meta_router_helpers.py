"""Regression tests for the shared MetaRouter helpers.

Locks the behavior of `MetaRouter._resolve_query_data` and `_normalize_row`
(extracted from 14 routers in commits 5507b09 / 51440e5) and proves they are
equivalent to the original inline blocks they replaced.

Runnable directly (`python tests/test_meta_router_helpers.py`) or via pytest.
"""
import copy

import torch.nn as nn

from llmrouter.models.meta_router import MetaRouter


class _DummyRouter(MetaRouter):
    """Minimal concrete MetaRouter (no YAML/data) for unit-testing helpers."""

    def route_batch(self, batch):  # pragma: no cover - not exercised
        return []

    def route_single(self, batch):  # pragma: no cover - not exercised
        return None


# --- reference implementations: the ORIGINAL inline logic, verbatim ---------
def _orig_resolve(self, batch):
    if batch is not None:
        query_data = batch if isinstance(batch, list) else [batch]
    else:
        if hasattr(self, "query_data_test") and self.query_data_test is not None:
            query_data = copy.copy(self.query_data_test)
        else:
            return []  # original returned [] directly (helper returns None -> caller returns [])
    return query_data


def _orig_normalize(row, task_name):
    if isinstance(row, dict):
        row_copy = copy.copy(row)
        original_query = row_copy.get("query", "")
        row_task_name = row_copy.get("task_name", task_name)
    else:
        row_copy = {"query": str(row)}
        original_query = str(row)
        row_task_name = task_name
    return row_copy, original_query, row_task_name


def _make():
    return _DummyRouter(model=nn.Identity())


def test_resolve_explicit_batch():
    r = _make()
    assert r._resolve_query_data({"query": "q"}) == [{"query": "q"}]          # dict -> wrapped
    assert r._resolve_query_data([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]  # list -> as-is


def test_resolve_fallback_to_test_set():
    r = _make()
    r.query_data_test = [{"query": "x"}]
    out = r._resolve_query_data(None)
    assert out == [{"query": "x"}]
    assert out is not r.query_data_test  # must be a copy, not the same object


def test_resolve_no_data_returns_none():
    r = _make()  # no query_data_test attribute set
    assert r._resolve_query_data(None) is None
    r.query_data_test = None
    assert r._resolve_query_data(None) is None


def test_resolve_empty_test_set_edge():
    r = _make()
    r.query_data_test = []  # falsy but not None -> resolves to [] (not the None branch)
    assert r._resolve_query_data(None) == []


def test_normalize_dict_and_non_dict():
    r = _make()
    assert r._normalize_row({"query": "hi", "task_name": "t1"}, "fallback") == (
        {"query": "hi", "task_name": "t1"}, "hi", "t1",
    )
    assert r._normalize_row({"query": "hi"}, "fallback") == (
        {"query": "hi"}, "hi", "fallback",  # task_name falls back
    )
    assert r._normalize_row("raw string", "fallback") == (
        {"query": "raw string"}, "raw string", "fallback",
    )


def test_equivalence_to_original_inline_logic():
    """The crux: helpers must match the original blocks across an input matrix."""
    r = _make()

    # _resolve_query_data: None-result <=> original []-result; else equal.
    resolve_cases = [{"query": "a"}, [{"query": "b"}], [], None]
    for qdt in ([{"query": "t"}], [], None, "MISSING"):
        if qdt == "MISSING":
            if hasattr(r, "query_data_test"):
                del r.query_data_test
        else:
            r.query_data_test = qdt
        for batch in resolve_cases:
            new = r._resolve_query_data(batch)
            old = _orig_resolve(r, batch)
            # map helper's None -> [] to compare with original's [] sentinel
            assert (new if new is not None else []) == old, (qdt, batch, new, old)

    # _normalize_row: must be identical across dict/non-dict/missing-keys.
    for row in [{"query": "q", "task_name": "t"}, {"query": "q"}, {}, "s", 42, None]:
        assert r._normalize_row(row, "fb") == _orig_normalize(row, "fb"), row


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
