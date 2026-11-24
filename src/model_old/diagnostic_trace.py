"""Optional forward diagnostic trace for inline tensor ops.

When disabled (default), ``record`` is a no-op and does not allocate.
When enabled via ``capture_diagnostic_trace``, callers append detached copies
of intermediate tensors without changing the live computation graph or values.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

import torch

_TRACE_BUF: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "explainable_diag_trace_buf", default=None
)
_TRACE_COUNTS: ContextVar[Optional[Dict[str, int]]] = ContextVar(
    "explainable_diag_trace_counts", default=None
)


def diagnostic_trace_enabled() -> bool:
    return _TRACE_BUF.get() is not None


def record(op: str, value: Any, *, kind: str = "tensor") -> None:
    """Append a detached snapshot of ``value`` under ``op`` if tracing is on.

    Repeated calls with the same ``op`` get a ``#N`` suffix (1-indexed) so hooks
    and inline sites that fire more than once remain distinguishable.
    """
    buf = _TRACE_BUF.get()
    if buf is None:
        return
    counts = _TRACE_COUNTS.get()
    if counts is None:
        counts = {}
        _TRACE_COUNTS.set(counts)
    n = counts.get(op, 0) + 1
    counts[op] = n
    name = op if n == 1 else f"{op}#{n}"

    entry: Dict[str, Any] = {"name": name, "op": op, "call_index": n, "kind": kind}
    if value is None:
        entry["available"] = False
        entry["reason"] = "value_is_none"
    elif torch.is_tensor(value):
        detached = value.detach()
        entry["available"] = True
        entry["tensor"] = detached
        entry["shape"] = list(detached.shape)
        entry["dtype"] = str(detached.dtype)
        entry["device"] = str(detached.device)
    elif isinstance(value, (int, float, bool)):
        entry["available"] = True
        entry["scalar"] = value
        entry["kind"] = "scalar"
    else:
        entry["available"] = False
        entry["reason"] = f"unsupported_type:{type(value).__name__}"
    buf.append(entry)


@contextmanager
def capture_diagnostic_trace() -> Iterator[List[Dict[str, Any]]]:
    """Enable tracing for the duration of the context; yield the buffer."""
    buf: List[Dict[str, Any]] = []
    token_buf = _TRACE_BUF.set(buf)
    token_counts = _TRACE_COUNTS.set({})
    try:
        yield buf
    finally:
        _TRACE_BUF.reset(token_buf)
        _TRACE_COUNTS.reset(token_counts)
