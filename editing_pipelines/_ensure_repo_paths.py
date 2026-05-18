"""Put the checkout containing ``paths.py``, optional PATH_TO_GRAFT override, and seed-gnn on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_DONE = False


def bootstrap() -> None:
    global _DONE
    if _DONE:
        return
    checkout_root = Path(__file__).resolve().parents[1]
    c_s = str(checkout_root)
    if c_s not in sys.path:
        sys.path.insert(0, c_s)

    import paths as _paths  # noqa: E402

    graft = _paths.path_to_graft().resolve()
    g_s = str(graft)
    if g_s not in sys.path:
        sys.path.insert(0, g_s)

    seed = _paths.seed_gnn_dir().resolve()
    s_s = str(seed)
    if s_s not in sys.path:
        sys.path.insert(0, s_s)

    _DONE = True


__all__ = ["bootstrap"]
