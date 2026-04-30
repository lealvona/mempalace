"""MemPalace as a Hermes Agent memory provider plugin.

Hermes loads this file from ``~/.hermes/plugins/mempalace/__init__.py`` after
``hermes plugins install lealvona/mempalace``. The actual MemoryProvider
implementation and the eight tool handlers live in
``mempalace/hermes_provider.py`` — this file is a thin re-export so the
Hermes plugin loader can find the canonical ``register(ctx)`` entry point
without breaking the underlying ``mempalace`` Python package layout.

The ``mempalace`` runtime package is installed via the ``pip_dependencies``
declaration in ``plugin.yaml`` (pinned to this same fork's ``master`` branch
so the bundled ``hermes_provider`` module is guaranteed to be available).
"""

from __future__ import annotations

from mempalace.hermes_provider import (  # noqa: F401  re-exported for Hermes
    MemPalaceMemoryProvider,
    register,
)

__all__ = ["MemPalaceMemoryProvider", "register"]
