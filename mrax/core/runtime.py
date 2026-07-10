from __future__ import annotations

import gc
import logging

import jax

_LOG = logging.getLogger(__name__)


def release_memory(*, gc_collect: bool = True, clear_jax_cache: bool = True) -> None:
    """Release Python/JAX caches to reduce peak memory in long workflows."""

    if gc_collect:
        gc.collect()
    if clear_jax_cache:
        try:
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        except Exception:
            _LOG.debug("Could not clear JAX caches", exc_info=True)


__all__ = ["release_memory"]
