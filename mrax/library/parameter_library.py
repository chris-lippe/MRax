from __future__ import annotations

import os
import tomllib
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_LIBRARY_ENV = "MRAX_PARAMETER_LIBRARY"
_PRIVATE_LIBRARY_ENV = "MRAX_PRIVATE_PARAMETER_LIBRARY"
_DEFAULT_LIBRARY_PATH = Path(__file__).with_suffix(".toml")


@lru_cache(maxsize=None)
def _load_library(path: str) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


@lru_cache(maxsize=None)
def _load_resolved_library(base_path: str, private_path: str) -> dict[str, Any]:
    library = _load_library(base_path)
    if not private_path:
        return library
    overlay_path = Path(private_path)
    if not overlay_path.exists():
        raise FileNotFoundError(f"Private parameter library not found at {overlay_path}")
    return _deep_merge(library, _load_library(str(overlay_path)))


def load_parameter_library(path: str | Path | None = None, *, reload: bool = False) -> dict[str, Any]:
    """Load the parameter library TOML file and return a deep copy.

    ``MRAX_PARAMETER_LIBRARY`` replaces the built-in library. ``MRAX_PRIVATE_PARAMETER_LIBRARY``
    overlays local additions, such as lab-specific pool definitions, on top of that base.
    """
    target = Path(path or os.environ.get(_LIBRARY_ENV, _DEFAULT_LIBRARY_PATH)).expanduser()
    private_env = os.environ.get(_PRIVATE_LIBRARY_ENV, "")
    private = str(Path(private_env).expanduser()) if private_env else ""
    if reload:
        _load_library.cache_clear()
        _load_resolved_library.cache_clear()
    if not target.exists():
        raise FileNotFoundError(f"Parameter library not found at {target}")
    return deepcopy(_load_resolved_library(str(target), private))


def _get_model_section(model_name: str, library: dict[str, Any] | None = None) -> dict[str, Any]:
    library = library or load_parameter_library()
    models = library.get("models", {})
    if model_name not in models:
        raise KeyError(f"Model '{model_name}' not found in parameter library")
    section = models[model_name]
    if not isinstance(section, dict):
        raise TypeError(f"Model section for '{model_name}' must be a dict")
    return section


def get_model_param_set(
    model_name: str,
    key: str = "default",
    *,
    library: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    section = _get_model_section(model_name, library=library)
    if key in ("pools", "defaults"):
        raise KeyError(f"'{key}' is reserved for model metadata in the parameter library")
    if key not in section:
        raise KeyError(f"Parameter key '{key}' not found for model '{model_name}'")
    param_set = section[key]
    if not isinstance(param_set, dict):
        raise TypeError(f"Parameter set '{key}' for model '{model_name}' must be a dict")
    return deepcopy(param_set)


def get_model_defaults(
    model_name: str,
    *,
    library: dict[str, Any] | None = None,
) -> dict[str, Any]:
    section = _get_model_section(model_name, library=library)
    defaults = section.get("defaults", {})
    if defaults is None:
        return {}
    if not isinstance(defaults, dict):
        raise TypeError(f"Defaults section for model '{model_name}' must be a dict")
    return deepcopy(defaults)


def get_model_pools(
    model_name: str,
    *,
    library: dict[str, Any] | None = None,
) -> dict[str, Any]:
    section = _get_model_section(model_name, library=library)
    pools = section.get("pools", {})
    if pools is None:
        return {}
    if not isinstance(pools, dict):
        raise TypeError(f"Pools section for model '{model_name}' must be a dict")
    return deepcopy(pools)


def _apply_context(param_specs: dict[str, dict[str, Any]], context: Mapping[str, float] | None) -> dict[str, dict[str, Any]]:
    if not context:
        return param_specs
    updated: dict[str, dict[str, Any]] = {}
    for name, spec in param_specs.items():
        if not isinstance(spec, Mapping):
            raise TypeError(f"Param spec for '{name}' must be a mapping")
        cfg = dict(spec)
        scale_key = cfg.pop("scale_key", None)
        if scale_key and scale_key in context:
            cfg["scale"] = float(context[scale_key])
        init_scale_key = cfg.pop("init_scale_key", None)
        if init_scale_key and init_scale_key in context and "init" in cfg:
            cfg["init"] = _scale_context_value(cfg["init"], float(context[init_scale_key]))
        bounds_scale_key = cfg.pop("bounds_scale_key", None)
        if bounds_scale_key and bounds_scale_key in context:
            bounds = cfg.get("bounds")
            if bounds is not None:
                scale_val = float(context[bounds_scale_key])
                cfg["bounds"] = [float(val) * scale_val for val in bounds]
                if "init" in cfg:
                    cfg["init"] = _scale_context_value(cfg["init"], scale_val)
        updated[name] = cfg
    return updated


def _merge_param_specs(
    base: dict[str, dict[str, Any]],
    overrides: Mapping[str, Mapping[str, Any] | None] | None,
) -> dict[str, dict[str, Any]]:
    if not overrides:
        return base
    merged = deepcopy(base)
    for name, override in overrides.items():
        if override is None:
            merged.pop(name, None)
            continue
        if not isinstance(override, Mapping):
            raise TypeError(f"Override for '{name}' must be a mapping")
        existing = merged.get(name)
        if isinstance(existing, Mapping):
            merged[name] = {**existing, **override}
        else:
            merged[name] = dict(override)
    return merged


def _scale_context_value(value: Any, scale: float) -> Any:
    if isinstance(value, list):
        return [_scale_context_value(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_context_value(item, scale) for item in value)
    return float(value) * scale


def resolve_param_specs(
    model_name: str,
    key: str = "default",
    *,
    context: Mapping[str, float] | None = None,
    overrides: Mapping[str, Mapping[str, Any] | None] | None = None,
    library: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve parameter specs for a model, applying context and overrides."""
    specs = get_model_param_set(model_name, key=key, library=library)
    specs = _apply_context(specs, context)
    specs = _merge_param_specs(specs, overrides)
    return specs


__all__ = [
    "get_model_defaults",
    "get_model_param_set",
    "get_model_pools",
    "load_parameter_library",
    "resolve_param_specs",
]
