"""Device discovery and explicit placement for optimization entry points."""

from __future__ import annotations

import copy
import inspect
import os
from dataclasses import fields, is_dataclass
from functools import wraps
from typing import TypeAlias

import jax
import optax

DeviceLike: TypeAlias = str | jax.Device
_constant_buffers_configured = False


def _configure_constant_buffers():
    """Keep large captured arrays in runtime buffers on JAX versions supporting it."""
    global _constant_buffers_configured
    if _constant_buffers_configured:
        return
    if (hasattr(jax.config, 'jax_use_simplified_jaxpr_constants') and
            'JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS' not in os.environ):
        jax.config.update('jax_use_simplified_jaxpr_constants', True)
    _constant_buffers_configured = True


def get_available_devices(platform: str | None = None) -> list[jax.Device]:
    """Return local JAX devices, ordered CPU first by default.

    ``platform`` accepts ``"cpu"``, ``"gpu"`` (or ``"cuda"``), and ``"tpu"``.
    An unavailable backend returns an empty list. Discovery preserves JAX's
    default backend and precision settings.
    """
    if platform is not None:
        if not isinstance(platform, str):
            raise TypeError("platform must be a string or None")
        platform = platform.lower()
        platform = "gpu" if platform == "cuda" else platform
        if platform not in ("cpu", "gpu", "tpu"):
            raise ValueError("platform must be 'cpu', 'gpu', 'cuda', or 'tpu'")
    result = []
    for backend in (platform,) if platform else ("cpu", "gpu", "tpu"):
        try:
            result.extend(jax.local_devices(backend=backend))
        except RuntimeError:
            continue
    return result


def resolve_device(device: DeviceLike = "cpu") -> jax.Device:
    """Resolve a local device object or a backend name with optional index."""
    if isinstance(device, jax.Device):
        if device not in get_available_devices(device.platform):
            raise ValueError("Optimization requires an available local device")
        return device
    if not isinstance(device, str):
        raise TypeError("device must be a JAX device or a string such as 'cpu' or 'gpu:0'")
    parts = device.lower().split(":")
    if len(parts) > 2 or (len(parts) == 2 and not parts[1].isdigit()):
        raise ValueError("device must name a backend with an optional nonnegative index")
    devices = get_available_devices(parts[0])
    index = int(parts[1]) if len(parts) == 2 else 0
    if index >= len(devices):
        raise ValueError(f"Requested optimization device '{device}' is unavailable")
    return devices[index]


def _move(value, device, memo=None):
    """Move JAX buffers into copies of the caller's containers."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, jax.Array):
        result = jax.device_put(value, device)
    elif isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((key, _move(item, device, memo)) for key, item in value.items())
    elif isinstance(value, tuple):
        items = [_move(item, device, memo) for item in value]
        result = type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    elif isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_move(item, device, memo) for item in value)
    elif is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        memo[id(value)] = result
        attributes = {field.name: getattr(value, field.name) for field in fields(value)}
        attributes.update(getattr(value, "__dict__", {}))
        for name, item in attributes.items():
            object.__setattr__(result, name, _move(item, device, memo))
    else:
        # NumPy arrays stay on the host until a JAX conversion in the device
        # context, preserving host float64 validation and relative weights.
        return value
    memo[id(value)] = result
    return result


def device_function(function):
    """Run an entry point with its explicit device and restored caller context."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        device = resolve_device(bound.arguments.get("device", "cpu"))
        memo = {}
        with jax.default_device(device):
            for name, value in list(bound.arguments.items()):
                if name != "device":
                    bound.arguments[name] = _move(value, device, memo)
            bound.arguments["device"] = device
            return function(*bound.args, **bound.kwargs)

    return wrapped


def bind_device(function, device):
    """Bind deferred optimizer callbacks to the selected device."""
    original = getattr(function, "_mrax_device_function", function)

    @wraps(original)
    def wrapped(*args, **kwargs):
        with jax.default_device(device):
            memo = {}
            return original(*_move(args, device, memo), **_move(kwargs, device, memo))

    wrapped._mrax_device_function = original
    return wrapped


def place_optimizer(optimizer, device):
    """Place initialization and updates, including custom Optax transforms."""
    def place(function, *, extra=False):
        original = getattr(function, "_mrax_optimizer_function", function)
        # The outer fitting JIT binds callable ExtraArgs as static functions.
        compiled = original if extra else jit_on_device(original, device)
        wrapped = bind_device(compiled, device)
        wrapped._mrax_optimizer_function = original
        return wrapped
    extra = isinstance(optimizer, optax.GradientTransformationExtraArgs)
    cls = optax.GradientTransformationExtraArgs if extra else optax.GradientTransformation
    return cls(place(optimizer.init), place(optimizer.update, extra=extra))


def jit_on_device(function, device):
    """Compile array-only callbacks on the requested local device."""
    _configure_constant_buffers()
    sharding = jax.sharding.SingleDeviceSharding(device)
    return jax.jit(function, in_shardings=sharding, out_shardings=sharding)
