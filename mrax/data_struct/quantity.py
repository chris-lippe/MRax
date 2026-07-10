from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass
class Quantity:
    """Value paired with its SI unit string."""

    values: Any
    si_unit: str

    def __post_init__(self):
        if not isinstance(self.values, jnp.ndarray):
            try:
                self.values = jnp.asarray(self.values)
            except Exception as exc:  # pragma: no cover - defensive branch
                raise TypeError("Quantity.values must be convertible to jax.numpy.ndarray") from exc
        if not isinstance(self.si_unit, str) or not self.si_unit.strip():
            raise ValueError("Quantity.si_unit must be a non-empty string naming the SI unit")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    def as_array(self) -> Any:
        return self.values

    def with_values(self, values: Any) -> "Quantity":
        return Quantity(jnp.asarray(values), self.si_unit)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=copy)


def require_quantity_dict(mapping: dict[str, Any], context: str) -> dict[str, Quantity]:
    """Validate that all values are Quantity instances and return a typed dict."""
    for key, value in mapping.items():
        if not isinstance(value, Quantity):
            raise TypeError(f"{context} '{key}' must be provided as a Quantity with an SI unit")
    return mapping  # type: ignore[return-value]


def unwrap_values(mapping: dict[str, Quantity]) -> dict[str, jnp.ndarray]:
    """Extract raw arrays from a dict of quantities."""
    return {k: v.values for k, v in mapping.items()}


def attach_units(fields: dict[str, jnp.ndarray], units: dict[str, str]) -> dict[str, Quantity]:
    """Pair raw parameter maps with their SI unit strings."""
    missing = [key for key in fields if key not in units]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing SI unit definitions for parameters: {missing_str}")
    return {key: Quantity(val, units[key]) for key, val in fields.items()}
