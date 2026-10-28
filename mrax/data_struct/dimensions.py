from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .quantity import Quantity

SPATIAL_DIMS = ("x", "y", "z")
TEMPORAL_DIM = "t"
FRAME_DIM = "frame"


@dataclass(frozen=True)
class DimensionSpec:
    name: str
    observable: str | None = None
    allow_singleton: bool = True


DEFAULT_DIM_SPECS_BY_EXP_TYPE: dict[str, tuple[DimensionSpec, ...]] = {
    "anatomical": (),
    "dce": (DimensionSpec(TEMPORAL_DIM, None),),
    "ir": (DimensionSpec("ti", "TI"),),
    "fair_asl": (DimensionSpec("ti", "TI"), DimensionSpec("fair", "fair_label")),
    "vtr": (DimensionSpec("tr", "TR"),),
    "vfa": (DimensionSpec("flip", "flip_angle"),),
    "mse": (DimensionSpec("te", "TE"),),
    "mge": (DimensionSpec("te", "TE"),),
    # Diffusion acquisitions use a per-volume list of bvals/bvecs.
    # Keep them as frames so incomplete shells and
    # arbitrary direction tables remain navigable one acquisition at a time.
    "dti": (),
    "dki": (),
    "sat_trans": (
        DimensionSpec(TEMPORAL_DIM, None),
        DimensionSpec("b1", "sat_b1"),
        DimensionSpec("offset", "freq_offsets"),
    ),
}


DEFAULT_DIM_SPECS_BY_MODEL_NAME: dict[str, tuple[DimensionSpec, ...]] = {
    "t1_ir": (DimensionSpec("ti", "TI"),),
    "fair_asl": DEFAULT_DIM_SPECS_BY_EXP_TYPE["fair_asl"],
    "t1_vtr": (DimensionSpec("tr", "TR"),),
    "t1_vfa": (DimensionSpec("flip", "flip_angle"),),
    "t2_se": (DimensionSpec("te", "TE"),),
    "t2star_se": (DimensionSpec("te", "TE"),),
    "dti": DEFAULT_DIM_SPECS_BY_EXP_TYPE["dti"],
    "dki": DEFAULT_DIM_SPECS_BY_EXP_TYPE["dki"],
    "wasabi": DEFAULT_DIM_SPECS_BY_EXP_TYPE["sat_trans"],
    "multipool_lorentzian": DEFAULT_DIM_SPECS_BY_EXP_TYPE["sat_trans"],
    "zaiss_full": DEFAULT_DIM_SPECS_BY_EXP_TYPE["sat_trans"],
}


def _observable_values(observables: Mapping[str, Quantity] | None, key: str | None):
    if not observables or not key:
        return None
    val = observables.get(key)
    if val is None:
        return None
    if isinstance(val, Quantity):
        return np.asarray(val.values)
    return np.asarray(val)


def _observable_length(
    observables: Mapping[str, Quantity] | None, key: str | None, *, frame_count: int
) -> tuple[int | None, bool]:
    values = _observable_values(observables, key)
    if values is None:
        return None, False
    if values.ndim == 0:
        return 1, False
    length = int(values.shape[0])
    if length == frame_count:
        return length, True
    return length, False


def _unique_count(values: np.ndarray, *, decimals: int = 6) -> int:
    if values.ndim == 0:
        return 1
    rounded = np.round(values.astype(np.float64), decimals=decimals)
    return int(np.unique(rounded).size)


def get_default_dim_specs(exp_type: str | None, model_name: str | None = None) -> tuple[DimensionSpec, ...]:
    if exp_type and exp_type in DEFAULT_DIM_SPECS_BY_EXP_TYPE:
        return DEFAULT_DIM_SPECS_BY_EXP_TYPE[exp_type]
    if model_name:
        if model_name in DEFAULT_DIM_SPECS_BY_MODEL_NAME:
            return DEFAULT_DIM_SPECS_BY_MODEL_NAME[model_name]
        if model_name.startswith("ab_mt"):
            return DEFAULT_DIM_SPECS_BY_EXP_TYPE["sat_trans"]
    return ()


def infer_non_spatial_dims(
    frame_count: int,
    observables: Mapping[str, Quantity] | None,
    dim_specs: Sequence[DimensionSpec],
    *,
    allow_time_fallback: bool = True,
    fallback_dim: str = FRAME_DIM,
) -> list[tuple[str, int]]:
    if frame_count <= 0:
        return [(fallback_dim, 1)]

    sizes: list[tuple[str, int]] = []
    per_frame_specs: list[DimensionSpec] = []
    known_product = 1

    for spec in dim_specs:
        length, is_per_frame = _observable_length(observables, spec.observable, frame_count=frame_count)
        if length is None:
            size = 1 if spec.allow_singleton else 0
        elif is_per_frame and len(dim_specs) > 1:
            per_frame_specs.append(spec)
            size = 1
        else:
            if length > frame_count or (length <= 0):
                size = 0
            else:
                size = length
        if size <= 0:
            return [(fallback_dim, frame_count)]
        sizes.append((spec.name, int(size)))
        if size > 1:
            known_product *= int(size)

    # Try to infer axis sizes from per-frame observables when they repeat.
    if per_frame_specs and known_product <= frame_count:
        for spec in dim_specs:
            if spec not in per_frame_specs:
                continue
            values = _observable_values(observables, spec.observable)
            if values is None or values.ndim == 0:
                continue
            uniq = _unique_count(values)
            if uniq <= 1:
                continue
            if frame_count % uniq != 0:
                continue
            candidate = uniq
            if frame_count % (known_product * candidate) != 0:
                continue
            updated = False
            for j, (name, size) in enumerate(sizes):
                if name == spec.name and size == 1:
                    sizes[j] = (name, candidate)
                    known_product *= candidate
                    updated = True
                    break
            if updated:
                per_frame_specs.remove(spec)
                if known_product == frame_count:
                    break

    if known_product == 0:
        return [(fallback_dim, frame_count)]

    if frame_count % known_product != 0:
        return [(fallback_dim, frame_count)]

    remainder = frame_count // known_product
    has_observable_axis = any(name != TEMPORAL_DIM and int(size) > 1 for name, size in sizes)
    if remainder > 1 and has_observable_axis:
        # Prefer an explicit temporal repeat axis only after an observable axis
        # explains part of the frame grid. Bare frame stacks should stay "frame".
        for idx, (name, size) in enumerate(sizes):
            if name == TEMPORAL_DIM and size == 1:
                sizes[idx] = (name, remainder)
                remainder = 1
                break
    if remainder > 1:
        if allow_time_fallback and has_observable_axis:
            sizes.append((TEMPORAL_DIM, remainder))
            remainder = 1
        else:
            return [(fallback_dim, frame_count)]

    if not sizes:
        return [(fallback_dim, frame_count)]
    return sizes


def build_experiment_dims(
    frame_count: int,
    spatial_shape: Iterable[int],
    observables: Mapping[str, Quantity] | None,
    *,
    exp_type: str | None = None,
    model_name: str | None = None,
    fallback_dim: str = FRAME_DIM,
) -> dict[str, int]:
    specs = get_default_dim_specs(exp_type, model_name)
    non_spatial = infer_non_spatial_dims(
        frame_count,
        observables,
        specs,
        fallback_dim=TEMPORAL_DIM if exp_type == "dce" else fallback_dim,
    )
    dims: dict[str, int] = {}
    for name, size in non_spatial:
        dims[name] = int(size)

    shape = tuple(int(s) for s in spatial_shape)
    if len(shape) == 3:
        x, y, z = shape
        dims.update({"x": x, "y": y, "z": z})
    elif len(shape) == 2:
        x, y = shape
        dims.update({"x": x, "y": y})
    elif len(shape) == 1:
        dims.update({"x": shape[0]})
    else:
        for idx, size in enumerate(shape):
            dims[f"dim_{idx}"] = size
    return dims


def dim_observable_map(
    exp_type: str | None, model_name: str | None = None
) -> dict[str, str | None]:
    specs = get_default_dim_specs(exp_type, model_name)
    return {spec.name: spec.observable for spec in specs}


def non_spatial_names(dims: Mapping[str, int]) -> list[str]:
    return [name for name in dims.keys() if name not in SPATIAL_DIMS]


def spatial_names(dims: Mapping[str, int]) -> list[str]:
    return [name for name in dims.keys() if name in SPATIAL_DIMS]


def spatial_shape_from_dims(dims: Mapping[str, int]) -> tuple[int, ...]:
    names = spatial_names(dims)
    return tuple(int(dims[name]) for name in names)


def non_spatial_shape_from_dims(dims: Mapping[str, int]) -> tuple[int, ...]:
    names = non_spatial_names(dims)
    return tuple(int(dims[name]) for name in names)


def frame_count_from_dims(dims: Mapping[str, int]) -> int:
    sizes = non_spatial_shape_from_dims(dims)
    if not sizes:
        return 1
    prod = 1
    for size in sizes:
        prod *= int(size)
    return int(prod)


def reshape_frames_to_dims(
    data: np.ndarray,
    dims: Mapping[str, int],
) -> np.ndarray:
    non_spatial = non_spatial_shape_from_dims(dims)
    spatial = spatial_shape_from_dims(dims)
    target = non_spatial + spatial
    if tuple(int(v) for v in data.shape) == target:
        return data
    frame_count = frame_count_from_dims(dims)
    if data.ndim == len(spatial) + 1 and int(data.shape[0]) == frame_count:
        return data.reshape(target)
    return data


def reshape_dims_to_frames(
    data: np.ndarray,
    dims: Mapping[str, int],
    *,
    broadcast_spatial: bool = True,
) -> np.ndarray:
    """Reshape a dim-ordered array into ``(frame, *spatial)`` layout."""

    non_spatial = non_spatial_shape_from_dims(dims)
    spatial = spatial_shape_from_dims(dims)
    frame_count = frame_count_from_dims(dims)
    target = non_spatial + spatial
    arr = np.asarray(data)

    if arr.ndim == len(spatial) + 1 and int(arr.shape[0]) == frame_count:
        return arr
    if tuple(int(v) for v in arr.shape) == target:
        return arr.reshape((frame_count,) + spatial)
    if broadcast_spatial and arr.ndim == len(spatial) and tuple(arr.shape) == spatial:
        return np.broadcast_to(arr, (frame_count,) + spatial)
    return arr


def reshape_observables_to_dims(
    observables: Mapping[str, Quantity],
    dims: Mapping[str, int],
    dim_to_observable: Mapping[str, str | None] | None = None,
    *,
    atol: float = 1e-6,
) -> dict[str, Quantity]:
    if not observables:
        return {}
    frame_count = frame_count_from_dims(dims)
    if frame_count <= 0:
        return dict(observables)
    non_spatial = non_spatial_shape_from_dims(dims)
    names = non_spatial_names(dims)
    if not non_spatial:
        return dict(observables)

    dim_map = dim_to_observable or {}
    updated: dict[str, Quantity] = {}

    for key, qty in observables.items():
        arr = np.asarray(qty.values)
        target_dim = None
        for dim_name, obs_name in dim_map.items():
            if obs_name == key:
                target_dim = dim_name
                break
        if arr.ndim == 1 and int(arr.shape[0]) == frame_count and target_dim in names:
            axis = names.index(target_dim)
            try:
                reshaped = arr.reshape(non_spatial)
            except Exception:
                updated[key] = qty
                continue
            other_axes = [idx for idx in range(len(names)) if idx != axis]
            ref = reshaped
            for ax in sorted(other_axes, reverse=True):
                ref = np.take(ref, 0, axis=ax)
            rebuilt = ref
            for ax in sorted(other_axes):
                rebuilt = np.expand_dims(rebuilt, axis=ax)
            rebuilt = np.broadcast_to(rebuilt, reshaped.shape)
            if np.allclose(reshaped, rebuilt, atol=atol, rtol=0.0):
                updated[key] = qty.with_values(ref)
            else:
                updated[key] = qty
        else:
            updated[key] = qty
    return updated


def expand_observables_to_frames(
    observables: Mapping[str, Quantity],
    dims: Mapping[str, int],
    dim_to_observable: Mapping[str, str | None] | None = None,
) -> dict[str, np.ndarray]:
    if not observables:
        return {}
    non_spatial = non_spatial_shape_from_dims(dims)
    spatial = spatial_shape_from_dims(dims)
    names = non_spatial_names(dims)
    frame_count = frame_count_from_dims(dims)
    dim_map = dim_to_observable or {}

    expanded: dict[str, np.ndarray] = {}
    for key, qty in observables.items():
        arr = np.asarray(qty.values)
        if arr.ndim == 0:
            expanded[key] = arr
            continue
        if arr.ndim == 1:
            if arr.shape[0] == frame_count:
                expanded[key] = arr
                continue
            if arr.shape[0] == 1:
                expanded[key] = arr
                continue
            target_dim = None
            for dim_name, obs_name in dim_map.items():
                if obs_name == key and dim_name in names:
                    target_dim = dim_name
                    break
            if target_dim is None:
                matches = [name for name, size in dims.items() if name in names and size == arr.shape[0]]
                if len(matches) == 1:
                    target_dim = matches[0]
            if target_dim is not None and non_spatial:
                axis = names.index(target_dim)
                reshape = (1,) * len(non_spatial)
                reshape = list(reshape)
                reshape[axis] = arr.shape[0]
                broadcast = np.broadcast_to(arr.reshape(tuple(reshape)), non_spatial)
                expanded[key] = broadcast.reshape((frame_count,))
                continue
            expanded[key] = arr
            continue
        if spatial and arr.ndim >= len(spatial) + 1 and arr.shape[-len(spatial):] == spatial and non_spatial:
            target_dim = None
            for dim_name, obs_name in dim_map.items():
                if obs_name == key and dim_name in names:
                    target_dim = dim_name
                    break
            if target_dim is None:
                matches = [name for name, size in dims.items() if name in names and size == arr.shape[0]]
                if len(matches) == 1:
                    target_dim = matches[0]
                elif matches:
                    if "t" in matches:
                        target_dim = "t"
                    elif "frame" in matches:
                        target_dim = "frame"
                    else:
                        target_dim = matches[0]
            if target_dim is not None:
                axis = names.index(target_dim)
                reshape = [1] * len(non_spatial) + list(spatial)
                reshape[axis] = arr.shape[0]
                spatial_size = int(np.prod(spatial))
                expected = int(arr.shape[0]) * spatial_size
                if int(arr.size) == expected:
                    arr_rs = arr.reshape((arr.shape[0],) + spatial).reshape(tuple(reshape))
                    broadcast = np.broadcast_to(arr_rs, non_spatial + spatial)
                    expanded[key] = broadcast.reshape((frame_count,) + spatial)
                    continue
        if non_spatial and arr.shape[: len(non_spatial)] == non_spatial:
            expanded[key] = arr.reshape((frame_count,) + arr.shape[len(non_spatial) :])
            continue
        if spatial and arr.shape == spatial:
            expanded[key] = arr
            continue
        expanded[key] = arr
    return expanded


def pick_observable_for_frames(
    observables: Mapping[str, Quantity],
    frame_count: int | None,
    *,
    preferred_keys: Sequence[str] = ("TI", "TE", "TR", "freq_offsets", "offsets"),
) -> tuple[np.ndarray | None, str | None]:
    """Pick an observable array matching the frame count (useful for spectra plots)."""

    if not observables:
        return None, None

    preferred = list(preferred_keys)
    ordered_keys = preferred + [k for k in observables.keys() if k not in preferred]

    def _match(values: np.ndarray) -> bool:
        if frame_count is None:
            return True
        if values.ndim == 0:
            return True
        return int(values.shape[0]) == int(frame_count)

    for key in ordered_keys:
        values = _observable_values(observables, key)
        if values is None:
            continue
        if _match(values):
            return np.asarray(values), key

    first_key = next(iter(observables.keys()))
    values = _observable_values(observables, first_key)
    return (None if values is None else np.asarray(values)), first_key


__all__ = [
    "DimensionSpec",
    "SPATIAL_DIMS",
    "TEMPORAL_DIM",
    "FRAME_DIM",
    "build_experiment_dims",
    "dim_observable_map",
    "expand_observables_to_frames",
    "frame_count_from_dims",
    "get_default_dim_specs",
    "infer_non_spatial_dims",
    "non_spatial_names",
    "non_spatial_shape_from_dims",
    "pick_observable_for_frames",
    "reshape_dims_to_frames",
    "reshape_frames_to_dims",
    "reshape_observables_to_dims",
    "spatial_names",
    "spatial_shape_from_dims",
]
