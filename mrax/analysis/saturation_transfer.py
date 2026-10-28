from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import DataSet, Experiment, Quantity, Study
from mrax.data_struct.dimensions import (
    non_spatial_names,
    reshape_dims_to_frames,
    reshape_frames_to_dims,
    spatial_shape_from_dims,
)


@dataclass(frozen=True)
class ZSpectraResult:
    """Reference-normalized saturation-transfer data computed on demand."""

    values: jnp.ndarray
    reference_image: jnp.ndarray
    reference_name: str | None


def compute_z_spectra(
    data: np.ndarray,
    reference_image: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """Return ``data / reference_image`` as a derived array."""

    arr = np.asarray(data, dtype=np.float32)
    ref = np.asarray(reference_image, dtype=np.float32)
    ref_safe = np.where(np.abs(ref) < float(epsilon), float(epsilon), ref)
    return jnp.asarray(arr / ref_safe.reshape((1,) + tuple(ref_safe.shape)))


def _quantity_scalar(
    observables: dict[str, Quantity] | None,
    key: str,
    *,
    reducer: str = "first",
    fallback: float,
) -> float:
    if not observables or key not in observables:
        return fallback
    arr = np.asarray(observables[key].values)
    if arr.size == 0:
        return fallback
    if reducer == "max":
        return float(np.nanmax(arr))
    if reducer == "min":
        return float(np.nanmin(arr))
    return float(arr.ravel()[0])


def _resolve_study_index(study: Study, target: str | int) -> int:
    if isinstance(target, int):
        if target < 0 or target >= len(study.experiments):
            raise IndexError("Study target index out of range")
        return int(target)
    names = list(study.names or [])
    if target not in names:
        raise KeyError(f"Study does not contain a scan named {target!r}")
    return names.index(target)


def choose_z_reference_scan(study: Study, target: str | int) -> int | None:
    """Pick the single-frame saturation-transfer reference scan for ``target``."""

    target_idx = _resolve_study_index(study, target)
    target_exp = study.experiments[target_idx]
    if target_exp.exp_type != "sat_trans":
        return None

    target_shape = spatial_shape_from_dims(target_exp.dims)
    candidates: list[tuple[float, float, int]] = []
    for idx, exp in enumerate(study.experiments):
        if idx == target_idx or exp.exp_type != "sat_trans" or int(exp.frame_count) != 1:
            continue
        if spatial_shape_from_dims(exp.dims) != target_shape:
            continue
        amp_val = _quantity_scalar(exp.observables, "sat_b1", reducer="first", fallback=1e9)
        tr_val = _quantity_scalar(exp.observables, "TR", reducer="max", fallback=0.0)
        candidates.append((amp_val, -tr_val, idx))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def get_z_spectra(
    study: Study,
    target: str | int,
    *,
    reference: str | int | None = None,
    epsilon: float = 1e-6,
) -> ZSpectraResult:
    """Compute reference-normalized spectra for a saturation-transfer study entry."""

    target_idx = _resolve_study_index(study, target)
    ref_idx = choose_z_reference_scan(study, target_idx) if reference is None else _resolve_study_index(study, reference)
    if ref_idx is None:
        raise ValueError("No matching single-frame saturation-transfer reference scan was found")

    target_exp = study.experiments[target_idx]
    ref_exp = study.experiments[ref_idx]
    target_data = reshape_dims_to_frames(study.datasets[target_idx].data, target_exp.dims)
    ref_data = reshape_dims_to_frames(study.datasets[ref_idx].data, ref_exp.dims)
    if ref_data.shape[0] != 1:
        raise ValueError("Saturation-transfer reference scan must contain exactly one frame")

    z = compute_z_spectra(target_data, ref_data[0], epsilon=epsilon)
    z = reshape_frames_to_dims(z, target_exp.dims)
    return ZSpectraResult(
        values=jnp.asarray(z),
        reference_image=jnp.asarray(ref_data[0]),
        reference_name=(study.names[ref_idx] if study.names else None),
    )


def dataset_with_z_spectra(
    study: Study,
    target: str | int,
    *,
    reference: str | int | None = None,
    epsilon: float = 1e-6,
    replace_data: bool = False,
) -> DataSet:
    """Return a copy of a dataset carrying lazily computed Z-spectra in ``processed``."""

    target_idx = _resolve_study_index(study, target)
    result = get_z_spectra(study, target_idx, reference=reference, epsilon=epsilon)
    source = study.datasets[target_idx]
    processed = dict(source.processed or {})
    processed["z_spectra"] = result.values
    if result.reference_name is not None:
        processed["reference_scan_no"] = result.reference_name
    processed["z_reference_image"] = result.reference_image
    return DataSet(
        data=result.values if replace_data else source.data,
        name=source.name,
        noise_level=source.noise_level,
        processed=processed,
        kspace=source.kspace,
        trajectory=source.trajectory,
        density=source.density,
        sampling_mask=source.sampling_mask,
        kspace_metadata=source.kspace_metadata,
    )


def experiment_with_ab_mt_z_prev(
    experiment: Experiment,
    z_spectra: np.ndarray,
    reference_image: np.ndarray | None = None,
) -> Experiment:
    """Return ``experiment`` with AB-MT runtime observables derived from Z-spectra."""

    dim_names = non_spatial_names(experiment.dims)
    axis = dim_names.index("offset") if "offset" in dim_names else (0 if dim_names else None)
    z_values = jnp.asarray(z_spectra)
    z_prev = jnp.roll(z_values, shift=1, axis=axis) if axis is not None else z_values
    observables: dict[str, Quantity] = {
        "z_prev": Quantity(z_prev, "1"),
    }
    if reference_image is not None:
        observables["z_reference_image"] = Quantity(jnp.asarray(reference_image), "1")
    return experiment.insert_observables(observables)


def prepare_ab_mt_z_spectra(
    study: Study,
    target: str | int,
    *,
    reference: str | int | None = None,
    epsilon: float = 1e-6,
) -> tuple[Experiment, DataSet, ZSpectraResult]:
    """Prepare derived image-domain AB-MT inputs from imported study data."""

    target_idx = _resolve_study_index(study, target)
    result = get_z_spectra(study, target_idx, reference=reference, epsilon=epsilon)
    source = study.datasets[target_idx]
    processed = dict(source.processed or {})
    processed["z_spectra"] = result.values
    processed["z_reference_image"] = result.reference_image
    if result.reference_name is not None:
        processed["reference_scan_no"] = result.reference_name
    dataset = DataSet(
        data=result.values,
        name=source.name,
        noise_level=source.noise_level,
        processed=processed,
        kspace=source.kspace,
        trajectory=source.trajectory,
        density=source.density,
        sampling_mask=source.sampling_mask,
        kspace_metadata=source.kspace_metadata,
    )
    experiment = experiment_with_ab_mt_z_prev(
        study.experiments[target_idx],
        result.values,
        reference_image=result.reference_image,
    )
    return experiment, dataset, result


__all__ = [
    "ZSpectraResult",
    "choose_z_reference_scan",
    "compute_z_spectra",
    "dataset_with_z_spectra",
    "experiment_with_ab_mt_z_prev",
    "get_z_spectra",
    "prepare_ab_mt_z_spectra",
]
