from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from mrax.data_struct.quantity import Quantity


@dataclass
class DTIMaps:
    """Derived diffusion tensor maps from fitted DTI parameters (eigenvalues sorted descending)."""

    tensor: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    adc: np.ndarray
    fa: np.ndarray
    md: np.ndarray
    ad: np.ndarray
    rd: np.ndarray
    trace: np.ndarray


def _as_array(val: object) -> np.ndarray:
    if isinstance(val, Quantity):
        return val.values
    return val


def _array_module(*values: object):
    for value in values:
        if isinstance(value, jnp.ndarray):
            return jnp
    return np


def tensor_from_fields(fields: Mapping[str, object]) -> np.ndarray:
    """Assemble a symmetric diffusion tensor from fitted parameter fields."""
    dxx = _as_array(fields["dxx"])
    dyy = _as_array(fields["dyy"])
    dzz = _as_array(fields["dzz"])
    dxy = _as_array(fields["dxy"])
    dxz = _as_array(fields["dxz"])
    dyz = _as_array(fields["dyz"])
    xp = _array_module(dxx, dyy, dzz, dxy, dxz, dyz)
    dxx, dyy, dzz, dxy, dxz, dyz = (
        xp.asarray(dxx),
        xp.asarray(dyy),
        xp.asarray(dzz),
        xp.asarray(dxy),
        xp.asarray(dxz),
        xp.asarray(dyz),
    )
    row0 = xp.stack([dxx, dxy, dxz], axis=-1)
    row1 = xp.stack([dxy, dyy, dyz], axis=-1)
    row2 = xp.stack([dxz, dyz, dzz], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


def dti_eigendecomposition(fields: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of the diffusion tensor."""
    tensor = tensor_from_fields(fields)
    xp = _array_module(tensor)
    evals, evecs = xp.linalg.eigh(tensor)
    return evals, evecs


def _sort_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xp = _array_module(eigenvalues, eigenvectors)
    order = xp.argsort(eigenvalues, axis=-1)[..., ::-1]
    evals_sorted = xp.take_along_axis(eigenvalues, order, axis=-1)
    evecs_sorted = xp.take_along_axis(eigenvectors, order[..., None, :], axis=-1)
    return evals_sorted, evecs_sorted


def dti_adc_fa(
    eigenvalues: np.ndarray,
    *,
    fa_nan_on_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ADC (mean diffusivity) and FA from tensor eigenvalues."""
    xp = _array_module(eigenvalues)
    md = xp.mean(eigenvalues, axis=-1)
    numerator = 1.5 * xp.sum((eigenvalues - md[..., None]) ** 2, axis=-1)
    denom = xp.sum(eigenvalues**2, axis=-1)
    fa = xp.sqrt(numerator / denom)
    fa = xp.where(xp.isfinite(fa), fa, 0.0)
    if fa_nan_on_negative:
        fa = xp.where(xp.any(eigenvalues < 0, axis=-1), xp.nan, fa)
    return md, fa


def dti_metrics_from_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    fa_nan_on_negative: bool = True,
) -> dict[str, np.ndarray]:
    """Compute common DTI scalar metrics from tensor eigenvalues."""
    xp = _array_module(eigenvalues)
    evals_sorted = xp.sort(eigenvalues, axis=-1)[..., ::-1]
    md, fa = dti_adc_fa(evals_sorted, fa_nan_on_negative=fa_nan_on_negative)
    ad = evals_sorted[..., 0]
    rd = xp.mean(evals_sorted[..., 1:], axis=-1)
    trace = xp.sum(evals_sorted, axis=-1)
    return {"adc": md, "md": md, "fa": fa, "ad": ad, "rd": rd, "trace": trace}


def compute_dti_maps(
    fields: Mapping[str, object],
    *,
    mask: np.ndarray | None = None,
) -> DTIMaps:
    """Compute tensor, eigenvalues/vectors, ADC, and FA from fitted fields."""
    tensor = tensor_from_fields(fields)
    xp = _array_module(tensor)
    # Enforce symmetry and stabilize numeric issues.
    tensor = 0.5 * (tensor + xp.swapaxes(tensor, -1, -2))

    if xp is np:
        spatial_shape = tensor.shape[:-2]
        flat = tensor.reshape((-1, 3, 3))
        evals = np.full((flat.shape[0], 3), np.nan, dtype=float)
        evecs = np.full((flat.shape[0], 3, 3), np.nan, dtype=float)

        finite = np.isfinite(flat).all(axis=(1, 2))
        if np.any(finite):
            try:
                evals_ok, evecs_ok = np.linalg.eigh(flat[finite])
                evals[finite] = evals_ok
                evecs[finite] = evecs_ok
            except np.linalg.LinAlgError:
                # Fall back to per-voxel decomposition on failure.
                for idx, mat in enumerate(flat):
                    if not finite[idx]:
                        continue
                    try:
                        w, v = np.linalg.eigh(mat)
                        evals[idx] = w
                        evecs[idx] = v
                    except np.linalg.LinAlgError:
                        evals[idx] = np.nan
                        evecs[idx] = np.nan

        evals = evals.reshape(spatial_shape + (3,))
        evecs = evecs.reshape(spatial_shape + (3, 3))
        evals, evecs = _sort_eigenpairs(evals, evecs)
        metrics = dti_metrics_from_eigenvalues(evals)
        adc = metrics["adc"]
        fa = metrics["fa"]
        md = metrics["md"]
        ad = metrics["ad"]
        rd = metrics["rd"]
        trace = metrics["trace"]
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            adc = np.where(mask, adc, np.nan)
            fa = np.where(mask, fa, np.nan)
            md = np.where(mask, md, np.nan)
            ad = np.where(mask, ad, np.nan)
            rd = np.where(mask, rd, np.nan)
            trace = np.where(mask, trace, np.nan)
    else:
        evals, evecs = jnp.linalg.eigh(tensor)
        evals, evecs = _sort_eigenpairs(evals, evecs)
        metrics = dti_metrics_from_eigenvalues(evals)
        adc = metrics["adc"]
        fa = metrics["fa"]
        md = metrics["md"]
        ad = metrics["ad"]
        rd = metrics["rd"]
        trace = metrics["trace"]
        if mask is not None:
            mask_arr = jnp.asarray(mask, dtype=bool)
            adc = jnp.where(mask_arr, adc, jnp.nan)
            fa = jnp.where(mask_arr, fa, jnp.nan)
            md = jnp.where(mask_arr, md, jnp.nan)
            ad = jnp.where(mask_arr, ad, jnp.nan)
            rd = jnp.where(mask_arr, rd, jnp.nan)
            trace = jnp.where(mask_arr, trace, jnp.nan)
    return DTIMaps(
        tensor=tensor,
        eigenvalues=evals,
        eigenvectors=evecs,
        adc=adc,
        fa=fa,
        md=md,
        ad=ad,
        rd=rd,
        trace=trace,
    )


def fit_dti_log_linear(
    data: object,
    bvals: object,
    bvecs: object | None = None,
    *,
    mask: np.ndarray | None = None,
    min_signal: float = 1e-6,
    b0_threshold: float = 50.0,
) -> dict[str, np.ndarray]:
    """Fit DTI fields with a voxelwise log-linear least-squares estimate.

    When the b-vector design is rank-deficient, an isotropic ADC estimate from
    b-values is returned instead of a poorly constrained tensor.
    """

    arr = np.asarray(data, dtype=float)
    if arr.ndim < 2:
        raise ValueError("DTI data must be shaped (frame, *spatial)")
    b = np.asarray(bvals.values if isinstance(bvals, Quantity) else bvals, dtype=float).reshape((-1,))
    if int(arr.shape[0]) != int(b.size):
        raise ValueError(f"DTI data has {arr.shape[0]} frames but bvals has {b.size} values")
    spatial_shape = tuple(int(v) for v in arr.shape[1:])
    voxel_count = int(np.prod(spatial_shape))
    flat = arr.reshape((int(b.size), voxel_count))
    valid = np.isfinite(flat) & (flat > float(min_signal)) & np.isfinite(b)[:, None]
    safe = np.clip(flat, float(min_signal), None)
    log_signal = np.log(safe)

    design = dti_design_matrix(b, bvecs, b0_threshold=b0_threshold)
    use_tensor = design is not None and dti_design_rank(b, bvecs, b0_threshold=b0_threshold) >= 7

    if use_tensor and design is not None:
        beta = _masked_linear_solve(design, log_signal, valid)
        ln_s0 = beta[0]
        fields = {
            "s0": np.exp(ln_s0).reshape(spatial_shape),
            "dxx": beta[1].reshape(spatial_shape),
            "dyy": beta[2].reshape(spatial_shape),
            "dzz": beta[3].reshape(spatial_shape),
            "dxy": beta[4].reshape(spatial_shape),
            "dxz": beta[5].reshape(spatial_shape),
            "dyz": beta[6].reshape(spatial_shape),
        }
    else:
        fields = _fit_isotropic_adc(log_signal, b, valid, spatial_shape, b0_threshold=b0_threshold)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != spatial_shape:
            raise ValueError(f"DTI mask shape {mask_arr.shape} does not match spatial shape {spatial_shape}")
        for key, value in list(fields.items()):
            fill = 0.0 if key == "s0" else np.nan
            fields[key] = np.where(mask_arr, value, fill)

    metrics = compute_dti_maps(fields, mask=mask)
    fields.update(
        {
            "adc": np.asarray(metrics.adc),
            "md": np.asarray(metrics.md),
            "fa": np.asarray(metrics.fa),
            "ad": np.asarray(metrics.ad),
            "rd": np.asarray(metrics.rd),
            "trace": np.asarray(metrics.trace),
        }
    )
    return fields


def dti_design_matrix(
    bvals: object,
    bvecs: object | None,
    *,
    b0_threshold: float = 50.0,
) -> np.ndarray | None:
    """Return the log-linear DTI design matrix, or ``None`` for unusable b-vectors."""

    b = np.asarray(bvals.values if isinstance(bvals, Quantity) else bvals, dtype=float).reshape((-1,))
    if bvecs is None:
        return None
    g = np.asarray(bvecs.values if isinstance(bvecs, Quantity) else bvecs, dtype=float)
    if g.ndim == 1:
        if g.size % 3 != 0:
            return None
        g = g.reshape((-1, 3))
    if g.ndim != 2 or int(g.shape[0]) != int(b.size) or int(g.shape[1]) != 3:
        return None
    g = _normalize_bvecs_np(g, b, b0_threshold=b0_threshold)
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    return np.column_stack(
        [
            np.ones_like(b),
            -b * gx * gx,
            -b * gy * gy,
            -b * gz * gz,
            -2.0 * b * gx * gy,
            -2.0 * b * gx * gz,
            -2.0 * b * gy * gz,
        ]
    )


def dti_design_rank(
    bvals: object,
    bvecs: object | None,
    *,
    b0_threshold: float = 50.0,
) -> int:
    """Return the rank of the full DTI design including the S0 intercept."""

    b = np.asarray(bvals.values if isinstance(bvals, Quantity) else bvals, dtype=float).reshape((-1,))
    design = dti_design_matrix(b, bvecs, b0_threshold=b0_threshold)
    if design is None:
        return 0
    rows = np.isfinite(b)
    if not np.any(rows):
        return 0
    return int(np.linalg.matrix_rank(design[rows]))


def _masked_linear_solve(design: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    params = design.shape[1]
    out = np.full((params, y.shape[1]), np.nan, dtype=float)
    if y.shape[1] and np.all(valid == valid[:, [0]]):
        row_mask = valid[:, 0]
        if int(np.count_nonzero(row_mask)) >= params:
            a = design[row_mask]
            if np.linalg.matrix_rank(a) >= params:
                rhs = y[row_mask]
                try:
                    return np.linalg.lstsq(a, rhs, rcond=None)[0]
                except np.linalg.LinAlgError:
                    return np.linalg.pinv(a) @ rhs
    for idx in range(y.shape[1]):
        row_mask = valid[:, idx]
        if int(np.count_nonzero(row_mask)) < params:
            continue
        a = design[row_mask]
        if np.linalg.matrix_rank(a) < params:
            continue
        rhs = y[row_mask, idx]
        try:
            out[:, idx] = np.linalg.lstsq(a, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            out[:, idx] = np.linalg.pinv(a) @ rhs
    return out


def _fit_isotropic_adc(
    log_signal: np.ndarray,
    bvals: np.ndarray,
    valid: np.ndarray,
    spatial_shape: Sequence[int],
    *,
    b0_threshold: float,
) -> dict[str, np.ndarray]:
    design = np.column_stack([np.ones_like(bvals), -bvals])
    beta = _masked_linear_solve(design, log_signal, valid)
    ln_s0 = beta[0]
    adc = beta[1]

    b0 = bvals < float(b0_threshold)
    if np.any(b0):
        b0_valid = valid[b0]
        b0_logs = np.where(b0_valid, log_signal[b0], np.nan)
        ln_s0_b0 = np.nanmean(b0_logs, axis=0)
        ln_s0 = np.where(np.isfinite(ln_s0_b0), ln_s0_b0, ln_s0)

    adc = np.where(np.isfinite(adc), adc, 0.0)
    adc = np.clip(adc, -0.01, 0.01)
    shape = tuple(int(v) for v in spatial_shape)
    zeros = np.zeros(shape, dtype=float)
    adc_map = adc.reshape(shape)
    return {
        "s0": np.exp(ln_s0).reshape(shape),
        "dxx": adc_map,
        "dyy": adc_map,
        "dzz": adc_map,
        "dxy": zeros,
        "dxz": zeros,
        "dyz": zeros,
    }


def _normalize_bvecs_np(bvecs: np.ndarray, bvals: np.ndarray, *, b0_threshold: float) -> np.ndarray:
    out = np.asarray(bvecs, dtype=float).copy()
    norms = np.linalg.norm(out, axis=1)
    nonzero = norms > 0
    out[nonzero] = out[nonzero] / norms[nonzero, None]
    out[np.asarray(bvals) < float(b0_threshold)] = 0.0
    return out


__all__ = [
    "DTIMaps",
    "compute_dti_maps",
    "dti_design_matrix",
    "dti_design_rank",
    "dti_adc_fa",
    "dti_eigendecomposition",
    "dti_metrics_from_eigenvalues",
    "fit_dti_log_linear",
    "tensor_from_fields",
]
