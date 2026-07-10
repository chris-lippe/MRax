from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from mrax.core.loss_builder import _extract_spatial_metadata
from mrax.data_struct.experiment import Experiment
from mrax.data_struct.quantity import Quantity, require_quantity_dict

from .variational import (
    Array,
    _align_kspace_global_phase_by_frame,
    _coerce_reference_image,
    _ensure_coil_axis,
    _infer_sampling_mask_cartesian,
    build_operator,
)


def _infer_reference_image_for_z(
    *,
    spatial_shape: tuple[int, int],
    sensitivities: Array | np.ndarray | None,
    trajectory: Array | np.ndarray | None,
    density: Array | np.ndarray | None,
    nufft_eps: float,
    reference_image: Array | np.ndarray | Quantity | None,
    reference_kspace: Array | np.ndarray | None,
    reference_mask: Array | np.ndarray | None,
    initial_reference_image: Array | np.ndarray | None,
) -> Array:
    """Infer a reference magnitude image used to normalize AB-MT Z-spectra."""

    if reference_kspace is None:
        if reference_image is None:
            raise ValueError("reference_image is required when reference_kspace is not provided")
        ref_arr = reference_image.values if isinstance(reference_image, Quantity) else reference_image
        ref = _coerce_reference_image(ref_arr, spatial_shape)
        if tuple(int(v) for v in ref.shape) != spatial_shape:
            raise ValueError(
                f"reference_image shape {tuple(int(v) for v in ref.shape)} does not match spatial_shape {spatial_shape}"
            )
        return ref

    if initial_reference_image is not None:
        ref0 = _coerce_reference_image(initial_reference_image, spatial_shape)
        if tuple(int(v) for v in ref0.shape) != spatial_shape:
            raise ValueError(
                f"initial_reference_image shape {tuple(int(v) for v in ref0.shape)} does not match spatial_shape {spatial_shape}"
            )
        return ref0

    if reference_image is not None:
        ref_arr = reference_image.values if isinstance(reference_image, Quantity) else reference_image
        ref0 = _coerce_reference_image(ref_arr, spatial_shape)
        if tuple(int(v) for v in ref0.shape) != spatial_shape:
            raise ValueError(
                f"reference_image shape {tuple(int(v) for v in ref0.shape)} does not match spatial_shape {spatial_shape}"
            )
        return ref0

    y_ref = jnp.asarray(reference_kspace)
    if y_ref.ndim == 4 and int(y_ref.shape[0]) == 1:
        y_ref = y_ref[0]
    if y_ref.ndim not in (2, 3):
        raise ValueError("reference_kspace must have shape (Ny, Nx), (coils, Ny, Nx), or (1, coils, Ny, Nx)")
    if tuple(int(v) for v in y_ref.shape[-2:]) != spatial_shape:
        raise ValueError(
            f"reference_kspace spatial shape {tuple(int(v) for v in y_ref.shape[-2:])} does not match {spatial_shape}"
        )

    mask_ref = reference_mask
    if trajectory is None and mask_ref is None:
        mask_ref = _infer_sampling_mask_cartesian(y_ref)

    op_ref = build_operator(
        spatial_shape=spatial_shape,
        sensitivities=sensitivities,
        mask=mask_ref,
        trajectory=trajectory,
        density=density,
        nufft_eps=nufft_eps,
    )

    coil_count_ref = op_ref.coil_count
    y_ref_has_coils = y_ref.ndim >= 3 and int(y_ref.shape[-3]) == coil_count_ref
    y_ref_coils, _ = _ensure_coil_axis(y_ref, has_coils=y_ref_has_coils, spatial_ndim=2, coil_axis=-3)
    ref0 = jnp.abs(op_ref.adjoint(y_ref_coils))
    if ref0.ndim == 3 and int(ref0.shape[0]) == 1:
        ref0 = ref0[0]
    return ref0


def infer_ab_mt_observables_from_kspace(
    kspace: Array | np.ndarray,
    *,
    experiment: Experiment,
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    nufft_eps: float = 1e-6,
    reference_image: Array | np.ndarray | Quantity | None = None,
    reference_kspace: Array | np.ndarray | None = None,
    reference_mask: Array | np.ndarray | None = None,
    initial_reference_image: Array | np.ndarray | None = None,
    frame_phase_correction: str | None = "none",
) -> dict[str, Quantity]:
    """Infer AB-MT observables (``z_prev`` + ``z_reference_image``) from raw k-space.

    ``z_prev`` is computed from a fast adjoint reconstruction of the same k-space used in the
    data term: ``z = |A^H y| / |S0|`` and ``z_prev = roll(z, 1, axis=frame)``.

    Returns a dict suitable for merging into ``Experiment.observables``.
    """

    full_spatial_shape, _ = _extract_spatial_metadata(experiment)
    spatial_shape = tuple(int(v) for v in full_spatial_shape[:2])
    if len(spatial_shape) != 2:
        raise ValueError("infer_ab_mt_observables_from_kspace currently supports 2D spatial_shape only")

    y = jnp.asarray(kspace)
    if trajectory is None and mask is None:
        mask = _infer_sampling_mask_cartesian(y)

    op = build_operator(
        spatial_shape=spatial_shape,
        sensitivities=sensitivities,
        mask=mask,
        trajectory=trajectory,
        density=density,
        nufft_eps=nufft_eps,
    )

    if sensitivities is None and y.ndim >= 4 and int(y.shape[-3]) > 1:
        raise ValueError("Multi-coil k-space requires sensitivities to infer z_prev")

    coil_count = op.coil_count
    y_has_coils = y.ndim >= 3 and int(y.shape[-3]) == coil_count
    y_coils, _ = _ensure_coil_axis(y, has_coils=y_has_coils, spatial_ndim=2, coil_axis=-3)
    if not op.is_non_cartesian:
        y_coils = op._apply_mask(y_coils)

    expected_frames = experiment.frame_count if hasattr(experiment, "frame_count") else int(
        experiment.dims.get("frame", int(y_coils.shape[0]))
    )
    y_coils = _align_kspace_global_phase_by_frame(
        y_coils,
        expected_frames=int(expected_frames),
        mode=str(frame_phase_correction or "none"),
    )

    ref0 = _infer_reference_image_for_z(
        spatial_shape=spatial_shape,
        sensitivities=sensitivities,
        trajectory=trajectory,
        density=density,
        nufft_eps=nufft_eps,
        reference_image=reference_image,
        reference_kspace=reference_kspace,
        reference_mask=reference_mask,
        initial_reference_image=initial_reference_image,
    )
    ref_base = jnp.abs(ref0) + 1e-6

    img_est = jnp.abs(op.adjoint(y_coils))
    if img_est.ndim == 2:
        img_est = img_est.reshape((1,) + img_est.shape)
    z_est = img_est / ref_base.reshape((1,) + ref_base.shape)
    z_prev = jnp.roll(z_est, shift=1, axis=0)
    return {
        "z_prev": Quantity(z_prev, "1"),
        "z_reference_image": Quantity(ref_base, "1"),
    }


def prepare_ab_mt_experiment(
    kspace: Array | np.ndarray,
    *,
    experiment: Experiment,
    z_prev_source: Literal["auto", "kspace", "none"] = "auto",
    sensitivities: Array | np.ndarray | None = None,
    mask: Array | np.ndarray | None = None,
    trajectory: Array | np.ndarray | None = None,
    density: Array | np.ndarray | None = None,
    nufft_eps: float = 1e-6,
    reference_image: Array | np.ndarray | Quantity | None = None,
    reference_kspace: Array | np.ndarray | None = None,
    reference_mask: Array | np.ndarray | None = None,
    initial_reference_image: Array | np.ndarray | None = None,
    frame_phase_correction: str | None = "none",
) -> Experiment:
    """Return an experiment with AB-MT observables inserted as needed.

    - ``auto``: keep existing ``z_prev`` if available, otherwise infer from k-space.
    - ``kspace``: always infer ``z_prev`` + ``z_reference_image`` from k-space.
    - ``none``: require ``z_prev`` to already be present in ``experiment.observables``.
    """

    mode = str(z_prev_source or "auto").lower()
    if mode not in ("auto", "kspace", "none"):
        raise ValueError("z_prev_source must be one of: auto, kspace, none")

    base_obs = require_quantity_dict(experiment.observables, "Observable")
    has_prev = ("z_prev" in base_obs) or ("Z_prev" in base_obs)

    if mode == "none" and not has_prev:
        raise ValueError("z_prev_source='none' requires experiment.observables['z_prev']")

    merged_obs = dict(base_obs)
    updated = False
    if "z_prev" not in merged_obs and "Z_prev" in merged_obs:
        merged_obs["z_prev"] = merged_obs["Z_prev"]
        updated = True

    inject_prev = mode == "kspace" or (mode == "auto" and not has_prev)
    if inject_prev:
        merged_obs.update(
            infer_ab_mt_observables_from_kspace(
                kspace,
                experiment=experiment,
                sensitivities=sensitivities,
                mask=mask,
                trajectory=trajectory,
                density=density,
                nufft_eps=nufft_eps,
                reference_image=reference_image,
                reference_kspace=reference_kspace,
                reference_mask=reference_mask,
                initial_reference_image=initial_reference_image,
                frame_phase_correction=frame_phase_correction,
            )
        )
        updated = True

    if not updated:
        return experiment

    exp2 = Experiment(dims=dict(experiment.dims), observables=merged_obs, exp_type=experiment.exp_type)
    for attr in ("study_globals", "_mrax_dim_observables"):
        if hasattr(experiment, attr):
            setattr(exp2, attr, getattr(experiment, attr))
    return exp2


__all__ = [
    "infer_ab_mt_observables_from_kspace",
    "prepare_ab_mt_experiment",
]
