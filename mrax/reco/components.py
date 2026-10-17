"""Magnitude/phase accessors for combined and per-coil reconstructions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .phase import unwrap_phase_image

RAW_RECONSTRUCTION_KEY = "raw_reconstruction"
RAW_RECONSTRUCTION_COILS_KEY = "raw_reconstruction_coils"
RAW_RECONSTRUCTION_COMPLEX_KEY = "raw_reconstruction_complex"
RAW_RECONSTRUCTION_FREQUENCY_KEY = "raw_reconstruction_frequency_hz"


def reconstruction_coil_images(processed: Mapping[str, Any] | None) -> np.ndarray | None:
    """Return standardized ``(frame, coil, *spatial)`` complex coil images."""

    value = dict(processed or {}).get(RAW_RECONSTRUCTION_COILS_KEY)
    if value is None:
        return None
    coils = np.asarray(value)
    if coils.ndim < 4:
        raise ValueError(
            "MRax per-coil reconstructions must use (frame, coil, *spatial) layout."
        )
    return coils


def reconstruction_coil_count(processed: Mapping[str, Any] | None) -> int:
    """Return the number of available reconstructed receiver coils."""

    coils = reconstruction_coil_images(processed)
    return 0 if coils is None else int(coils.shape[1])


def reconstruction_variant(
    processed: Mapping[str, Any] | None,
    *,
    component: str = "magnitude",
    coil: int | None = None,
    frame: int | None = None,
    unwrap_phase: bool = False,
    spatial_ndim: int | None = None,
) -> np.ndarray:
    """Return a real magnitude or phase reconstruction.

    ``coil=None`` selects the combined reconstruction. Its magnitude preserves the
    existing MRax/vendor-style combined magnitude. Combined phase is taken from a
    stored complex reconstruction when available, otherwise from the coherent sum
    of the channel-corrected coil images. Phase values are in radians in
    ``[-pi, pi]`` unless ``unwrap_phase=True``. Spatial unwrapping is applied
    independently over leading frame/observable axes.
    """

    payload = dict(processed or {})
    component_key = str(component or "magnitude").strip().lower()
    if component_key not in {"magnitude", "phase"}:
        raise ValueError("Reconstruction component must be 'magnitude' or 'phase'.")

    def select_frame(value: np.ndarray) -> np.ndarray:
        if frame is None:
            return value
        frame_index = int(frame)
        if frame_index < 0 or frame_index >= int(value.shape[0]):
            raise IndexError(
                f"Reconstruction frame {frame_index + 1} is outside the available "
                f"range 1..{int(value.shape[0])}."
            )
        return value[frame_index : frame_index + 1]

    if coil is None and component_key == "magnitude":
        frequency = payload.get(RAW_RECONSTRUCTION_FREQUENCY_KEY)
        if frequency is not None:
            return select_frame(np.asarray(frequency)).astype(np.float32, copy=False)
        combined = payload.get(RAW_RECONSTRUCTION_KEY)
        if combined is not None:
            return select_frame(np.abs(np.asarray(combined))).astype(np.float32, copy=False)

    coils = reconstruction_coil_images(payload)
    if coil is not None:
        if coils is None:
            raise RuntimeError("Per-coil MRax reconstruction images are unavailable.")
        coil_index = int(coil)
        if coil_index < 0 or coil_index >= int(coils.shape[1]):
            raise IndexError(
                f"Reconstruction coil {coil_index + 1} is outside the available "
                f"range 1..{int(coils.shape[1])}."
            )
        complex_image = select_frame(np.asarray(coils[:, coil_index]))
    else:
        complex_value = payload.get(RAW_RECONSTRUCTION_COMPLEX_KEY)
        if complex_value is not None:
            complex_image = select_frame(np.asarray(complex_value))
        elif coils is not None:
            complex_image = select_frame(np.sum(coils, axis=1))
        else:
            combined = payload.get(RAW_RECONSTRUCTION_KEY)
            if combined is None:
                raise RuntimeError("MRax reconstruction images are unavailable.")
            complex_image = select_frame(np.asarray(combined))
            if not np.iscomplexobj(complex_image) and component_key == "phase":
                raise RuntimeError(
                    "Combined reconstruction phase is unavailable because this stored "
                    "payload contains magnitude data only."
                )

    if component_key == "phase":
        phase = np.angle(complex_image).astype(np.float32, copy=False)
        if unwrap_phase:
            phase_weights = np.abs(complex_image).astype(np.float32, copy=False)
            if coil is None and payload.get(RAW_RECONSTRUCTION_KEY) is not None:
                phase_weights = select_frame(
                    np.abs(np.asarray(payload[RAW_RECONSTRUCTION_KEY]))
                ).astype(np.float32, copy=False)
            return unwrap_phase_image(
                phase,
                spatial_ndim=spatial_ndim,
                weights=phase_weights,
            )
        return phase
    return np.abs(complex_image).astype(np.float32, copy=False)


def available_reconstruction_variants(
    processed: Mapping[str, Any] | None,
    *,
    spatial_ndim: int | None = None,
    unwrap_phase: bool = True,
    include_wrapped_phase: bool = True,
) -> list[tuple[str, str, int | None, np.ndarray]]:
    """Return exportable ``(name, component, coil, array)`` reconstruction variants."""

    variants: list[tuple[str, str, int | None, np.ndarray]] = []
    try:
        variants.append(
            (
                "combined_magnitude",
                "magnitude",
                None,
                reconstruction_variant(processed, component="magnitude"),
            )
        )
    except (RuntimeError, ValueError):
        return variants
    try:
        variants.append(
            (
                "combined_phase",
                "phase",
                None,
                reconstruction_variant(
                    processed,
                    component="phase",
                    unwrap_phase=unwrap_phase,
                    spatial_ndim=spatial_ndim,
                ),
            )
        )
    except RuntimeError:
        pass
    if unwrap_phase and include_wrapped_phase:
        try:
            variants.append(
                (
                    "combined_phase_wrapped",
                    "phase",
                    None,
                    reconstruction_variant(processed, component="phase"),
                )
            )
        except RuntimeError:
            pass
    for coil_index in range(reconstruction_coil_count(processed)):
        for component in ("magnitude", "phase"):
            variants.append(
                (
                    f"coil_{coil_index + 1:03d}_{component}",
                    component,
                    coil_index,
                    reconstruction_variant(
                        processed,
                        component=component,
                        coil=coil_index,
                        unwrap_phase=bool(unwrap_phase and component == "phase"),
                        spatial_ndim=spatial_ndim,
                    ),
                )
            )
        if unwrap_phase and include_wrapped_phase:
            variants.append(
                (
                    f"coil_{coil_index + 1:03d}_phase_wrapped",
                    "phase",
                    coil_index,
                    reconstruction_variant(
                        processed,
                        component="phase",
                        coil=coil_index,
                    ),
                )
            )
    return variants


__all__ = [
    "RAW_RECONSTRUCTION_COILS_KEY",
    "RAW_RECONSTRUCTION_COMPLEX_KEY",
    "RAW_RECONSTRUCTION_KEY",
    "available_reconstruction_variants",
    "reconstruction_coil_count",
    "reconstruction_coil_images",
    "reconstruction_variant",
]
