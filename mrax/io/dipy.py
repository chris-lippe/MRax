from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _import_dipy():
    try:
        import dipy  # type: ignore
    except Exception as exc:  # pragma: no cover - invalid runtime environment
        raise ImportError(
            "dipy is required for DIPY I/O helpers. Reinstall MRax with `pip install mrax`."
        ) from exc
    return dipy


def load_dipy_nifti(path: str | Path, *, dtype: np.dtype | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file using DIPY's I/O helpers."""

    _import_dipy()
    from dipy.io.image import load_nifti  # type: ignore

    data, affine = load_nifti(str(path))
    if dtype is not None:
        data = data.astype(dtype)
    return data, affine


def save_dipy_nifti(
    path: str | Path,
    data: np.ndarray,
    affine: np.ndarray,
    *,
    dtype: np.dtype | None = None,
) -> None:
    """Save a NIfTI file using DIPY's I/O helpers."""

    _import_dipy()
    from dipy.io.image import save_nifti  # type: ignore

    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    save_nifti(str(path), arr, affine)


def load_dipy_gradients(
    bvals_path: str | Path,
    bvecs_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, "object"]:
    """Load diffusion gradients with DIPY and return (bvals, bvecs, gtab)."""

    _import_dipy()
    from dipy.core.gradients import gradient_table  # type: ignore
    from dipy.io.gradients import read_bvals_bvecs  # type: ignore

    bvals, bvecs = read_bvals_bvecs(str(bvals_path), str(bvecs_path))
    gtab = gradient_table(bvals, bvecs)
    return bvals, bvecs, gtab
