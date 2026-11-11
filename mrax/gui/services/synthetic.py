"""Synthetic quantitative phantom sessions."""

from __future__ import annotations

import numpy as np

from mrax.data_struct import Quantity
from mrax.gui.session import GuiSession

SYNTHETIC_T2_MS = 80.0


def synthetic_t2_session() -> GuiSession:
    """Create an anisotropic 3D phantom with eight echoes and known T2."""
    echo_times = np.array([10, 20, 40, 60, 80, 120, 160, 200], dtype=np.float32)
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 24),
        np.linspace(-1, 1, 32),
        np.linspace(-1, 1, 3),
        indexing="ij",
    )
    amplitude = 1e6 * (0.6 + 0.4 * np.exp(-3 * (x * x + y * y + 0.2 * z * z)))
    data = amplitude[None, ...] * np.exp(-echo_times[:, None, None, None] / SYNTHETIC_T2_MS)
    session = GuiSession()
    session.create_empty_study("Synthetic T2 phantom", field_strength_mhz=100)
    session.add_experiment(
        "T2 phantom",
        data.astype(np.float32),
        dims={"frame": 8, "x": 24, "y": 32, "z": 3},
        exp_type="t2_se",
        model_name="t2_se",
        observables={"TE": Quantity(echo_times, "ms")},
        extra_metadata={
            "spatial_geometry": {
                "version": 1,
                "axis_convention": "native_v1",
                "affine_convention": "nifti_xyz_ras",
                "status": "unknown",
                "source": "synthetic",
                "spatial_shape": [24, 32, 3],
                "voxel_sizes": [0.7, 0.9, 2.0],
            }
        },
    )
    return session


def synthetic_tutorial_session() -> GuiSession:
    """T2 workflow practice with an independent, prepared Cartesian acquisition."""
    from mrax.data_struct import AcquisitionData
    from mrax.reco import fftnc

    session = synthetic_t2_session()
    session.rename_study("Tutorial: synthetic T2")
    image = np.asarray(session.image_array, dtype=np.complex64)
    samples = np.asarray(fftnc(image[:, None], axes=(-3, -2, -1)))
    acquisition = AcquisitionData(samples, tuple(image.shape[1:]), ("x", "y", "z"))
    acquisition.require_prepared()
    session.current_experiment.update(raw_kspace=samples, acquisition=acquisition)
    session.current_experiment["extra_metadata"]["synthetic_reference"] = {
        "T2_ms": SYNTHETIC_T2_MS,
        "frequency_MHz": 100,
        "description": "Independent T2 phantom",
    }
    session.study.datasets[0].kspace = samples
    session.study.datasets[0].acquisition = acquisition
    return session


def synthetic_multimodal_session() -> GuiSession:
    """Independent signed/complex, coil and motion fixtures on declared grids."""
    from mrax.data_struct import AcquisitionData
    from mrax.reco import fftnc

    session = synthetic_t2_session()
    session.rename_study("Synthetic multimodal study")
    base = np.asarray(session.image_array[:2], dtype=np.float64) / 1e6 - 0.5
    complex_image = (base + 1j * np.flip(base, axis=2)).astype(np.complex128)
    coils = np.stack([complex_image, complex_image * np.exp(1j * 0.4)], axis=1)
    magnitude = np.sqrt(np.sum(np.abs(coils) ** 2, axis=1))
    geometry = dict(session.current_experiment["spatial_geometry"])
    samples = np.asarray(fftnc(coils, axes=(-3, -2, -1)))
    session.add_experiment(
        "Complex coils",
        complex_image,
        dims={"frame": 2, "x": 24, "y": 32, "z": 3},
        model_name="identity",
        kspace=samples,
        processed={"raw_reconstruction": magnitude, "raw_reconstruction_coils": coils},
        extra_metadata={"spatial_geometry": geometry},
    )
    acquisition = AcquisitionData(samples, tuple(complex_image.shape[1:]), ("x", "y", "z"))
    session.current_experiment["acquisition"] = acquisition
    session.study.datasets[-1].acquisition = acquisition
    signed = np.real(complex_image).copy()
    signed[0, 0, 0, 0] = np.nan
    signed[0, 0, 1, 0] = np.inf
    session.add_experiment(
        "Signed with missing samples",
        signed,
        dims={"frame": 2, "x": 24, "y": 32, "z": 3},
        model_name="identity",
        extra_metadata={"spatial_geometry": geometry},
    )
    y, x = np.mgrid[:32, :40]
    fixed = (
        np.exp(-((x - 15) ** 2 + (y - 12) ** 2) / 30)
        + 0.6 * np.exp(-((x - 27) ** 2 + (y - 24) ** 2) / 12)
    ).astype(np.float32)
    moving = np.zeros_like(fixed)
    moving[:, 3:] = fixed[:, :-3]
    session.add_experiment(
        "Motion reference",
        np.stack([fixed] * 4),
        dims={"frame": 4, "x": 32, "y": 40},
        model_name="identity",
    )
    session.add_experiment(
        "Motion shifted",
        np.stack([np.roll(fixed, i, axis=1) for i in range(4)]),
        dims={"frame": 4, "x": 32, "y": 40},
        model_name="identity",
    )
    session.select_experiment("T2 phantom")
    return session
