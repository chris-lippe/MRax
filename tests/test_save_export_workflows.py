"""Synthetic save export contracts and numerical references."""

import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

from mrax.data_struct import (
    DataSet,
    Experiment,
    Quantity,
)
from mrax.io.bundle import extract_bundle, write_bundle
from mrax.io.image_export import export_image_files
from mrax.io.nifti import (
    export_nifti_4d,
    export_nifti_series,
    load_nifti,
    mrax_to_nifti_volume,
    nifti_spatial_geometry,
    save_nifti,
)


def test_nifti_single_volume_and_series_round_trip(tmp_path):
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    affine = np.diag([1.5, 2.0, 1.0, 1.0])
    path = tmp_path / "image.nii.gz"
    save_nifti(path, image, affine)
    loaded, loaded_affine = load_nifti(path)
    np.testing.assert_allclose(loaded, image)
    np.testing.assert_allclose(loaded_affine, affine, atol=1e-6)
    geometry = nifti_spatial_geometry(loaded_affine, spatial_shape=image.shape)
    assert geometry.voxel_sizes == (2.0, 1.5)
    assert geometry.axis_anatomical_labels == ("y", "x")

    experiment = Experiment(
        dims={"echo": 2, "x": 3, "y": 4},
        observables={"TE": Quantity(jnp.asarray([10.0, 20.0]), "ms")},
        exp_type="mse",
    )
    dataset = DataSet(jnp.stack([jnp.asarray(image), jnp.asarray(image + 1)]), name="nifti")
    outputs = export_nifti_series(experiment, dataset, tmp_path / "series", affine=affine)
    output_4d = export_nifti_4d(experiment, dataset, tmp_path / "series_4d.nii.gz", affine=affine)

    assert len(outputs) == 2
    assert all(path.is_file() for path in outputs)
    assert output_4d.is_file()


def test_bundle_failure_preserves_previous_file_and_rejects_traversal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "array.bin").write_bytes(b"x" * 2048)
    target = tmp_path / "study.mrax"
    target.write_bytes(b"previous contents")

    def cancel(*_args):
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        write_bundle(target, source, progress_callback=cancel, overwrite=True)
    assert target.read_bytes() == b"previous contents"
    assert not list(tmp_path.glob("*.partial"))
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("../outside", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(target, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_dicom_export_preserves_calibrated_pixels_units_and_geometry(tmp_path):
    import pydicom

    data = np.linspace(-12.5, 93.7, 2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    affine = np.diag([1.2, 2.3, 3.4, 1.0])
    affine[:3, 3] = [10.0, -20.0, 30.0]
    paths = export_image_files(
        tmp_path / "image.dcm",
        data,
        dims={"t": 2, "x": 3, "y": 4, "z": 5},
        affine=affine,
        format="dicom",
        unit="Hz",
    )
    assert len(paths) == 6
    series = [pydicom.dcmread(path) for path in paths]
    assert len({ds.StudyInstanceUID for ds in series}) == 1
    assert len({ds.FrameOfReferenceUID for ds in series}) == 1
    assert len({ds.SeriesInstanceUID for ds in series}) == 2
    for index in range(2):
        expected = mrax_to_nifti_volume(data[index])
        for z, ds in enumerate(series[index * 3 : (index + 1) * 3]):
            pixels = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(
                ds.RescaleIntercept
            )
            np.testing.assert_allclose(
                pixels, expected[:, :, z].T, atol=float(ds.RescaleSlope) * 0.51
            )
            np.testing.assert_allclose(ds.ImagePositionPatient, [-10.0, 20.0, 30.0 + z * 3.4])
            np.testing.assert_allclose(ds.PixelSpacing, [2.3, 1.2])
            assert ds.RescaleType == "Hz" and not ds.PatientID
