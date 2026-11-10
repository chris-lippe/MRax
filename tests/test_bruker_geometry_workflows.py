"""Independent synthetic pixel, physical-coordinate and raw reconstruction checks."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mrax.gui.session import GuiSession
from mrax.io.bruker_frames import frame_parameter, slice_packages
from mrax.io.bruker_images import bruker_spatial_geometry, load_2dseq
from mrax.io.bruker_import import import_bruker_experiment, import_bruker_study
from mrax.io.image_export import export_image_files
from mrax.reco.acquisition import reconstruct_acquisition

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])
ORIENTATIONS = (
    np.eye(3),
    np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
)


def _array(key, value):
    values = np.asarray(value)
    return f"##${key}=( {', '.join(str(n) for n in values.shape)} )\n" + " ".join(
        str(x) for x in values.ravel()
    )


def _scan(root, *, shape=(7, 5), counts=(3,), echoes=2, echo_first=True, reverse=False):
    """Create an asymmetric phantom and an independently encoded complex FID."""
    scan = root / "1"
    pdata = scan / "pdata" / "1"
    pdata.mkdir(parents=True)
    nx, ny = shape
    ns = sum(counts)
    matrices, positions, physical_slice = [], [], []
    for package, count in enumerate(counts):
        matrix = ORIENTATIONS[package]
        order = list(reversed(range(count))) if reverse else list(range(count))
        for z in order:
            matrices.append(matrix)
            positions.append(np.asarray([11.0, -7.0, 19.0]) + matrix[2] * (z * 2.5))
            physical_slice.append((package, z))
    coordinates = (
        [(echo, z) for z in range(ns) for echo in range(echoes)]
        if echo_first
        else [(echo, z) for echo in range(echoes) for z in range(ns)]
    )
    images = np.stack(
        [
            5 + 100 * echo + 13 * z + np.arange(nx)[:, None] * 3 + np.arange(ny)[None, :]
            for echo, z in coordinates
        ]
    ).astype(np.float32)
    np.concatenate([image.ravel(order="F") for image in images]).astype("<f4").tofile(
        pdata / "2dseq"
    )
    entries = (
        [(echoes, "FG_ECHO", 0, 0), (ns, "FG_SLICE", 0, 2)]
        if echo_first
        else [(ns, "FG_SLICE", 0, 2), (echoes, "FG_ECHO", 2, 0)]
    )
    starts = np.cumsum((0,) + counts[:-1])
    (pdata / "visu_pars").write_text(
        "\n".join(
            [
                "##$VisuCoreDim=2",
                _array("VisuCoreSize", shape),
                _array("VisuCoreExtent", [nx * 0.8, ny * 1.2]),
                f"##$VisuCoreFrameCount={len(images)}",
                "##$VisuCoreWordType=_32BIT_FLOAT",
                "##$VisuCoreByteOrder=littleEndian",
                "##$VisuCoreFrameType=MAGNITUDE_IMAGE",
                "##$VisuCoreDataSlope=1",
                "##$VisuCoreDataOffs=0",
                "##$VisuCoreFrameThickness=2",
                "##$VisuFGOrderDesc=( 2 )\n"
                + "\n".join(
                    f"({count}, <{kind}>, <>, {first}, {length})"
                    for count, kind, first, length in entries
                ),
                "##$VisuGroupDepVals=( 2 )\n(<VisuCorePosition>, 0) (<VisuCoreOrientation>, 0)",
                _array("VisuCorePosition", positions),
                _array("VisuCoreOrientation", np.asarray(matrices).reshape(ns, 9)),
                "##$VisuCoreSlicePacksSlices=( "
                + str(len(counts))
                + " )\n"
                + " ".join(
                    f"({start}, {count})" for start, count in zip(starts, counts, strict=True)
                ),
                "##$VisuCoreDiskSliceOrder="
                + ("disk_reverse_slice_order" if reverse else "disk_normal_slice_order"),
            ]
        )
    )
    (scan / "method").write_text(
        "\n".join(
            [
                "##$Method=<FLASH>",
                "##$PVM_SpatDimEnum=<2D>",
                _array("PVM_EncMatrix", shape),
                "##$PVM_EncNReceivers=1",
                f"##$PVM_NEchoImages={echoes}",
                _array("PVM_EchoTime", np.arange(1, echoes + 1) * 12),
                _array("PVM_EncSteps1", np.arange(ny) - ny // 2),
                _array("PVM_SPackArrNSlices", counts),
                f"##$PVM_NSPacks={len(counts)}",
            ]
        )
    )
    object_order = np.roll(np.arange(len(images)), 1)
    (scan / "acqp").write_text(
        "\n".join(
            [
                "##$ACQ_dim=2",
                f"##$NI={len(images)}",
                "##$NR=1",
                f"##$NSLICES={ns}",
                "##$GO_raw_data_format=GO_32BIT_FLOAT",
                "##$GO_block_size=continuous",
                _array("ACQ_size", [nx * 2, ny]),
                _array("ACQ_obj_order", object_order),
            ]
        )
    )
    (pdata / "reco").write_text(
        "\n".join(
            [
                _array("RECO_ft_size", shape),
                _array("RECO_size", shape),
                _array("RECO_rotate", [0.5, 0.5]),
                "##$RECO_transposition=0",
            ]
        )
    )
    # The raw stream is [phase, acquisition object, receiver, read] and does
    # not use MRax's forward operator or image-formatting implementation.
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    ) * np.sqrt(nx * ny)
    stream = kspace[object_order].transpose(2, 0, 1).ravel()
    words = np.empty(stream.size * 2, dtype="<f4")
    words[::2], words[1::2] = stream.real, stream.imag
    words.tofile(scan / "fid")
    return scan, images, coordinates, np.asarray(positions), matrices, physical_slice


@pytest.mark.parametrize("shape", [(7, 5), (4, 4)])
@pytest.mark.parametrize("echo_first", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_slice_and_echo_groups_use_pixel_coordinates_and_measured_slice_positions(
    tmp_path, shape, echo_first, reverse
):
    scan, frames, coordinates, positions, matrices, _ = _scan(
        tmp_path, shape=shape, echo_first=echo_first, reverse=reverse
    )
    data, visu, spatial = load_2dseq(scan / "pdata/1")
    assert data.shape == (2,) + shape + (3,)
    geometry = bruker_spatial_geometry(visu, spatial_shape=spatial)
    affine = np.asarray(geometry.affine)
    assert not geometry.frame_affines
    assert geometry.voxel_sizes == pytest.approx((0.8, 1.2, 2.5))
    for echo in range(2):
        indices = [i for i, (e, _) in enumerate(coordinates) if e == echo]
        indices.sort(key=lambda i: positions[coordinates[i][1]] @ matrices[0][2])
        for z, source_index in enumerate(indices):
            np.testing.assert_array_equal(data[echo, :, :, z], frames[source_index])
            source_slice = coordinates[source_index][1]
            expected = LPS_TO_RAS @ (positions[source_slice] + np.asarray([2 * 0.8, 1 * 1.2, 0]))
            np.testing.assert_allclose((affine @ [z, 1, 2, 1])[:3], expected)


@pytest.mark.parametrize("echo_first", [True, False])
def test_multiple_packages_import_as_separate_spatial_stacks_and_reconstruct_without_transpose(
    tmp_path, echo_first
):
    scan, _, _, _, _, _ = _scan(tmp_path, counts=(3, 2, 1), echo_first=echo_first)
    hashes = {
        path: hashlib.sha256(path.read_bytes()).digest()
        for path in scan.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="multiple slice packages"):
        import_bruker_experiment(scan, load_raw_kspace=False)
    study, metadata = import_bruker_study(tmp_path, field_strength_mhz=100, load_raw_kspace=True)
    assert len(study.datasets) == 3
    assert [metadata["experiments"][name]["native_image_plane"] for name in study.names] == [
        "axial",
        "coronal",
        "sagittal",
    ]
    for package, (name, experiment, _, dataset) in enumerate(study.iter_entries()):
        assert metadata["experiments"][name]["slice_package"] == package
        assert dataset.acquisition.geometry == dataset.spatial_geometry
        result = reconstruct_acquisition(dataset.acquisition)
        expected = np.asarray(dataset.data).reshape(result.image.shape)
        np.testing.assert_allclose(result.image, expected, rtol=2e-5, atol=2e-4)
        assert experiment.spatial_shape == ((7, 5, (3, 2, 1)[package]) if package < 2 else (7, 5))
    assert all(
        hashlib.sha256(path.read_bytes()).digest() == digest for path, digest in hashes.items()
    )


def test_gui_preserves_package_geometry_through_lazy_raw_loading(tmp_path):
    from mrax.gui.services.workspace import StudyWorkspace
    from mrax.web.app import update_view
    from mrax.web.schemas import ViewCommand

    _scan(tmp_path, counts=(3, 2, 1))
    session = GuiSession()
    session.import_bruker_study(tmp_path, field_strength_mhz=100)
    assert len(session.experiment_names) == 3
    for name, plane in zip(session.experiment_names, ("axial", "coronal", "sagittal"), strict=True):
        session.select_experiment(name)
        original = np.asarray(session.current_experiment["mr_image"]).copy()
        assert session.native_image_plane == plane
        assert session.image_plane_dims == ("y", "x")
        assert session.image_plane_axis_labels == {
            "axial": ("y", "x"),
            "coronal": ("z", "x"),
            "sagittal": ("z", "y"),
        }[plane]
        spatial = tuple(
            session.experiment_dims[axis]
            for axis in ("x", "y", "z")
            if axis in session.experiment_dims
        )
        volume = original.reshape((-1,) + spatial)[0]
        np.testing.assert_array_equal(
            session.frame_image(), volume[..., 0].T if len(spatial) == 3 else volume.T
        )
        if not session.supports_image_planes():
            workspace = StudyWorkspace("geometry", "Geometry", session)
            update_view(workspace, ViewCommand(plane=plane, revision=0))
            with pytest.raises(ValueError, match="native image plane"):
                update_view(workspace, ViewCommand(plane="axial", revision=0))
        assert session.load_raw_data_for_experiment(name, require_acquisition=True)
        acquisition = session.current_experiment["acquisition"]
        result = reconstruct_acquisition(acquisition)
        np.testing.assert_allclose(
            result.image, original.reshape(result.image.shape), rtol=2e-5, atol=2e-4
        )
        np.testing.assert_array_equal(session.current_experiment["mr_image"], original)


@pytest.mark.parametrize("shape", [(5, 3), (4, 4)])
def test_core_transposition_describes_storage_not_an_extra_anatomical_rotation(tmp_path, shape):
    scan, frames, _, _, _, _ = _scan(tmp_path, shape=shape, counts=(2,), echoes=1)
    pdata = scan / "pdata/1"
    with (pdata / "visu_pars").open("a") as handle:
        handle.write("\n##$VisuCoreTransposition=( 2 )\n0 1\n")
    stored = [frames[0], frames[1].T if shape[0] != shape[1] else frames[1]]
    np.concatenate([image.ravel(order="F") for image in stored]).astype("<f4").tofile(
        pdata / "2dseq"
    )
    data, _, spatial = load_2dseq(pdata)
    assert spatial == shape + (2,)
    np.testing.assert_array_equal(data[0], np.moveaxis(frames, 0, -1))


def test_rectangular_storage_transposition_matches_the_raw_reconstruction(tmp_path):
    scan, frames, _, _, _, _ = _scan(tmp_path, shape=(5, 7), counts=(2,), echoes=1)
    pdata = scan / "pdata/1"
    # Storage and raw encoding are 5x7; the declared logical image is 7x5.
    with (pdata / "visu_pars").open("a") as handle:
        handle.write("\n" + _array("VisuCoreSize", [7, 5]) + "\n##$VisuCoreTransposition=1\n")
    experiment, dataset, _, _ = import_bruker_experiment(scan)
    expected = np.moveaxis(frames.transpose(0, 2, 1), 0, -1)[None]
    np.testing.assert_array_equal(np.asarray(dataset.data).reshape(expected.shape), expected)
    result = reconstruct_acquisition(dataset.acquisition)
    np.testing.assert_allclose(result.image, expected, rtol=2e-5, atol=2e-4)
    assert experiment.spatial_shape == (7, 5, 2)


def test_exported_package_pixels_keep_their_patient_coordinates(tmp_path):
    nib = pytest.importorskip("nibabel")
    counts = (3, 2, 1)
    _, _, _, positions, matrices, _ = _scan(tmp_path / "input", counts=counts)
    study, _ = import_bruker_study(
        tmp_path / "input", field_strength_mhz=100, load_raw_kspace=False
    )
    for package, (_, experiment, _, dataset) in enumerate(study.iter_entries()):
        paths = export_image_files(
            tmp_path / f"package-{package}.nii.gz",
            dataset.data,
            dims=experiment.dims,
            geometry=dataset.spatial_geometry,
        )
        assert len(paths) == 2
        image = nib.load(paths[0])
        native = np.asarray(dataset.data).reshape((-1,) + experiment.spatial_shape)[0]
        z = min(counts[package] - 1, 1)
        index = (z, 1, 2) if native.ndim == 3 else (1, 2, 0)
        assert np.asanyarray(image.dataobj)[index] == (
            native[2, 1, z] if native.ndim == 3 else native[2, 1]
        )
        first = sum(counts[:package])
        patient = positions[first + z] + matrices[first].T @ np.asarray([2 * 0.8, 1.2, 0])
        np.testing.assert_allclose(
            (image.affine @ (*index, 1))[:3], LPS_TO_RAS @ patient, atol=1e-6
        )


@pytest.mark.parametrize("code", [0, 1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_anisotropic_3d_frames_preserve_coordinates_and_storage_transpositions(
    tmp_path, code, reverse
):
    shape = (5, 3, 2)
    x, y, z = np.indices(shape)
    expected = np.stack([x + 10 * y + 100 * z + 1000 * frame for frame in range(2)]).astype("<f4")
    frames = expected[..., ::-1] if reverse else expected
    first, second = ((0, 2) if code == 3 else (code - 1, code)) if code else (0, 0)
    if code:
        frames = np.swapaxes(frames, first + 1, second + 1)
    np.concatenate([frame.ravel(order="F") for frame in frames]).tofile(tmp_path / "2dseq")
    origins = np.asarray([[4.0, -8.0, 12.0], [6.0, -8.0, 12.0]])
    rotation = ORIENTATIONS[2]
    (tmp_path / "visu_pars").write_text(
        "\n".join(
            [
                "##$VisuCoreDim=3",
                "##$VisuCoreFrameCount=2",
                _array("VisuCoreSize", shape),
                _array("VisuCoreExtent", [5.0, 6.0, 8.0]),
                "##$VisuCoreWordType=_32BIT_FLOAT",
                "##$VisuCoreDataSlope=1",
                "##$VisuCoreDataOffs=0",
                _array("VisuCoreOrientation", rotation.reshape(1, 9)),
                _array("VisuCorePosition", origins),
                f"##$VisuCoreTransposition={code}",
                "##$VisuCoreDiskSliceOrder="
                + ("disk_reverse_slice_order" if reverse else "disk_normal_slice_order"),
            ]
        )
    )
    data, visu, spatial = load_2dseq(tmp_path)
    np.testing.assert_array_equal(data, expected)
    geometry = bruker_spatial_geometry(visu, spatial_shape=spatial)
    assert len(geometry.frame_affines) == 2
    assert geometry.axis_anatomical_labels == ("y", "z", "x")
    assert geometry.voxel_sizes == (1.0, 2.0, 4.0)
    for frame, affine in enumerate(geometry.frame_affines):
        patient = origins[frame] + rotation.T @ np.asarray([2.0, 2.0, 4.0])
        np.testing.assert_allclose((np.asarray(affine) @ [1, 1, 2, 1])[:3], LPS_TO_RAS @ patient)


def test_frame_dependent_geometry_uses_declared_dependency_not_flat_tiling():
    visu = {
        "VisuCoreFrameCount": 6,
        "VisuFGOrderDesc": [[2, "FG_ECHO", "", 0, 0], [3, "FG_SLICE", "", 0, 1]],
        "VisuGroupDepVals": [["VisuCorePosition", 0]],
        "VisuCorePosition": [[0, 0, 4], [0, 0, 7], [0, 0, 10]],
    }
    np.testing.assert_array_equal(
        frame_parameter(visu, "VisuCorePosition", 3)[:, 2], [4, 4, 7, 7, 10, 10]
    )


def test_image_only_product_remains_importable_without_pretending_raw_geometry_is_known(tmp_path):
    scan, _, _, _, _, _ = _scan(tmp_path)
    (scan / "pdata/1/reco").unlink()
    _, dataset, _, metadata = import_bruker_experiment(scan, load_raw_kspace=True)
    assert dataset.data is not None
    assert dataset.acquisition is None
    assert metadata["raw_import"]["reason"] == "missing_reconstruction_recipe"
    session = GuiSession()
    session.import_bruker_study(tmp_path, field_strength_mhz=100)
    assert not session.can_lazy_load_raw_data(require_acquisition=True)


def test_image_frame_reordering_keeps_complete_raw_slice_groups(tmp_path):
    from mrax.io.bruker_import import _attach_acquisition

    scan, _, _, _, _, _ = _scan(tmp_path, counts=(2,), echoes=3)
    _, dataset, _, _ = import_bruker_experiment(scan, load_raw_kspace=False)
    reference = np.asarray(dataset.data).reshape((3, 7, 5, 2))
    order = np.asarray([2, 0, 1])
    _attach_acquisition(
        dataset,
        scan,
        "1",
        definition_paths=(),
        profiles=None,
        strict_raw=True,
        image_frame_order=order,
    )
    result = reconstruct_acquisition(dataset.acquisition)
    np.testing.assert_allclose(result.image, reference[order], rtol=2e-5, atol=2e-4)


def test_incomplete_or_overlapping_package_descriptions_are_rejected():
    visu = {
        "VisuCoreDim": 2,
        "VisuCoreFrameCount": 4,
        "VisuFGOrderDesc": [[4, "FG_SLICE", "", 0, 0]],
        "VisuCoreSlicePacksSlices": [[0, 3], [2, 2]],
    }
    with pytest.raises(ValueError, match="partition"):
        slice_packages(visu)


def test_irregular_slice_positions_are_not_misrepresented_as_a_regular_volume(tmp_path):
    scan, _, _, _, _, _ = _scan(tmp_path, counts=(3,), echoes=1)
    pdata = scan / "pdata/1"
    with (pdata / "visu_pars").open("a") as handle:
        handle.write("\n" + _array("VisuCorePosition", [[0, 0, 0], [0, 0, 2], [0, 0, 5]]) + "\n")
    data, visu, spatial = load_2dseq(pdata)
    assert data.shape[-1] == 3
    with pytest.raises(ValueError, match="regular spatial stack"):
        bruker_spatial_geometry(visu, spatial_shape=spatial)


def test_tilted_stack_export_preserves_the_complete_affine(tmp_path):
    nib = pytest.importorskip("nibabel")
    scan, _, _, _, _, _ = _scan(tmp_path / "input", counts=(3,), echoes=1)
    positions = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 2.0], [1.0, 0.0, 4.0]])
    with (scan / "pdata/1/visu_pars").open("a") as handle:
        handle.write("\n" + _array("VisuCorePosition", positions) + "\n")
    experiment, dataset, _, _ = import_bruker_experiment(scan, load_raw_kspace=False)
    paths = export_image_files(
        tmp_path / "tilted.nii.gz",
        dataset.data,
        dims=experiment.dims,
        geometry=dataset.spatial_geometry,
    )
    image = nib.load(paths[0])
    np.testing.assert_allclose(
        (image.affine @ [2, 1, 2, 1])[:3], LPS_TO_RAS @ (positions[2] + [1.6, 1.2, 0]), atol=1e-6
    )
    assert int(image.header["sform_code"]) != 0


def test_package_geometry_survives_workspace_save_and_reload(tmp_path):
    from mrax.gui.services.persistence import load_workspace_bundle, save_workspace_bundle
    from mrax.gui.services.workspace import StudyWorkspace

    _scan(tmp_path / "input", counts=(3, 2, 1))
    session = GuiSession()
    session.import_bruker_study(tmp_path / "input", field_strength_mhz=100)
    workspace = StudyWorkspace("before", "Geometry", session)
    path = save_workspace_bundle(workspace, tmp_path / "geometry.mrax")
    loaded = load_workspace_bundle(path, tmp_path / "loaded", key="after")
    assert loaded.session.experiment_names == session.experiment_names
    for name in session.experiment_names:
        before = session.data["experiments"][name]
        after = loaded.session.data["experiments"][name]
        assert before["spatial_geometry"] == after["spatial_geometry"]
        assert before["extra_metadata"]["slice_package"] == after["extra_metadata"]["slice_package"]
        np.testing.assert_array_equal(after["mr_image"], before["mr_image"])
