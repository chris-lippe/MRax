"""Independent synthetic pixel, physical-coordinate and raw reconstruction checks."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mrax.io.bruker_images import bruker_spatial_geometry, load_2dseq
from mrax.io.bruker_import import import_bruker_experiment, import_bruker_study
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



def test_sampled_bruker_reconstruction_keeps_native_rectangular_axes(tmp_path):
    shape = (9, 7)
    scan, images, *_ = _scan(tmp_path, shape=shape, counts=(2,), echoes=1)
    pdata = scan / "pdata" / "1"
    (scan / "method").write_text(
        "##$Method=<SPIRAL>\n##$PVM_EncNReceivers=1\n"
        + _array("PVM_SPackArrNSlices", [2])
    )
    (scan / "acqp").write_text(
        "##$ACQ_dim=1\n##$NI=2\n##$NR=1\n"
        "##$GO_raw_data_format=GO_32BIT_FLOAT\n##$GO_block_size=continuous\n"
        + _array("ACQ_size", [2 * np.prod(shape)])
    )
    (pdata / "reco").write_text(
        "\n".join(
            [
                _array("RECO_size", shape),
                _array("RecoRegridNFtSize", shape),
                _array("RECO_rotate", [0.5, 0.5]),
                "##$RecoRegridN=Yes",
                f"##$RecoRegridNPoints={np.prod(shape)}",
                "##$RecoRegridNInterleaves=1",
                "##$RecoRegridNDim=2",
                "##$RecoRegridNTrajType=<Spiral>",
                "##$RecoRegridNDensCorr=<Pre>",
            ]
        )
    )
    coordinates = np.stack(
        np.meshgrid(
            *[np.fft.fftshift(np.fft.fftfreq(size)) for size in shape],
            indexing="ij",
        ),
        axis=-1,
    )
    coordinates.astype("<f8").tofile(scan / "traj")
    samples = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    ).ravel()
    words = np.empty(samples.size * 2, dtype="<f4")
    words[::2], words[1::2] = samples.real, samples.imag
    words.tofile(scan / "fid")
    _, dataset, _, _ = import_bruker_experiment(scan, load_raw_kspace=True)
    assert dataset.acquisition is not None
    result = reconstruct_acquisition(
        dataset.acquisition, method="nufft", density_compensation=False
    )
    expected = np.moveaxis(images, 0, -1)[None, ...]
    actual = np.abs(np.asarray(result.image))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        actual / np.linalg.norm(actual),
        expected / np.linalg.norm(expected),
        atol=2e-6,
    )


@pytest.mark.parametrize("echo_first", [True, False])
def test_multiple_packages_import_on_one_axis_and_reconstruct_without_transpose(
    tmp_path, echo_first
):
    scan, _, _, _, _, _ = _scan(tmp_path, counts=(3, 2, 1), echo_first=echo_first)
    hashes = {
        path: hashlib.sha256(path.read_bytes()).digest()
        for path in scan.rglob("*")
        if path.is_file()
    }
    study, metadata = import_bruker_study(tmp_path, field_strength_mhz=100, load_raw_kspace=True)
    assert study.names == ["1"]
    assert [
        package["native_image_plane"] for package in metadata["experiments"]["1"]["slice_packages"]
    ] == [
        "axial",
        "coronal",
        "sagittal",
    ]
    for _name, experiment, _, dataset in study.iter_entries():
        assert experiment.dims["slice_package"] == 3
        assert dataset.acquisition.geometry == dataset.spatial_geometry
        result = reconstruct_acquisition(dataset.acquisition)
        expected = np.asarray(dataset.data).reshape(result.image.shape)
        np.testing.assert_allclose(result.image, expected, rtol=2e-5, atol=2e-4)
        assert experiment.spatial_shape == (7, 5, 3)
    assert all(
        hashlib.sha256(path.read_bytes()).digest() == digest for path, digest in hashes.items()
    )


def test_rectangular_storage_transposition_matches_the_raw_reconstruction(tmp_path):
    scan, frames, _, _, _, _ = _scan(tmp_path, shape=(5, 7), counts=(2,), echoes=1)
    pdata = scan / "pdata/1"
    # Storage and raw encoding are 5x7. The declared logical image is 7x5.
    with (pdata / "visu_pars").open("a") as handle:
        handle.write("\n" + _array("VisuCoreSize", [7, 5]) + "\n##$VisuCoreTransposition=1\n")
    experiment, dataset, _, _ = import_bruker_experiment(scan)
    expected = np.moveaxis(frames.transpose(0, 2, 1), 0, -1)[None]
    np.testing.assert_array_equal(np.asarray(dataset.data).reshape(expected.shape), expected)
    result = reconstruct_acquisition(dataset.acquisition)
    np.testing.assert_allclose(result.image, expected, rtol=2e-5, atol=2e-4)
    assert experiment.spatial_shape == (7, 5, 2)


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


def test_irregular_slice_positions_are_not_misrepresented_as_a_regular_volume(tmp_path):
    scan, _, _, _, _, _ = _scan(tmp_path, counts=(3,), echoes=1)
    pdata = scan / "pdata/1"
    with (pdata / "visu_pars").open("a") as handle:
        handle.write("\n" + _array("VisuCorePosition", [[0, 0, 0], [0, 0, 2], [0, 0, 5]]) + "\n")
    data, visu, spatial = load_2dseq(pdata)
    assert data.shape[-1] == 3
    with pytest.raises(ValueError, match="regular spatial stack"):
        bruker_spatial_geometry(visu, spatial_shape=spatial)
