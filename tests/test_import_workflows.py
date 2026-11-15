"""Synthetic import contracts and numerical references."""

import warnings

import jax.numpy as jnp
import numpy as np
from _synthetic import (
    _write_classic_dicom_slice,
    _write_minimal_bruker_scan,
)

from mrax.data_struct import Quantity
from mrax.io import (
    discover_dicom_series,
    import_bruker_experiment,
    import_bruker_study,
    import_pulseq_experiment,
    load_dicom_series,
)


def _write_multicycle_bruker_slice_scan(root):
    scan = root / "1"
    pdata = scan / "pdata" / "1"
    pdata.mkdir(parents=True)
    frame_count, slice_count, x, y, coil_count = 3, 2, 3, 2, 2
    object_order = np.asarray([1, 0], dtype=int)

    images = np.empty((frame_count, x, y, slice_count), dtype=np.float32)
    base = np.arange(x * y, dtype=np.float32).reshape((x, y)) + 1.0
    for frame_index in range(frame_count):
        for slice_index in range(slice_count):
            images[frame_index, :, :, slice_index] = base + 10.0 * frame_index + 100.0 * slice_index

    (scan / "method").write_text(
        "\n".join(
            [
                "##TITLE=method",
                "##$Method=<Bruker:MGE>",
                "##$PVM_SpatDimEnum=<2D>",
                "##$PVM_EncMatrix=( 2 )",
                f"{x} {y}",
                f"##$PVM_EncNReceivers={coil_count}",
                f"##$PVM_NRepetitions={frame_count}",
                "##$PVM_NSPacks=1",
                "##$PVM_SPackArrNSlices=( 1 )",
                f"{slice_count}",
                f"##$PVM_ObjOrderList=( {slice_count} )",
                " ".join(str(value) for value in object_order.tolist()),
                f"##$PVM_EncSteps1=( {y} )",
                "-1 0",
                "##$PVM_EncUseMultiRec=<Yes>",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    (scan / "acqp").write_text(
        "\n".join(
            [
                "##TITLE=acqp",
                "##$ACQ_dim=2",
                f"##$NI={slice_count}",
                f"##$NR={frame_count}",
                f"##$NSLICES={slice_count}",
                "##$GO_raw_data_format=GO_32BIT_FLOAT",
                "##$ACQ_jobs=( 9 )",
                f"{2 * x} 0 0 0 0 0 {frame_count * slice_count * y} 1 <job0>",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    (pdata / "reco").write_text(
        "\n".join(
            [
                "##TITLE=reco",
                f"##$RECO_inp_size=( 2 )\n{x} {y}",
                f"##$RECO_size=( 2 )\n{x} {y}",
                f"##$RECO_ft_size=( 2 )\n{x} {y}",
                f"##$RecoSortRange=( 2 )\n{y} {slice_count}",
                f"##$RecoSortSize=( 2 )\n{y} {slice_count}",
                f"##$RecoSortMaps=( {y + slice_count} )",
                "0 1 1 0",
                f"##$RecoObjectsPerRepetition={slice_count}",
                f"##$RECO_transposition=( {slice_count} )\n0 0",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    (pdata / "visu_pars").write_text(
        "\n".join(
            [
                "##TITLE=visu_pars",
                f"##$VisuCoreFrameCount={frame_count * slice_count}",
                "##$VisuCoreDim=2",
                f"##$VisuCoreSize=( 2 )\n{x} {y}",
                "##$VisuFGOrderDesc=( 2 )",
                f"({slice_count}, <FG_SLICE>, <>, 0, 2)",
                f"({frame_count}, <FG_CYCLE>, <>, 2, 0)",
                "##$VisuCoreDataSlope=( 1 )\n1",
                "##$VisuCoreDataOffs=( 1 )\n0",
                "##$VisuCoreFrameType=( 1 )\nMAGNITUDE_IMAGE",
                "##$VisuCoreWordType=_16BIT_SGN_INT",
                "##$VisuCoreByteOrder=littleEndian",
                f"##$VisuCoreTransposition=( {slice_count} )\n0 0",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )

    stored_frames = np.moveaxis(images, -1, 1).reshape((frame_count * slice_count, x, y))
    np.concatenate([frame.reshape((-1,), order="F") for frame in stored_frames]).astype(
        np.int16
    ).tofile(pdata / "2dseq")

    scaled_images = images * np.sqrt(float(x * y))
    image_objects = np.moveaxis(scaled_images, -1, 1)
    kspace = np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(image_objects, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    ).transpose((0, 1, 3, 2))
    coil_kspace = np.zeros((frame_count, slice_count, coil_count, y, x), dtype=np.complex64)
    coil_kspace[:, :, 0] = kspace
    acquired = np.empty_like(coil_kspace)
    for acquired_slice, output_slice in enumerate(object_order.tolist()):
        acquired[:, acquired_slice] = coil_kspace[:, output_slice]
    blocks = np.transpose(acquired, (0, 3, 1, 2, 4))
    interleaved = np.empty(blocks.shape + (2,), dtype=np.float32)
    interleaved[..., 0] = blocks.real
    interleaved[..., 1] = blocks.imag
    interleaved.reshape((-1,)).tofile(scan / "rawdata.job0")
    return scan, images


def test_synthetic_bruker_study_import_workflow(tmp_path):
    root = tmp_path / "bruker"
    _write_minimal_bruker_scan(root)

    study, metadata = import_bruker_study(
        root,
        study_globals={"field_strength_mhz": Quantity(jnp.asarray(400.0), "MHz")},
        load_raw_kspace=False,
    )

    assert study.names == ["1"]
    assert study.experiments[0].exp_type == "mse"
    assert study.datasets[0].shape == (1, 2, 2)
    np.testing.assert_allclose(np.asarray(study.datasets[0].data[0]), [[10.0, 14.0], [12.0, 16.0]])

    exp_meta = metadata["experiments"]["1"]
    assert exp_meta["method"]["Method"] == "MSME"
    assert exp_meta["acqp"]["ACQ_dim"] == 2
    assert exp_meta["image_scaling"]["applied_to"] == "DataSet.data"


def test_bruker_multicycle_slice_stack_sorts_vendor_images_and_raw_kspace(tmp_path):
    scan, expected_images = _write_multicycle_bruker_slice_scan(tmp_path / "bruker")

    experiment, dataset, _, metadata = import_bruker_experiment(scan, load_raw_kspace=True)

    assert experiment.dims == {"frame": 3, "x": 3, "y": 2, "z": 2}
    np.testing.assert_allclose(np.asarray(dataset.data), expected_images)
    assert dataset.kspace.shape == (6, 2, 2, 3)
    assert dataset.acquisition.samples.shape == dataset.kspace.shape
    assert metadata["raw_import"]["reconstruction_status"] == "not_run"
    assert "raw_reconstruction" not in (dataset.processed or {})
    from mrax.reco import reconstruct_acquisition

    result = reconstruct_acquisition(dataset.acquisition)
    np.testing.assert_allclose(
        np.asarray(result.image),
        expected_images,
        rtol=1e-5,
        atol=1e-5,
    )
    coil_images = np.asarray(result.coil_images)
    assert coil_images.shape == (3, 2, 3, 2, 2)
    assert np.iscomplexobj(coil_images)
    np.testing.assert_allclose(np.abs(coil_images[:, 0]), expected_images, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(coil_images[:, 1], 0.0, atol=1e-6)


def test_standard_dicom_series_import_builds_mrax_experiment(tmp_path):
    series_uid = "1.2.826.0.1.3680043.10.1000.10"
    raw_frames = {}
    for echo_index in range(2):
        for slice_index in range(2):
            raw_frames[(echo_index, slice_index)] = _write_classic_dicom_slice(
                tmp_path / f"image_{echo_index}_{slice_index}.dcm",
                series_uid=series_uid,
                echo_index=echo_index,
                slice_index=slice_index,
            )

    discovered = discover_dicom_series(tmp_path)
    experiment, dataset, metadata = load_dicom_series(tmp_path)

    assert len(discovered) == 1
    assert discovered[0].file_count == 4
    assert experiment.exp_type == "mse"
    assert experiment.dims == {"te": 2, "x": 2, "y": 3, "z": 2}
    assert dataset.shape == (2, 2, 3, 2)
    np.testing.assert_allclose(np.asarray(experiment.observables["TE"].values), [10.0, 20.0])
    expected = raw_frames[(0, 0)].astype(float) * 2.0 + 5.0
    np.testing.assert_allclose(np.asarray(dataset.data)[0, ..., 0], expected)
    np.testing.assert_allclose(
        np.asarray(metadata["affine"]),
        np.asarray(
            [
                [0.0, -3.0, 0.0, 0.0],
                [0.0, 0.0, -2.0, 0.0],
                [2.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert metadata["adapter"] == "standard"
    assert dataset.spatial_geometry is not None
    assert dataset.spatial_geometry.voxel_sizes == (2.0, 3.0, 2.5)
    assert dataset.spatial_geometry.axis_anatomical_labels == ("y", "x", "z")
    assert "study_instance_uid" not in metadata
    assert experiment._mrax_dicom_metadata is metadata


def test_synthetic_pulseq_import_workflow(tmp_path):
    seq = tmp_path / "demo.seq"
    seq.write_text(
        "\n".join(
            [
                "[DEFINITIONS]",
                "Name demo_mse",
                "TE 8 24",
                "Matrix 2 2",
                "[ADC]",
                "1 2 0 0 0 0 0",
                "[BLOCKS]",
                "1 0 0 0 0 0 1",
                "2 0 0 0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )
    image_data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        experiment, dataset, model, metadata = import_pulseq_experiment(
            seq,
            image_data=image_data,
            spatial_shape=(2, 2),
            override_type="mse",
            name="pulseq_demo",
        )

    assert experiment.exp_type == "mse"
    assert dataset.name == "pulseq_demo"
    assert dataset.shape == (2, 2, 2)
    assert model.name == "t2_se"
    assert metadata["path"] == seq
    assert metadata["pulseq"]["adc_counts"] == (2, 2)
    np.testing.assert_array_equal(np.asarray(dataset.data), image_data)
