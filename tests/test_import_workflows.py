import warnings

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import Quantity
from mrax.io import load_bruker_study, load_pulseq_experiment


def _write_minimal_bruker_scan(root):
    scan = root / "1"
    pdata = scan / "pdata" / "1"
    pdata.mkdir(parents=True)
    (scan / "method").write_text("##TITLE=method\n##$Method=<MSME>\n##END=\n", encoding="utf-8")
    (scan / "acqp").write_text("##TITLE=acqp\n##$ACQ_dim=2\n##END=\n", encoding="utf-8")
    (pdata / "reco").write_text("##TITLE=reco\n##$RECO_mode=<USER>\n##END=\n", encoding="utf-8")
    (pdata / "visu_pars").write_text(
        "\n".join(
            [
                "##TITLE=visu_pars",
                "##$VisuCoreFrameCount=1",
                "##$VisuCoreDim=2",
                "##$VisuCoreSize=( 2 )",
                "2 2",
                "##$VisuCoreDataSlope=( 1 )",
                "2",
                "##$VisuCoreDataOffs=( 1 )",
                "10",
                "##$VisuCoreFrameType=( 1 )",
                "MAGNITUDE_IMAGE",
                "##$VisuCoreWordType=_16BIT_SGN_INT",
                "##$VisuCoreByteOrder=littleEndian",
                "##END=",
            ]
        ),
        encoding="utf-8",
    )
    np.arange(4, dtype=np.int16).tofile(pdata / "2dseq")


def test_synthetic_bruker_study_import_workflow(tmp_path):
    root = tmp_path / "bruker"
    _write_minimal_bruker_scan(root)

    study, metadata = load_bruker_study(
        root,
        study_globals={"field_strength_mhz": Quantity(jnp.asarray(400.0), "MHz")},
        load_raw_kspace=False,
    )

    assert study.names == ["1"]
    assert study.experiments[0].exp_type == "mse"
    assert study.datasets[0].shape == (1, 2, 2)
    np.testing.assert_allclose(np.asarray(study.datasets[0].data[0]), [[10.0, 12.0], [14.0, 16.0]])

    exp_meta = metadata["experiments"]["1"]
    assert exp_meta["method"]["Method"] == "MSME"
    assert exp_meta["acqp"]["ACQ_dim"] == 2
    assert exp_meta["image_scaling"]["applied_to"] == "DataSet.data"


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
        experiment, dataset, model, metadata = load_pulseq_experiment(
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
