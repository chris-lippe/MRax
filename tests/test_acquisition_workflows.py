"""Synthetic acquisition contracts and numerical references."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from mrax.data_struct import (
    AcquisitionData,
    DataSet,
    Experiment,
    Model,
    Quantity,
    ReconstructionRecipe,
    Study,
)
from mrax.data_struct.storage import load_study_results, save_study_results
from mrax.io import load_acquisition, save_acquisition
from mrax.reco import (
    ReconstructionPipeline,
    build_operator,
    fftnc,
    ifftnc,
    reconstruct_acquisition,
)
from mrax.reco.acquisition import complete_partial_fourier
from mrax.reco.epi import (
    correct_epi_entropy,
    correct_readout_phase,
)


def test_explicit_acquisition_fft_and_adjoint_identity():
    shape = (5, 6, 3)
    rng = np.random.default_rng(17)
    image = (rng.normal(size=(2, 3) + shape) + 1j * rng.normal(size=(2, 3) + shape)).astype(
        "complex64"
    )
    op = build_operator(spatial_shape=shape, independent_coils=True)
    samples = op.forward(image)
    acq = AcquisitionData(np.asarray(samples), shape, tuple(f"axis_{i}" for i in range(len(shape))))
    result = reconstruct_acquisition(acq)
    np.testing.assert_allclose(result.coil_images, image, atol=2e-6)
    probe = jnp.asarray(
        rng.normal(size=samples.shape) + 1j * rng.normal(size=samples.shape), dtype=samples.dtype
    )
    np.testing.assert_allclose(
        jnp.vdot(samples, probe),
        jnp.vdot(image, op.adjoint(probe, kspace_has_coils=True)),
        atol=2e-5,
    )


def test_complex_mask_and_unnormalized_sensitivity_adjoint():
    rng = np.random.default_rng(12)
    image = rng.normal(size=(2, 7, 5)) + 1j * rng.normal(size=(2, 7, 5))
    sensitivity = rng.normal(size=(3, 7, 5)) + 1j * rng.normal(size=(3, 7, 5))
    samples = rng.normal(size=(2, 3, 7, 5)) + 1j * rng.normal(size=(2, 3, 7, 5))
    mask = np.ones((2, 1, 7, 5), complex)
    mask[0, :, ::2] = 0
    mask[1, :, 1::2] = 1j
    op = build_operator(spatial_shape=(7, 5), sensitivities=sensitivity, mask=mask)
    np.testing.assert_allclose(
        jnp.vdot(op.forward(image), samples),
        jnp.vdot(image, op.adjoint(samples, kspace_has_coils=True)),
        rtol=2e-6,
        atol=2e-5,
    )


def test_partial_fourier_preserves_measured_complex_values():
    shape = (17, 16)
    x, y = np.meshgrid(np.arange(17), np.arange(16), indexing="ij")
    image = np.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 12) * np.exp(0.03j * x)
    complete = fftnc(image)
    mask = np.ones(shape, bool)
    mask[:5] = False
    measured = jnp.where(mask, complete, 0)
    filled = complete_partial_fourier(measured, mask, iterations=20)
    np.testing.assert_array_equal(np.asarray(filled)[mask], np.asarray(measured)[mask])
    assert np.linalg.norm(np.asarray(filled - complete)) < np.linalg.norm(
        np.asarray(measured - complete)
    )


def test_editable_reconstruction_plan_reuses_measured_data_and_shared_solvers(tmp_path):
    samples = np.ones((1, 1, 9, 8), np.complex64)
    acquired = np.ones((9, 8), bool)
    acquired[:2] = False
    acq = AcquisitionData(
        samples,
        (9, 8),
        ("phase", "read"),
        acquired=acquired,
        recipe=ReconstructionRecipe(partial_fourier="pocs", phase_window="hann"),
    )
    plan = ReconstructionPipeline.from_acquisition(acq)
    np.testing.assert_array_equal(plan.run(acq).image, reconstruct_acquisition(acq).image)
    path = tmp_path / "plan.toml"
    path.write_text(
        'schema_version=1\n[[stages]]\nkind="variational"\n'
        '[stages.options]\nnum_steps=3\noptimizer_name="fista"\nregularizer="tv"\nregularizer_weight=0.01\n'
        '[[stages]]\nkind="format"\n[[stages]]\nkind="combine"\n'
    )
    plan = ReconstructionPipeline.from_toml(path)
    result = plan.run(acq)
    direct = reconstruct_acquisition(
        replace(acq, recipe=replace(acq.recipe, partial_fourier="none")),
        method="variational",
        num_steps=3,
        optimizer_name="fista",
        regularizer="tv",
        regularizer_weight=0.01,
    )
    np.testing.assert_array_equal(result.image, direct.image)
    np.testing.assert_array_equal(acq.samples, samples)
    np.testing.assert_array_equal(acq.acquired, acquired)
    assert result.history == direct.history


def test_referenceless_epi_phase_fit_removes_injected_ghost_without_reference():

    y, x = np.meshgrid(np.arange(48) - 24, np.arange(40) - 20, indexing="ij")
    image = (
        np.exp(-((y - 2) ** 2 / 30 + (x + 3) ** 2 / 20))
        + 0.7 * np.exp(-((y + 5) ** 2 / 12 + (x - 5) ** 2 / 8))
    )[None]
    samples = fftnc(image)
    negative = np.broadcast_to((np.arange(48) % 2 == 0)[:, None], (48, 40))
    corrupted = correct_readout_phase(samples, negative, [-0.4, -7.0])
    corrected, report = correct_epi_entropy(corrupted, negative)
    before = np.linalg.norm(np.abs(ifftnc(corrupted)) - image)
    after = np.linalg.norm(np.abs(ifftnc(corrected)) - image)
    assert after < before * 0.01
    assert report["entropy_after"] < report["entropy_before"]
    # The data-independent ambiguity is explicit in the phase-prior option.
    shifted, _ = correct_epi_entropy(corrupted, negative, phase_prior=(np.pi, 0.0))
    np.testing.assert_allclose(np.abs(ifftnc(shifted)), np.roll(image, 24, axis=-2), atol=1e-3)


def test_acquisition_archive_and_study_roundtrip_preserve_encoding_metadata(tmp_path):
    samples = np.ones((2, 1, 4, 3), dtype=np.complex128) * (1 + 2j)
    recipe = ReconstructionRecipe(
        frame_image_shifts=((0.25, -0.1), (0.5, -0.2)), frame_permutations=((0, 1), (0, 1))
    )
    acq = AcquisitionData(
        samples,
        (4, 3),
        ("phase", "read"),
        acquired=np.ones((2, 1, 4, 3), bool),
        statistical_weights=np.full((4, 3), 0.5),
        frame_coordinates={"echo_seconds": np.asarray([0.01, 0.02])},
        recipe=recipe,
        readout_polarity=np.asarray([1, -1, 1, -1])[:, None],
        auxiliary_samples={"navigator": np.ones((2, 1, 3, 6), complex) * (2 + 3j)},
    )
    path = tmp_path / "acquisition.npz"
    save_acquisition(path, acq)
    restored = load_acquisition(path)
    assert restored.samples.dtype == samples.dtype
    np.testing.assert_array_equal(restored.samples, samples)
    np.testing.assert_array_equal(restored.acquired, acq.acquired)
    np.testing.assert_array_equal(restored.statistical_weights, acq.statistical_weights)
    np.testing.assert_array_equal(restored.readout_polarity, acq.readout_polarity)
    np.testing.assert_array_equal(
        restored.auxiliary_samples["navigator"], acq.auxiliary_samples["navigator"]
    )
    assert restored.recipe == recipe
    np.testing.assert_array_equal(restored.frame_coordinates["echo_seconds"], [0.01, 0.02])
    exp = Experiment(dims={"frame": 2, "x": 4, "y": 3})
    model = Model(
        name="synthetic",
        parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda *args: np.ones((2, 4, 3)),
        parameter_units={"level": "1"},
    )
    study = Study(
        experiments=[exp],
        datasets=[DataSet(jnp.ones((2, 4, 3)), acquisition=acq)],
        models=[model],
        names=["synthetic"],
        study_globals={"field_strength_mhz": Quantity(100.0, "MHz")},
    )
    manifest = save_study_results(
        study, [{"name": "synthetic", "model": model.name, "fields": {}}], tmp_path / "study"
    )
    result = load_study_results(manifest)["experiments"]["synthetic"]
    np.testing.assert_array_equal(result["raw_kspace"], samples)
    assert result["acquisition"].recipe == recipe
