"""Independently generated acquisition-contract and reconstruction regressions."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrax.core import run_fit
from mrax.data_struct import (
    AcquisitionData,
    DataSet,
    Experiment,
    Model,
    Quantity,
    ReadoutModel,
    ReconstructionRecipe,
    SpatialGeometry,
    Study,
)
from mrax.data_struct.storage import load_study_results, save_study_results
from mrax.io import load_acquisition, save_acquisition
from mrax.io.bruker_acquisition import read_cartesian_acquisition
from mrax.io.bruker_sampling import _density_from_jacobian_trajectory
from mrax.io.bruker_storage import decode_records
from mrax.io.definitions import DefinitionLibrary
from mrax.reco import (
    ReconstructionPipeline,
    ReconstructionStage,
    build_operator,
    fftnc,
    ifftnc,
    radial_cell_density,
    radial_profile_density,
    reconstruct_acquisition,
    solve_model_based,
    solve_variational,
)
from mrax.reco.acquisition import apply_image_shift, complete_partial_fourier
from mrax.reco.epi import (
    correct_epi_entropy,
    correct_readout_phase,
    estimate_navigator_phase,
    resample_readout,
)
from mrax.reco.gridding import grid_kaiser_bessel
from mrax.reco.processing import pad_centered_kspace


@pytest.mark.parametrize("shape", [(7,), (5, 8), (5, 6, 3)])
def test_explicit_acquisition_fft_and_adjoint_identity(shape):
    rng = np.random.default_rng(17)
    image = (rng.normal(size=(2, 3) + shape) + 1j * rng.normal(size=(2, 3) + shape)).astype("complex64")
    op = build_operator(spatial_shape=shape, independent_coils=True)
    samples = op.forward(image)
    acq = AcquisitionData(np.asarray(samples), shape, tuple(f"axis_{i}" for i in range(len(shape))))
    result = reconstruct_acquisition(acq)
    np.testing.assert_allclose(result.coil_images, image, atol=2e-6)
    probe = jnp.asarray(rng.normal(size=samples.shape) + 1j * rng.normal(size=samples.shape), dtype=samples.dtype)
    np.testing.assert_allclose(jnp.vdot(samples, probe), jnp.vdot(image, op.adjoint(probe, kspace_has_coils=True)), atol=2e-5)


def test_complex_mask_and_unnormalized_sensitivity_adjoint():
    rng = np.random.default_rng(12)
    image = rng.normal(size=(2, 7, 5)) + 1j * rng.normal(size=(2, 7, 5))
    sensitivity = rng.normal(size=(3, 7, 5)) + 1j * rng.normal(size=(3, 7, 5))
    samples = rng.normal(size=(2, 3, 7, 5)) + 1j * rng.normal(size=(2, 3, 7, 5))
    mask = np.ones((2, 1, 7, 5), complex)
    mask[0, :, ::2] = 0
    mask[1, :, 1::2] = 1j
    op = build_operator(spatial_shape=(7, 5), sensitivities=sensitivity, mask=mask)
    np.testing.assert_allclose(jnp.vdot(op.forward(image), samples),
        jnp.vdot(image, op.adjoint(samples, kspace_has_coils=True)), rtol=2e-6, atol=2e-5)


@pytest.mark.parametrize("source,target", [(5, 6), (6, 7), (5, 8), (8, 9)])
def test_zero_fill_preserves_centered_dc_across_parity(source, target):
    kspace = np.zeros((source, source), complex)
    kspace[source // 2, source // 2] = 1
    padded = pad_centered_kspace(kspace, (target, target))
    acquired = pad_centered_kspace(np.ones(kspace.shape, bool), (target, target))
    assert padded[target // 2, target // 2] == 1
    assert np.sum(acquired) == source**2


def test_valid_zero_measurements_and_nonfinite_unacquired_samples():
    samples = np.zeros((2, 1, 5, 6), complex)
    acquired = np.ones(samples.shape, bool)
    acquired[0, :, :2] = False
    samples[~acquired] = np.nan
    acq = AcquisitionData(samples, (5, 6), ("phase", "read"), acquired=acquired)
    np.testing.assert_array_equal(reconstruct_acquisition(acq).image, 0)
    with pytest.raises(ValueError, match="positive weight"):
        solve_variational(np.zeros((5, 6), complex), mask=np.zeros((5, 6)), num_steps=0)
    with pytest.raises(ValueError, match="broadcast"):
        replace(acq, acquired=np.ones((3,) + samples.shape, bool))


def test_fractional_shift_uses_signed_frequencies():
    shape = (9, 10)
    image = np.zeros((1, 1) + shape, complex)
    image[0, 0, 4, 5] = 1
    samples = fftnc(image)
    shift = (.35, -.7)
    shifted = apply_image_shift(samples, shift, spatial_ndim=2)
    from scipy.ndimage import fourier_shift
    expected = np.fft.ifftn(fourier_shift(np.fft.fftn(image[0, 0]), shift))
    from mrax.reco.variational import ifftnc
    np.testing.assert_allclose(ifftnc(shifted)[0, 0], expected, atol=2e-7)


def test_partial_fourier_preserves_measured_complex_values():
    shape = (17, 16)
    x, y = np.meshgrid(np.arange(17), np.arange(16), indexing="ij")
    image = np.exp(-((x - 8)**2 + (y - 8)**2) / 12) * np.exp(.03j * x)
    complete = fftnc(image)
    mask = np.ones(shape, bool)
    mask[:5] = False
    measured = jnp.where(mask, complete, 0)
    filled = complete_partial_fourier(measured, mask, iterations=20)
    np.testing.assert_array_equal(np.asarray(filled)[mask], np.asarray(measured)[mask])
    assert np.linalg.norm(np.asarray(filled - complete)) < np.linalg.norm(np.asarray(measured - complete))


@pytest.mark.parametrize("weighting", ["step", "ramp"])
def test_homodyne_uses_encoding_dc_and_symmetric_weight_pair(weighting):
    from mrax.reco import ifftnc
    coordinate = np.arange(33) - 16
    image = np.exp(-coordinate**2 / 25) * np.exp(.7j)
    samples = fftnc(image, axes=(-1,))
    acquired = coordinate >= -6
    reconstructed = ifftnc(complete_partial_fourier(jnp.where(acquired, samples, 0), acquired,
        spatial_ndim=1, method="homodyne", phase_window="hann", weighting=weighting), axes=(-1,))
    np.testing.assert_allclose(reconstructed, image, atol=3e-5)


def test_editable_reconstruction_plan_reuses_measured_data_and_shared_solvers(tmp_path):
    samples = np.ones((1, 1, 9, 8), np.complex64)
    acquired = np.ones((9, 8), bool)
    acquired[:2] = False
    acq = AcquisitionData(samples, (9, 8), ("phase", "read"), acquired=acquired,
        recipe=ReconstructionRecipe(partial_fourier="pocs", phase_window="hann"))
    plan = ReconstructionPipeline.from_acquisition(acq)
    np.testing.assert_array_equal(plan.run(acq).image, reconstruct_acquisition(acq).image)
    path = tmp_path / "plan.toml"
    path.write_text('schema_version=1\n[[stages]]\nkind="variational"\n'
        '[stages.options]\nnum_steps=3\noptimizer_name="fista"\nregularizer="tv"\nregularizer_weight=0.01\n'
        '[[stages]]\nkind="format"\n[[stages]]\nkind="combine"\n')
    plan = ReconstructionPipeline.from_toml(path)
    result = plan.run(acq)
    direct = reconstruct_acquisition(replace(acq, recipe=replace(acq.recipe, partial_fourier="none")),
        method="variational", num_steps=3, optimizer_name="fista", regularizer="tv", regularizer_weight=.01)
    np.testing.assert_array_equal(result.image, direct.image)
    np.testing.assert_array_equal(acq.samples, samples)
    np.testing.assert_array_equal(acq.acquired, acquired)
    assert result.history == direct.history


def test_reconstruction_plan_rejects_unknown_options_and_invalid_stage_order():
    with pytest.raises(ValueError, match="Unknown options"):
        ReconstructionStage("partial_fourier", {"invented": True})
    with pytest.raises(ValueError, match="invalid order"):
        ReconstructionPipeline((ReconstructionStage("format"), ReconstructionStage("fft"),
            ReconstructionStage("combine")))
    with pytest.raises(ValueError, match="exactly one solver"):
        ReconstructionPipeline((ReconstructionStage("fft"), ReconstructionStage("variational"),
            ReconstructionStage("format"), ReconstructionStage("combine")))


def test_readout_resampling_preserves_coordinates_and_zeros_missing_support():
    x = np.arange(5, dtype=float)
    samples = (2 * x + 1j * x)[None]
    result = resample_readout(samples, x, [-1., .5, 1.5, 5.])
    np.testing.assert_allclose(result, [[0., 1 + .5j, 3 + 1.5j, 0.]])
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_readout(samples, x[::-1], x)


def test_navigator_phase_fit_uses_hybrid_space_and_radian_units():
    x = (np.arange(48) - 24) / 48
    image = np.exp(-x**2 / .04)
    positive = fftnc(image, axes=(-1,))
    negative = fftnc(image * np.exp(1j * (.4 + 3.2 * x)), axes=(-1,))
    np.testing.assert_allclose(estimate_navigator_phase(positive, negative), [.4, 3.2], atol=2e-6)


def test_referenceless_epi_phase_fit_removes_injected_ghost_without_reference():
    from mrax.reco import ifftnc
    y, x = np.meshgrid(np.arange(48) - 24, np.arange(40) - 20, indexing="ij")
    image = (np.exp(-((y - 2)**2 / 30 + (x + 3)**2 / 20)) + .7 * np.exp(-((y + 5)**2 / 12 + (x - 5)**2 / 8)))[None]
    samples = fftnc(image)
    negative = np.broadcast_to((np.arange(48) % 2 == 0)[:, None], (48, 40))
    corrupted = correct_readout_phase(samples, negative, [-.4, -7.])
    corrected, report = correct_epi_entropy(corrupted, negative)
    before = np.linalg.norm(np.abs(ifftnc(corrupted)) - image)
    after = np.linalg.norm(np.abs(ifftnc(corrected)) - image)
    assert after < before * .01
    assert report["entropy_after"] < report["entropy_before"]
    # The data-independent ambiguity is explicit in the phase-prior option.
    shifted, _ = correct_epi_entropy(corrupted, negative, phase_prior=(np.pi, 0.))
    np.testing.assert_allclose(np.abs(ifftnc(shifted)), np.roll(image, 24, axis=-2), atol=1e-3)


def test_gridding_is_periodic_at_nyquist_boundaries():
    samples = jnp.asarray([1. + .5j])
    first = grid_kaiser_bessel(samples, np.asarray([[.49, .2]]), (8, 8), chunk_size=1)
    equivalent = grid_kaiser_bessel(samples, np.asarray([[-.51, .2]]), (8, 8), chunk_size=1)
    np.testing.assert_allclose(first, equivalent, atol=2e-6)
    assert np.any(np.asarray(first)[0] != 0)


def test_kaiser_bessel_deapodization_has_finite_analytic_limit():
    from mrax.reco.variational import _kaiser_bessel_deapodization_1d
    correction = _kaiser_bessel_deapodization_1d(8, full_size=8, offset=0,
        kernel_width=4., shape_parameter=np.pi, strength=1.)
    np.testing.assert_allclose(correction[6], np.sinh(np.pi) / np.pi, rtol=2e-6)
    assert np.isfinite(correction).all()


def test_epi_event_profiles_preserve_polarity_and_rectangular_frame_groups(monkeypatch):
    import mrax.io.bruker_epi as epi
    from mrax.io.bruker_storage import DecodedRecords
    order = np.asarray([0, 3, 1, 4, 2, 5])
    records = (order[:, None, None] + 1) * np.ones((6, 1, 24), complex)
    monkeypatch.setattr(epi, "decode_records", lambda *args, **kwargs:
        DecodedRecords(records, "synthetic", "fid", 0))
    method = {"PVM_EpiMatrix": [6, 4, 0, 0], "PVM_EpiNEchoes": 4,
        "PVM_EpiNShots": 1, "PVM_EncSteps1": [-2, -1, 0, 1],
        "PVM_SPackArrNSlices": [2]}
    acqp = {"NI": 6, "NR": 1, "ACQ_dim": 2, "ACQ_obj_order": order}
    reco = {"RECO_size": [6, 4], "RECO_ft_size": [6, 4], "RECO_rotate": [.5, .5]}
    visu = {"VisuCoreDim": 2, "VisuCoreSize": [6, 4], "VisuCoreFrameCount": 6,
        "VisuFGOrderDesc": [[3, "FG_ECHO", "", 0, 0], [2, "FG_SLICE", "", 0, 0]]}
    acq, report = epi.read_epi_acquisition(None, method, acqp, reco, visu)
    np.testing.assert_array_equal(acq.samples[:, 0, 0, 0], np.arange(1, 7))
    np.testing.assert_array_equal(np.broadcast_to(acq.readout_polarity, acq.samples.shape)[0, 0, :, 0], [-1, 1, -1, 1])
    assert acq.recipe.frame_permutations == ((1, 0),) * 6
    assert acq.recipe.output_frame_indices == (0, 3, 1, 4, 2, 5)
    assert reconstruct_acquisition(acq).image.shape == (3, 6, 4, 2)
    assert report["event_profile"] == "epi.alternating_readouts"


@pytest.mark.parametrize("ndim", [2, 3])
def test_radial_cells_cover_ball_include_origin_and_preserve_geometric_units(ndim):
    radius = np.linspace(0, .7, 8)
    direction = np.asarray([[1., 0., 0.], [-1., 0., 0.]])[:, :ndim]
    trajectory = (direction[:, None] * radius[None, :, None]).reshape((-1, ndim))
    weights = radial_cell_density(trajectory, points_per_readout=8)
    assert np.all(weights.reshape((-1, 8))[:, 0] > 0)
    expected = np.pi * .75**2 if ndim == 2 else 4 * np.pi / 3 * .75**3
    np.testing.assert_allclose(weights.sum(), expected)
    duplicated = radial_cell_density(np.concatenate((trajectory, trajectory)), points_per_readout=8)
    np.testing.assert_allclose(duplicated.sum(), expected)


def test_radial_cells_account_for_irregular_spacing_and_explicit_normalization():
    radius = np.asarray([0., .01, .06, .2, .5])
    trajectory = np.stack((radius, np.zeros_like(radius)), axis=-1)
    weights = radial_cell_density(trajectory, points_per_readout=5, target_sum=1.)
    np.testing.assert_allclose(weights.sum(), 1.)
    assert weights[-1] / weights[-2] > radius[-1] / radius[-2]
    with pytest.raises(ValueError, match="strictly increasing"):
        radial_cell_density(trajectory[::-1], points_per_readout=5)


@pytest.mark.parametrize("algorithm", ["fista", "split_bregman"])
def test_complex_tv_solver_preserves_phase_and_reports_full_final_objective(algorithm):
    target = jnp.asarray([[1j, 1j], [-1j, -1j]])
    op = build_operator(spatial_shape=(2, 2))
    result, history = solve_variational(op.forward(target), spatial_shape=(2, 2),
        initial_image=jnp.zeros((2, 2)), regularizer="tv", regularizer_weight=.1,
        optimizer_name=algorithm, num_steps=400, return_history=True)
    np.testing.assert_allclose(result, target * .9, atol=5e-4)
    expected = .5 * jnp.mean(jnp.abs(result - target)**2) + .1 * jnp.mean(jnp.abs(jnp.diff(result, axis=0))) / 2
    np.testing.assert_allclose(history[-1], expected, atol=2e-6)


@pytest.mark.parametrize("byte_order", ["little", "big"])
def test_joint_receiver_block_padding_and_echo_slice_layout(tmp_path, byte_order):
    read, phase, receivers, slices, echoes = 8, 6, 2, 3, 2
    objects = slices * echoes
    order = [0, 1, 4, 5, 2, 3]
    lines = []
    truth = np.empty((objects, receivers, phase, read), complex)
    dtype = np.dtype("i4").newbyteorder("<" if byte_order == "little" else ">")
    for ky in range(phase):
        for obj in order:
            row = np.zeros(256, dtype=dtype)
            for coil in range(receivers):
                signal = np.arange(read) + 100 * obj + 10 * coil + 1000 * ky
                values = signal + 1j * (signal + 1)
                truth[obj, coil, ky] = values
                start = coil * read * 2
                row[start:start + 2 * read:2] = signal
                row[start + 1:start + 2 * read:2] = signal + 1
            lines.append(row)
    np.stack(lines).astype(dtype).tofile(tmp_path / "fid")
    method = {"PVM_EncNReceivers": receivers, "PVM_EncMatrix": [read, phase],
        "PVM_NEchoImages": echoes, "PVM_SPackArrNSlices": slices}
    acqp = {"GO_raw_data_format": "GO_32BIT_SGN_INT", "ACQ_byte_order": byte_order,
        "GO_block_size": "Standard_KBlock_Format", "ACQ_dim": 2,
        "ACQ_size": [read * 2, phase], "NI": objects, "NR": 1, "ACQ_obj_order": order}
    reco = {"RECO_ft_size": [read, phase], "RECO_size": [read, phase], "RECO_rotate": [.5, .5]}
    acq, report = read_cartesian_acquisition(tmp_path, method, acqp, reco, {"VisuCoreDim": 2})
    np.testing.assert_array_equal(acq.samples, truth)
    assert report["padding_bytes_per_record"] == 1024 - receivers * read * 8
    image = reconstruct_acquisition(acq).image
    assert image.shape == (echoes, phase, read, slices)
    with (tmp_path / "fid").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="scalar words"):
        decode_records(tmp_path, method, acqp)


def test_definition_library_rejects_ambiguity_and_executable_configuration(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text('schema_version=1\n[[profiles]]\nname="other"\nrole="storage"\nmatch={stream="fid",block_format="continuous"}\nsettings={reader="fid_records",alignment_bytes=1}\n')
    library = DefinitionLibrary([path])
    features = {"stream": "fid", "block_format": "continuous"}
    with pytest.raises(ValueError, match="Ambiguous"):
        library.select("storage", features)
    assert library.select("storage", features, profile="other").name == "other"
    path.write_text('schema_version=1\nexecute="arbitrary Python"\n')
    with pytest.raises(ValueError, match="Unsupported"):
        DefinitionLibrary([path])


def test_continuous_negative_scan_shift_selects_tail_of_each_receiver(tmp_path):
    blocks = np.full((2, 2, 20), 9999, dtype="<i4")
    expected = np.arange(16).reshape((2, 2, 4)) + 1j
    blocks[..., -8::2] = expected.real
    blocks[..., -7::2] = expected.imag
    blocks.tofile(tmp_path / "fid")
    records = decode_records(tmp_path, {"PVM_EncNReceivers": 2, "PVM_DigNp": 4},
        {"GO_raw_data_format": "GO_32BIT_SGN_INT", "GO_block_size": "continuous",
         "ACQ_size": 20, "ACQ_scan_shift": -1, "ACQ_scan_size": "ACQ_phase_factor_scans"})
    np.testing.assert_array_equal(records.samples, expected)
    complete = decode_records(tmp_path, {"PVM_EncNReceivers": 2, "PVM_DigNp": 4},
        {"GO_raw_data_format": "GO_32BIT_SGN_INT", "GO_block_size": "continuous",
         "ACQ_size": 20, "ACQ_scan_shift": -1, "ACQ_scan_size": "ACQ_phase_factor_scans"}, retain_record=True)
    assert complete.samples.shape == (2, 2, 10)
    np.testing.assert_array_equal(complete.samples[..., -4:], expected)


def test_rotated_spiral_jacobian_uses_radial_growth():
    t = np.linspace(.05, 1, 2001)
    angle = 20 * t
    trajectory = np.stack((t * np.cos(angle), t * np.sin(angle)), axis=-1)
    weights = _density_from_jacobian_trajectory(trajectory, points_per_interleave=len(t),
        interleaves=1, spatial_shape=(16, 16))
    expected = t * (256 / np.sum(t))
    np.testing.assert_allclose(weights[2:-2], expected[2:-2], rtol=2e-4)


def test_gridding_retains_density_weights_instead_of_dividing_them_out():
    trajectory = jnp.zeros((2, 2))
    samples = jnp.asarray([[1 + 2j, 3 - 1j]])
    grid = grid_kaiser_bessel(samples, trajectory, (8, 8), density=jnp.asarray([1., 2.]), chunk_size=2)
    np.testing.assert_allclose(grid[0, 4, 4], 7, atol=1e-6)
    doubled = grid_kaiser_bessel(samples, trajectory, (8, 8), density=jnp.asarray([2., 4.]), chunk_size=2)
    np.testing.assert_allclose(doubled, 2 * grid, atol=1e-6)


def test_full_sample_model_based_and_image_fit_agree_with_roi_outside_signal():
    exp = Experiment(dims={"frame": 1, "x": 4, "y": 3})
    model = Model(name="synthetic_linear", parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda p, offsets, observables=None: p["level"][None], parameter_units={"level": "1"},
        parameter_specs={"level": {"transform": "identity"}})
    image = jnp.arange(12, dtype=jnp.float32).reshape((1, 4, 3)) / 12 + 1
    roi = np.zeros((4, 3))
    roi[1:3, 1:] = 1
    op = build_operator(spatial_shape=(4, 3))
    fields, _ = run_fit(exp, model, DataSet(image), mask=roi, num_steps=1000)
    direct = solve_model_based(op.forward(image), experiment=exp, model=model,
        roi_mask=roi, spatial_shape=(4, 3), num_steps=1000)
    np.testing.assert_allclose(direct["level"].values[roi > 0], fields["level"].values[roi > 0], atol=1e-4)


def test_acquisition_archive_and_study_roundtrip_preserve_encoding_metadata(tmp_path):
    samples = np.ones((2, 1, 4, 3), dtype=np.complex128) * (1 + 2j)
    recipe = ReconstructionRecipe(frame_image_shifts=((.25, -.1), (.5, -.2)),
        frame_permutations=((0, 1), (0, 1)))
    acq = AcquisitionData(samples, (4, 3), ("phase", "read"),
        acquired=np.ones((2, 1, 4, 3), bool),
        statistical_weights=np.full((4, 3), .5),
        frame_coordinates={"echo_seconds": np.asarray([.01, .02])}, recipe=recipe,
        readout_polarity=np.asarray([1, -1, 1, -1])[:, None],
        auxiliary_samples={"navigator": np.ones((2, 1, 3, 6), complex) * (2 + 3j)})
    path = tmp_path / "acquisition.npz"
    save_acquisition(path, acq)
    restored = load_acquisition(path)
    assert restored.samples.dtype == samples.dtype
    np.testing.assert_array_equal(restored.samples, samples)
    np.testing.assert_array_equal(restored.acquired, acq.acquired)
    np.testing.assert_array_equal(restored.statistical_weights, acq.statistical_weights)
    np.testing.assert_array_equal(restored.readout_polarity, acq.readout_polarity)
    np.testing.assert_array_equal(restored.auxiliary_samples["navigator"], acq.auxiliary_samples["navigator"])
    assert restored.recipe == recipe
    np.testing.assert_array_equal(restored.frame_coordinates["echo_seconds"], [.01, .02])
    exp = Experiment(dims={"frame": 2, "x": 4, "y": 3})
    model = Model(name="synthetic", parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
                  model_apply=lambda *args: np.ones((2, 4, 3)), parameter_units={"level": "1"})
    study = Study(experiments=[exp], datasets=[DataSet(jnp.ones((2, 4, 3)), acquisition=acq)],
        models=[model], names=["synthetic"], study_globals={"field_strength_mhz": Quantity(100., 'MHz')})
    manifest = save_study_results(study, [{"name": "synthetic", "model": model.name, "fields": {}}], tmp_path / "study")
    result = load_study_results(manifest)["experiments"]["synthetic"]
    np.testing.assert_array_equal(result["raw_kspace"], samples)
    assert result["acquisition"].recipe == recipe


def test_import_only_facade_preserves_image_import_without_reconstructing(tmp_path, monkeypatch):
    from test_import_workflows import _write_minimal_bruker_scan

    import mrax.reco.acquisition as reconstruction
    from mrax.io import import_bruker_experiment

    _write_minimal_bruker_scan(tmp_path)
    folder = tmp_path / "1"
    (folder / "method").write_text('##$Method=<MSME>\n##$PVM_EncMatrix=( 2 )\n2 2\n##$PVM_EncNReceivers=1\n')
    (folder / "acqp").write_text('##$ACQ_dim=2\n##$ACQ_size=( 2 )\n4 2\n##$NI=1\n##$NR=1\n##$GO_raw_data_format=GO_32BIT_SGN_INT\n##$GO_block_size=Standard_KBlock_Format\n')
    (folder / "pdata/1/reco").write_text('##$RECO_ft_size=( 2 )\n2 2\n##$RECO_size=( 2 )\n2 2\n')
    blocks = np.zeros((2, 256), dtype="<i4")
    blocks[:, :4] = [[1, 2, 3, 4], [5, 6, 7, 8]]
    blocks.tofile(folder / "fid")

    def unexpected(*args, **kwargs):
        raise AssertionError("Import must not reconstruct")

    monkeypatch.setattr(reconstruction, "reconstruct_acquisition", unexpected)
    exp, dataset, model, metadata = import_bruker_experiment(folder)
    exp.validate(dataset)
    np.testing.assert_array_equal(dataset.data, np.asarray([[[10, 12], [14, 16]]]))
    assert dataset.acquisition.samples.shape == (1, 1, 2, 2)
    assert metadata["raw_import"]["status"] == "loaded"
    assert metadata["raw_import"]["reconstruction_status"] == "not_run"


def test_unified_reconstruction_keeps_image_units_and_unmasked_forward_image():
    exp = Experiment(dims={"frame": 1, "x": 4, "y": 3})
    model = Model(name="synthetic_linear", parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda p, offsets, observables=None: p["level"][None], parameter_units={"level": "1"})
    target = jnp.arange(12, dtype=jnp.float32).reshape((1, 4, 3)) / 12 + 1
    scale = .25
    samples = build_operator(spatial_shape=(4, 3)).forward(target) / scale
    acq = AcquisitionData(np.asarray(samples), (4, 3), ("phase", "read"),
        recipe=ReconstructionRecipe(output_scale=scale))
    direct = reconstruct_acquisition(acq)
    variational = reconstruct_acquisition(acq, method="variational", regularizer="none",
        optimizer_name="fista", num_steps=2)
    roi = np.zeros((4, 3))
    roi[1:3, 1:] = 1
    modeled = reconstruct_acquisition(acq, method="model_based", experiment=exp,
        model=model, roi_mask=roi, num_steps=1000)
    np.testing.assert_allclose(direct.image, target, atol=1e-6)
    np.testing.assert_allclose(variational.image, target, atol=1e-6)
    np.testing.assert_allclose(modeled.image, target, atol=1e-4)
    np.testing.assert_array_equal(np.asarray(modeled.parameters["level"].values)[roi == 0], 0)


def test_sampled_shift_uses_coordinates_and_frame_specific_displacements():
    trajectory = np.asarray([[0., 0.], [.125, -.25], [-.4, .2]])
    samples = np.ones((2, 3, 3), complex)
    shifts = np.asarray([[1.25, -.75], [-2., .5]])
    shifted = apply_image_shift(samples, shifts, spatial_ndim=2, trajectory=trajectory)
    expected = samples * np.exp(-2j * np.pi * (shifts @ trajectory.T))[:, None]
    np.testing.assert_allclose(shifted, expected, atol=5e-7)
    np.testing.assert_allclose(apply_image_shift(shifted, -shifts, spatial_ndim=2,
        trajectory=trajectory), samples, atol=5e-7)


def test_elliptical_support_uses_the_centered_fourier_origin():
    from mrax.reco.variational import elliptical_support_mask

    shape = (8, 10, 6)
    coordinates = np.indices(shape)
    radius = sum(((axis - size // 2) / (size / 2))**2
                 for axis, size in zip(coordinates, shape, strict=True))
    np.testing.assert_array_equal(elliptical_support_mask(shape), radius <= 1)


def test_delayed_readout_inverse_uses_both_polarities_and_preserves_measurements(tmp_path):
    from mrax.reco.readout import invert_readout_profiles

    rng = np.random.default_rng(123)
    size, profiles = 12, 3
    angles = np.arange(profiles) * np.pi / profiles
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    frequency = (np.arange(size) - size // 2) / size
    target = (directions[:, None] * frequency[None, :, None]).reshape((-1, 2))
    time = (np.arange(20) + .4) / (2 * size)
    coordinates = np.concatenate((time, -time))
    positions = np.arange(size) - size // 2
    matrix = np.exp(-2j * np.pi * coordinates[:, None] * positions) / np.sqrt(size)
    projection = rng.normal(size=(1, 2, profiles, size)) + 1j * rng.normal(size=(1, 2, profiles, size))
    measured = np.einsum("...pn,mn->...pm", projection, matrix).reshape((1, 2, -1)).astype("complex64")
    trajectory = (directions[:, None] * coordinates[None, :, None]).reshape((-1, 2))
    readout = ReadoutModel(coordinates, size, target, size, 1e-7)
    synthesized, report = invert_readout_profiles(measured, readout)
    expected = fftnc(jnp.asarray(projection), axes=(-1,)).reshape((1, 2, -1))
    np.testing.assert_allclose(synthesized, expected, atol=1e-6)
    assert report["rank"] == size and report["retained_condition"] < 10
    acq = AcquisitionData(measured, (size, size), ("x", "y"), kind="sampled",
        trajectory=trajectory, readout_model=readout,
        recipe=ReconstructionRecipe(profile_inversion=True, image_shift=(.5, -.25)))
    plan = ReconstructionPipeline.from_acquisition(acq)
    assert plan.stages[0].kind == "profile_inversion"
    result = plan.run(acq)
    from mrax.reco.density import radial_profile_density

    control = replace(acq, samples=np.asarray(expected), trajectory=target,
        density=radial_profile_density(target, points_per_readout=size, center_out=False),
        readout_model=None, recipe=replace(acq.recipe, profile_inversion=False))
    np.testing.assert_allclose(result.image, reconstruct_acquisition(control).image, atol=2e-6)
    archive = tmp_path / "readout.npz"
    save_acquisition(archive, acq)
    loaded = load_acquisition(archive)
    np.testing.assert_array_equal(loaded.samples, measured)
    np.testing.assert_array_equal(loaded.readout_model.target_trajectory, target)
    np.testing.assert_array_equal(loaded.readout_model.coordinates, coordinates)


def test_model_based_layout_preserves_slices_repeats_and_frame_sensitivities():
    from mrax.reco.layout import image_to_encoding

    spatial = (4, 3, 2)
    frames, repeats, slices = 2, 2, 2
    native_frames = frames * repeats * slices
    recipe = ReconstructionRecipe(output_permutation=(1, 0), stack_slices=slices,
        average_repeats=repeats, coil_combine="sense")
    acq = AcquisitionData(np.zeros((native_frames, 2, 3, 4), "complex64"), (3, 4),
        ("phase", "read"), recipe=recipe)
    field = jnp.arange(np.prod(spatial), dtype=jnp.float32).reshape(spatial) / 24 + .5
    factor = jnp.asarray([.7, 1.3])
    image = field[None] * factor[:, None, None, None]
    native = image_to_encoding(image, acq)
    phase = jnp.arange(native_frames)[:, None, None] / 10
    sensitivity = jnp.stack([jnp.ones_like(native), jnp.exp(1j * phase) * jnp.ones_like(native)], axis=1) / jnp.sqrt(2.)
    operator = build_operator(spatial_shape=(3, 4), sensitivities=sensitivity)
    samples = operator.forward(native)
    rng = np.random.default_rng(21)
    probe = jnp.asarray(rng.normal(size=samples.shape) + 1j * rng.normal(size=samples.shape))
    np.testing.assert_allclose(jnp.vdot(samples, probe), jnp.vdot(native,
        operator.adjoint(probe, kspace_has_coils=True)), atol=2e-5)
    acq = replace(acq, samples=np.asarray(samples))
    direct = reconstruct_acquisition(acq, sensitivities=sensitivity)
    np.testing.assert_allclose(direct.image, image, atol=1e-6)
    experiment = Experiment(dims={"frame": frames, "x": 4, "y": 3, "z": 2})
    model = Model(name="synthetic_layout", parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda p, offsets, observables=None: p["level"][None] * factor[:, None, None, None],
        parameter_units={"level": "1"}, parameter_constant_dims={"level": ("frame",)})
    result = reconstruct_acquisition(acq, method="model_based", experiment=experiment,
        model=model, sensitivities=sensitivity, num_steps=800)
    np.testing.assert_allclose(result.image, image, atol=2e-4)
    np.testing.assert_allclose(result.parameters["level"].values,
        np.broadcast_to(field, result.parameters["level"].values.shape), atol=2e-4)
    selected = acq.select_frames([1])
    np.testing.assert_allclose(reconstruct_acquisition(selected,
        sensitivities=sensitivity[4:]).image, image[1:], atol=1e-6)


def test_nonuniform_readout_operator_has_true_adjoint_and_measured_sample_solution():
    rng = np.random.default_rng(29)
    shape = (8, 10)
    frequency = np.fft.fftshift(np.fft.fftfreq(shape[-1]))
    coordinates = np.stack((frequency + .003 * np.sin(np.arange(10)),
                            frequency + .004 * np.cos(np.arange(10))))
    polarity = np.broadcast_to(np.where(np.arange(8) % 2, -1, 1)[None, None, :, None], (2, 1) + shape)
    phase = (.2, 1.3)
    sensitivity = np.stack((np.ones(shape), 1j * np.ones(shape))) / np.sqrt(2)
    target = (rng.normal(size=(2,) + shape) + 1j * rng.normal(size=(2,) + shape)).astype("complex64")
    operator = build_operator(spatial_shape=shape, sensitivities=sensitivity,
        readout_coordinates=coordinates, readout_polarity=polarity, readout_phase=np.asarray(phase))
    samples = operator.forward(target)
    probe = jnp.asarray(rng.normal(size=samples.shape) + 1j * rng.normal(size=samples.shape))
    np.testing.assert_allclose(jnp.vdot(samples, probe), jnp.vdot(target,
        operator.adjoint(probe, kspace_has_coils=True)), atol=3e-5)
    mask = np.ones((2, 1) + shape)
    mask[0, :, :2] = 0
    masked = build_operator(spatial_shape=shape, sensitivities=sensitivity, mask=mask,
        readout_coordinates=coordinates, readout_polarity=polarity, readout_phase=np.asarray(phase))
    np.testing.assert_allclose(jnp.vdot(masked.forward(target), probe), jnp.vdot(target,
        masked.adjoint(probe, kspace_has_coils=True)), atol=3e-5)
    result = solve_variational(samples, spatial_shape=shape, sensitivities=sensitivity,
        readout_coordinates=coordinates, readout_polarity=polarity, readout_phase=np.asarray(phase),
        regularizer="none", optimizer_name="fista", num_steps=40)
    np.testing.assert_allclose(result, target, atol=5e-5)
    acquisition = AcquisitionData(np.asarray(samples), shape, ("phase", "read"),
        readout_coordinates=coordinates, readout_polarity=polarity,
        recipe=ReconstructionRecipe(epi_phase_correction="fixed", epi_phase_prior=phase, coil_combine="sense"))
    solved = ReconstructionPipeline.from_acquisition(acquisition, method="variational",
        regularizer="none", optimizer_name="fista", num_steps=40).run(acquisition, sensitivities=sensitivity)
    np.testing.assert_allclose(solved.image, target, atol=5e-5)


@pytest.mark.parametrize("ndim", [2, 3])
def test_radial_profile_density_integrates_normalized_volume_without_angular_average(ndim):
    points = 32
    directions = np.eye(ndim)
    radius = (np.arange(points) + .5) / (2 * points)
    trajectory = (directions[:, None] * radius[None, :, None]).reshape((-1, ndim))
    weights = radial_profile_density(trajectory, points_per_readout=points)
    np.testing.assert_allclose(weights.reshape((-1, points)).sum(axis=-1), 1., atol=1e-12)
    signed = (np.arange(2 * points) + .5 - points) / (2 * points)
    full = (directions[:, None] * signed[None, :, None]).reshape((-1, ndim))
    weights = radial_profile_density(full, points_per_readout=2 * points, center_out=False)
    np.testing.assert_allclose(weights.reshape((-1, 2 * points)).sum(axis=-1), 2., atol=1e-12)


def test_inverse_fourier_deapodization_retains_kernel_normalization_and_crop_origin():
    from scipy.integrate import quad
    from scipy.special import i0

    from mrax.reco.variational import _kaiser_bessel_deapodization_1d

    width, beta, size = 3., 6., 32
    mass = quad(lambda u: i0(beta * np.sqrt(max(0, 1 - (2 * u / width)**2))) / i0(beta), -width / 2, width / 2)[0]
    dc = _kaiser_bessel_deapodization_1d(1, full_size=size, offset=size // 2,
        kernel_width=width, shape_parameter=beta, strength=1., normalization="inverse_fft")
    np.testing.assert_allclose(dc, size / mass, rtol=2e-6)
    kwargs = dict(full_size=size, kernel_width=width, shape_parameter=beta, strength=1.)
    full = _kaiser_bessel_deapodization_1d(size, offset=0, **kwargs)
    crop = _kaiser_bessel_deapodization_1d(4, offset=3, **kwargs)
    np.testing.assert_allclose(crop, full[3:7], rtol=1e-6)


def test_radial_gridding_absolute_units_follow_fourier_and_profile_conventions():
    size, rays, points = 16, 64, 64
    angle = np.arange(rays) * 2 * np.pi / rays
    directions = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    radius = (np.arange(points) + .5) / (2 * points)
    trajectory = (directions[:, None] * radius[None, :, None]).reshape((-1, 2))
    position = np.stack(np.meshgrid(np.arange(size) - size // 2,
        np.arange(size) - size // 2, indexing="ij"), axis=-1).reshape((-1, 2))
    image = np.exp(-np.sum(position**2, axis=-1) / 8).reshape((size, size))
    samples = np.exp(-2j * np.pi * (trajectory @ position.T)) @ image.ravel()
    factor = rays * 8 / (2 * np.pi)
    recipe = ReconstructionRecipe(gridding_shape=(2 * size, 2 * size), output_shape=(size, size),
        gridding_kernel_width=3., gridding_shape_parameter=6., deapodize=True,
        deapodization_normalization="inverse_fft", output_scale=1 / (2 * size),
        encoding_scale=factor / size)
    acq = AcquisitionData(samples.astype("complex64")[None, None], (size, size), ("x", "y"),
        kind="sampled", trajectory=trajectory,
        density=radial_profile_density(trajectory, points_per_readout=points), recipe=recipe)
    recovered = np.asarray(reconstruct_acquisition(acq).image)[0]
    assert np.linalg.norm(recovered - factor * image) / np.linalg.norm(factor * image) < .005
    assert abs(recovered[size // 2, size // 2] / factor - 1) < .002


def test_frequency_product_preserves_signed_hz_coil_phase_and_echo_pair_selection():
    from mrax.reco.products import frequency_from_echoes

    shape, spacing = (12, 14, 10), .003
    x, y, z = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape), indexing="ij")
    field = 300 * x + 60 * y - 20 * z
    amplitude = np.exp(-(x*x + y*y + z*z))
    coil_phase = np.asarray([.3, -1.2, 2.])[:, None, None, None]
    first = amplitude[None] * np.exp(1j * coil_phase) / np.sqrt(3)
    second = first * np.exp(2j * np.pi * field[None] * spacing)
    coils = np.stack((first, second)).astype("complex64")
    mask = x*x + y*y + z*z < .8
    frequency, reliable = frequency_from_echoes(coils, spacing, spatial_ndim=3, mask=mask)
    np.testing.assert_allclose(np.asarray(frequency)[0][mask], field[mask], atol=.002)
    np.testing.assert_array_equal(np.asarray(frequency)[0][~mask], 0)
    np.testing.assert_array_equal(reliable[0], mask)
    acquisition = AcquisitionData(np.asarray(fftnc(jnp.asarray(coils), axes=(-3, -2, -1))),
        shape, ("x", "y", "z"), recipe=ReconstructionRecipe(output_product="frequency_hz", echo_spacing_seconds=spacing))
    result = ReconstructionPipeline.from_acquisition(acquisition).run(acquisition)
    np.testing.assert_allclose(result.image[0], field, atol=.003)
    assert acquisition.select_frames([0]).samples.shape[0] == 2
    with pytest.raises(ValueError, match="frame"):
        acquisition.select_frames([1])


def test_import_definitions_support_explicit_version_and_setting_predicates(tmp_path):
    document = tmp_path / "definitions.toml"
    document.write_text('''schema_version = 1
[[profiles]]
name = "synthetic.feature_range"
role = "synthetic_role"
match = {software = {prefix = "System-", version_at_least = "2.1", version_before = "3.0"}, count = {at_least = 2, less_than = 8}, absent = {exists = false}}
settings = {resolver = "synthetic"}
''')
    library = DefinitionLibrary([document])
    assert library.select("synthetic_role", {"software": "System-2.4.1", "count": 3}).name == "synthetic.feature_range"
    for features in [{"software": "System-3.0", "count": 3}, {"software": "System-2.4", "count": 8},
                     {"software": "System-2.4", "count": 3, "absent": 0}]:
        with pytest.raises(ValueError, match="No supported"):
            library.select("synthetic_role", features)


@pytest.mark.parametrize("transpose", [0, 1, 2, 3])
def test_cartesian_volume_transposition_respects_declared_coordinate_pair(monkeypatch, transpose):
    import mrax.io.bruker_acquisition as reader
    from mrax.io.bruker_storage import DecodedRecords

    rng = np.random.default_rng(59)
    shape = (3, 4, 5)
    native = (rng.normal(size=(2, 1) + shape) + 1j * rng.normal(size=(2, 1) + shape)).astype("complex64")
    grid = np.asarray(fftnc(jnp.asarray(native), axes=(-3, -2, -1))) * np.sqrt(np.prod(shape))
    records = grid.transpose((2, 3, 0, 1, 4)).reshape((-1, 1, 5))
    monkeypatch.setattr(reader, "decode_records", lambda *args, **kwargs: DecodedRecords(records, "synthetic", "fid", 0))
    method = {"PVM_EncMatrix": [5, 4, 3], "PVM_EncSteps1": [-2, -1, 0, 1], "PVM_EncSteps2": [-1, 0, 1]}
    acqp = {"ACQ_dim": 3, "NI": 2, "NR": 1}
    reco = {"RECO_size": [5, 4, 3], "RECO_ft_size": [5, 4, 3], "RECO_transposition": transpose}
    acq, _ = reader.read_cartesian_acquisition(".", method, acqp, reco, {"VisuCoreDim": 3})
    expected = native.transpose((0, 1, 4, 3, 2))
    if transpose:
        expected = np.swapaxes(expected, transpose - 1 + 2, (transpose % 3) + 2)
    result = reconstruct_acquisition(acq)
    np.testing.assert_allclose(result.coil_images, expected, atol=2e-6)


def test_quality_guided_phase_paths_ignore_noisy_masked_boundaries():
    from mrax.reco.phase_quality import quality_guided_unwrap

    rng = np.random.default_rng(67)
    x, y = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 43), indexing="ij")
    truth = 6 * x + 2 * y
    magnitude = np.exp(-2 * (x*x + y*y))
    noisy = truth + rng.normal(size=truth.shape) * .025 / magnitude
    wrapped = np.angle(np.exp(1j * noisy))
    mask = magnitude > .15
    recovered = np.asarray(quality_guided_unwrap(wrapped, spatial_ndim=2, mask=mask, weights=magnitude))
    reliable = magnitude > .5
    np.testing.assert_allclose(recovered[reliable], noisy[reliable], atol=1e-6)
    np.testing.assert_array_equal(recovered[~mask], 0)


@pytest.mark.parametrize("with_sensitivities", [False, True])
def test_variational_auto_step_is_independent_of_measured_signal_scale(with_sensitivities):
    rng = np.random.default_rng(71)
    shape = (9, 10)
    target = (rng.normal(size=(2,) + shape) + 1j * rng.normal(size=(2,) + shape)).astype("complex64")
    sensitivities = None
    if with_sensitivities:
        sensitivities = np.stack((np.ones(shape), 1j * (1 + np.arange(10)[None] / 20) * np.ones(shape))).astype("complex64")
    operator = build_operator(spatial_shape=shape, sensitivities=sensitivities)
    samples = operator.forward(target)
    mask = np.ones(samples.shape, bool)
    mask[..., shape[0] // 2, :] = False
    recovered = []
    for scale in (1., 1e8):
        result = solve_variational(samples * scale, spatial_shape=shape,
            sensitivities=sensitivities, mask=mask, initial_image=jnp.zeros_like(target),
            regularizer="none", optimizer_name="fista", num_steps=4)
        recovered.append(np.asarray(result) / scale)
    np.testing.assert_allclose(recovered[0], recovered[1], atol=3e-6, rtol=1e-5)


def test_direct_readout_quadrature_retains_uniform_pixel_noise_variance():
    from mrax.reco.epi import regrid_readouts

    size = 14
    coordinates = np.stack((np.linspace(-.45, .45, size)**3 / .45**2,
                            np.linspace(-.46, .46, size)**3 / .46**2))
    basis = np.eye(size, dtype="complex64")[:, None, None, :]
    polarity = np.ones_like(basis, dtype=np.int8)
    preview, support = regrid_readouts(basis, coordinates, polarity,
        np.ones_like(basis, bool), kernel="adjoint")
    transformed = np.asarray(jnp.fft.fftshift(jnp.fft.ifft(jnp.fft.ifftshift(preview, axes=-1), axis=-1, norm="ortho"), axes=-1))[:, 0, 0]
    pixel_variance = np.sum(abs(transformed)**2, axis=0)
    np.testing.assert_allclose(pixel_variance, np.full(size, pixel_variance.mean()), rtol=2e-6)
    assert support.all()


def test_nearest_gridding_preserves_density_and_uses_periodic_nearest_coordinates():
    from mrax.reco import regrid_nearest

    trajectory = jnp.asarray([[.49, .49], [-.5, -.5], [0., 0.]])
    samples = jnp.asarray([1., 2., 3.])
    grid, weights = regrid_nearest(samples, trajectory, (8, 8), density=jnp.asarray([2., 4., 1.]))
    assert grid[0, 0] == 10 and weights[0, 0] == 6
    assert grid[4, 4] == 3


def test_model_based_honors_statistical_weights_and_rejects_soft_sampling_masks():
    shape = (3, 4)
    pattern = jnp.arange(12, dtype=jnp.float32).reshape(shape) / 12 + .2
    experiment = Experiment(dims={"frame": 1, "x": 3, "y": 4})
    model = Model(name="synthetic_weighted_encoding",
        parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda p, offsets, observables=None: p["level"][None] * pattern,
        parameter_units={"level": "1"}, parameter_constant_dims={"level": ("x", "y")})
    encoded = fftnc(pattern)
    noise = jnp.zeros_like(encoded).at[1, 1].set(1. + 2j).at[1, 2].set(.4)
    samples = (2 * encoded + noise)[None, None]
    weights = jnp.ones(shape).at[1, 1].set(10.)
    expected = jnp.real(jnp.sum(weights * jnp.conj(encoded) * samples[0, 0])) / jnp.sum(weights * jnp.abs(encoded)**2)
    acq = AcquisitionData(np.asarray(samples), shape, ("x", "y"), statistical_weights=np.asarray(weights))
    result = reconstruct_acquisition(acq, method="model_based", experiment=experiment,
        model=model, num_steps=1000)
    np.testing.assert_allclose(result.parameters["level"].values, expected, atol=2e-5)
    with pytest.raises(ValueError, match="binary"):
        solve_model_based(samples, experiment=experiment, model=model, mask=jnp.full(shape, .5), num_steps=0)


def test_sampled_model_based_keeps_missing_support_out_of_the_objective():
    with jax.default_device(jax.devices("cpu")[0]):
        _check_sampled_model_based_support()


def _check_sampled_model_based_support():
    shape = (3, 4)
    pattern = jnp.arange(12, dtype=jnp.float32).reshape(shape) / 12 + .2
    trajectory = np.asarray([[0., 0.], [.1, .2], [-.3, .2], [.1, -.3], [-.2, -.1]], "float32")
    operator = build_operator(spatial_shape=shape, trajectory=trajectory)
    samples = operator.forward(2 * pattern[None])
    mask = np.asarray([True, True, True, False, True])
    samples = samples.at[..., 3].set(1e6 + 1e6j)
    experiment = Experiment(dims={"frame": 1, "x": 3, "y": 4})
    model = Model(name="synthetic_sampled_mask",
        parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda p, offsets, observables=None: p["level"][None] * pattern,
        parameter_units={"level": "1"}, parameter_constant_dims={"level": ("x", "y")})
    acquisition = AcquisitionData(np.asarray(samples), shape, ("x", "y"), kind="sampled",
        trajectory=trajectory, acquired=mask)
    result = reconstruct_acquisition(acquisition, method="model_based", experiment=experiment,
        model=model, num_steps=1000)
    np.testing.assert_allclose(result.parameters["level"].values, 2., atol=2e-5)


def test_continuous_digitizer_selection_retains_inactive_adc_samples(tmp_path):
    values = np.asarray([1., 2., 3., 4., 5., 6., 7., 8.], dtype="<f4")
    values.tofile(tmp_path / "fid")
    method = {"PVM_EncNReceivers": 1, "PVM_DigNp": 2}
    acqp = {"ACQ_size": [8], "GO_raw_data_format": "GO_32BIT_FLOAT", "GO_block_size": "continuous",
        "ACQ_scan_size": "ACQ_phase_factor_scans", "ACQ_scan_shift": -1}
    decoded = decode_records(tmp_path, method, acqp)
    np.testing.assert_array_equal(decoded.samples, np.asarray([[[5 + 6j, 7 + 8j]]]))
    np.testing.assert_array_equal(decoded.auxiliary_samples["inactive_adc"], np.asarray([[[1 + 2j, 3 + 4j]]]))


def test_dataset_uses_declared_geometry_for_three_dimensional_spatial_shape():
    data = DataSet(jnp.zeros((2, 3, 4, 5)), spatial_geometry=SpatialGeometry(spatial_shape=(3, 4, 5)))
    assert data.spatial_shape == (3, 4, 5)


def test_explicit_coordinate_events_preserve_sparse_encoding_echoes_and_output_flips(monkeypatch):
    from mrax.io.bruker_storage import DecodedRecords
    from mrax.reco.acquisition import format_coil_images
    from mrax.reco.layout import image_to_encoding

    rng = np.random.default_rng(42)
    measured = rng.normal(size=(2, 2, 4, 6)) + 1j * rng.normal(size=(2, 2, 4, 6))
    records = DecodedRecords(measured.transpose(2, 0, 1, 3).reshape(8, 2, 6), "synthetic", "job", 0)
    monkeypatch.setattr("mrax.io.bruker_acquisition.decode_records", lambda *args, **kwargs: records)
    method = {"PVM_EncMatrix": [6, 5, 7], "PVM_NEchoImages": 2,
        "PVM_EncGenTotalSteps": 4, "PVM_EncGenSteps1": [-2, -1, 0, 1],
        "PVM_EncGenSteps2": [0, 1, -1, 0]}
    acqp = {"ACQ_dim": 3, "NI": 2, "NR": 1, "NECHOES": 2}
    reco = {"RECO_ft_size": [6, 5, 7], "RECO_size": [6, 5, 7]}
    acquisition, report = read_cartesian_acquisition("unused", method, acqp, reco,
        {"VisuCoreDim": 3, "VisuCoreDiskSliceOrder": "disk_reverse_slice_order"})
    assert report["event_profile"] == "cartesian.coordinate_events"
    assert acquisition.recipe.output_flips == (2,)
    assert np.sum(acquisition.acquired) == 24
    np.testing.assert_array_equal(acquisition.samples[:, :, np.array([3, 4, 2, 3]), np.array([0, 1, 2, 3])], measured)
    acquisition = replace(acquisition, recipe=replace(acquisition.recipe, output_scale=1))
    encoded_image = ifftnc(acquisition.samples[:, :1])
    formatted = format_coil_images(encoded_image, acquisition.recipe, spatial_ndim=3)
    np.testing.assert_allclose(image_to_encoding(formatted[:, 0], acquisition), encoded_image[:, 0], atol=1e-6)


def test_object_geometry_parameters_repeat_across_acquisition_repetitions(monkeypatch):
    from mrax.io.bruker_storage import DecodedRecords

    records = DecodedRecords(np.ones((12, 1, 4), complex), "synthetic", "fid", 0)
    monkeypatch.setattr("mrax.io.bruker_acquisition.decode_records", lambda *args, **kwargs: records)
    method = {"PVM_EncMatrix": [4, 2], "PVM_SPackArrNSlices": 2}
    acqp = {"ACQ_dim": 2, "NI": 2, "NR": 3}
    reco = {"RECO_ft_size": [4, 2], "RECO_size": [4, 2], "RECO_rotate": [.5, .5, .25, .75]}
    acquisition, _ = read_cartesian_acquisition("unused", method, acqp, reco, {"VisuCoreDim": 2})
    assert acquisition.recipe.frame_image_shifts == ((-.5, 0.), (.5, 0.)) * 3


def test_reconstruction_graph_composes_serial_windows_without_counting_parallel_receivers():
    from mrax.io.bruker_graph import spectral_window_powers
    from mrax.reco.windows import spectral_window

    nodes, edges = [], []
    for pass_index, axis in [(0, 0), (1, 0), (2, 1)]:
        for receiver in range(3):
            node = f"W{receiver}"
            nodes.append(["signal", pass_index,
                f"RecoSineWinMultFilter {node}{{winDirection={axis};SSB=1;square=false;}}", ""])
            edges.extend([["signal", pass_index, f"Input{receiver}->{node}"],
                ["signal", pass_index, f"{node}->Output{receiver}"]])
    graph = {"RecoStageNodes": nodes, "RecoStageEdges": edges}
    assert spectral_window_powers(graph, 2) == (2, 1)
    window = spectral_window((9, 11), ("sine", "sine"), (1, 1), (2, 1))
    expected = np.sin(np.pi * np.arange(9) / 9)[:, None]**2 * np.sin(np.pi * np.arange(11) / 11)[None]
    np.testing.assert_allclose(window, expected, rtol=1e-6, atol=2e-7)
    cyclic = {**graph, "RecoStageEdges": edges + [["signal", 0, "Output0->Input0"]]}
    with pytest.raises(ValueError, match="cycle"):
        spectral_window_powers(cyclic, 2)


def test_pulseq_raw_import_uses_acquisition_contract_without_reconstructing(monkeypatch):
    from pathlib import Path

    from mrax.io.import_definitions import NormalizedMetadata, reshape_rare_samples
    from mrax.io.pulseq_import import PulseqSequenceInfo, read_pulseq_acquisition

    info = PulseqSequenceInfo(path=Path("synthetic.seq"), definitions={}, trajectory=None,
        trajectory_physical=None, adc_times=None, adc_counts=(4, 4, 4),
        sequence_kind="cartesian", sequence_type="standard", spatial_shape=(3, 4),
        normalized_metadata=NormalizedMetadata())
    monkeypatch.setattr("mrax.io.pulseq_import.read_pulseq", lambda *args: info)
    monkeypatch.setattr("mrax.io.pulseq_import.read_pulseq_sidecar_metadata", lambda *args: {})

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not reconstruct")

    monkeypatch.setattr("mrax.reco.acquisition.reconstruct_acquisition", forbidden)
    acquisition = read_pulseq_acquisition("synthetic.seq", np.zeros(12, complex))
    assert acquisition.samples.shape == (1, 1, 3, 4)
    acquisition.require_prepared()
    assert acquisition.acquired is None
    # Cartesian coordinates retain a missing interior line, including when all
    # measured values happen to be zero.
    trajectory = np.concatenate([np.column_stack([np.arange(4) / 4 - .5, np.full(4, ky)])
        for ky in [-1 / 3, 1 / 3]])
    support = reshape_rare_samples(np.ones(8, bool), trajectory=trajectory,
        adc_counts=(4, 4), spatial_shape=(3, 4))
    np.testing.assert_array_equal(support, [[True] * 4, [False] * 4, [True] * 4])


def test_pocs_utility_defaults_to_estimated_complex_phase():
    from mrax.reco import pocs_complete_kspace

    rng = np.random.default_rng(83)
    image = (rng.uniform(.5, 1.5, (9, 11)) * np.exp(.8j)).astype("complex64")
    kspace = np.asarray(fftnc(jnp.asarray(image)))
    acquired = np.ones(kspace.shape, bool)
    acquired[:2] = False
    completed = pocs_complete_kspace(np.where(acquired, kspace, 0), mask=acquired, iterations=40)
    np.testing.assert_allclose(completed, kspace, atol=1e-5)


def test_unified_variational_uses_sensitivities_with_multislice_output():
    shape = (6, 8)
    target = jnp.arange(2 * 48, dtype=jnp.float32).reshape((2,) + shape) / 96 + .5
    sensitivities = jnp.stack([jnp.ones(shape), 2j * jnp.ones(shape)])
    samples = build_operator(spatial_shape=shape, sensitivities=sensitivities).forward(target)
    recipe = ReconstructionRecipe(coil_combine="sense", stack_slices=2,
        frame_permutations=((1, 0), (1, 0)))
    acquisition = AcquisitionData(np.asarray(samples), shape, ("phase", "read"), recipe=recipe)
    result = reconstruct_acquisition(acquisition, method="variational", sensitivities=sensitivities,
        regularizer="none", optimizer_name="fista", num_steps=2)
    expected = jnp.transpose(target, (2, 1, 0))[None]
    np.testing.assert_allclose(result.image, expected, atol=1e-6)
