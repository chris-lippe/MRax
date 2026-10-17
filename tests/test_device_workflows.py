"""Synthetic device-selection regressions; no measurement fixtures."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax import get_available_devices
from mrax.core import (
    build_loss,
    debug_fit_prototype,
    execute_study_plan,
    plan_study_fits,
    run_batch_studies,
    run_fit,
)
from mrax.core.fit_execution import execute_optimizer
from mrax.core.loss_builder import build_study_losses
from mrax.core.tv_fit import make_tv_steps
from mrax.data_struct import DataSet, Experiment, Model, Quantity, Study
from mrax.devices import resolve_device
from mrax.opt import make_optimizer
from mrax.preprocessing import regularized_denoise
from mrax.preprocessing.drift import (
    linear_drift_correct_dataset,
    linear_drift_correct_framestack,
    negative_mean_squared_error,
)
from mrax.reco.variational import build_operator, solve_model_based, solve_variational


def _device(backend):
    devices = get_available_devices(backend)
    if not devices:
        pytest.skip(f"{backend} is unavailable")
    return devices[0]


def _problem():
    exp = Experiment(dims={"frame": 1, "x": 2, "y": 1})
    exp.observables["gain"] = Quantity(jnp.asarray(1.0), "1")
    model = Model(
        name="synthetic_level", parameter_units={"level": "1"},
        parameter_specs={"level": {"transform": "identity"}},
        parametrization_factory=lambda shape: ({"level": jnp.zeros(shape)}, lambda p: p),
        model_apply=lambda fields, offsets, observables=None: fields["level"][None]
        * getattr(observables["gain"], "values", observables["gain"]),
    )
    return exp, model, DataSet(jnp.ones(exp.shape))


def _assert_device(tree, device):
    arrays = [x for x in jax.tree.leaves(tree) if isinstance(x, jax.Array)]
    assert arrays and all(x.devices() == {device} for x in arrays)


def test_device_discovery_returns_local_devices_cpu_first():
    found = get_available_devices()
    assert found and found[0].platform == "cpu"
    assert len(found) == len(set(found))
    assert get_available_devices("cpu") == jax.local_devices(backend="cpu")
    assert get_available_devices("cuda") == get_available_devices("gpu")
    for device in found:
        assert resolve_device(device) == device


@pytest.mark.parametrize("entry", [run_fit, build_loss, execute_optimizer, make_optimizer,
                                     make_tv_steps, debug_fit_prototype, solve_variational,
                                     solve_model_based, regularized_denoise,
                                     linear_drift_correct_framestack, linear_drift_correct_dataset,
                                     execute_study_plan, run_batch_studies, build_study_losses])
def test_optimizer_entry_points_have_explicit_cpu_default(entry):
    assert inspect.signature(entry).parameters["device"].default == "cpu"


@pytest.mark.parametrize("selector", ["gpu:99999", "cpu:-1", "cpu:abc", "cpu:0:0", "unknown"])
def test_invalid_or_unavailable_device_is_rejected_before_fitting(selector):
    exp, model, data = _problem()
    with pytest.raises(ValueError):
        run_fit(exp, model, data, device=selector, num_steps=0)


def test_unavailable_backend_returns_empty_list_and_selection_raises(monkeypatch):
    def local_devices(*, backend):
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(jax, "local_devices", local_devices)
    assert get_available_devices("gpu") == []
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("gpu")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("selector_kind", ["name", "index", "object"])
@pytest.mark.parametrize("optimizer", ["adam", "fista", "split_bregman"])
def test_fitting_explicit_device_including_true_tv(backend, selector_kind, optimizer):
    target = _device(backend)
    selector = target if selector_kind == "object" else backend + (":0" if selector_kind == "index" else "")
    exp, model, data = _problem()
    config = {"level": [{"type": "tv", "axes": (0,), "weight": 0.1}]} if optimizer != "adam" else {}
    fields, history, info = run_fit(exp, model, data, optimizer_name=optimizer, num_steps=8,
                                    regularizer_configs=config, device=selector, return_fit_info=True)
    _assert_device(fields["level"].values, target)
    _assert_device(info.state.params, target)
    _assert_device(info.state.optimizer_state, target)
    assert info.devices == (str(target),)
    assert np.isfinite(history).all()


def test_default_cpu_moves_gpu_inputs_without_mutation_or_global_context_change():
    gpu, cpu = _device("gpu"), _device("cpu")
    with jax.default_device(gpu):
        exp, model, data = _problem()
        original = data.data
        gain = exp.observables["gain"].values
        fields, _, info = run_fit(exp, model, data, num_steps=2, return_fit_info=True)
        assert jax.config.jax_default_device == gpu
        assert data.data is original and original.device == gpu
        assert exp.observables["gain"].values is gain and gain.device == gpu
        _assert_device(fields["level"].values, cpu)
        _assert_device(info.state.params, cpu)
        _assert_device(make_optimizer("adam", 0.1).init({"level": original}), cpu)


def test_cpu_placement_handles_captured_gpu_parametrization_constants():
    gpu, cpu = _device("gpu"), _device("cpu")
    exp, model, data = _problem()
    offset = jax.device_put(jnp.ones((2, 1)), gpu)
    model.parametrization_factory = lambda shape: (
        {"raw": jnp.zeros(shape)}, lambda p: {"level": p["raw"] + offset}
    )
    fields, _ = run_fit(exp, model, data, device="cpu", num_steps=1)
    _assert_device(fields["level"].values, cpu)
    np.testing.assert_allclose(fields["level"].values, 1.0)


def test_resume_moves_parameters_and_moments_between_cpu_and_gpu():
    cpu, gpu = _device("cpu"), _device("gpu")
    exp, model, data = _problem()
    _, _, first = run_fit(exp, model, data, device=gpu, num_steps=4, return_fit_info=True)
    resumed, _, last = run_fit(exp, model, data, device=cpu, num_steps=4,
                              initial_state=first.state, return_fit_info=True)
    whole, _ = run_fit(exp, model, data, device=cpu, num_steps=8)
    _assert_device(last.state.params, cpu)
    _assert_device(last.state.optimizer_state, cpu)
    _assert_device(first.state.params, gpu)
    np.testing.assert_allclose(resumed["level"].values, whole["level"].values, rtol=1e-5)
    assert last.state.step == 8


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("kind", ["sgd", "custom_adam"])
def test_executor_factory_and_custom_optimizer_respect_device(backend, kind):
    device = _device(backend)
    params = {"level": jnp.asarray(0.0)}
    def loss(p):
        return 0.5 * (p["level"] - 1.0) ** 2
    optimizer = make_optimizer("sgd", 0.1, device=device) if kind == "sgd" else optax.adam(0.1)
    result, _, info = execute_optimizer(loss, params, optimizer, num_steps=200, device=device)
    _assert_device(result, device)
    if kind != "sgd":
        _assert_device(info.state.optimizer_state, device)
    np.testing.assert_allclose(result["level"], 1.0, atol=0.001)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_reconstruction_and_denoising_use_selected_device(backend):
    device = _device(backend)
    target = jnp.ones((2, 2))
    op = build_operator(spatial_shape=(2, 2), kspace=jnp.zeros((2, 2)))
    image = solve_variational(op.forward(target), spatial_shape=(2, 2),
                                     regularizer="l1", regularizer_weight=0.1,
                                     num_steps=60, device=device)
    _assert_device(image, device)
    np.testing.assert_allclose(image, 0.9, atol=0.001)
    denoised = regularized_denoise(target, num_steps=4, device=device)
    _assert_device(denoised, device)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("optimizer", ["adam", "fista", "split_bregman"])
def test_model_based_reconstruction_and_diagnostic_forward_device(backend, optimizer):
    device = _device(backend)
    exp, model, data = _problem()
    op = build_operator(spatial_shape=(2, 1), kspace=jnp.zeros((2, 1)))
    fields = solve_model_based(op.forward(jnp.ones((1, 2, 1))), experiment=exp,
                                     model=model, num_steps=2, device=device)
    _assert_device(fields["level"].values, device)
    config = {"level": [{"type": "tv", "axes": (0,), "weight": 0.1}]} if optimizer != "adam" else {}
    report = debug_fit_prototype(exp, model, data, device=device, optimizer_name=optimizer,
                                regularizer_configs=config)
    assert report.optimizer_name == optimizer


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("dataset_entry", [False, True])
def test_drift_correction_forwards_selected_device_to_custom_factory(backend, dataset_entry):
    device = _device(backend)
    contexts = []
    class AdditiveTransform:
        def identity_parameters(self, image):
            return jnp.zeros(1)
        def apply(self, image, parameters):
            return image + parameters[0]
    def factory(rate):
        contexts.append(jax.config.jax_default_device)
        return optax.sgd(rate)
    times = np.linspace(0.0, 1.0, 6)
    frames = np.stack([np.full((2, 2), -2.5 * t, dtype=np.float32) for t in times])
    kwargs = dict(device=device, transform=AdditiveTransform(), metric=negative_mean_squared_error,
                  times=times, optimizer_factory=factory, step_size=0.6, num_steps=40)
    if dataset_entry:
        exp = Experiment(dims={"t": 6, "x": 2, "y": 2})
        result = linear_drift_correct_dataset(exp, DataSet(frames), **kwargs)
        corrected = result.dataset.data
    else:
        corrected = linear_drift_correct_framestack(frames, **kwargs).corrected
    assert contexts and all(context == device for context in contexts)
    np.testing.assert_allclose(corrected, 0.0, atol=1e-4)


def test_device_context_is_restored_after_validation_error():
    gpu = _device("gpu")
    with jax.default_device(gpu):
        with pytest.raises(ValueError):
            make_optimizer("unknown", 0.1, device="cpu")
        assert jax.config.jax_default_device == gpu


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_returned_model_callable_keeps_selected_device_outside_fit_context(backend):
    device = _device(backend)
    exp, model, data = _problem()
    fields, _, model_fn = run_fit(exp, model, data, num_steps=2,
                                  return_model_callable=True, device=device)
    other = _device("gpu" if backend == "cpu" else "cpu")
    with jax.default_device(other):
        prediction = model_fn(fields)
        _assert_device(prediction, device)
        assert jax.config.jax_default_device == other


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_study_and_batch_runners_forward_device_and_allow_explicit_overrides(backend, monkeypatch, tmp_path):
    import mrax.core.batch as batch
    import mrax.core.study_runner as runner
    device = _device(backend)
    exp, model, data = _problem()
    study = Study([exp], [data], [model], names=["synthetic"],
                  study_globals={"field_strength_mhz": Quantity(1.0, "MHz")})
    losses = build_study_losses(study, device=device)
    _assert_device(losses[0]["init_params"], device)
    selections = []
    def fit(experiment, model, dataset, **kwargs):
        selections.append(resolve_device(kwargs["device"]))
        return {"level": Quantity(jnp.ones((2, 1)), "1")}, []
    monkeypatch.setattr(runner, "run_fit", fit)
    plan = plan_study_fits(study)
    execute_study_plan(plan, device=device)
    assert selections[-1] == device
    execute_study_plan(plan, device=device, run_fit_overrides={"synthetic": {"device": "cpu"}})
    assert selections[-1] == _device("cpu")
    (tmp_path / "studies" / "synthetic").mkdir(parents=True)
    monkeypatch.setattr(batch, "save_study_results", lambda *args, **kwargs: tmp_path / "manifest.json")
    run_batch_studies(tmp_path / "studies", lambda path: study, tmp_path / "output", device=device)
    assert selections[-1] == device


@pytest.mark.parametrize("method", ["run_current_fit", "run_preview_fit"])
def test_gui_fit_methods_have_cpu_default_and_forward_explicit_device(method, monkeypatch):
    from mrax.analysis.roi import ROI
    module = pytest.importorskip("mrax.gui.session")
    session = module.GuiSession()
    exp, model, data = _problem()
    session.create_empty_study("synthetic")
    session.add_experiment("synthetic", np.ones(exp.shape), dims=exp.dims)
    session.add_roi(ROI("all", np.ones((2, 1), dtype=bool), dims=("x", "y")))
    monkeypatch.setattr(session, "build_fit_inputs", lambda *args, **kwargs: (exp, model, data, None))
    seen = []
    class CalledFit(Exception):
        pass
    def fit(*args, **kwargs):
        seen.append(kwargs["device"])
        raise CalledFit()
    monkeypatch.setattr(module, "run_fit", fit)
    entry = getattr(session, method)
    assert inspect.signature(entry).parameters["device"].default == "cpu"
    with pytest.raises(CalledFit):
        entry(device="gpu:0", num_steps=1)
    assert seen == ["gpu:0"]


def test_importing_library_and_readers_does_not_initialize_a_backend():
    import os
    import subprocess
    import sys

    script = 'import mrax.library; import mrax.io; import mrax.preprocessing'
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, timeout=45,
        env={**os.environ, 'JAX_PLATFORMS': 'mrax_unavailable_backend'})
