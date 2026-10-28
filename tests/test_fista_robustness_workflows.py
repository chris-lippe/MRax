"""Independent synthetic domain and bounded-influence regressions."""
import inspect
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mrax.core import build_loss, run_fit
from mrax.core.data_losses import pseudo_huber_loss
from mrax.data_struct import DataSet, Experiment, Model, Quantity
from mrax.opt._fista import make_fista_kernel


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_domain_restart_retains_certified_curvature(backtracking):
    def loss(p):
        return -jnp.log(p["x"][0])
    p = {"x": jnp.asarray([1e-8])}
    opt, step, _ = make_fista_kernel(loss, loss, {"x": (1e-12, 1.)}, {"x": ()}, .02,
                                      restart="gradient", backtracking=backtracking)
    s = opt.init(p)
    s.update(previous={"x": jnp.asarray([1.])}, y={"x": jnp.asarray([-1.])},
             t=jnp.asarray(10.), rate=jnp.asarray(1e-20), safe_rate=jnp.asarray(1e-20),
             grow=jnp.asarray(True), count=jnp.asarray(10, jnp.int32), cached_loss=loss(p))
    _, q, ns, status = jax.jit(step)(p, s)
    assert int(status) == 0 and int(ns["count"]) == 11
    assert float(loss(q)) < float(loss(p))
    assert np.isfinite(float(ns["rate"])) and float(ns["rate"]) > 0


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_invalid_accepted_domain_preserves_checkpoint(backtracking):
    def loss(p):
        return -jnp.log(p["x"][0])
    p = {"x": jnp.asarray([-1.])}
    opt, step, _ = make_fista_kernel(loss, loss, {"x": (-2., 2.)}, {"x": ()}, .02,
                                      restart="gradient", backtracking=backtracking)
    state = opt.init(p)
    _, q, new, status = jax.jit(step)(p, state)
    assert int(status) == 3
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_unresolvable_nonstationary_step_is_not_counted(backtracking):
    def loss(p):
        return 1e10 * p["x"][0]
    p = {"x": jnp.asarray([1.], jnp.float32)}
    opt, step, _ = make_fista_kernel(loss, loss, {"x": (-2., 2.)}, {"x": ()}, .02,
                                      restart="gradient", backtracking=backtracking)
    state = opt.init(p)
    state.update(rate=jnp.asarray(1e-30, jnp.float32), safe_rate=jnp.asarray(1e-30, jnp.float32))
    _, q, new, status = jax.jit(step)(p, state)
    assert int(status) == 4
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_overflowed_majorization_comparison_is_not_a_certificate(backtracking):
    def loss(p):
        return jnp.sum(jnp.sin(1e20 * p['x']))
    p = {'x': jnp.zeros(4, jnp.float32)}
    opt, step, _ = make_fista_kernel(loss, loss, {'x': (-np.inf, np.inf)}, {'x': ()}, .02,
        backtracking=backtracking, max_backtracking=1)
    state = opt.init(p)
    _, q, new, status = jax.jit(step)(p, state)
    assert int(status) == 3
    for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backtracking", ["classic", "full"])
def test_rounding_guard_uses_least_precise_parameter_dtype(backtracking):
    enable = getattr(jax, 'enable_x64', None)
    if enable is None:
        from jax.experimental import enable_x64
        enable = enable_x64
    with enable():
        def loss(p):
            return p['single'][0] + p['double'][0]**2
        p = {'single': jnp.asarray([1.], jnp.float32), 'double': jnp.asarray([0.], jnp.float64)}
        opt, step, _ = make_fista_kernel(loss, loss, {k: (-2., 2.) for k in p}, {k: () for k in p}, .02,
            backtracking=backtracking)
        state = opt.init(p)
        state.update(rate=jnp.asarray(1e-10, jnp.float64), safe_rate=jnp.asarray(1e-10, jnp.float64))
        _, q, new, status = jax.jit(step)(p, state)
        assert int(status) == 4
        for a, b in zip(jax.tree.leaves((p, state)), jax.tree.leaves((q, new)), strict=True):
            np.testing.assert_array_equal(a, b)


def test_group_pseudo_huber_bounds_spectral_influence_and_ignores_missing():
    weights = jnp.ones((8, 2))
    baseline = jnp.zeros((8, 2))
    def f(r):
        return pseudo_huber_loss(r, weights, delta=.1, group_size=4)
    gradient = jax.jit(jax.grad(f))
    norm = []
    for amplitude in (1., 100., 10000.):
        r = baseline.at[:4, 0].set(amplitude)
        norm.append(float(jnp.linalg.norm(gradient(r)[:4, 0])))
    assert max(norm) / min(norm) < 1.01
    r = jnp.asarray([[.01, jnp.nan], [.02, jnp.inf]])
    w = jnp.asarray([[1., 0.], [1., 0.]])
    assert np.isfinite(float(pseudo_huber_loss(r, w, delta=.1, group_size=2)))


def test_pseudo_huber_quadratic_limit_and_zero_gradient():
    r = jnp.asarray([[.01, -.02], [.03, .04]])
    w = jnp.asarray([[1., .5], [0., 2.]])
    actual = pseudo_huber_loss(r, w, delta=100., group_size=2)
    expected = jnp.sum(w * r*r) / jnp.sum(w)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    zero = jax.grad(lambda x: pseudo_huber_loss(x, w, delta=.1, group_size=2))(jnp.zeros_like(r))
    np.testing.assert_array_equal(zero, jnp.zeros_like(r))


def _problem():
    values = jnp.asarray([1., 1.1, .9, 15.]).reshape(4, 1, 1)
    exp = Experiment(dims={"frame": 4, "x": 1, "y": 1})
    exp.observables["offsets"] = Quantity(jnp.arange(4), "1")
    exp._mrax_dim_observables = {"frame": "offsets"}
    model = Model(name="synthetic_location", parameter_units={"level": "1"},
                  parametrization_factory=lambda shape: ({"level": jnp.full(shape, .5)}, lambda p: p),
                  model_apply=lambda fields, offsets, observables=None: jnp.broadcast_to(fields["level"], exp.shape))
    return exp, model, DataSet(values)


@pytest.mark.parametrize("optimizer", ["adam", "fista", "injected_adam"])
def test_custom_data_loss_reaches_named_and_injected_optimizers(optimizer):
    exp, model, data = _problem()
    extra = {"optimizer": optax.adam(.1)} if optimizer == "injected_adam" else {"optimizer_name": optimizer}
    result = run_fit(exp, model, data, num_steps=200, step_size=.1,
                     data_loss=partial(pseudo_huber_loss, delta=.1), **extra)
    # Public run_fit returns fields first; robust fitting should resist the
    # independently generated fourth observation instead of reaching mean 4.5.
    fields = result[0]
    value = np.asarray(fields["level"].values if hasattr(fields["level"], "values") else fields["level"])
    assert np.isfinite(value).all() and float(value.mean()) < 2.


def test_custom_data_loss_contract_is_checked():
    exp, model, data = _problem()
    with pytest.raises(TypeError, match="callable"):
        build_loss(exp, model, data, data_loss=2.)
    loss, p, _ = build_loss(exp, model, data, data_loss=lambda r, w: r)
    with pytest.raises(ValueError, match="real scalar"):
        loss(p)


def test_new_objective_option_preserves_positional_device_arguments():
    for function in (build_loss, run_fit):
        signature = inspect.signature(function)
        assert signature.parameters['data_loss'].kind == inspect.Parameter.KEYWORD_ONLY
        for name in ('device', 'reference_image'):
            assert signature.parameters[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    # Preserve the original public positional device/reference-image slots;
    # the additional FISTA choices belong after the keyword-only boundary.
    signature = inspect.signature(run_fit)
    positional = [p.name for p in signature.parameters.values()
                  if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[-3:] == ['fit_active_only', 'device', 'reference_image']
    for name in ('fista_restart', 'fista_backtracking'):
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
