"""Synthetic fista contracts and numerical references."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.optimize import minimize

from mrax.opt.optimizers import make_optimizer
from mrax.opt.tv import TVOperator, tv_prox


def _paper_reference(matrix, target, initial, weight, rate, steps):
    """Literal Algorithm 4.1, with a known Lipschitz constant and L1 prox."""
    x, y, t = initial.copy(), initial.copy(), 1.0
    trace = []
    for _ in range(steps):
        z = y - rate * matrix.T @ (matrix @ y - target)
        new = np.sign(z) * np.maximum(np.abs(z) - rate * weight, 0)
        new_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = new + (t - 1) / new_t * (new - x)
        x, t = new, new_t
        trace.append(x.copy())
    return np.asarray(trace)


def test_fista_matches_original_paper_trace():
    rng = np.random.default_rng(24)
    matrix = rng.normal(size=(14, 9)).astype(np.float32)
    target = rng.normal(size=14).astype(np.float32)
    rate = 0.9 / np.linalg.norm(matrix, 2) ** 2
    initial = np.zeros(9, np.float32)
    expected = _paper_reference(matrix, target, initial, 0.05, rate, 40)

    def loss(p):
        return 0.5 * jnp.sum((matrix @ p["x"] - target) ** 2)

    opt = make_optimizer(
        "fista", rate, regularizer_weight=0.05, fista_restart="none", fista_backtracking="classic"
    )

    def run():
        def body(carry, _):
            p, state = carry
            updates, state = opt.update(None, state, p, value_fn=loss)
            p = optax.apply_updates(p, updates)
            return (p, state), p["x"]

        p = {"x": jnp.asarray(initial)}
        return jax.lax.scan(body, (p, opt.init(p)), None, length=40)[1]

    actual = jax.jit(run)()
    np.testing.assert_allclose(actual, expected, atol=8e-7, rtol=1e-5)


def test_full_backtracking_matches_independent_published_recurrence():
    matrix = np.asarray([[1.0, 0.3], [0.3, 0.1]], dtype=np.float64)
    target = np.asarray([1.0, -0.2])
    x, previous, t, last_rate = np.zeros(2), np.zeros(2), 1.0, 4.0
    expected = []
    for iteration in range(50):
        rate = last_rate * (1.5 if iteration else 1.0)
        while True:
            current_t = (
                1.0 if not iteration else 0.5 * (1 + np.sqrt(1 + 4 * t * t * last_rate / rate))
            )
            y = x + ((t - 1) / current_t) * (x - previous)
            grad = matrix.T @ (matrix @ y - target)
            forward = y - rate * grad
            new = np.sign(forward) * np.maximum(np.abs(forward) - rate * 0.01, 0)
            f_y = 0.5 * np.sum((matrix @ y - target) ** 2)
            f_new = 0.5 * np.sum((matrix @ new - target) ** 2)
            d = new - y
            if f_new - f_y - grad @ d <= d @ d / (2 * rate) + 1e-15:
                break
            rate /= 2
        previous, x, t, last_rate = x, new, current_t, rate
        expected.append(x.copy())
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64 as enable
    with enable():

        def loss(p):
            return 0.5 * jnp.sum((matrix @ p["x"] - target) ** 2)

        opt = make_optimizer(
            "fista", 4.0, regularizer_weight=0.01, fista_restart="none", fista_backtracking="full"
        )

        def run():
            def body(carry, _):
                p, state = carry
                update, state = opt.update(None, state, p, value_fn=loss)
                p = optax.apply_updates(p, update)
                return (p, state), p["x"]

            p = {"x": jnp.zeros(2, dtype=jnp.float64)}
            return jax.lax.scan(body, (p, opt.init(p)), None, length=50)[1]

        actual = np.asarray(jax.jit(run)())
        # Near the solution, roundoff-aware line-search decisions can differ
        # between independent reduction implementations. The resolvable trace
        # must agree tightly; the complete trace agrees to numerical accuracy.
        resolved = np.linalg.norm(np.asarray(expected) - expected[-1], axis=1) > 1e-5
        assert np.count_nonzero(resolved) >= 10
        np.testing.assert_allclose(
            actual[resolved], np.asarray(expected)[resolved], rtol=2e-10, atol=2e-11
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("masked", [True])
def test_fgp_box_tv_prox_matches_independent_convex_epigraph(order, masked):
    values = np.asarray([-0.8, 0.9, -0.3, 1.2, 0.2], dtype=np.float64)
    mask = np.asarray([1, 1, 0, 1, 1]) if masked else None
    # One spatial dimension gives a linear-program epigraph for |D x|.
    enable = getattr(jax, "enable_x64", None)
    if enable is None:
        from jax.experimental import enable_x64 as enable
    with enable():
        op = TVOperator.create(values.shape, order=order, mask=mask, weight=1.7, dtype=jnp.float64)
        matrix = np.stack([np.asarray(op.gradient(jnp.eye(5)[i]))[:, 0] for i in range(5)], axis=1)
        weights = np.asarray(op.weights)

        def objective(v):
            return 0.5 * np.sum((v[:5] - values) ** 2) + weights @ v[5:]

        def jacobian(v):
            return np.r_[v[:5] - values, weights]

        constraints = [
            {"type": "ineq", "fun": lambda v: v[5:] - matrix @ v[:5]},
            {"type": "ineq", "fun": lambda v: v[5:] + matrix @ v[:5]},
        ]
        reference = minimize(
            objective,
            np.r_[np.zeros(5), np.ones(5)],
            jac=jacobian,
            constraints=constraints,
            bounds=[(-0.5, 0.7)] * 5 + [(0, None)] * 5,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        assert reference.success
        solve = jax.jit(
            lambda v: tv_prox(
                v, (op,), 1.0, (-0.5, 0.7), tolerance=1e-12, max_steps=4096, method="fgp"
            )
        )
        actual, duals, gap, count = solve(jnp.asarray(values))
        np.testing.assert_allclose(actual, reference.x[:5], atol=3e-6, rtol=0)
        assert float(gap) <= 1e-12 * (1 + reference.fun) * 1.01
        warm, _, warm_gap, warm_count = jax.jit(
            lambda v: tv_prox(v, (op,), 1.0, (-0.5, 0.7), duals, tolerance=1e-12, method="fgp")
        )(jnp.asarray(values))
        np.testing.assert_array_equal(warm, actual)
        assert int(warm_count) == 0 and float(warm_gap) <= float(gap)
