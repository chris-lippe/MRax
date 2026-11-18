"""Synthetic split bregman contracts and numerical references."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from mrax.opt.optimizers import make_optimizer
from mrax.opt.tv import TVOperator


def _same_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_split_bregman_matches_independent_dense_recurrence():
    with jax.enable_x64():
        matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]])
        target = np.asarray([1.2, -0.7])
        weights = np.asarray([0.3, 0.1])
        rho = 0.8
        scale = 1.0

        def loss(p):
            residual = p["x"] - target
            return 0.5 * residual @ matrix @ residual

        optimizer = make_optimizer(
            "split_bregman",
            1.0,
            regularizer_weight={"x": weights},
            bregman_penalty=rho,
            proximal_tolerance=1e-14,
        )
        params = {"x": jnp.zeros(2, jnp.float64)}
        state = optimizer.init(params)
        x = np.zeros(2)
        split = scale * x
        dual = np.zeros(2)
        np.testing.assert_array_equal(state["solver"]["duals"]["0"][0][..., 0], dual)
        for _ in range(8):
            x = np.linalg.solve(
                matrix + rho * scale**2 * np.eye(2), matrix @ target + rho * scale * (split - dual)
            )
            argument = scale * x + dual
            split = np.sign(argument) * np.maximum(np.abs(argument) - weights / (rho * scale), 0)
            dual = dual + scale * x - split
            updates, state = optimizer.update(None, state, params, value_fn=loss)
            params = optax.apply_updates(params, updates)
            assert int(state["status"]) == 0
            np.testing.assert_allclose(params["x"], x, atol=3e-13, rtol=0)
            np.testing.assert_allclose(state["solver"]["splits"]["0"][0][..., 0], split, atol=3e-13)
            np.testing.assert_allclose(state["solver"]["duals"]["0"][0][..., 0], dual, atol=3e-13)
            assert float(state["solver"]["penalty"]) == rho


def test_true_tv_two_pixel_solution_and_kkt():
    target = jnp.asarray([-1.0, 1.0], jnp.complex64) * (1 + 1j) / jnp.sqrt(2.0)
    operator = TVOperator.create((2,), weight=0.5)

    def smooth(p):
        return 0.5 * jnp.sum(jnp.abs(p["x"] - target) ** 2)

    def loss(p):
        return smooth(p) + operator.value(p["x"])

    bounds = (-np.inf, np.inf)
    optimizer = make_optimizer(
        "split_bregman", 1.0, proximal_context=(loss, smooth, {"x": bounds}, {"x": (operator,)})
    )
    params = {"x": jnp.zeros_like(target)}
    state = optimizer.init(params)

    @jax.jit
    def solve(params, state):
        def update(carry, _):
            point, state = carry
            _, state = optimizer.update(None, state, point, value_fn=smooth)
            return (state["accepted"], state), state["status"]

        return jax.lax.scan(update, (params, state), None, length=80)

    (result, state), statuses = solve(params, state)
    np.testing.assert_array_equal(statuses, 0)
    np.testing.assert_allclose(result["x"], 0.75 * target, atol=8e-6, rtol=0)
    assert bool(optimizer.update._mrax_convergence(result, state, 2e-5, value_fn=smooth))


def test_schedule_pause_freezes_primal_split_and_multiplier_and_resumes():
    def rate(count):
        return jnp.where((count == 1) | (count == 2), 0.0, 1.0)

    def loss(p):
        return 0.5 * jnp.sum((p["x"] - 2.0) ** 2)

    optimizer = make_optimizer("split_bregman", rate, regularizer_weight=0.2)
    point = {"x": jnp.asarray([0.0, 1.0])}
    state = optimizer.init(point)
    changes, state = optimizer.update(None, state, point, value_fn=loss)
    point = optax.apply_updates(point, changes)
    for _ in range(2):
        before = copy.copy(state)
        changes, state = optimizer.update(None, state, point, value_fn=loss)
        np.testing.assert_array_equal(changes["x"], 0)
        for name in ["splits", "duals", "rate", "penalty"]:
            _same_tree(before["solver"][name], state["solver"][name])
    changes, state = optimizer.update(None, state, point, value_fn=loss)
    assert np.any(np.asarray(changes["x"]) != 0)
    assert int(state["solver"]["count"]) == 4
