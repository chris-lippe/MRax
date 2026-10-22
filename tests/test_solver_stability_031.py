"""Release safeguards: known optima, callback API, masks and checkpoint parity."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_optimizer_options_workflows import _problem

from mrax.core.active_parameters import pack_problem
from mrax.core.fit_execution import execute_optimizer
from mrax.core.loss_builder import run_fit
from mrax.data_struct import Quantity
from mrax.library.default_models import make_AB_MT_n_model
from mrax.opt._proximal import make_proximal_kernel
from mrax.opt.optimizers import make_optimizer
from mrax.opt.tv import TVOperator


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
@pytest.mark.parametrize('scale', [1., 1e-8, 1e-14])
def test_extraargs_known_convex_optimum_and_resume(name, scale):
    target=jnp.asarray([2., -3.])
    def loss(p):
        return scale * .5 * jnp.sum((p['x']-target)**2)
    # Scaling the entire objective also scales an absolute augmented penalty.
    # Keeping rho fixed would change conditioning, not the minimizer.
    options=dict(regularizer_weight=.5*scale, box_constraints=(-2., 2.), bregman_penalty=scale)
    # Deliberately unsafe initial rate must be reduced by line search.
    opt=make_optimizer(name, 20./scale, **options)
    start={'x':jnp.zeros(2)}
    result, history, info=execute_optimizer(loss,start,opt,num_steps=1000)
    np.testing.assert_allclose(result['x'],[1.5,-2.],atol=1e-3)
    assert np.isfinite(history).all() and np.isfinite(info.final_loss)
    assert info.final_loss / scale == pytest.approx(2.375, abs=2e-4)
    partial,_,first=execute_optimizer(loss,start,opt,num_steps=37)
    resumed,_,_=execute_optimizer(loss,partial,opt,num_steps=963,initial_state=first.state)
    np.testing.assert_array_equal(resumed['x'],result['x'])


@pytest.mark.parametrize('order',[1,2])
def test_packed_tv_value_gradient_adjoint_and_no_mask_bridges(order):
    mask=np.ones((3,7),bool)
    mask[:,3]=False
    p={'x':jnp.asarray(np.random.default_rng(4301).normal(size=(3,7)),jnp.float32)}
    op=TVOperator.create((3,7),order=order,mask=mask,weight=.5)
    def unpack(x):return x
    unpack._mrax_smooth_loss=lambda x:jnp.sum(jnp.where(mask,(x['x']-1.)**2,0))
    unpack._mrax_tv_operators={'x':(op,)}
    def loss(x):
        return unpack._mrax_smooth_loss(x)+op.value(x['x'])
    packed_loss,packed,u=pack_problem(loss,p,unpack,mask)
    packed_op=u._mrax_tv_operators['x'][0]
    np.testing.assert_allclose(packed_op.value(packed['x']),op.value(p['x']),rtol=1e-6)
    np.testing.assert_allclose(packed_loss(packed),loss(p),rtol=1e-6)
    grad=jax.grad(loss)(p)['x'][mask]
    np.testing.assert_allclose(jax.grad(packed_loss)(packed)['x'],grad,rtol=1e-6)
    dual=jnp.ones_like(packed_op.gradient(packed['x']))*.37
    np.testing.assert_allclose(jnp.vdot(packed_op.gradient(packed['x']),dual),
                               jnp.vdot(packed['x'],packed_op.adjoint(dual)),atol=1e-6)


@pytest.mark.parametrize('name',['fista','split_bregman','adam'])
def test_active_fit_equivalence_and_resume(name):
    exp,model,data=_problem([[[1.],[9.],[2.]],[[2.],[8.],[3.]]])
    model.parameter_specs={'level':{'transform':'identity','init':0.,'scale':1.}}
    mask=jnp.asarray([[1.],[0.],[1.]])
    config={'level':[{'type':'tv','weight':.01}]}
    options=dict(optimizer_name=name,num_steps=40,step_size=.02,mask=mask,
                 regularizer_configs=config,return_fit_info=True)
    full,_,_=run_fit(exp,model,data,**options)
    packed,_,info=run_fit(exp,model,data,fit_active_only=True,**options)
    # Default curvature coordinates depend on original full shapes and remain
    # fixed before packing; raw Adam is unchanged on active pixels.
    np.testing.assert_allclose(packed['level'].values,full['level'].values,atol=2e-5)
    options.update(num_steps=20)
    _,_,partial=run_fit(exp,model,data,fit_active_only=True,**options)
    resumed,_,_=run_fit(exp,model,data,fit_active_only=True,initial_state=partial.state,**options)
    np.testing.assert_array_equal(resumed['level'].values,packed['level'].values)
    assert info.state.active_layout
    with pytest.raises(ValueError,match='layout'):
        run_fit(exp,model,data,fit_active_only=True,initial_state=partial.state,
                **dict(options,mask=jnp.asarray([[0.],[1.],[1.]])))


@pytest.mark.parametrize('name',['fista','split_bregman'])
def test_custom_convex_projection_and_box(name):
    # Nonseparable projection onto the sum-zero plane plus a box. Sequential
    # clipping is incorrect; the combined prox must solve the intersection.
    def prox(x, w):
        return x-jnp.mean(x)
    def loss(p):
        return .5*jnp.sum((p['x']-jnp.asarray([4.,1.,-1.]))**2)
    opt=make_optimizer(name,.1,regularizer_prox=prox,box_constraints=(-1.,1.))
    result,_,_=execute_optimizer(loss,{'x':jnp.zeros(3)},opt,num_steps=300)
    np.testing.assert_allclose(result['x'],[1.,0.,-1.],atol=1e-4)


@pytest.mark.parametrize('name',['fista','split_bregman'])
def test_double_precision_loss_with_single_precision_parameters(name):
    enable = getattr(jax, 'enable_x64', None)
    if enable is None:
        from jax.experimental import enable_x64
        enable = enable_x64
    with enable():
        def loss(p):
            return jnp.sum((p['x']-jnp.asarray(2.,jnp.float64))**2)
        start={'x':jnp.zeros((2,),jnp.float32)}
        result,_,_=execute_optimizer(loss,start,make_optimizer(name,.1),num_steps=60)
        assert result['x'].dtype==jnp.float32
        np.testing.assert_allclose(result['x'],2.,atol=2e-3)


@pytest.mark.parametrize('name',['fista','split_bregman'])
def test_gradient_clipping_cannot_certify_a_distant_stationary_point(name):
    def loss(p):
        return .5*jnp.sum((p['x']-2.)**2)
    opt=make_optimizer(name,.1,grad_clip_norm=1e-12)
    result,_,info=execute_optimizer(loss,{'x':jnp.zeros(2)},opt,num_steps=12,
        convergence_tolerance=1e-6,parameter_tolerance=1e-5,convergence_patience=2)
    assert not info.converged and info.termination_reason=='max_steps'
    assert float(jnp.linalg.norm(jax.grad(loss)(result)['x']))>1.


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
@pytest.mark.parametrize('option,value', [
    ('proximal_tolerance', 0.),
    ('proximal_max_steps', 0),
    ('proximal_max_step_size', -1.),
    ('bregman_inner_steps', 0),
])
def test_raw_fit_validates_proximal_options(name, option, value):
    exp, model, data = _problem([[[1.]], [[2.]]])
    with pytest.raises(ValueError):
        run_fit(exp, model, data, optimizer_name=name, num_steps=0, **{option: value})


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
def test_raw_fit_honors_proximal_step_cap(name):
    exp, model, data = _problem([[[1.]], [[2.]]])
    _, _, info = run_fit(exp, model, data, optimizer_name=name, num_steps=5,
                         step_size=2., proximal_max_step_size=.01, return_fit_info=True)
    assert float(info.state.optimizer_state['solver']['rate']) <= .010001


def test_raw_bregman_fit_honors_inner_budget():
    exp, model, data = _problem([[[1.]], [[2.]]])
    losses = []
    for inner_steps in (1, 3):
        _, _, info = run_fit(exp, model, data, optimizer_name='split_bregman',
                             step_size=.01, num_steps=3, bregman_inner_steps=inner_steps,
                             return_fit_info=True)
        losses.append(info.final_loss)
    assert losses[1] < losses[0]


@pytest.mark.parametrize('spin_lock,expected,expected_gradient', [
    (False, [0.9581428766, 0.1448954195, 0.3239752054],
     [-1.3465567827, -1.0522072315, -1.1726071835]),
    (True, [1.0181690454, 0.5023628473, 0.3643416464],
     [-1.4368637800, -4.2174105644, -1.3498690128]),
])
def test_ab_mt_preserves_prepatch_signal_and_gradient(spin_lock, expected, expected_gradient):
    # Frozen from the pre-patch implementation with independent synthetic inputs.
    model = make_AB_MT_n_model(pools=[])
    params, unpack = model.parametrization_factory((1, 1))
    fields = {**unpack(params), 'a': jnp.full((1, 1), .01), 'b': jnp.full((1, 1), .04),
              'sin_a_eff': jnp.full((1, 1), .8), 't_delay': jnp.full((1, 1), 20.)}
    observations = {
        'freq_offsets': Quantity(jnp.array([-2., .5, 3.]), 'ppm'),
        'field_strength_mhz': Quantity(120., 'MHz'),
        'sat_time': Quantity(jnp.array([200., 100., 300.]), 'ms'),
        'sat_b1': Quantity(jnp.array([1.5, 2.5, 3.5]), 'uT'),
        'TR': Quantity(jnp.array([900., 1000., 1100.]), 'ms'),
        'r1_map': Quantity(jnp.ones((1, 1)), '1/s'),
        'spin_lock': spin_lock,
    }

    def signal(b):
        return model.model_apply({**fields, 'b': b}, None, observations, _allow_direct=True)

    np.testing.assert_allclose(signal(fields['b']).reshape(-1), expected, rtol=3e-6)
    np.testing.assert_allclose(jax.jacrev(signal)(fields['b']).reshape(-1), expected_gradient,
                               rtol=3e-6)


def test_fista_reuses_trial_objective():
    calls = []

    def smooth(params):
        return .5 * jnp.sum((params['x'] - 2.)**2)

    def objective(params):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return smooth(params)

    params = {'x': jnp.zeros(2)}
    optimizer, step, _ = make_proximal_kernel(objective, smooth, {'x': (-np.inf, np.inf)},
                                             {'x': ()}, 'fista', .1)
    _, _, _, status = jax.jit(step)(params, optimizer.init(params))
    jax.effects_barrier()
    assert int(status) == 0
    assert len(calls) == 2


def test_ab_mt_default_bound_keeps_saturated_parameter_gradients_finite():
    model = make_AB_MT_n_model(pools=[])
    params, unpack = model.parametrization_factory((1, 1))
    params['sin_a_eff_raw'] = jnp.full_like(params['sin_a_eff_raw'], 50.)
    observations = {
        'freq_offsets': Quantity(jnp.array([-1., 1.]), 'ppm'),
        'field_strength_mhz': Quantity(120., 'MHz'),
        'sat_time': Quantity(jnp.array([100., 100.]), 'ms'),
        'sat_b1': Quantity(jnp.array([3., 3.]), 'uT'),
        'TR': Quantity(jnp.array([1000., 1000.]), 'ms'),
        'r1_map': Quantity(jnp.ones((1, 1)), '1/s'),
    }

    def signal(raw):
        return jnp.sum(model.model_apply(unpack(raw), None, observations, _allow_direct=True))

    value, gradients = jax.value_and_grad(signal)(params)
    assert np.isfinite(value)
    assert all(np.isfinite(gradient).all() for gradient in jax.tree.leaves(gradients))
    assert bool(jnp.all(unpack(params)['sin_a_eff'] < 1.))


@pytest.mark.parametrize('initial,expected_steps,expected_checks', [(0., 5, 0), (2., 2, 2)])
def test_expensive_convergence_check_waits_for_small_updates(initial, expected_steps, expected_checks):
    calls = []

    def loss(params):
        return .5 * jnp.sum((params['x'] - 2.)**2)

    def convergence(params, state, tolerance):
        jax.debug.callback(lambda: calls.append(1), ordered=True)
        return jnp.asarray(True)

    _, _, info = execute_optimizer(loss, {'x': jnp.full(2, initial)}, make_optimizer('sgd', .1),
        num_steps=5, convergence_tolerance=1e-10, parameter_tolerance=1e-10,
        convergence_patience=2, convergence_check=convergence)
    jax.effects_barrier()
    assert info.iterations == expected_steps
    assert len(calls) == expected_checks


@pytest.mark.parametrize('name', ['fista', 'split_bregman'])
def test_checkpoint_resume_accepts_earlier_redundant_state_fields(name):
    def loss(params):
        return .5 * jnp.sum((params['x'] - 2.)**2)

    start = {'x': jnp.zeros(2)}
    optimizer = make_optimizer(name, .1, regularizer_weight=.2)
    expected, _, _ = execute_optimizer(loss, start, optimizer, num_steps=12)
    params, _, partial = execute_optimizer(loss, start, optimizer, num_steps=5)
    old_solver = {**partial.state.optimizer_state['solver'], 'x_ref': {'0': params['x']}}
    if name == 'fista':
        old_solver['x'] = {'0': params['x']}
    old_state = replace(partial.state,
                        optimizer_state={**partial.state.optimizer_state, 'solver': old_solver})
    resumed, _, info = execute_optimizer(loss, params, optimizer, num_steps=7, initial_state=old_state)
    np.testing.assert_array_equal(resumed['x'], expected['x'])
    assert info.state.step == 12
