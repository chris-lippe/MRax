"""Compatibility adapter; the solver implementation lives in mrax.opt._proximal."""
from mrax.devices import bind_device, device_function
from mrax.opt.optimizers import make_optimizer


@device_function
def make_tv_steps(loss, smooth_loss, bounds, operators, name, step_size, *,
                  tolerance=1e-8, max_prox_steps=2048, penalty=1., inner_steps=1,
                  device="cpu", grad_clip_norm=None, maximum_rate=None):
    opt = make_optimizer(name, step_size, device=device,
        proximal_context=(loss, smooth_loss, bounds, operators),
        proximal_tolerance=tolerance, proximal_max_steps=max_prox_steps,
        bregman_penalty=penalty, bregman_inner_steps=inner_steps,
        grad_clip_norm=grad_clip_norm, proximal_max_step_size=maximum_rate)
    def step(params, state):
        _, state = opt.update(None, state, params)
        return state["value"], state["accepted"], state, state["status"]
    return opt, bind_device(step, device), bind_device(opt.update._mrax_convergence, device)
