from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np

from mrax.data_struct import Experiment, Model
from mrax.data_struct.dimensions import build_experiment_dims, dim_observable_map
from mrax.library.default_models import (
    make_dki_kurtosis_model,
    make_dti_tensor_model,
    make_fair_asl_model,
    make_identity_model,
    make_t1_ir_model,
    make_t1_vfa_model,
    make_t1_vtr_model,
    make_t2_se_model,
    make_t2star_se_model,
)

ModelFactory = Callable[..., Model]


@dataclass(frozen=True)
class DefaultExperiment:
    name: str
    model_factory: ModelFactory
    allowed_observables: frozenset[str] | None


DEFAULT_SCALE = 1e6

DEFAULT_EXPERIMENTS: Dict[str, DefaultExperiment] = {
    "anatomical": DefaultExperiment(
        name="anatomical",
        model_factory=make_identity_model,
        allowed_observables=frozenset(),
    ),
    "dce": DefaultExperiment(
        name="dce",
        model_factory=make_identity_model,
        allowed_observables=frozenset({"TE", "TR", "flip_angle", "scan_time"}),
    ),
    "ir": DefaultExperiment(
        name="ir",
        model_factory=lambda scale=DEFAULT_SCALE: make_t1_ir_model(scale=scale),
        allowed_observables=frozenset({"TI"}),
    ),
    "fair_asl": DefaultExperiment(
        name="fair_asl",
        model_factory=lambda scale=DEFAULT_SCALE: make_fair_asl_model(scale=scale),
        allowed_observables=frozenset(
            {"TI", "fair_label", "t1_blood", "t1_blood_ms", "partition_coeff", "partition_coefficient", "lambda"}
        ),
    ),
    "vtr": DefaultExperiment(
        name="vtr",
        model_factory=lambda scale=DEFAULT_SCALE: make_t1_vtr_model(scale=scale),
        allowed_observables=frozenset({"TR"}),
    ),
    "vfa": DefaultExperiment(
        name="vfa",
        model_factory=lambda scale=DEFAULT_SCALE: make_t1_vfa_model(scale=scale),
        allowed_observables=frozenset({"TR", "flip_angle"}),
    ),
    "mse": DefaultExperiment(
        name="mse",
        model_factory=lambda scale=DEFAULT_SCALE: make_t2_se_model(scale=scale),
        allowed_observables=frozenset({"TE"}),
    ),
    "mge": DefaultExperiment(
        name="mge",
        model_factory=lambda scale=DEFAULT_SCALE: make_t2star_se_model(scale=scale),
        allowed_observables=frozenset({"TE", "flip_angle", "TR"}),
    ),
    "sat_trans": DefaultExperiment(
        name="sat_trans",
        model_factory=make_identity_model,
        allowed_observables=frozenset(
            {
                "freq_offsets",
                "sat_b1",
                "sat_time",
                "spin_lock",
                "SpinLock",
                "TR",
                "r1_map",
                "t1_map",
                "b0_map",
                "B0_map",
                "b0_shift",
                "b0_shift_ppm",
                "b1_map",
                "B1_map",
            }
        ),
    ),
    "dti": DefaultExperiment(
        name="dti",
        model_factory=make_dti_tensor_model,
        allowed_observables=frozenset({"bvals", "bvecs", "TE", "TR"}),
    ),
    "dki": DefaultExperiment(
        name="dki",
        model_factory=make_dki_kurtosis_model,
        allowed_observables=frozenset({"bvals", "bvecs", "TE", "TR"}),
    ),
}

def get_default_experiment(exp_type: str) -> DefaultExperiment | None:
    """Return the default experiment definition for a given type."""
    return DEFAULT_EXPERIMENTS.get(exp_type)


def default_model_for_type(exp_type: str, *, scale: float | None = None,
                           observables=None, image_data=None) -> Model:
    """Build the default model for an experiment type."""
    spec = get_default_experiment(exp_type)
    if spec is None:
        return make_identity_model()
    if scale is not None:
        from inspect import signature

        if "scale" in signature(spec.model_factory).parameters:
            model = spec.model_factory(scale=scale)
        else:
            model = spec.model_factory()
    else:
        model = spec.model_factory()
    if exp_type == 'ir' and image_data is not None and observables is not None and 'TI' in observables:
        seed = _inversion_recovery_seed(model, observables['TI'], image_data)
        if seed is not None:
            model = make_t1_ir_model(scale=DEFAULT_SCALE if scale is None else scale,
                param_overrides={'t1': {'init': seed}})
    return model


def _inversion_recovery_seed(model, times, image_data):
    """Profile the ideal IR amplitude to avoid starting beyond the wrong null."""
    if getattr(times, 'si_unit', 'ms').strip().lower() != 'ms':
        return None
    times = np.asarray(getattr(times, 'values', times), float).reshape(-1)
    data = np.asarray(image_data)
    if len(np.unique(times)) < 3 or data.ndim < 2 or data.shape[0] != len(times) or not np.isfinite(times).all() or np.any(times < 0):
        return None
    values = data.reshape((len(times), -1))
    dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    values = np.abs(values[:, ::max(1, int(np.ceil(values.shape[1] / 262144)))].astype(dtype))
    peak = np.max(values, where=np.isfinite(values), initial=0.)
    if peak <= 0:
        return None
    valid = np.isfinite(values)
    present = np.any(valid, axis=1)
    if np.count_nonzero(present) < 3:
        return None
    curve = np.sqrt(np.sum(np.where(valid, values / peak, 0.)**2, axis=1) /
                    np.maximum(np.sum(valid, axis=1), 1))
    times, curve = times[present], curve[present]
    bounds = model.parametrization_factory.parameter_specs['t1']['bounds']
    candidates = np.geomspace(*bounds, 256)
    basis = np.abs(1. - 2. * np.exp(-times[None] / candidates[:, None]))
    amplitudes = (basis @ curve) / np.maximum(np.sum(basis**2, axis=1), 1e-20)
    errors = np.sum((amplitudes[:, None] * basis - curve)**2, axis=1)
    return float(candidates[np.argmin(errors)])


def filter_observables_for_type(exp_type: str, observables: Mapping[str, object] | None) -> dict[str, object]:
    """Limit observables to the defaults for the experiment type."""
    if observables is None:
        return {}
    spec = get_default_experiment(exp_type)
    allowed = spec.allowed_observables if spec is not None else None
    if allowed is None:
        return dict(observables)
    return {k: v for k, v in observables.items() if k in allowed}


def build_default_experiment(
    exp_type: str,
    frame_count: int,
    spatial_shape: Iterable[int],
    observables: Optional[Mapping[str, object]] = None,
) -> Experiment:
    """Create an Experiment with default dims and filtered observables."""
    filtered = filter_observables_for_type(exp_type, observables)
    dims = build_experiment_dims(
        frame_count,
        spatial_shape,
        filtered,
        exp_type=exp_type,
        model_name=exp_type,
    )
    experiment = Experiment(dims=dims, observables=filtered, exp_type=exp_type)
    dim_map = dim_observable_map(exp_type, exp_type)
    experiment._mrax_dim_observables = dim_map
    return experiment


__all__ = [
    "DEFAULT_EXPERIMENTS",
    "build_default_experiment",
    "default_model_for_type",
    "filter_observables_for_type",
    "get_default_experiment",
]
