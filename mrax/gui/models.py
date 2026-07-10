"""Built-in model registry used by the GUI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Sequence

from mrax.data_struct.model import Model

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelChoice:
    """A model option exposed in the GUI."""

    key: str
    label: str
    factory: Callable[[], Model]


def _choices() -> tuple[ModelChoice, ...]:
    from mrax.library import (
        make_AB_MT_n_model,
        make_exp_decay_model,
        make_fair_asl_model,
        make_identity_model,
        make_multipool_lorentzian_model,
        make_t1_ir_model,
        make_t1_vfa_model,
        make_t1_vtr_model,
        make_t2_se_model,
        make_t2star_se_model,
        make_wasabi_model,
        make_zaiss_full_model,
    )
    from mrax.library.default_models import make_dki_kurtosis_model, make_dti_tensor_model

    scale = 1e6
    return (
        ModelChoice("identity", "Identity", make_identity_model),
        ModelChoice("t1_ir", "T1 inversion recovery", lambda: make_t1_ir_model(scale=scale)),
        ModelChoice("t1_vtr", "T1 variable TR", lambda: make_t1_vtr_model(scale=scale)),
        ModelChoice("t1_vfa", "T1 variable flip angle", lambda: make_t1_vfa_model(scale=scale)),
        ModelChoice("exp_decay", "Exponential decay", lambda: make_exp_decay_model(scale=scale)),
        ModelChoice("t2_se", "T2 spin echo", lambda: make_t2_se_model(scale=scale)),
        ModelChoice("t2star_se", "T2* decay", lambda: make_t2star_se_model(scale=scale)),
        ModelChoice("fair_asl", "FAIR ASL", lambda: make_fair_asl_model(scale=scale)),
        ModelChoice("dti", "Diffusion tensor", make_dti_tensor_model),
        ModelChoice("dki", "Diffusion kurtosis", make_dki_kurtosis_model),
        ModelChoice("wasabi", "WASABI / B0-B1", make_wasabi_model),
        ModelChoice("multipool_lorentzian", "Multipool Lorentzian", make_multipool_lorentzian_model),
        ModelChoice("ab_mt_n", "AB-MT+n", make_AB_MT_n_model),
        ModelChoice("zaiss_full", "Zaiss full analytical", make_zaiss_full_model),
    )


@lru_cache(maxsize=1)
def model_choices() -> tuple[ModelChoice, ...]:
    """Return supported built-in model choices."""

    return _choices()


def model_choice_map() -> dict[str, ModelChoice]:
    """Return model choices keyed by normalized name."""

    choices = {choice.key: choice for choice in model_choices()}
    aliases = {
        "t2star": "t2star_se",
        "t2*": "t2star_se",
        "diffusion": "dti",
        "dti_tensor": "dti",
        "kurtosis": "dki",
        "AB_MT_n": "ab_mt_n",
        "ab_mt": "ab_mt_n",
        "zaiss": "zaiss_full",
        "zaiss_analytical": "zaiss_full",
        "full_analytical": "zaiss_full",
    }
    for alias, target in aliases.items():
        choices[alias.lower()] = choices[target]
    return choices


def normalize_model_key(name: str | None) -> str | None:
    """Resolve a user/manifest model name to a GUI registry key when possible."""

    if not name:
        return None
    choices = model_choice_map()
    key = str(name).strip()
    choice = choices.get(key) or choices.get(key.lower())
    return choice.key if choice is not None else None


def make_model(
    name: str | None,
    *,
    param_overrides: dict[str, dict[str, object]] | None = None,
    pools: Sequence[str] | None = None,
    pool_overrides: dict[str, dict[str, object]] | None = None,
    parameter_constant_dims: dict[str, tuple[int | str, ...]] | None = None,
) -> Model:
    """Instantiate a built-in model by registry key or manifest name."""

    key = normalize_model_key(name) or "identity"
    param_overrides = _filter_param_overrides_for_model(key, param_overrides)
    if param_overrides or pools is not None:
        from mrax.library import (
            make_AB_MT_n_model,
            make_exp_decay_model,
            make_fair_asl_model,
            make_multipool_lorentzian_model,
            make_t1_ir_model,
            make_t1_vfa_model,
            make_t1_vtr_model,
            make_t2_se_model,
            make_t2star_se_model,
            make_wasabi_model,
            make_zaiss_full_model,
        )
        from mrax.library.default_models import make_dki_kurtosis_model, make_dti_tensor_model

        scale = 1e6
        kwargs = {"param_overrides": param_overrides} if param_overrides else {}
        if key == "t1_ir":
            model = make_t1_ir_model(scale=scale, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "t1_vtr":
            model = make_t1_vtr_model(scale=scale, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "t1_vfa":
            model = make_t1_vfa_model(scale=scale, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "exp_decay":
            model = make_exp_decay_model(scale=scale, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "t2_se":
            model = make_t2_se_model(scale=scale, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "t2star_se":
            model = make_t2star_se_model(scale=scale, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "fair_asl":
            model = make_fair_asl_model(scale=scale, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "dti":
            model = make_dti_tensor_model(**kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "dki":
            model = make_dki_kurtosis_model(**kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "wasabi":
            model = make_wasabi_model(**kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "multipool_lorentzian":
            model = make_multipool_lorentzian_model(pools=pools, pool_overrides=pool_overrides, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "ab_mt_n":
            model = make_AB_MT_n_model(pools=pools, pool_overrides=pool_overrides, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
        if key == "zaiss_full":
            model = make_zaiss_full_model(pools=pools, pool_overrides=pool_overrides, **kwargs)
            return _with_constant_dims(model, parameter_constant_dims)
    return _with_constant_dims(model_choice_map()[key].factory(), parameter_constant_dims)


def _filter_param_overrides_for_model(
    key: str,
    param_overrides: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, object]] | None:
    """Drop stale GUI overrides from another selected model."""

    if not param_overrides or key in {"multipool_lorentzian", "ab_mt_n", "zaiss_full"}:
        return param_overrides
    try:
        from mrax.library.parameter_library import resolve_param_specs

        valid = set(resolve_param_specs(key).keys())
    except Exception:
        return param_overrides
    filtered = {name: cfg for name, cfg in param_overrides.items() if name in valid}
    return filtered or None


def _with_constant_dims(
    model: Model,
    parameter_constant_dims: dict[str, tuple[int | str, ...]] | None,
) -> Model:
    if parameter_constant_dims:
        model.parameter_constant_dims = parameter_constant_dims
    return model


def model_default_pools(name: str | None) -> tuple[str, ...]:
    key = normalize_model_key(name) or "identity"
    try:
        from mrax.library.parameter_library import get_model_defaults

        return tuple(str(item) for item in get_model_defaults(key).get("pools", ()))
    except Exception:
        return ()


def model_available_pools(name: str | None) -> tuple[str, ...]:
    key = normalize_model_key(name) or "identity"
    try:
        from mrax.library.parameter_library import get_model_pools

        return tuple(str(item) for item in get_model_pools(key).keys())
    except Exception:
        return ()


def model_parameter_specs(
    name: str | None,
    *,
    pools: Sequence[str] | None = None,
    param_overrides: dict[str, dict[str, object]] | None = None,
    pool_overrides: dict[str, dict[str, object]] | None = None,
    parameter_constant_dims: dict[str, tuple[int | str, ...]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return GUI-editable field-space parameter specs for the selected model."""

    model_key = normalize_model_key(name) or "identity"
    model = make_model(
        name,
        pools=pools,
        param_overrides=param_overrides,
        pool_overrides=pool_overrides,
        parameter_constant_dims=parameter_constant_dims,
    )
    params, unpack = model.parametrization_factory((1, 1))
    fields = unpack(params)
    specs: dict[str, dict[str, Any]] = {}
    for key, value in fields.items():
        block_until_ready = getattr(value, "block_until_ready", None)
        arr = block_until_ready() if block_until_ready is not None else value
        try:
            initial = float(arr.reshape((-1,))[0])
        except Exception:
            initial = 0.0
        specs[str(key)] = {
            "initial": initial,
            "bounds": None,
            "unit": (model.parameter_units or {}).get(str(key), ""),
        }
    try:
        from mrax.library.parameter_library import resolve_param_specs

        base_specs = resolve_param_specs(model_key, overrides=param_overrides)
        for param_name, cfg in base_specs.items():
            if param_name in specs and cfg.get("bounds") is not None:
                specs[param_name]["bounds"] = list(cfg["bounds"])
    except Exception:
        _LOG.debug("Ignored optional operation failure", exc_info=True)
    if model_key in {"multipool_lorentzian", "ab_mt_n", "zaiss_full"}:
        try:
            from mrax.library.parameter_library import get_model_pools

            available = get_model_pools(model_key)
            for pool_name in pools or model_default_pools(model_key):
                pool_spec = dict(available.get(str(pool_name), {}))
                pool_spec.update((pool_overrides or {}).get(str(pool_name), {}))
                for raw_param, cfg in pool_spec.items():
                    if not isinstance(cfg, dict):
                        continue
                    param_name = f"{raw_param}_{pool_name}"
                    if param_name in specs and cfg.get("bounds") is not None:
                        specs[param_name]["bounds"] = list(cfg["bounds"])
        except Exception:
            _LOG.debug("Ignored optional operation failure", exc_info=True)
    return specs


__all__ = [
    "ModelChoice",
    "make_model",
    "model_choice_map",
    "model_choices",
    "model_available_pools",
    "model_default_pools",
    "model_parameter_specs",
    "normalize_model_key",
]
