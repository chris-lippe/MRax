from __future__ import annotations

from numbers import Real
from typing import Iterable, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from mrax.data_struct.model import Model
from mrax.data_struct.quantity import Quantity
from mrax.library.parameter_library import get_model_defaults, get_model_pools, resolve_param_specs

# Shared constants for saturation-transfer models
SAT_GAMMA = 2.0 * jnp.pi * 42.58  # rad/s/uT
SUPER_LORENTZIAN_LOOKUP_MIN = 1e-4
SUPER_LORENTZIAN_LOOKUP_MAX = 64.0
SUPER_LORENTZIAN_LOOKUP_SIZE = 2048
SUPER_LORENTZIAN_QUAD_ORDER = 96
SUPER_LORENTZIAN_MAGIC_EPS = 5e-3


def _normalize_abmt_line_shape_name(line_shape: object) -> str:
    """Normalize accepted AB-MT MT line-shape names."""
    key = str(line_shape).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lorentzian": "lorentzian",
        "gaussian": "gaussian",
        "super_lorentzian": "super_lorentzian",
        "superlorentzian": "super_lorentzian",
        "super_lor": "super_lorentzian",
        "superlor": "super_lorentzian",
    }
    if key not in aliases:
        allowed = "', '".join(sorted(aliases))
        raise ValueError(f"Unsupported AB-MT MT line_shape '{line_shape}'. Expected one of '{allowed}'")
    return aliases[key]


def _build_super_lorentzian_lookup() -> tuple[np.ndarray, np.ndarray]:
    """Prepare the semisolid lookup on the host without initializing a device."""
    # The ideal super-Lorentzian diverges at zero offset around the magic angle.
    # We tabulate a regularized kernel once and reuse it through interpolation so
    # the model stays finite and reasonably fast during iterative fitting.
    x_grid = np.geomspace(
        SUPER_LORENTZIAN_LOOKUP_MIN,
        SUPER_LORENTZIAN_LOOKUP_MAX,
        SUPER_LORENTZIAN_LOOKUP_SIZE - 1,
        dtype=np.float64,
    )
    nodes, weights = np.polynomial.legendre.leggauss(SUPER_LORENTZIAN_QUAD_ORDER)
    u_nodes = 0.5 * (nodes + 1.0)
    u_weights = 0.5 * weights
    denom = np.sqrt((3.0 * u_nodes**2 - 1.0) ** 2 + SUPER_LORENTZIAN_MAGIC_EPS**2)
    scaled = x_grid[:, None] / denom[None, :]
    kernel = np.sqrt(2.0 / np.pi) * np.exp(-2.0 * scaled**2) / denom[None, :]
    values = np.sum(kernel * u_weights[None, :], axis=1)
    x_lookup = np.concatenate(([0.0], x_grid))
    value_lookup = np.concatenate(([values[0]], values))
    x_lookup.setflags(write=False)
    value_lookup.setflags(write=False)
    return x_lookup, value_lookup


_SUPER_LORENTZIAN_LOOKUP_X, _SUPER_LORENTZIAN_LOOKUP_Y = _build_super_lorentzian_lookup()


def _arr(val):
    """Extract raw ndarray from Quantity or plain array."""
    return jnp.asarray(val.values if isinstance(val, Quantity) else val)


def _unit_map(keys: Iterable[str], unit: str) -> dict[str, str]:
    return {str(key): unit for key in keys}


def _bounded(raw, lo, hi):
    """Map raw values to [lo, hi] via sigmoid."""
    if isinstance(lo, Real) and isinstance(hi, Real) and lo < 0 and hi == -lo:
        return hi * jnn.tanh(0.5 * raw)
    return lo + (hi - lo) * jnn.sigmoid(raw)


def _inverse_transform(value, cfg):
    """Map constrained values back to raw space for initialization."""
    transform = cfg.get("transform", "softplus")
    arr = jnp.asarray(value)
    eps = 1e-6
    if transform == "softplus":
        scale = cfg.get("scale", 1.0)
        offset = cfg.get("offset", 0.0)
        scaled = (arr - offset) / scale
        scaled = jnp.maximum(scaled, eps)
        return scaled + jnp.log(-jnp.expm1(-scaled))
    if transform == "bounded":
        lo, hi = cfg["bounds"]
        if isinstance(lo, Real) and isinstance(hi, Real) and lo < 0 and hi == -lo:
            scaled = jnp.clip(arr / hi, -1.0 + 2.0 * eps, 1.0 - 2.0 * eps)
            return 2.0 * jnp.arctanh(scaled)
        lo_arr = jnp.asarray(lo)
        hi_arr = jnp.asarray(hi)
        denom = jnp.where(jnp.abs(hi_arr - lo_arr) < eps, 1.0, hi_arr - lo_arr)
        frac = (arr - lo_arr) / denom
        frac = jnp.clip(frac, eps, 1.0 - eps)
        return jnp.log(frac) - jnp.log1p(-frac)
    if transform == "tanh":
        scale = cfg.get("scale", 1.0)
        scaled = arr / scale
        scaled = jnp.clip(scaled, -1.0 + eps, 1.0 - eps)
        return jnp.arctanh(scaled)
    if transform == "linear":
        scale = cfg.get("scale", 1.0)
        offset = cfg.get("offset", 0.0)
        return (arr - offset) / scale
    if transform == "identity":
        return arr
    raise ValueError(f"Unknown transform '{transform}' for parameter initialization")


def _resolve_init_raw(cfg):
    if "init" in cfg:
        init_val = _arr(cfg["init"])
        return _inverse_transform(init_val, cfg)
    if "init_raw" in cfg:
        return _arr(cfg["init_raw"])
    return jnp.array(0.0)


def _require_observable(
    freq_offsets: object | None, observables: dict | None, keys: Iterable[str] | str, model_name: str,
    *, field_ndim: int = 2,
):
    """Resolve a required observable from freq_offsets or a mapping."""
    if isinstance(keys, str):
        keys = (keys,)
    candidate = freq_offsets
    if candidate is None and observables is not None:
        for key in keys:
            val = observables.get(key)
            if val is not None:
                candidate = val
                break
    if candidate is None:
        joined = "/".join(keys)
        raise ValueError(f"{joined} observable is required for the {model_name} model")
    return _arr(candidate).reshape((-1,) + (1,) * field_ndim)


def _field_ndim(fields, observables=None):
    """Count spatial axes, excluding fields' already-expanded frame axis."""
    if observables is not None and "_mrax_spatial_ndim" in observables:
        return int(observables["_mrax_spatial_ndim"])
    return max((_arr(value).ndim for value in fields.values()), default=0)


def _resolve_scalar_observable(
    observables: dict | None,
    keys: Iterable[str] | str,
    default: float,
) -> float:
    """Fetch a scalar observable by key(s), falling back to a default."""
    if isinstance(keys, str):
        keys = (keys,)
    if observables:
        for key in keys:
            if key in observables:
                arr = _arr(observables[key])
                if arr.ndim == 0:
                    return float(arr)
                return float(arr.reshape((-1,))[0])
    return float(default)


def _resolve_bool_observable(
    observables: dict | None,
    keys: Iterable[str] | str,
    default: bool,
) -> bool:
    """Fetch a scalar boolean observable by key(s), falling back to a default."""
    if isinstance(keys, str):
        keys = (keys,)
    if observables:
        for key in keys:
            if key in observables:
                value = observables[key]
                value = value.values if isinstance(value, Quantity) else value
                try:
                    arr = np.asarray(value)
                except jax.errors.TracerArrayConversionError as exc:
                    raise ValueError(f"Observable '{key}' must be static before JIT") from exc
                if arr.size == 0:
                    return bool(default)
                first = arr.reshape((-1,))[0]
                if not np.all(arr == first):
                    raise ValueError(f"Observable '{key}' must have one static Boolean value")
                if isinstance(first, (str, np.str_)):
                    token = str(first).strip().strip("<>").lower()
                    if token in ("on", "true", "1", "1.0"):
                        return True
                    if token in ("off", "false", "0", "0.0"):
                        return False
                elif first == 0 or first == 1:
                    return bool(first)
                raise ValueError(f"Observable '{key}' must be a Boolean value")
    return bool(default)


def _resolve_string_observable(
    observables: dict | None,
    keys: Iterable[str] | str,
    default: object | None = None,
) -> object | None:
    """Fetch a scalar string-like observable by key(s), falling back to a default."""
    if isinstance(keys, str):
        keys = (keys,)
    if observables:
        for key in keys:
            if key in observables:
                val = observables[key]
                if isinstance(val, Quantity):
                    val = val.values
                arr = np.asarray(val)
                if arr.size == 0:
                    return default
                return arr.reshape((-1,))[0].item()
    return default


def _resolve_field_strength_mhz(observables: dict | None) -> float:
    """Extract scanner field strength in MHz; raises if absent."""
    if observables is None or "field_strength_mhz" not in observables:
        raise ValueError(
            "field_strength_mhz is required; set it once via Study.study_globals['field_strength_mhz']"
        )
    field = observables.get("field_strength_mhz")
    arr = jnp.asarray(field.values if isinstance(field, Quantity) else field)
    return arr.reshape((-1,))[0]

def _extract_sat_observables(
    freq_offsets,
    observables: dict | None,
    model_name: str,
    require_sat_time: bool,
    require_sat_b1: bool,
    field_ndim: int = 2,
):
    """Prepare shared saturation-transfer observables."""
    freqs = _require_observable(freq_offsets, observables, ("freq_offsets",), model_name, field_ndim=field_ndim)  # ppm
    field_strength_mhz = _resolve_field_strength_mhz(observables)
    sat_time_s = None
    sat_b1_uT = None
    obs = observables or {}
    if "sat_time" in obs:
        sat_time_s = _arr(obs["sat_time"]).reshape((-1,) + (1,) * field_ndim) * 1e-3  # ms -> s
    elif require_sat_time:
        raise ValueError("sat_time observable is required for the saturation-transfer model")
    if "sat_b1" in obs:
        sat_b1_uT = _arr(obs["sat_b1"]).reshape((-1,) + (1,) * field_ndim)
    elif require_sat_b1:
        raise ValueError("sat_b1 observable is required for the saturation-transfer model")
    return freqs, field_strength_mhz, sat_time_s, sat_b1_uT

def _map_obs(obs: dict, key: str):
    val = obs.get(key)
    if val is None:
        return None
    arr = _arr(val)
    if arr.ndim == 0:
        arr = arr[None]
    return arr


def _apply_b0_correction(freqs_ppm: jnp.ndarray, observables: dict | None) -> jnp.ndarray:
    """Shift frequency offsets by any available B0 observable (ppm)."""
    if not observables:
        return freqs_ppm
    manual_shift = observables.get("b0_shift_ppm", observables.get("b0_shift"))
    manual_arr = None
    if manual_shift is not None:
        manual_arr = _arr(manual_shift)
        if manual_arr.ndim == 0:
            manual_arr = manual_arr[None]
    for key in ("b0_map", "B0_map", "delta"):
        val = observables.get(key)
        if val is None:
            continue
        b0 = _arr(val)
        if b0.ndim == 0:
            b0 = b0[None]
        if manual_arr is not None:
            b0 = b0 + manual_arr
        return freqs_ppm - b0[None, ...]
    if manual_arr is not None:
        return freqs_ppm - manual_arr[None, ...]
    return freqs_ppm

def _make_parametrization_factory(param_specs: dict[str, dict[str, object]]):
    """Create a parametrization_factory from parameter specs."""

    def factory(spatial_shape):
        params = {}
        for name, cfg in param_specs.items():
            if cfg.get("trainable", True):
                init_raw = _resolve_init_raw(cfg)
                init_arr = jnp.asarray(init_raw)
                if init_arr.shape == ():
                    params[f"{name}_raw"] = jnp.full(spatial_shape, init_arr)
                else:
                    params[f"{name}_raw"] = jnp.broadcast_to(init_arr, spatial_shape)

        def unpack(p):
            fields = {}
            for name, cfg in param_specs.items():
                if not cfg.get("trainable", True):
                    value = cfg.get("value", 0.0)
                    arr = _arr(value)
                    if arr.shape == ():
                        arr = jnp.full(spatial_shape, arr)
                    fields[name] = arr
                    continue

                raw = p[f"{name}_raw"]
                transform = cfg.get("transform", "softplus")
                if transform == "softplus":
                    scale = cfg.get("scale", 1.0)
                    offset = cfg.get("offset", 0.0)
                    fields[name] = scale * jnn.softplus(raw) + offset
                elif transform == "bounded":
                    lo, hi = cfg["bounds"]
                    fields[name] = _bounded(raw, lo, hi)
                elif transform == "tanh":
                    scale = cfg.get("scale", 1.0)
                    fields[name] = scale * jnn.tanh(raw)
                elif transform == "linear":
                    scale = cfg.get("scale", 1.0)
                    offset = cfg.get("offset", 0.0)
                    fields[name] = scale * raw + offset
                elif transform == "identity":
                    fields[name] = raw
                else:
                    raise ValueError(f"Unknown transform '{transform}' for parameter '{name}'")
            return fields

        unpack.parameter_specs = param_specs
        return params, unpack

    factory.parameter_specs = param_specs
    return factory


def _units_from_specs(model_name: str, param_specs: dict[str, dict[str, object]]):
    units = {}
    for name, cfg in param_specs.items():
        unit = cfg.get("unit")
        if unit is None:
            raise ValueError(f"Missing unit for parameter '{name}' in model '{model_name}'")
        units[name] = str(unit)
    return units


def _sat_trans_observable_units() -> dict[str, str]:
    return {
        "freq_offsets": "ppm",
        "offsets": "ppm",
        "offset": "ppm",
        "field_strength_mhz": "MHz",
        "sat_time": "ms",
        "sat_b1": "uT",
        "b0_map": "ppm",
        "B0_map": "ppm",
        "delta": "ppm",
        "b0_shift_ppm": "ppm",
        "b0_shift": "ppm",
        "b1_map": "%",
        "B1_map": "%",
    }


def _merge_pool_spec(base: dict | None, override: dict) -> dict:
    merged = dict(base or {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
            if 'bounds' in value and not {'init', 'init_raw'} & value.keys():
                initial = merged[key].get('init')
                lower, upper = value['bounds']
                if initial is not None and not np.all((np.asarray(initial) > lower) & (np.asarray(initial) < upper)):
                    merged[key].pop('init', None)
        else:
            merged[key] = value
    return merged


def _normalize_multipool_pool_spec(pool_name: str, spec: dict) -> dict[str, dict[str, object]]:
    if not isinstance(spec, dict):
        raise TypeError(f"Pool spec for '{pool_name}' must be a dict")
    if any(isinstance(spec.get(key), dict) for key in ("amp", "gamma", "dw")):
        return {key: dict(val) for key, val in spec.items() if isinstance(val, dict)}

    if "amp" not in spec or "gamma" not in spec:
        raise ValueError(f"Pool spec for '{pool_name}' must define amp/gamma bounds")
    amp_bounds = spec["amp"]
    gamma_bounds = spec["gamma"]
    dw_val = spec.get("dw", 0.0)
    train_dw = bool(spec.get("train_dw", False))

    amp_spec = {"transform": "bounded", "bounds": tuple(amp_bounds), "unit": "1"}
    gamma_spec = {"transform": "bounded", "bounds": tuple(gamma_bounds), "unit": "ppm"}

    if isinstance(dw_val, (tuple, list)):
        dw_lo, dw_hi = float(dw_val[0]), float(dw_val[1])
    else:
        dw_lo = dw_hi = float(dw_val)
    if train_dw:
        dw_spec = {"transform": "bounded", "bounds": (dw_lo, dw_hi), "unit": "ppm", "trainable": True}
    else:
        dw_spec = {"transform": "identity", "trainable": False, "value": dw_lo, "unit": "ppm"}

    return {"amp": amp_spec, "gamma": gamma_spec, "dw": dw_spec}


def _normalize_ab_mt_pool_spec(pool_name: str, spec: dict) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if not isinstance(spec, dict):
        raise TypeError(f"Pool spec for '{pool_name}' must be a dict")
    meta: dict[str, object] = {}
    if "enable_mt_offset" in spec:
        meta["enable_mt_offset"] = bool(spec["enable_mt_offset"])
    if "rate_key" in spec:
        meta["rate_key"] = str(spec["rate_key"])
    line_shape = spec.get("line_shape", spec.get("lineshape"))
    if line_shape is not None:
        meta["line_shape"] = _normalize_abmt_line_shape_name(line_shape)
    params: dict[str, dict[str, object]] = {}

    if "f" in spec:
        f_val = spec["f"]
        if isinstance(f_val, dict):
            params["f"] = dict(f_val)
        else:
            params["f"] = {"transform": "bounded", "bounds": tuple(f_val), "unit": "1"}

    rate_key = meta.get("rate_key")
    if rate_key is None:
        if "r2" in spec:
            rate_key = "r2"
        elif "k" in spec:
            rate_key = "k"
    if rate_key is not None:
        meta["rate_key"] = str(rate_key)
        if rate_key in spec:
            rate_val = spec[rate_key]
            if isinstance(rate_val, dict):
                params[str(rate_key)] = dict(rate_val)
            else:
                params[str(rate_key)] = {"transform": "bounded", "bounds": tuple(rate_val), "unit": "ppm"}

    if "dw" in spec:
        dw_val = spec["dw"]
        if isinstance(dw_val, dict):
            params["dw"] = dict(dw_val)
        else:
            train_dw = bool(spec.get("train_dw", False))
            if isinstance(dw_val, (tuple, list)):
                dw_lo, dw_hi = float(dw_val[0]), float(dw_val[1])
            else:
                dw_lo = dw_hi = float(dw_val)
            if train_dw or isinstance(dw_val, (tuple, list)):
                params["dw"] = {"transform": "bounded", "bounds": (dw_lo, dw_hi), "unit": "ppm"}
            else:
                params["dw"] = {"transform": "identity", "trainable": False, "value": dw_lo, "unit": "ppm"}

    return params, meta


def _param_spec_from_value(
    value: object,
    *,
    unit: str,
    train: bool = False,
) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (tuple, list)):
        lo, hi = float(value[0]), float(value[1])
        return {"transform": "bounded", "bounds": (lo, hi), "unit": unit}
    val = float(value)
    if train:
        return {"transform": "identity", "unit": unit, "init": val}
    return {"transform": "identity", "trainable": False, "value": val, "unit": unit}


def _normalize_zaiss_pool_spec(pool_name: str, spec: dict) -> dict[str, dict[str, object]]:
    if not isinstance(spec, dict):
        raise TypeError(f"Pool spec for '{pool_name}' must be a dict")

    params: dict[str, dict[str, object]] = {}
    if "f" in spec:
        f_val = spec["f"]
        if isinstance(f_val, dict):
            params["f"] = dict(f_val)
        else:
            params["f"] = _param_spec_from_value(f_val, unit="1")

    for key in ("r1", "r2", "k", "dw"):
        if key not in spec:
            continue
        train_key = f"train_{key}"
        params[key] = _param_spec_from_value(
            spec[key],
            unit="ppm",
            train=bool(spec.get(train_key, False)),
        )

    if "r2" not in params:
        params["r2"] = {"transform": "identity", "trainable": False, "value": 0.0, "unit": "ppm"}
    return params


def _resolve_mt_line_shape(line_shape: object | None) -> str:
    """Resolve the MT pool line-shape name."""
    if line_shape is None:
        return "lorentzian"
    return _normalize_abmt_line_shape_name(line_shape)


def _super_lorentzian_lineshape(delta: jnp.ndarray, r2_mt: jnp.ndarray, eps: float) -> jnp.ndarray:
    """Evaluate a regularized semisolid super-Lorentzian kernel."""
    r2_safe = jnp.maximum(jnp.abs(r2_mt), eps)
    scaled = jnp.abs(delta) / r2_safe
    scaled_flat = jnp.reshape(scaled, (-1,))
    scaled_clipped = jnp.clip(scaled_flat, 0.0, SUPER_LORENTZIAN_LOOKUP_MAX)
    kernel_flat = jnp.interp(scaled_clipped, _SUPER_LORENTZIAN_LOOKUP_X, _SUPER_LORENTZIAN_LOOKUP_Y)
    kernel_flat = jnp.where(scaled_flat > SUPER_LORENTZIAN_LOOKUP_MAX, 0.0, kernel_flat)
    kernel = jnp.reshape(kernel_flat, scaled.shape)
    return kernel / r2_safe


def _mt_rrf(
    delta: jnp.ndarray,
    omega1: jnp.ndarray,
    r2_mt: jnp.ndarray,
    *,
    line_shape: str,
    eps: float,
) -> jnp.ndarray:
    """Compute the MT pool RF absorption rate for the selected line shape."""
    if line_shape == "gaussian":
        return omega1**2 * jnp.sqrt(jnp.pi / 2.0) / (r2_mt + eps) * jnp.exp(-((delta / (2.0 * r2_mt + eps)) ** 2))
    if line_shape == "super_lorentzian":
        return omega1**2 * _super_lorentzian_lineshape(delta, r2_mt, eps)
    return omega1**2 * r2_mt / (r2_mt**2 + delta**2 + eps)


def _make_relaxometry_model(
    name: str,
    observable_keys: Iterable[str] | str,
    param_specs: dict[str, dict[str, object]],
    signal_fn,
):
    parametrization_factory = _make_parametrization_factory(param_specs)
    units = _units_from_specs(name, param_specs)
    if isinstance(observable_keys, str):
        obs_keys = (observable_keys,)
    else:
        obs_keys = tuple(observable_keys)
    obs_units = _unit_map(obs_keys, "ms")

    def model_apply(fields, freq_offsets, observables=None):
        times = _require_observable(freq_offsets, observables, observable_keys, name, field_ndim=_field_ndim(fields, observables))
        return signal_fn(times, fields)

    return Model(
        name=name,
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units=units,
        required_observables=obs_keys,
        observable_units=obs_units,
    )


def _require_diffusion_observables(observables: dict | None, model_name: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    if observables is None:
        raise ValueError(f"bvals/bvecs observables are required for the {model_name} model")
    bvals = observables.get("bvals", observables.get("bval"))
    bvecs = observables.get("bvecs", observables.get("bvec"))
    if bvals is None or bvecs is None:
        raise ValueError(f"bvals/bvecs observables are required for the {model_name} model")
    bvals_arr = _arr(bvals).reshape((-1,))
    bvecs_arr = _arr(bvecs)
    if bvecs_arr.ndim == 1:
        if bvecs_arr.size % 3 != 0:
            raise ValueError(f"bvecs must be shape (N, 3) for the {model_name} model")
        bvecs_arr = bvecs_arr.reshape((-1, 3))
    if bvecs_arr.ndim != 2 or bvecs_arr.shape[1] != 3:
        raise ValueError(f"bvecs must be shape (N, 3) for the {model_name} model")
    if bvecs_arr.shape[0] == 1 and bvals_arr.size > 1:
        bvecs_arr = jnp.broadcast_to(bvecs_arr, (bvals_arr.size, 3))
    if bvecs_arr.shape[0] != bvals_arr.size:
        raise ValueError(
            f"bvecs length {bvecs_arr.shape[0]} does not match bvals length {bvals_arr.size} for {model_name}"
        )
    return bvals_arr, bvecs_arr


def _dti_tensor_from_fields(fields: dict[str, jnp.ndarray]) -> jnp.ndarray:
    dxx = _arr(fields["dxx"])
    dyy = _arr(fields["dyy"])
    dzz = _arr(fields["dzz"])
    dxy = _arr(fields["dxy"])
    dxz = _arr(fields["dxz"])
    dyz = _arr(fields["dyz"])
    row0 = jnp.stack([dxx, dxy, dxz], axis=-1)
    row1 = jnp.stack([dxy, dyy, dyz], axis=-1)
    row2 = jnp.stack([dxz, dyz, dzz], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def _dti_metrics_from_tensor(tensor: jnp.ndarray) -> dict[str, jnp.ndarray]:
    evals = jnp.linalg.eigvalsh(tensor)
    evals_sorted = jnp.sort(evals, axis=-1)[..., ::-1]
    md = jnp.mean(evals_sorted, axis=-1)
    numerator = 1.5 * jnp.sum((evals_sorted - md[..., None]) ** 2, axis=-1)
    denom = jnp.sum(evals_sorted**2, axis=-1)
    fa = jnp.sqrt(jnp.where(denom > 0, numerator / denom, 0.0))
    ad = evals_sorted[..., 0]
    rd = jnp.mean(evals_sorted[..., 1:], axis=-1)
    trace = jnp.sum(evals_sorted, axis=-1)
    return {"adc": md, "md": md, "fa": fa, "ad": ad, "rd": rd, "trace": trace}


def _make_sat_trans_model(
    name: str,
    param_specs: dict[str, dict[str, object]],
    signal_fn,
    derived_maps_fn=None,
    require_sat_time: bool = False,
    require_sat_b1: bool = False,
    required_observables: Sequence[str] | None = None,
    observable_units: dict[str, str] | None = None,
):
    parametrization_factory = _make_parametrization_factory(param_specs)
    units = _units_from_specs(name, param_specs)
    obs_units = _sat_trans_observable_units()
    if observable_units:
        obs_units.update({str(k): str(v) for k, v in observable_units.items()})
    required = ["freq_offsets", "field_strength_mhz"]
    if require_sat_time:
        required.append("sat_time")
    if require_sat_b1:
        required.append("sat_b1")
    if required_observables:
        for key in required_observables:
            if key not in required:
                required.append(str(key))

    def model_apply(fields, freq_offsets, observables=None):
        freqs, field_strength, sat_time_s, sat_b1_uT = _extract_sat_observables(
            freq_offsets, observables, name, require_sat_time, require_sat_b1, _field_ndim(fields, observables)
        )
        return signal_fn(fields, freqs, observables or {}, field_strength, sat_time_s, sat_b1_uT)

    model = Model(
        name=name,
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units=units,
        required_observables=tuple(required),
        observable_units=obs_units,
    )
    if derived_maps_fn is not None:
        def derived_maps(fields, observables=None):
            field_strength = _resolve_field_strength_mhz(observables)
            return derived_maps_fn(fields, field_strength, observables or {})
        model.derived_maps = derived_maps  # type: ignore[attr-defined]
    return model


def make_t1_ir_model(
    scale: float = 1.0,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """Inversion recovery T1 model.

    ``scale`` controls the softplus magnitude for amplitude/bias maps to match data range.
    """
    param_specs = resolve_param_specs(
        "t1_ir",
        key=param_key,
        context={"scale": scale},
        overrides=param_overrides,
    )

    def signal_fn(ti, fields):
        a = fields["amplitude"][None, ...]
        t1 = fields["t1"][None, ...]
        return jnp.abs(a - 2.0 * a * jnp.exp(-ti / t1)) + 1e-6

    return _make_relaxometry_model("t1_ir", ("TI",), param_specs, signal_fn)


def make_fair_asl_model(
    scale: float = 1.0,
    t1_blood_ms: float = 2430.0,
    partition_coeff: float = 90.0,
    perfusion_scale: float = 60000.0,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """FAIR ASL model with selective/non-selective T1 maps and derived perfusion map."""
    param_specs = resolve_param_specs(
        "fair_asl",
        key=param_key,
        context={"scale": scale},
        overrides=param_overrides,
    )
    parametrization_factory = _make_parametrization_factory(param_specs)
    units = _units_from_specs("fair_asl", param_specs)

    def model_apply(fields, freq_offsets, observables=None):
        ti = _require_observable(freq_offsets, observables, ("TI",), "fair_asl", field_ndim=_field_ndim(fields, observables))
        labels = _require_observable(None, observables, ("fair_label", "fair_labels"), "fair_asl", field_ndim=_field_ndim(fields, observables))
        non_mask = labels >= 0.5
        a_sel = fields["amplitude_sel"][None, ...]
        a_non = fields["amplitude_non"][None, ...]
        t1_sel = fields["t1_sel"][None, ...]
        t1_non = fields["t1_non"][None, ...]
        amp = jnp.where(non_mask, a_non, a_sel)
        t1 = jnp.where(non_mask, t1_non, t1_sel)
        return jnp.abs(amp - 2.0 * amp * jnp.exp(-ti / t1)) + 1e-6

    def derived_maps(fields, observables=None):
        t1_sel = _arr(fields["t1_sel"])
        t1_non = _arr(fields["t1_non"])
        t1_blood = _resolve_scalar_observable(
            observables,
            ("t1_blood", "t1_blood_ms", "T1_blood"),
            t1_blood_ms,
        )
        part_coeff = _resolve_scalar_observable(
            observables,
            ("partition_coeff", "partition_coefficient", "lambda"),
            partition_coeff,
        )
        eps = 1e-6
        perf = (
            part_coeff
            * perfusion_scale
            * (t1_non / (t1_blood + eps))
            * (1.0 / (t1_sel + eps) - 1.0 / (t1_non + eps))
        )
        return {"t1_sel": t1_sel, "t1_non": t1_non, "perfusion_map": perf}

    model = Model(
        name="fair_asl",
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units=units,
        required_observables=("TI", "fair_label"),
        observable_units={
            "TI": "ms",
            "fair_label": "1",
            "fair_labels": "1",
            "t1_blood": "ms",
            "t1_blood_ms": "ms",
            "T1_blood": "ms",
            "partition_coeff": "1",
            "partition_coefficient": "1",
            "lambda": "1",
        },
    )
    model.derived_maps = derived_maps  # type: ignore[attr-defined]
    return model


def make_t1_vtr_model(
    scale: float = 1.0,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """Variable-TR T1 model."""
    param_specs = resolve_param_specs(
        "t1_vtr",
        key=param_key,
        context={"scale": scale},
        overrides=param_overrides,
    )

    def signal_fn(tr, fields):
        a = fields["amplitude"][None, ...]
        t1 = fields["t1"][None, ...]
        bias = fields["bias"][None, ...]
        return a * (1.0 - jnp.exp(-tr / t1)) + bias

    return _make_relaxometry_model("t1_vtr", ("TR",), param_specs, signal_fn)


def make_t1_vfa_model(
    scale: float = 1.0,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """Variable flip-angle T1 model using spoiled GRE steady state."""
    param_specs = resolve_param_specs(
        "t1_vfa",
        key=param_key,
        context={"scale": scale},
        overrides=param_overrides,
    )
    parametrization_factory = _make_parametrization_factory(param_specs)
    units = _units_from_specs("t1_vfa", param_specs)

    def model_apply(fields, freq_offsets, observables=None):
        tr = _require_observable(None, observables, ("TR",), "t1_vfa", field_ndim=_field_ndim(fields, observables))
        flip = _require_observable(
            freq_offsets,
            observables,
            ("flip_angle", "flip_angles", "FA", "alpha"),
            "t1_vfa",
            field_ndim=_field_ndim(fields, observables),
        )
        alpha = jnp.deg2rad(flip)
        t1 = fields["t1"][None, ...]
        e1 = jnp.exp(-tr / t1)
        numerator = (1.0 - e1) * jnp.sin(alpha)
        denominator = 1.0 - jnp.cos(alpha) * e1
        amp = fields["amplitude"][None, ...]
        bias = fields["bias"][None, ...]
        return amp * numerator / (denominator + 1e-6) + bias

    return Model(
        name="t1_vfa",
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units=units,
        required_observables=("TR", "flip_angle"),
        observable_units={
            "TR": "ms",
            "flip_angle": "deg",
            "flip_angles": "deg",
            "FA": "deg",
            "alpha": "deg",
        },
    )


def make_t2_se_model(
    scale: float = 1.0,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """T2 spin-echo model backed by the shared exponential-decay model."""

    return make_exp_decay_model(
        scale=scale,
        param_key=param_key,
        param_overrides=param_overrides,
        name="t2_se",
        decay_parameter="t2",
        decay_bounds=(5.0, 200.0),
    )


def make_t2star_se_model(
    scale: float = 1.0,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """T2* gradient-echo decay model backed by the shared exponential-decay model."""

    return make_exp_decay_model(
        scale=scale,
        param_key=param_key,
        param_overrides=param_overrides,
        name="t2star_se",
        decay_parameter="t2star",
        decay_bounds=(1.0, 200.0),
    )


def make_exp_decay_model(
    scale: float = 1.0,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
    *,
    name: str = "exp_decay",
    decay_parameter: str = "decay_time",
    decay_bounds: tuple[float, float] | None = None,
):
    """Mono-exponential decay model ``S = amplitude * exp(-TE / tau) + bias``.

    ``make_t2_se_model`` and ``make_t2star_se_model`` use this factory with only
    the decay parameter renamed to ``t2`` or ``t2star``.
    """

    decay_parameter = str(decay_parameter)
    param_specs = resolve_param_specs(
        "exp_decay",
        key=param_key,
        context={"scale": scale},
    )
    decay_spec = dict(param_specs.pop("decay_time"))
    if decay_bounds is not None:
        lo, hi = (float(v) for v in decay_bounds)
        decay_spec["bounds"] = (lo, hi)
        if "init" in decay_spec:
            decay_spec["init"] = 0.5 * (lo + hi)
    param_specs[decay_parameter] = decay_spec
    param_specs = _apply_param_overrides(param_specs, param_overrides, decay_parameter=decay_parameter)

    def signal_fn(te, fields):
        a = fields["amplitude"][None, ...]
        tau = fields[decay_parameter][None, ...]
        c = fields["bias"][None, ...]
        return a * jnp.exp(-te / tau) + c

    return _make_relaxometry_model(name, ("TE",), param_specs, signal_fn)


def _apply_param_overrides(
    param_specs: dict[str, dict[str, object]],
    overrides: dict[str, dict[str, object]] | None,
    *,
    decay_parameter: str | None = None,
) -> dict[str, dict[str, object]]:
    if not overrides:
        return param_specs
    merged = {key: dict(value) for key, value in param_specs.items()}
    for name, override in overrides.items():
        target = decay_parameter if decay_parameter and name == "decay_time" else str(name)
        if override is None:
            merged.pop(target, None)
            continue
        existing = merged.get(target)
        if isinstance(existing, dict):
            merged[target] = {**existing, **override}
        else:
            merged[target] = dict(override)
    return merged


def make_dti_tensor_model(
    scale: float = 1e4,
    diffusivity_scale: float = 1e-3,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """Basic diffusion tensor model (DTI).

    Signal model: ``S = S0 * exp(-b * g^T D g)`` with a symmetric 3x3 diffusion tensor.
    """
    param_specs = resolve_param_specs(
        "dti",
        key=param_key,
        context={"scale": scale, "diffusivity_scale": diffusivity_scale},
        overrides=param_overrides,
    )
    parametrization_factory = _make_parametrization_factory(param_specs)
    units = _units_from_specs("dti", param_specs)

    def model_apply(fields, freq_offsets, observables=None):
        bvals, bvecs = _require_diffusion_observables(observables, "dti")
        spatial_ndim = fields["s0"].ndim
        reshape = (bvals.shape[0],) + (1,) * spatial_ndim
        b = bvals.reshape(reshape)
        gx = bvecs[:, 0].reshape(reshape)
        gy = bvecs[:, 1].reshape(reshape)
        gz = bvecs[:, 2].reshape(reshape)
        dxx = fields["dxx"]
        dyy = fields["dyy"]
        dzz = fields["dzz"]
        dxy = fields["dxy"]
        dxz = fields["dxz"]
        dyz = fields["dyz"]
        dt = (
            dxx * gx**2
            + dyy * gy**2
            + dzz * gz**2
            + 2.0 * dxy * gx * gy
            + 2.0 * dxz * gx * gz
            + 2.0 * dyz * gy * gz
        )
        s0 = fields["s0"]
        return s0[None, ...] * jnp.exp(-b * dt) + 1e-6

    def derived_maps(fields, observables=None):
        tensor = _dti_tensor_from_fields(fields)
        return _dti_metrics_from_tensor(tensor)

    model = Model(
        name="dti",
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units=units,
        required_observables=("bvals", "bvecs"),
        observable_units={
            "bvals": "s/mm^2",
            "bval": "s/mm^2",
            "bvecs": "1",
            "bvec": "1",
        },
    )
    model.derived_maps = derived_maps  # type: ignore[attr-defined]
    model._mrax_param_specs = param_specs  # type: ignore[attr-defined]
    diff_unit = units.get("dxx", "1")
    model.derived_map_units = {
        "adc": diff_unit,
        "md": diff_unit,
        "ad": diff_unit,
        "rd": diff_unit,
        "trace": diff_unit,
        "fa": "1",
    }
    return model


def make_dki_kurtosis_model(
    scale: float = 1e4,
    diffusivity_scale: float = 1e-3,
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """Diffusion kurtosis model (DKI) with isotropic kurtosis."""
    param_specs = resolve_param_specs(
        "dki",
        key=param_key,
        context={"scale": scale, "diffusivity_scale": diffusivity_scale},
        overrides=param_overrides,
    )
    parametrization_factory = _make_parametrization_factory(param_specs)
    units = _units_from_specs("dki", param_specs)

    def model_apply(fields, freq_offsets, observables=None):
        bvals, bvecs = _require_diffusion_observables(observables, "dki")
        spatial_ndim = fields["s0"].ndim
        reshape = (bvals.shape[0],) + (1,) * spatial_ndim
        b = bvals.reshape(reshape)
        gx = bvecs[:, 0].reshape(reshape)
        gy = bvecs[:, 1].reshape(reshape)
        gz = bvecs[:, 2].reshape(reshape)
        dxx = fields["dxx"]
        dyy = fields["dyy"]
        dzz = fields["dzz"]
        dxy = fields["dxy"]
        dxz = fields["dxz"]
        dyz = fields["dyz"]
        dt = (
            dxx * gx**2
            + dyy * gy**2
            + dzz * gz**2
            + 2.0 * dxy * gx * gy
            + 2.0 * dxz * gx * gz
            + 2.0 * dyz * gy * gz
        )
        k = fields["kurtosis"]
        s0 = fields["s0"]
        return s0[None, ...] * jnp.exp(-b * dt + (b**2) * (dt**2) * k / 6.0) + 1e-6

    def derived_maps(fields, observables=None):
        tensor = _dti_tensor_from_fields(fields)
        maps = _dti_metrics_from_tensor(tensor)
        maps["mk"] = _arr(fields["kurtosis"])
        return maps

    model = Model(
        name="dki",
        model_apply=model_apply,
        parametrization_factory=parametrization_factory,
        parameter_units=units,
        required_observables=("bvals", "bvecs"),
        observable_units={
            "bvals": "s/mm^2",
            "bval": "s/mm^2",
            "bvecs": "1",
            "bvec": "1",
        },
    )
    model.derived_maps = derived_maps  # type: ignore[attr-defined]
    diff_unit = units.get("dxx", "1")
    model.derived_map_units = {
        "adc": diff_unit,
        "md": diff_unit,
        "ad": diff_unit,
        "rd": diff_unit,
        "trace": diff_unit,
        "fa": "1",
        "mk": "1",
    }
    return model


def make_wasabi_model(
    name: str = "wasabi",
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    param_specs = resolve_param_specs(
        "wasabi",
        key=param_key,
        overrides=param_overrides,
    )
    def signal_fn(fields, freqs, observables, field_strength_mhz, sat_time_s, sat_b1_uT):
        c = fields["c"][None, ...]
        d = fields["d"][None, ...]
        b1 = fields["b1"][None, ...]
        delta = fields["delta"][None, ...] # ppm
        if sat_time_s is None:
            raise ValueError("sat_time observable is required for the WASABI model")
        tp = sat_time_s
        denom = (freqs - delta)*field_strength_mhz * (2.0 * jnp.pi)  # ppm -> rad/s
        denom = jnp.where(jnp.abs(denom) < 1e-6, 1e-6, denom)
        term1 = jnp.arctan(SAT_GAMMA * b1 / denom)
        term2 = jnp.sqrt((SAT_GAMMA * b1) ** 2 + denom**2) * tp / 2.0
        return jnp.abs(c - d * (jnp.sin(term1) ** 2) * (jnp.sin(term2) ** 2)) + 1e-6

    def derived_maps_fn(fields, field_strength_mhz, observables):
        b1 = fields["b1"].values if isinstance(fields["b1"], Quantity) else fields["b1"]
        delta = fields["delta"].values if isinstance(fields["delta"], Quantity) else fields["delta"]
        sat_b1 = observables.get("sat_b1")
        if sat_b1 is None:
            raise ValueError("sat_b1 observable is required for the WASABI model")
        nominal = float(_arr(sat_b1).reshape((-1,))[0])
        b1_map = (b1 - nominal) / nominal * 100.0  # percent deviation from nominal
        b0_map = delta  # ppm
        return {"B1_map": b1_map, "B0_map": b0_map}

    return _make_sat_trans_model(
        name=name,
        param_specs=param_specs,
        signal_fn=signal_fn,
        derived_maps_fn=derived_maps_fn,
        require_sat_time=True,
        require_sat_b1=True,
    )


def make_multipool_lorentzian_model(
    pools: Sequence[str] | None = None,
    pool_overrides: dict | None = None,
    name: str = "multipool_lorentzian",
    baseline_range: tuple[float, float] = (0.3, 1.0),
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
):
    """
    Multi-pool Lorentzian Z-spectra model for CEST/CESL scans.

    Parameters
    ----------
    pools : Sequence[str] | None
        Ordered pool names to include. Defaults to ["water", "mt"].
    pool_overrides : dict | None
        Optional mapping ``pool -> {amp, gamma, dw, train_dw}`` or full param-spec dicts
        to tweak ranges or add new pools.
    name : str
        Model name to attach to the returned ``Model``.
    baseline_range : tuple[float, float]
        Lower/upper bounds for the baseline term.
    param_key : str
        Parameter-set key to pull from the parameter library.
    param_overrides : dict | None
        Param-spec overrides (per-parameter) merged on top of the library.
    """
    defaults = get_model_defaults("multipool_lorentzian")
    base_pool_defs = get_model_pools("multipool_lorentzian")
    pool_order = list(pools) if pools is not None else list(defaults.get("pools", ("water", "mt")))

    pool_defs: dict[str, dict[str, dict[str, object]]] = {}
    for pool_name in pool_order:
        base_spec = base_pool_defs.get(pool_name)
        normalized_base = _normalize_multipool_pool_spec(pool_name, base_spec) if base_spec is not None else None
        override_spec = None
        if pool_overrides and pool_name in pool_overrides:
            override_spec = _normalize_multipool_pool_spec(pool_name, pool_overrides[pool_name])
        if normalized_base is None and override_spec is None:
            raise ValueError(f"Unknown pool '{pool_name}' requested for {name}")
        if normalized_base is None:
            pool_defs[pool_name] = override_spec
        elif override_spec is None:
            pool_defs[pool_name] = normalized_base
        else:
            pool_defs[pool_name] = _merge_pool_spec(normalized_base, override_spec)

    def lorentzian(freq_offsets, amp, center, gamma):
        half = gamma / 2.0
        return amp * half**2 / ((freq_offsets - center) ** 2 + half**2)

    def signal_fn(fields, freqs, observables, field_strength_mhz, sat_time_s, sat_b1_uT):
        freqs_ppm = _apply_b0_correction(freqs, observables)
        freqs_angular = freqs_ppm * (2.0 * jnp.pi)

        # Saturation-transfer offsets are treated as angular ppm (2*pi * ppm).
        # Empirical linewidth/rate parameters remain in the units declared by
        # the parameter library, matching the AB-MT convention below.
        spec_sum = jnp.zeros_like(freqs_angular)
        for pool_name in pool_order:
            amp = fields[f"amp_{pool_name}"][None, ...]
            gamma = fields[f"gamma_{pool_name}"][None, ...]
            dw_val_ppm = fields.get(f"dw_{pool_name}", 0.0)
            dw = 2.0 * jnp.pi * jnp.asarray(dw_val_ppm)[None, ...]
            spec_sum += lorentzian(freqs_angular, amp, dw, gamma)
        baseline = fields.get("baseline", 1.0)
        return baseline - spec_sum

    param_specs = resolve_param_specs(
        "multipool_lorentzian",
        key=param_key,
        overrides=param_overrides,
    )
    if baseline_range is not None and (param_overrides is None or "baseline" not in param_overrides):
        if "baseline" in param_specs:
            param_specs["baseline"] = {**param_specs["baseline"], "bounds": tuple(baseline_range)}

    for pool_name in pool_order:
        spec = pool_defs[pool_name]
        for param_name, cfg in spec.items():
            if not isinstance(cfg, dict):
                continue
            param_specs[f"{param_name}_{pool_name}"] = cfg

    return _make_sat_trans_model(
        name=name,
        param_specs=param_specs,
        signal_fn=signal_fn,
    )

def make_AB_MT_n_model(
    pools: Sequence[str] | None = None,
    pool_overrides: dict | None = None,
    name: str = "ab_mt_n",
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
    enable_water_offset: bool | None = None,
    water_offset_bounds: tuple[float, float] = (-1.0, 1.0),
    enable_water_offset_linear_drift: bool = False,
    water_offset_drift_bounds: tuple[float, float] = (-0.05, 0.05),
):
    """
    AB-MT+n Z-spectra model for CEST/CESL scans.

    Parameters
    ----------
    pools : Sequence[str] | None
        Ordered CEST pool names to include. Defaults to ["mt"].
    pool_overrides : dict | None
        Optional mapping ``pool -> {...}`` to tweak pool parameter specs or metadata
        such as ``enable_mt_offset`` and ``line_shape``.
    name : str
        Model name to attach to the returned ``Model``.
    param_key : str
        Parameter-set key to pull from the parameter library.
    param_overrides : dict | None
        Param-spec overrides (per-parameter) merged on top of the library.
    enable_water_offset : bool | None
        Add a trainable residual water-offset parameter ``dw_water`` in ppm.
        If ``None``, this is enabled when ``param_overrides`` defines
        ``dw_water``.
    water_offset_bounds : tuple[float, float]
        Default bounded range for ``dw_water`` when no explicit override is
        provided.
    enable_water_offset_linear_drift : bool
        Add a trainable ``dw_water_drift`` slope in ppm/min. The measured
        ``b0_map`` observable remains the fixed intercept, so the modeled
        water shift is ``b0_map + dw_water_drift * water_offset_time``.
    water_offset_drift_bounds : tuple[float, float]
        Default bounded range in ppm/min for ``dw_water_drift``.
    """
    defaults = get_model_defaults("ab_mt_n")
    base_pool_defs = get_model_pools("ab_mt_n")
    pool_order = list(pools) if pools is not None else list(defaults.get("pools", ("mt",)))
    water_offset_enabled = (
        bool(enable_water_offset)
        if enable_water_offset is not None
        else bool(param_overrides and "dw_water" in param_overrides)
    )
    water_offset_linear_drift_enabled = bool(enable_water_offset_linear_drift)

    pool_defs: dict[str, dict[str, dict[str, object]]] = {}
    pool_meta: dict[str, dict[str, object]] = {}
    for pool_name in pool_order:
        base_spec = base_pool_defs.get(pool_name)
        base_params, base_meta = (
            _normalize_ab_mt_pool_spec(pool_name, base_spec)
            if base_spec is not None
            else ({}, {})
        )
        override_params = None
        override_meta: dict[str, object] = {}
        if pool_overrides and pool_name in pool_overrides:
            override_params, override_meta = _normalize_ab_mt_pool_spec(pool_name, pool_overrides[pool_name])
        if not base_params and override_params is None:
            raise ValueError(f"Unknown pool '{pool_name}' requested for {name}")
        if not base_params:
            pool_defs[pool_name] = override_params
            pool_meta[pool_name] = override_meta
        elif override_params is None:
            pool_defs[pool_name] = base_params
            pool_meta[pool_name] = base_meta
        else:
            pool_defs[pool_name] = _merge_pool_spec(base_params, override_params)
            pool_meta[pool_name] = {**base_meta, **override_meta}

    eps = 1e-8
    mt_meta = pool_meta.get("mt", {})
    default_mt_offset = bool(mt_meta.get("enable_mt_offset", False))
    default_mt_line_shape = mt_meta.get("line_shape")
    mt_rate_key = mt_meta.get("rate_key")
    if mt_rate_key is None:
        mt_rate_key = "r2" if "r2" in pool_defs.get("mt", {}) else "k"

    pool_rate_keys: dict[str, str] = {}
    for pool_name in pool_order:
        meta = pool_meta.get(pool_name, {})
        rate_key = meta.get("rate_key")
        if rate_key is None:
            rate_key = "r2" if "r2" in pool_defs[pool_name] else "k"
        pool_rate_keys[pool_name] = str(rate_key)

    def offres_contrib(freqs, w1, f, k, dw):
        denom = 4.0 * w1**2 * dw**2 + (k**2 + dw**2) ** 2
        denom = denom + eps
        a = 2.0 * dw**3 * f * k / denom
        b = dw**2 * f * k * (dw**2 + k**2) / denom
        d = dw**2 * f * k * (3.0 * dw**2 - k**2) / denom
        return (d - a * freqs) / ((freqs - dw) ** 2 + k**2 + w1**2 + eps) + (b + a * freqs) / (
            freqs**2 + w1**2 + eps
        )

    def signal_fn(fields, freqs, observables, field_strength_mhz, sat_time_s, sat_b1_uT):
        if sat_time_s is None:
            raise ValueError("sat_time observable is required for the AB-MT+n model")
        if sat_b1_uT is None:
            raise ValueError("sat_b1 observable is required for the AB-MT+n model")

        t_delay_ppm = fields["t_delay"][None, ...] * 1e-3 * field_strength_mhz
        rep_time_ms = _require_observable(None, observables, ("TR", "tr"), name, field_ndim=_field_ndim(fields, observables))
        if rep_time_ms.shape[0] != freqs.shape[0]:
            rep_len = int(rep_time_ms.shape[0])
            freq_len = int(freqs.shape[0])
            if rep_len != 1 and freq_len % rep_len == 0:
                rep_time_ms = jnp.tile(rep_time_ms, (freq_len // rep_len,) + (1,) * (rep_time_ms.ndim - 1))
            elif rep_len != 1:
                raise ValueError(
                    f"TR length {rep_len} does not match freq_offsets length {freq_len} for {name}"
                )
        rec_time_ppm = (rep_time_ms * 1e-3 - sat_time_s) * field_strength_mhz - t_delay_ppm
        sat_time_ppm = sat_time_s * field_strength_mhz

        r1_map = _map_obs(observables, "r1_map")
        if r1_map is None:
            t1_map = _map_obs(observables, "t1_map")
            if t1_map is None:
                raise ValueError("r1_map or t1_map observable is required for the AB-MT+n model")
            r1_map = 1.0 / (t1_map * 1e-3 + eps) / field_strength_mhz  # (T1) ms -> (R1) ppm
        else:
            r1_map = r1_map / field_strength_mhz  # (R1) s^-1 -> (R1) ppm
        r1_obs = r1_map[None, ...]

        freqs_ppm = _apply_b0_correction(freqs, observables)
        if water_offset_linear_drift_enabled:
            drift_time_s = _require_observable(
                None,
                observables,
                ("water_offset_time", "b0_drift_time", "t_seconds"),
                name,
                field_ndim=_field_ndim(fields, observables),
            )
            drift_time_min = drift_time_s / 60.0
            freqs_ppm = freqs_ppm - fields["dw_water_drift"][None, ...] * drift_time_min
        if water_offset_enabled:
            # Residual B0 correction: Delta omega_corr = Delta omega - delta W.
            freqs_ppm = freqs_ppm - fields["dw_water"][None, ...]
        freqs = freqs_ppm * (2.0 * jnp.pi)

        b1_map = _map_obs(observables, "b1_map")
        b1_scale = 1.0
        if b1_map is not None:
            b1_scale = 1.0 + b1_map[None, ...] * 0.01
        omega1 = SAT_GAMMA * sat_b1_uT * b1_scale / field_strength_mhz  # uT -> ppm

        spin_lock = _resolve_bool_observable(observables, ("spin_lock", "SpinLock"), False)
        mt_offset = _resolve_bool_observable(observables, "mt_offset", default_mt_offset)
        mt_line_shape = _resolve_mt_line_shape(
            _resolve_string_observable(
                observables,
                ("mt_line_shape", "line_shape", "lineshape"),
                default_mt_line_shape,
            )
        )

        has_mt = "mt" in pool_order
        mt_rate = mt_rate_key if mt_rate_key is not None else "k"
        f_mt = fields["f_mt"][None, ...] if has_mt else jnp.zeros_like(freqs[:1])
        r2_mt = fields[f"{mt_rate}_mt"][None, ...] if has_mt else jnp.zeros_like(f_mt)
        a_coeff = fields["a"][None, ...]
        b_coeff = fields["b"][None, ...]
        sin_a_eff = fields["sin_a_eff"][None, ...]

        dw_mt = jnp.asarray(fields.get("dw_mt", 0.0))[None, ...] if (mt_offset and has_mt) else 0.0

        freq_abs = jnp.abs(freqs)
        denom = jnp.sqrt(freqs**2 + omega1**2 + eps)
        projection = jnp.ones_like(freqs) if spin_lock else freq_abs / denom
        r1_res = r1_obs * projection * freq_abs / denom

        if has_mt:
            delta_mt = freqs - dw_mt * 2.0 * jnp.pi
            rrf_mt = _mt_rrf(delta_mt, omega1, r2_mt, line_shape=mt_line_shape, eps=eps)
            r_ex_mt = rrf_mt * f_mt / (1.0 + f_mt) * freqs**2 / (freqs**2 + omega1**2 + eps)
        else:
            r_ex_mt = 0.0
        r1rho = (r1_obs * freqs**2 + b_coeff * omega1**2 + a_coeff * omega1**2 * freqs) / (
            freqs**2 + omega1**2 + eps
        )
        r1rho = r1rho + r_ex_mt

        cest_pools = [pool for pool in pool_order if pool != "mt"]
        for pool_name in cest_pools:
            f_val = fields[f"f_{pool_name}"][None, ...]
            rate_key = pool_rate_keys.get(pool_name, "k")
            k_val = fields[f"{rate_key}_{pool_name}"][None, ...]
            dw_val = fields[f"dw_{pool_name}"][None, ...] * 2.0 * jnp.pi
            divisor = 1.0 + f_mt if has_mt else 1.0
            r1rho += omega1**2 * offres_contrib(freqs, omega1, f_val, k_val, dw_val) / divisor

        z_ss = r1_res / (r1rho + eps)

        z_prev_obs = observables.get("z_prev", None)
        if z_prev_obs is None:
            z_prev_obs = observables.get("Z_prev", None)
        if z_prev_obs is not None:
            z_prev_arr = _arr(z_prev_obs)
            if z_prev_arr.ndim == 1:
                z_prev = z_prev_arr.reshape((-1,) + (1,) * _field_ndim(fields, observables))
            else:
                z_prev = z_prev_arr
        else:
            z_prev = jnp.ones_like(freqs)

        c_factor = jnp.exp(-r1_obs * t_delay_ppm)
        recovery = jnp.exp(-r1_obs * rec_time_ppm)
        sqrt_term = jnp.sqrt(jnp.maximum(1.0 / (sin_a_eff**2 + eps) - 1.0, 0.0))
        z_rec = 1.0 + (sqrt_term * z_prev - 1.0) * recovery
        z_model = 1.0 - c_factor + c_factor * ((z_rec * projection**2 - z_ss) * jnp.exp(-r1rho * sat_time_ppm) + z_ss)

        return z_model

    param_specs = resolve_param_specs(
        "ab_mt_n",
        key=param_key,
        overrides=param_overrides,
    )
    if water_offset_enabled:
        water_offset_spec = {
            "transform": "bounded",
            "bounds": tuple(water_offset_bounds),
            "unit": "ppm",
            "init": 0.0,
        }
        if "dw_water" in param_specs:
            water_offset_spec.update(param_specs["dw_water"])
        param_specs["dw_water"] = water_offset_spec
    if water_offset_linear_drift_enabled:
        water_offset_drift_spec = {
            "transform": "bounded",
            "bounds": tuple(water_offset_drift_bounds),
            "unit": "ppm/min",
            "init": 0.0,
        }
        if "dw_water_drift" in param_specs:
            water_offset_drift_spec.update(param_specs["dw_water_drift"])
        param_specs["dw_water_drift"] = water_offset_drift_spec

    for pool_name in pool_order:
        spec = pool_defs[pool_name]
        rate_key = pool_rate_keys[pool_name]
        for param_name in ("f", rate_key, "dw"):
            if param_name not in spec:
                raise ValueError(f"Pool '{pool_name}' missing '{param_name}' spec for {name}")
            param_specs[f"{param_name}_{pool_name}"] = spec[param_name]

    return _make_sat_trans_model(
        name=name,
        param_specs=param_specs,
        signal_fn=signal_fn,
        require_sat_time=True,
        require_sat_b1=True,
        required_observables=("TR",),
        observable_units={
            "TR": "ms",
            "r1_map": "1/s",
            "t1_map": "ms",
            "z_prev": "1",
            "Z_prev": "1",
            "z_reference_image": "1",
            "water_offset_time": "s",
            "b0_drift_time": "s",
            "t_seconds": "s",
            "spin_lock": "1",
            "SpinLock": "1",
            "mt_offset": "1",
            "mt_line_shape": "any",
            "line_shape": "any",
            "lineshape": "any",
        },
    )


def make_zaiss_full_model(
    pools: Sequence[str] | None = None,
    pool_overrides: dict | None = None,
    name: str = "zaiss_full",
    param_key: str = "default",
    param_overrides: dict[str, dict[str, object]] | None = None,
    enable_water_offset: bool | None = None,
    water_offset_bounds: tuple[float, float] = (-1.0, 1.0),
):
    """
    Full analytical cw saturation-transfer model following Zaiss et al.

    This model evaluates the approximate Bloch-McConnell Z-spectrum solution
    used as the starting point for AB-MT+n: ``R1rho = Reff + sum(Rex_i)`` with
    each exchange contribution described by the Zaiss/Trott-Palmer analytical
    rate term. Frequency offsets and pool chemical shifts are converted to the
    same angular-ppm convention used by the other MRax saturation-transfer
    models: ``delta_omega = 2*pi*freq_offsets_ppm`` and
    ``omega1 = gamma*sat_b1_uT/field_strength_mhz``.
    """
    defaults = get_model_defaults("zaiss_full")
    base_pool_defs = get_model_pools("zaiss_full")
    pool_order = list(pools) if pools is not None else list(defaults.get("pools", ("mt",)))
    water_offset_enabled = (
        bool(enable_water_offset)
        if enable_water_offset is not None
        else bool(param_overrides and "dw_water" in param_overrides)
    )

    pool_defs: dict[str, dict[str, dict[str, object]]] = {}
    for pool_name in pool_order:
        base_spec = base_pool_defs.get(pool_name)
        normalized_base = _normalize_zaiss_pool_spec(pool_name, base_spec) if base_spec is not None else None
        override_spec = None
        if pool_overrides and pool_name in pool_overrides:
            override_spec = _normalize_zaiss_pool_spec(pool_name, pool_overrides[pool_name])
        if normalized_base is None and override_spec is None:
            raise ValueError(f"Unknown pool '{pool_name}' requested for {name}")
        if normalized_base is None:
            pool_defs[pool_name] = override_spec
        elif override_spec is None:
            pool_defs[pool_name] = normalized_base
        else:
            pool_defs[pool_name] = _merge_pool_spec(normalized_base, override_spec)

    eps = 1e-8

    def r1_observable_ppm(observables, field_strength_mhz):
        r1_map = _map_obs(observables, "r1_map")
        if r1_map is not None:
            return r1_map[None, ...] / field_strength_mhz
        t1_map = _map_obs(observables, "t1_map")
        if t1_map is not None:
            return (1.0 / (t1_map * 1e-3 + eps) / field_strength_mhz)[None, ...]
        return None

    def zaiss_rex(freqs, omega1, f, k, r2, dw):
        gamma2_over4 = (r2 + k) / (k + eps) * omega1**2 + (r2 + k) ** 2
        direct_term = r2 + k * (dw**2 + r2 * (r2 + k)) / (freqs**2 + omega1**2 + eps)
        return f * omega1**2 * direct_term / (gamma2_over4 + (freqs - dw) ** 2 + eps)

    def signal_fn(fields, freqs, observables, field_strength_mhz, sat_time_s, sat_b1_uT):
        if sat_time_s is None:
            raise ValueError("sat_time observable is required for the Zaiss full analytical model")
        if sat_b1_uT is None:
            raise ValueError("sat_b1 observable is required for the Zaiss full analytical model")

        freqs_ppm = _apply_b0_correction(freqs, observables)
        if water_offset_enabled:
            freqs_ppm = freqs_ppm - fields["dw_water"][None, ...]
        freqs = freqs_ppm * (2.0 * jnp.pi)

        b1_map = _map_obs(observables, "b1_map")
        b1_scale = 1.0
        if b1_map is not None:
            b1_scale = 1.0 + b1_map[None, ...] * 0.01
        omega1 = SAT_GAMMA * sat_b1_uT * b1_scale / field_strength_mhz

        r1w = fields["r1w"][None, ...]
        r2w = fields["r2w"][None, ...]

        has_mt = "mt" in pool_order
        f_mt = fields["f_mt"][None, ...] if has_mt else jnp.zeros_like(freqs[:1])
        r1_obs = r1_observable_ppm(observables, field_strength_mhz)
        if r1_obs is None:
            r1_mt = fields["r1_mt"][None, ...] if (has_mt and "r1_mt" in fields) else r1w
            r1_obs = (r1w + f_mt * r1_mt) / (1.0 + f_mt) if has_mt else r1w

        reff = (r1w * freqs**2 + r2w * omega1**2) / (freqs**2 + omega1**2 + eps)
        r1rho = reff

        if has_mt:
            r1rho += zaiss_rex(
                freqs,
                omega1,
                f_mt,
                fields["k_mt"][None, ...],
                fields["r2_mt"][None, ...],
                fields["dw_mt"][None, ...] * 2.0 * jnp.pi,
            )

        divisor = 1.0 + f_mt if has_mt else 1.0
        for pool_name in pool_order:
            if pool_name == "mt":
                continue
            r1rho += zaiss_rex(
                freqs,
                omega1,
                fields[f"f_{pool_name}"][None, ...],
                fields[f"k_{pool_name}"][None, ...],
                fields[f"r2_{pool_name}"][None, ...],
                fields[f"dw_{pool_name}"][None, ...] * 2.0 * jnp.pi,
            ) / divisor

        spin_lock = _resolve_bool_observable(observables, ("spin_lock", "SpinLock"), False)
        freq_abs = jnp.abs(freqs)
        denom = jnp.sqrt(freqs**2 + omega1**2 + eps)
        cos_theta = freq_abs / denom
        projection = jnp.ones_like(freqs) if spin_lock else cos_theta
        r1_res = r1_obs * projection * cos_theta
        z_ss = r1_res / (r1rho + eps)
        sat_time_ppm = sat_time_s * field_strength_mhz
        z_initial = jnp.ones_like(freqs)
        return (projection**2 * z_initial - z_ss) * jnp.exp(-r1rho * sat_time_ppm) + z_ss

    param_specs = resolve_param_specs(
        "zaiss_full",
        key=param_key,
        overrides=param_overrides,
    )
    if water_offset_enabled:
        water_offset_spec = {
            "transform": "bounded",
            "bounds": tuple(water_offset_bounds),
            "unit": "ppm",
            "init": 0.0,
        }
        if "dw_water" in param_specs:
            water_offset_spec.update(param_specs["dw_water"])
        param_specs["dw_water"] = water_offset_spec

    for pool_name in pool_order:
        spec = pool_defs[pool_name]
        for param_name in ("f", "k", "r2", "dw"):
            if param_name not in spec:
                raise ValueError(f"Pool '{pool_name}' missing '{param_name}' spec for {name}")
            param_specs[f"{param_name}_{pool_name}"] = spec[param_name]
        if "r1" in spec:
            param_specs[f"r1_{pool_name}"] = spec["r1"]

    return _make_sat_trans_model(
        name=name,
        param_specs=param_specs,
        signal_fn=signal_fn,
        require_sat_time=True,
        require_sat_b1=True,
        observable_units={
            "r1_map": "1/s",
            "t1_map": "ms",
            "spin_lock": "1",
            "SpinLock": "1",
        },
    )


def make_identity_model(name: str = "identity") -> Model:
    """Trivial model that returns the amplitude field unchanged."""
    def model_apply(fields, offsets, observables=None):
        amp = fields["amplitude"]
        if offsets is None and observables:
            for val in observables.values():
                arr = jnp.asarray(val.values if isinstance(val, Quantity) else val)
                if arr.ndim == 1:
                    offsets = arr
                    break
        if offsets is None:
            return amp[None, ...]
        offs = jnp.asarray(offsets)
        if offs.ndim == 0:
            offs = offs[None]
        if amp.ndim >= 1 and int(amp.shape[0]) == int(offs.shape[0]):
            return amp
        return jnp.broadcast_to(amp, (int(offs.shape[0]),) + amp.shape)

    return Model(
        name=name,
        model_apply=model_apply,
        parametrization_factory=lambda spatial_shape: (
            {"amplitude": jnp.ones(spatial_shape)},
            lambda params: {"amplitude": params["amplitude"]},
        ),
        parameter_units={"amplitude": "1"},
    )


__all__ = [
    "make_fair_asl_model",
    "make_exp_decay_model",
    "make_t1_ir_model",
    "make_t1_vtr_model",
    "make_t1_vfa_model",
    "make_t2_se_model",
    "make_t2star_se_model",
    "make_dti_tensor_model",
    "make_dki_kurtosis_model",
    "make_wasabi_model",
    "make_multipool_lorentzian_model",
    "make_AB_MT_n_model",
    "make_zaiss_full_model",
    "make_identity_model",
]
