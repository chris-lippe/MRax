from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp

from mrax.data_struct import DataSet, Experiment, Model, Quantity, Study
from mrax.library import make_t1_ir_model, make_t2_se_model
from mrax.library.default_experiments import build_default_experiment


@dataclass
class Example:
    name: str
    dataset: DataSet
    experiment: Experiment
    model: Model
    truth_fields: dict | None = None


def _as_array(val):
    if isinstance(val, Quantity):
        return val.values
    return val


def _make_truth_maps(nx: int = 16, ny: int = 16):
    xg, yg = jnp.meshgrid(jnp.linspace(0.0, 1.0, nx), jnp.linspace(0.0, 1.0, ny), indexing="ij")
    amp = 0.9 + 0.35 * jnp.sin(2.0 * jnp.pi * xg) * jnp.cos(2.0 * jnp.pi * yg)  # ~[0.55, 1.25]

    # Limit sinusoid depth to keep T1/T2 well inside model bounds.
    t1_map = 1500.0 + 800.0 * jnp.sin(jnp.pi * xg) * jnp.cos(jnp.pi * yg)  # ~[700, 2300] ms
    t2_map = 70.0 + 40.0 * jnp.cos(jnp.pi * yg)  # ~[30, 110] ms

    bias_map = 0.05 * jnp.cos(2.0 * jnp.pi * (xg + yg))
    return {
        "t1_ir": {"amplitude": amp, "t1": t1_map},
        "t2_se": {"amplitude": 0.85 * amp + 0.15, "t2": t2_map, "bias": bias_map},
    }


def _build_example(
    name: str,
    offsets: jnp.ndarray,
    offset_key: str,
    offset_unit: str,
    model_factory,
    fields: dict,
    noise_level: float,
    rng,
    exp_type: str,
) -> Example:
    # First evaluate the clean signal to estimate intensity scale.
    base_model = model_factory(scale=1.0)
    clean = base_model.model_apply(fields, offsets)
    scale = float(jnp.percentile(jnp.abs(clean), 99))
    scale = max(1.0, scale)

    model = model_factory(scale=scale)
    noise = noise_level * jax.random.normal(rng, clean.shape)
    data = clean + noise

    dataset = DataSet(data=jnp.asarray(data), name=name, noise_level=noise_level)
    experiment = build_default_experiment(
        exp_type,
        frame_count=int(data.shape[0]),
        spatial_shape=data.shape[1:],
        observables={offset_key: Quantity(offsets, offset_unit)},
    )
    return Example(name=name, dataset=dataset, experiment=experiment, model=model, truth_fields=fields)


def make_example_suite(
    noise_level: float = 0.05, seed: int = 0, model_names: Sequence[str] | None = None
) -> List[Example]:
    names = tuple(model_names) if model_names is not None else ("t1_ir", "t2_se")
    truth = _make_truth_maps()
    rng = jax.random.PRNGKey(seed)
    examples: List[Example] = []

    for name in names:
        rng, sub = jax.random.split(rng)
        if name == "t1_ir":
            offsets = jnp.linspace(80.0, 1200.0, 10)
            examples.append(
                _build_example(
                    name,
                    offsets=offsets,
                    offset_key="TI",
                    offset_unit="ms",
                    model_factory=lambda scale=1.0: make_t1_ir_model(scale=scale),
                    fields=truth["t1_ir"],
                    noise_level=noise_level,
                    rng=sub,
                    exp_type="ir",
                )
            )
        elif name == "t2_se":
            offsets = jnp.linspace(8.0, 120.0, 12)
            examples.append(
                _build_example(
                    name,
                    offsets=offsets,
                    offset_key="TE",
                    offset_unit="ms",
                    model_factory=lambda scale=1.0: make_t2_se_model(scale=scale),
                    fields=truth["t2_se"],
                    noise_level=noise_level,
                    rng=sub,
                    exp_type="mse",
                )
            )
        else:
            raise ValueError(f"Unknown model '{name}'")

    return examples


def make_synthetic_study(
    noise_level: float = 0.05,
    seed: int = 0,
    model_names: Sequence[str] | None = None,
    study_globals: dict[str, Quantity] | None = None,
) -> Tuple[Study, dict]:
    """Build a ``Study`` from synthetic examples to mirror importer outputs."""
    examples = make_example_suite(noise_level=noise_level, seed=seed, model_names=model_names)
    study = Study.from_examples(examples, study_globals=study_globals)
    truth = {ex.name: ex.truth_fields for ex in examples if ex.truth_fields is not None}
    metadata = {
        "entries": examples,
        "source": "synthetic",
        "noise_level": noise_level,
        "seed": seed,
        "model_names": [ex.name for ex in examples],
        "truth_fields": truth,
    }
    return study, metadata
