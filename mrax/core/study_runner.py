from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

from mrax.core.fit_sanity import check_fit_sanity
from mrax.core.loss_builder import run_fit
from mrax.data_struct import DataSet, Experiment, Model, Study
from mrax.data_struct.quantity import require_quantity_dict
from mrax.devices import DeviceLike, resolve_device

_PASSTHROUGH_DICT_KWARGS = {
    "box_constraints",
    "initial_params",
    "regularizer_configs",
    "regularizer_type",
    "regularizer_weights",
    "study_globals",
}


@dataclass
class PlannedFit:
    """Planned experiment fit (ordered, optional cached fields)."""

    name: str
    experiment: Experiment
    model: Model
    dataset: DataSet
    cached_fields: dict | None = None


def plan_study_fits(
    study: Study,
    target_names: Sequence[str] | None = None,
    precomputed_fields: dict[str, dict] | None = None,
) -> list[PlannedFit]:
    """Return a sequential plan for a Study (optionally filtered/cached)."""

    entries = study.entries_by_name()
    order = study.execution_order(target_names)
    cached = precomputed_fields or {}
    return [
        PlannedFit(
            name=name,
            experiment=entries[name][0],
            model=entries[name][1],
            dataset=entries[name][2],
            cached_fields=cached.get(name),
        )
        for name in order
    ]


def execute_study_plan(
    plan: Sequence[PlannedFit],
    run_fit_kwargs: dict | None = None,
    run_fit_overrides: dict[str, dict] | None = None,
    *,
    skip_missing_observables: bool = True,
    warn_on_skip: bool = True,
    device: DeviceLike = "cpu",
):
    """Execute a planned Study fit in order, returning result entries and field cache.

    When ``skip_missing_observables`` is True, experiments missing required observables
    are skipped with an explanatory warning and a ``skipped`` result entry.

    ``run_fit_kwargs`` are passed to every fit. A value
    can be a mapping keyed by plan entry name to provide per-experiment values, except
    for dict-valued ``run_fit`` options such as ``regularizer_weights`` and
    ``study_globals``. Use ``run_fit_overrides`` for explicit per-experiment keyword
    overrides.
    """

    device = resolve_device(device)
    run_fit_kwargs = run_fit_kwargs or {}
    plan_names = {item.name for item in plan}
    results: list[dict] = []

    def _select_for_name(key, value, name):
        if (
            isinstance(value, dict)
            and key not in _PASSTHROUGH_DICT_KWARGS
            and value
            and set(value).issubset(plan_names)
        ):
            return value.get(name)
        return value

    for item in plan:
        if item.cached_fields is not None:
            fitted_fields = require_quantity_dict(item.cached_fields, f"Cached fields for {item.name}")
            losses = []
            cached = True
        else:
            kwargs = {key: _select_for_name(key, val, item.name) for key, val in run_fit_kwargs.items()}
            if run_fit_overrides and item.name in run_fit_overrides:
                kwargs.update(run_fit_overrides[item.name])
            kwargs.setdefault("device", device)
            if skip_missing_observables:
                report = check_fit_sanity(
                    item.experiment,
                    item.model,
                    item.dataset,
                    study_globals=kwargs.get("study_globals"),
                )
                if report.missing_observables or report.missing_any_of:
                    if warn_on_skip:
                        warnings.warn(
                            f"Skipping '{item.name}' (model={item.model.name}): missing observables. "
                            "Use Experiment.insert_observables(...) and re-run.\n"
                            + report.explain(),
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    results.append(
                        {
                            "name": item.name,
                            "model": item.model.name,
                            "fields": {},
                            "loss_history": [],
                            "cached": False,
                            "skipped": True,
                            "skip_reason": report.explain(),
                        }
                    )
                    continue
            fitted_fields, losses = run_fit(item.experiment, item.model, item.dataset, **kwargs)
            cached = False

        results.append(
            {
                "name": item.name,
                "model": item.model.name,
                "fields": fitted_fields,
                "loss_history": losses,
                "cached": cached,
                "spatial_geometry": (
                    item.dataset.spatial_geometry.to_metadata()
                    if item.dataset.spatial_geometry is not None
                    else None
                ),
            }
        )

    return results, {item["name"]: item["fields"] for item in results}


def collect_skipped_experiments(
    plan: Sequence[PlannedFit],
    results: Sequence[dict],
) -> dict[str, PlannedFit]:
    """Return a mapping of skipped experiment names to PlannedFit entries."""
    plan_by_name = {item.name: item for item in plan}
    skipped: dict[str, PlannedFit] = {}
    for entry in results:
        if entry.get("skipped"):
            name = entry.get("name")
            if name is None:
                continue
            if name not in plan_by_name:
                raise KeyError(f"Skipped experiment '{name}' not found in plan")
            skipped[name] = plan_by_name[name]
    return skipped


__all__ = [
    "PlannedFit",
    "plan_study_fits",
    "execute_study_plan",
    "collect_skipped_experiments",
]
