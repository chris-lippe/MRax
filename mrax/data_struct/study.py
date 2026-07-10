from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Sequence

from .dataset import DataSet
from .experiment import Experiment
from .model import Model
from .quantity import Quantity, require_quantity_dict

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mrax.core.synthetic import Example


@dataclass
class Study:
    """Collection of experiments/datasets/models that can be processed together."""

    experiments: Sequence[Experiment]
    datasets: Sequence[DataSet]
    models: Sequence[Model]
    names: Sequence[str] | None = None
    study_globals: Dict[str, Quantity] | None = None

    def __post_init__(self):
        n_exp = len(self.experiments)
        if not (n_exp == len(self.datasets) == len(self.models)):
            raise ValueError("Study expects the same number of experiments, datasets, and models")

        exp_list = list(self.experiments)
        ds_list = list(self.datasets)
        model_list = list(self.models)
        if self.names is None:
            name_list = [ds.name or f"experiment_{idx}" for idx, ds in enumerate(ds_list)]
        else:
            name_list = list(self.names)
            if len(name_list) != n_exp:
                raise ValueError("Number of provided names must match the number of experiments")
        if len(set(name_list)) != len(name_list):
            raise ValueError("Study entry names must be unique")

        for experiment, dataset in zip(exp_list, ds_list):
            experiment.validate(dataset)

        globals_map: dict[str, Quantity] = {}
        if self.study_globals:
            globals_map = require_quantity_dict(dict(self.study_globals), "Study global")

        if "field_strength_mhz" not in globals_map:
            warnings.warn(
                "Study created without field_strength_mhz. Models that depend on it will raise an error; "
                "set Study.study_globals['field_strength_mhz'] to a Quantity with MHz units.",
                RuntimeWarning,
                stacklevel=2,
            )

        object.__setattr__(self, "experiments", exp_list)
        object.__setattr__(self, "datasets", ds_list)
        object.__setattr__(self, "models", model_list)
        object.__setattr__(self, "names", name_list)
        object.__setattr__(self, "study_globals", globals_map)
        # Attach study globals to experiments/models so standalone run_fit can pick them up.
        for exp in exp_list:
            try:
                exp.study_globals = globals_map
            except Exception:
                _LOG.debug("Ignored optional operation failure", exc_info=True)
        for mdl in model_list:
            try:
                mdl._mrax_study_globals = globals_map
            except Exception:
                _LOG.debug("Ignored optional operation failure", exc_info=True)
            self._wrap_model_with_globals(mdl, globals_map)

    @staticmethod
    def _wrap_model_with_globals(model: Model, globals_map: dict[str, Quantity]):
        """Ensure model_apply/derived_maps always see study globals when provided."""
        if not globals_map or getattr(model, "_mrax_globals_wrapped", False):
            return

        if hasattr(model, "model_apply"):
            base_apply = model.model_apply

            def model_apply_with_globals(
                fields,
                offsets,
                observables=None,
                *args,
                base_apply=base_apply,
                globals_map=globals_map,
                **kwargs,
            ):
                merged = dict(globals_map)
                if observables:
                    merged.update(observables)
                return base_apply(fields, offsets, merged if merged else observables, *args, **kwargs)

            model.model_apply = model_apply_with_globals  # type: ignore[assignment]

        if hasattr(model, "derived_maps"):
            base_derived = model.derived_maps  # type: ignore[attr-defined]

            def derived_with_globals(
                fields,
                observables=None,
                *args,
                base_derived=base_derived,
                globals_map=globals_map,
                **kwargs,
            ):
                merged = dict(globals_map)
                if observables:
                    merged.update(observables)
                return base_derived(fields, merged if merged else observables, *args, **kwargs)

            model.derived_maps = derived_with_globals  # type: ignore[assignment]

        model._mrax_globals_wrapped = True

    def __len__(self) -> int:
        return len(self.experiments)

    def iter_entries(self) -> Iterator[tuple[str, Experiment, Model, DataSet]]:
        """Yield (name, experiment, model, dataset) tuples for the study."""
        for name, experiment, model, dataset in zip(
            self.names, self.experiments, self.models, self.datasets
        ):
            yield name, experiment, model, dataset

    def entries_by_name(self) -> dict[str, tuple[Experiment, Model, DataSet]]:
        return {
            name: (experiment, model, dataset)
            for name, experiment, model, dataset in zip(
                self.names, self.experiments, self.models, self.datasets
            )
        }

    def execution_order(self, targets: Sequence[str] | None = None) -> list[str]:
        """Return the study order (as provided), optionally filtered by targets."""

        if targets is None:
            return list(self.names)
        missing = [name for name in targets if name not in self.names]
        if missing:
            raise ValueError(f"Unknown experiment(s) requested in targets: {', '.join(missing)}")
        target_set = set(targets)
        return [name for name in self.names if name in target_set]

    def describe(self, targets: Sequence[str] | None = None) -> str:
        """Human-readable multi-line summary of the study configuration."""

        items = []
        by_name = self.entries_by_name()
        for name in self.execution_order(targets):
            experiment, model, dataset = by_name[name]
            obs_keys = ", ".join(sorted(experiment.observables.keys())) or "none"
            items.append(
                f"{name}: model={model.name}, exp_type={experiment.exp_type or 'unknown'}, dims={experiment.dims}, "
                f"observables=[{obs_keys}], dataset={dataset.name}"
            )
        return "\n".join(items)

    def help(self) -> str:
        """Return a quick-start help summary for study exploration."""
        lines = [
            "Study help",
            f"experiments={len(self.experiments)}",
            "use study.describe() to list experiments and metadata",
            "use experiment.describe(dataset) for detailed observable summaries",
            "use study.explain_sanity() to check fit readiness across experiments",
        ]
        return "\n".join(lines)

    def sanity_check(self, *, study_globals: Dict[str, Quantity] | None = None):
        """Run per-experiment sanity checks for this study."""
        from mrax.core.fit_sanity import check_fit_sanity

        globals_map = study_globals or self.study_globals
        reports: dict[str, object] = {}
        for name, experiment, model, dataset in self.iter_entries():
            reports[name] = check_fit_sanity(experiment, model, dataset, study_globals=globals_map)
        return reports

    def explain_sanity(self, *, study_globals: Dict[str, Quantity] | None = None) -> str:
        """Return a combined sanity-check summary for all experiments."""
        reports = self.sanity_check(study_globals=study_globals)
        lines = ["Study sanity check"]
        for name, report in reports.items():
            lines.append(f"{name}:")
            lines.append(report.explain())
        return "\n".join(lines)

    @classmethod
    def from_examples(cls, examples: Sequence["Example"], study_globals: Dict[str, Quantity] | None = None):
        experiments = [ex.experiment for ex in examples]
        datasets = [ex.dataset for ex in examples]
        models = [ex.model for ex in examples]
        names = [getattr(ex, "name", f"experiment_{idx}") for idx, ex in enumerate(examples)]
        return cls(
            experiments=experiments,
            datasets=datasets,
            models=models,
            names=names,
            study_globals=study_globals,
        )
