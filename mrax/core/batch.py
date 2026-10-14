from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from mrax.data_struct.storage import save_study_results
from mrax.data_struct.study import Study
from mrax.devices import DeviceLike, resolve_device

from .study_runner import execute_study_plan, plan_study_fits


def discover_study_dirs(studies_root: str | Path, selected: Sequence[str] | None = None) -> list[Path]:
    """Return study directories to process, optionally filtering by name."""

    root = Path(studies_root)
    if not root.exists():
        raise FileNotFoundError(f"Studies root '{root}' does not exist")

    if selected:
        selected_names = [Path(name).name for name in selected]
        missing = sorted(name for name in selected_names if not (root / name).is_dir())
        if missing:
            raise ValueError(f"Requested studies not found in {root}: {', '.join(missing)}")
        dirs = [root / name for name in selected_names]
    else:
        dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

    if not dirs:
        raise ValueError(f"No study folders found under {root}")
    return dirs


def run_batch_studies(
    studies_root: str | Path,
    study_loader: Callable[[Path], Study | tuple[Study, dict[str, Any] | None]],
    output_root: str | Path,
    selected: Sequence[str] | None = None,
    analysis_fn: Callable[..., list[dict[str, Any]]] | None = None,
    analysis_kwargs: dict[str, Any] | None = None,
    save_kwargs: dict[str, Any] | None = None,
    device: DeviceLike = "cpu",
) -> list[dict[str, Any]]:
    """Load studies from a folder, run analysis/fitting, and persist results.

    The batch runner assumes no interdependencies between studies; it simply loops through
    them. Provide a ``study_loader`` that converts a study folder into a ``Study`` (plus
    optional metadata), and an optional ``analysis_fn`` to override the default
    Study planner/executor flow. Results are saved under ``output_root/<study_name>`` using
    :func:`save_study_results`.

    Returns a list of dictionaries summarizing each run with ``study_name``,
    ``manifest_path``, ``results``, ``metadata``, and ``study_dir`` keys.
    """

    device = resolve_device(device)
    study_dirs = discover_study_dirs(studies_root, selected)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    analysis_kwargs = analysis_kwargs or {}
    save_kwargs = save_kwargs or {}

    summaries: list[dict[str, Any]] = []
    for study_dir in study_dirs:
        loaded = study_loader(study_dir)
        if isinstance(loaded, Study):
            study, metadata = loaded, None
        else:
            study, metadata = loaded
        metadata = metadata or {}

        if analysis_fn is None:
            target_names = analysis_kwargs.get("target_names")
            precomputed_fields = analysis_kwargs.get("precomputed_fields")
            fit_kwargs = {
                key: value for key, value in analysis_kwargs.items() if key not in ("target_names", "precomputed_fields")
            }
            plan = plan_study_fits(
                study,
                target_names=target_names,
                precomputed_fields=precomputed_fields,
            )
            results, _ = execute_study_plan(plan, run_fit_kwargs=fit_kwargs, device=device)
            used_targets = target_names
        else:
            results = analysis_fn(study, study_dir=study_dir, metadata=metadata, **analysis_kwargs)
            used_targets = analysis_kwargs.get("target_names") or analysis_kwargs.get("targets")

        save_opts = dict(save_kwargs)
        if "targets" not in save_opts and used_targets is not None:
            save_opts["targets"] = list(used_targets)

        study_name = save_opts.pop("study_name", study_dir.name)
        manifest_path = save_study_results(
            study,
            results,
            output_dir=out_root / study_name,
            study_name=study_name,
            **save_opts,
        )

        summaries.append(
            {
                "study_name": study_name,
                "manifest_path": manifest_path,
                "results": results,
                "metadata": metadata,
                "study_dir": study_dir,
            }
        )
    return summaries
