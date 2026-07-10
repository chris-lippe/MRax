from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from mrax.data_struct.storage import MANIFEST_NAME, load_study_results

from .roi import ROI, collect_roi_stats
from .stats import cohens_d, describe_values, fdr_bh, hedges_g, nonparametric_test, ttest


@dataclass(frozen=True)
class GroupSpec:
    name: str
    members: Sequence[str | Path]


def discover_study_manifests(output_root: str | Path) -> dict[str, Path]:
    """Return mapping of study name -> manifest path under output_root."""

    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(f"Output root '{root}' does not exist")

    manifests: dict[str, Path] = {}
    for entry in root.iterdir():
        if entry.is_dir():
            manifest = entry / MANIFEST_NAME
            if manifest.exists():
                manifests[entry.name] = manifest

    if manifests:
        return manifests

    for manifest in root.rglob(MANIFEST_NAME):
        if manifest.is_file():
            manifests[manifest.parent.name] = manifest
    return manifests


def _resolve_manifest(
    member: str | Path,
    *,
    output_root: str | Path | None = None,
    manifest_index: Mapping[str, Path] | None = None,
) -> Path:
    path = Path(member)
    if path.exists():
        if path.is_dir():
            manifest = path / MANIFEST_NAME
        else:
            manifest = path
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found for '{member}'")
        return manifest

    if output_root is None:
        raise FileNotFoundError(f"Cannot resolve manifest for '{member}' without output_root")

    index = manifest_index or {}
    if str(member) in index:
        return index[str(member)]

    manifest = Path(output_root) / str(member) / MANIFEST_NAME
    if manifest.exists():
        return manifest

    raise FileNotFoundError(f"Manifest not found for '{member}' in {output_root}")


def resolve_groups(
    groups: Mapping[str, Sequence[str | Path]],
    *,
    output_root: str | Path | None = None,
) -> dict[str, list[Path]]:
    """Resolve group members into manifest paths."""

    manifest_index = discover_study_manifests(output_root) if output_root else {}
    resolved: dict[str, list[Path]] = {}
    for name, members in groups.items():
        manifests = [
            _resolve_manifest(member, output_root=output_root, manifest_index=manifest_index)
            for member in members
        ]
        resolved[name] = manifests
    return resolved


def collect_subject_roi_stats(
    groups: Mapping[str, Sequence[str | Path]],
    *,
    experiment_name: str,
    rois: Sequence[ROI],
    map_names: Sequence[str] | None = None,
    stats: Sequence[str] = ("mean", "variance"),
    ddof: int = 1,
    output_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Collect per-subject ROI statistics across groups."""

    resolved = resolve_groups(groups, output_root=output_root)
    rows: list[dict[str, Any]] = []

    for group_name, manifests in resolved.items():
        for manifest in manifests:
            data = load_study_results(manifest)
            study_name = data.get("manifest", {}).get("study_name", manifest.parent.name)
            experiments = data.get("experiments", {})
            if experiment_name not in experiments:
                raise KeyError(f"Experiment '{experiment_name}' not found in {manifest}")
            exp_entry = experiments[experiment_name]
            maps = exp_entry.get("fitted_maps", {})
            if map_names is not None:
                maps = {k: v for k, v in maps.items() if k in map_names}
            meta = {
                "group": group_name,
                "study_name": study_name,
                "experiment": experiment_name,
                "manifest_path": str(manifest),
            }
            rows.extend(collect_roi_stats(maps, rois, stats=stats, ddof=ddof, metadata=meta))
    return rows


def summarize_group_stats(
    subject_rows: Sequence[Mapping[str, Any]],
    *,
    ddof: int = 1,
) -> list[dict[str, Any]]:
    """Summarize subject-level ROI stats within each group."""

    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    units: dict[tuple[str, str, str, str, str], str] = {}
    for row in subject_rows:
        key = (
            str(row.get("group")),
            str(row.get("experiment")),
            str(row.get("map")),
            str(row.get("roi")),
            str(row.get("stat")),
        )
        grouped.setdefault(key, []).append(float(row.get("value", float("nan"))))
        if "unit" in row:
            units[key] = str(row["unit"])

    summaries: list[dict[str, Any]] = []
    for (group, experiment, map_name, roi, stat), values in grouped.items():
        summary = describe_values(values, ddof=ddof)
        entry: dict[str, Any] = {
            "group": group,
            "experiment": experiment,
            "map": map_name,
            "roi": roi,
            "stat": stat,
        }
        entry.update(summary)
        unit = units.get((group, experiment, map_name, roi, stat))
        if unit:
            entry["unit"] = unit
        summaries.append(entry)
    return summaries


def compare_group_stats(
    subject_rows: Sequence[Mapping[str, Any]],
    *,
    paired: bool = False,
    equal_var: bool = False,
    nonparametric: bool = True,
    p_adjust: str | None = None,
) -> list[dict[str, Any]]:
    """Compute pairwise group comparisons for each map/ROI/stat."""

    grouped: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    units: dict[tuple[str, str, str, str], str] = {}

    for row in subject_rows:
        key = (str(row.get("experiment")), str(row.get("map")), str(row.get("roi")), str(row.get("stat")))
        stat = str(row.get("stat"))
        group = str(row.get("group"))
        grouped.setdefault(key, {}).setdefault(group, []).append(
            float(row.get("value", float("nan")))
        )
        if "unit" in row:
            units[key] = str(row["unit"])

    comparisons: list[dict[str, Any]] = []
    for (experiment, map_name, roi, stat), groups in grouped.items():
        for group_a, group_b in combinations(groups.keys(), 2):
            values_a = groups[group_a]
            values_b = groups[group_b]
            t_stat, t_p = ttest(values_a, values_b, paired=paired, equal_var=equal_var)
            effect = cohens_d(values_a, values_b, paired=paired)
            g = hedges_g(effect, len(values_a), len(values_b))
            np_stat = np_p = float("nan")
            np_name = ""
            if nonparametric:
                np_stat, np_p, np_name = nonparametric_test(values_a, values_b, paired=paired)
            entry = {
                "experiment": experiment,
                "map": map_name,
                "roi": roi,
                "stat": stat,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": float(len(values_a)),
                "n_b": float(len(values_b)),
                "mean_a": float(jnp.nanmean(jnp.asarray(values_a))) if values_a else float("nan"),
                "mean_b": float(jnp.nanmean(jnp.asarray(values_b))) if values_b else float("nan"),
                "mean_diff": float(jnp.nanmean(jnp.asarray(values_a)) - jnp.nanmean(jnp.asarray(values_b)))
                if values_a and values_b
                else float("nan"),
                "t_stat": t_stat,
                "t_p": t_p,
                "effect_size_d": effect,
                "effect_size_g": g,
            }
            if nonparametric:
                entry["np_test"] = np_name
                entry["np_stat"] = np_stat
                entry["np_p"] = np_p
            unit = units.get((experiment, map_name, roi, stat))
            if unit:
                entry["unit"] = unit
            comparisons.append(entry)

    if p_adjust == "fdr_bh" and comparisons:
        p_vals = [entry.get("t_p", float("nan")) for entry in comparisons]
        adj = fdr_bh(p_vals)
        for entry, q in zip(comparisons, adj):
            entry["t_p_fdr_bh"] = q

    return comparisons


def analyze_groups(
    groups: Mapping[str, Sequence[str | Path]],
    *,
    experiment_name: str,
    rois: Sequence[ROI],
    map_names: Sequence[str] | None = None,
    stats: Sequence[str] = ("mean", "variance"),
    ddof: int = 1,
    output_root: str | Path | None = None,
    paired: bool = False,
    equal_var: bool = False,
    nonparametric: bool = True,
    p_adjust: str | None = None,
) -> dict[str, Any]:
    """End-to-end group analysis: subject stats, group summaries, and comparisons."""

    subject_rows = collect_subject_roi_stats(
        groups,
        experiment_name=experiment_name,
        rois=rois,
        map_names=map_names,
        stats=stats,
        ddof=ddof,
        output_root=output_root,
    )
    summary_rows = summarize_group_stats(subject_rows, ddof=ddof)
    comparison_rows = compare_group_stats(
        subject_rows,
        paired=paired,
        equal_var=equal_var,
        nonparametric=nonparametric,
        p_adjust=p_adjust,
    )
    return {
        "subject_stats": subject_rows,
        "group_summary": summary_rows,
        "comparisons": comparison_rows,
    }


__all__ = [
    "GroupSpec",
    "analyze_groups",
    "collect_subject_roi_stats",
    "compare_group_stats",
    "discover_study_manifests",
    "resolve_groups",
    "summarize_group_stats",
]
