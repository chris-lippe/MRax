"""Read-only noise previews and the estimates actually used by a denoising run."""

from __future__ import annotations

import copy

import numpy as np

from mrax.analysis.roi import broadcast_mask
from mrax.preprocessing import estimate_noise

from .cloning import clone_workspace
from .scene import fitting_roi


def noise_mask(workspace, options, dims, shape):
    frozen = getattr(workspace, "_processing_noise_roi", None)
    names = options.get("noise_roi_names")
    if frozen is None and names is None:
        return None
    roi = frozen if frozen is not None else fitting_roi(workspace, names)
    if (
        frozen is None
        and options.get("input", "image") not in {"original", "reconstruction"}
        and not getattr(workspace, "_pipeline_evaluating", False)
    ):
        from .pipeline import project_roi

        roi = project_roi(workspace, roi)
    return broadcast_mask(roi.mask, shape, mask_dims=roi.dims, target_dims=tuple(dims))


def block_estimate(block, parameters, mask=None):
    model = parameters.get("noise_model", "auto")
    coils = parameters.get("noise_coils", 1)
    if np.iscomplexobj(block) and parameters.get("complex_mode", "real_imag") == "magnitude":
        block = np.abs(block)
    return estimate_noise(block, mask=mask, model=model, coils=coils)


def summarize(reports, *, blocks, sampled=None, scope="block"):
    if not reports:
        if scope == "roi":
            raise ValueError(
                "The noise ROI needs at least 16 finite, nonzero pixels in a filter block"
            )
        return {
            "sigma": 0.0,
            "range": [0.0, 0.0],
            "blocks": blocks,
            "estimated_blocks": 0,
            "scope": scope,
            "samples": 0,
            "warnings": ["No blocks contain enough nonzero samples to estimate noise."],
        }
    values = [report["sigma"] for report in reports]
    result = {
        "sigma": float(np.median(values)),
        "range": [
            float(min(r["range"][0] for r in reports)),
            float(max(r["range"][1] for r in reports)),
        ],
        "blocks": blocks,
        "estimated_blocks": len(reports) if sampled is None else sampled,
        "scope": scope,
        "samples": sum(r["samples"] for r in reports),
        "warnings": sorted({w for report in reports for w in report["warnings"]}),
    }
    if any("components" in r for r in reports):
        result["components"] = {
            key: float(np.median([r["components"][key]["sigma"] for r in reports]))
            for key in ("real", "imaginary")
        }
    if result["range"][1] > 2 * max(result["range"][0], np.finfo(float).tiny):
        warning = "Noise varies across regions or blocks. Review the range before using one sigma."
        if warning not in result["warnings"]:
            result["warnings"].append(warning)
    return result


def estimate_input(data, dims, axes, parameters, mask=None):
    axes = [list(dims).index(name) for name in axes]
    other = [axis for axis in range(data.ndim) if axis not in axes]
    order = other + axes
    values = data.transpose(order)
    masks = None if mask is None else mask.transpose(order)
    outer = tuple(data.shape[axis] for axis in other)
    count = int(np.prod(outer))
    if masks is None:
        indices = np.linspace(0, count - 1, min(64, count), dtype=int)
    else:
        eligible = np.sum(masks, axis=tuple(range(len(other), data.ndim))) >= 16
        candidates = np.flatnonzero(eligible)
        indices = candidates[
            np.linspace(0, len(candidates) - 1, min(64, len(candidates)), dtype=int)
        ]
    reports = []
    for index in indices:
        key = np.unravel_index(index, outer) if outer else ()
        report = block_estimate(values[key], parameters, None if masks is None else masks[key])
        if report.get("method") != "insufficient_samples":
            reports.append(report)
    return summarize(reports, blocks=count, scope="roi" if mask is not None else "block")


def preview_noise(workspace, options):
    from .pipeline import _evaluate, current_pipeline, install
    from .processing import _settings, _source, processing_methods
    from .processing_scope import freeze_scope, region
    from .view_data import scalar_component

    if options.get("method") not in {"bm3d", "bm4d"}:
        raise ValueError("Choose BM3D or BM4D before estimating noise")
    work = clone_workspace(workspace, readonly_arrays=True)
    scope_dims = work.session.experiment_dims
    if str(options.get("input", "image")).startswith("map:"):
        scope_dims = {
            name: size for name, size in scope_dims.items() if name in work.session.roi_dim_names
        }
    elif options.get("input") in {"original", "reconstruction"} and current_pipeline(work):
        scope_dims = current_pipeline(work)["source"]["dims"]
    options = freeze_scope(work.session, copy.deepcopy(options), scope_dims)
    pipeline = current_pipeline(work)
    # Estimate from the replacement step's input.
    # Preceding steps run only on an explicit preview.
    if pipeline and options.get("input", "image") == "image":
        requested = options.get("step_id")
        previous = next(
            (
                i
                for i, s in enumerate(pipeline["steps"])
                if s["id"] == requested
                or (
                    not requested
                    and s["family"] == "denoise"
                )
            ),
            len(pipeline["steps"]),
        )
        pipeline["steps"] = pipeline["steps"][:previous]
        install(work, pipeline["source"])
        if pipeline["steps"]:
            _evaluate(work, lambda *_args: None, run_fits=set(), execute=set())
    method = next(m for m in processing_methods(work.session) if m["id"] == options["method"])
    parameters = _settings(method, options.get("parameters", {}))
    partial = options.get("scope") in {"image", "package"}
    data, dims, _observables, _model, _unit, geometry, _rois = _source(
        work,
        str(options.get("input", "image")),
        "auto" if partial else options.get("component", "auto"),
    )
    axes = options.get("axes", list(work.session.spatial_dim_names))
    required = 2 if method["id"] == "bm3d" else 3
    if (
        not isinstance(axes, list)
        or len(axes) != required
        or len(set(axes)) != len(axes)
        or any(a not in dims for a in axes)
    ):
        raise ValueError(f"{method['id'].upper()} needs {required} distinct filter axes")
    mask = noise_mask(work, options, dims, data.shape)
    selection = region(work.session, options, dims, geometry)
    data = data[selection]
    if partial and options.get("component", "auto") != "auto":
        data = scalar_component(data, options["component"])
    scoped_dims = dict(zip(dims, data.shape))
    result = estimate_input(
        data, scoped_dims, axes, parameters, None if mask is None else mask[selection]
    )
    return copy.deepcopy(result)
