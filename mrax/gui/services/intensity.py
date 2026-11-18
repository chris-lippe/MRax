"""Reference display limits shared by frames of the same scalar image."""

import numpy as np

from mrax.gui.plotting import scalar_component


def fixed_range(cache, key, plane, data=None, component="auto"):
    if key in cache:
        return cache[key]
    if data is None:
        samples = np.asarray(plane)
    else:
        data = getattr(data, "values", data)
        if not hasattr(data, "shape"):
            data = np.asarray(data)
        # Bound both host transfers and temporary arrays, including for strided volumes.
        count = int(np.prod(data.shape))
        limit = 65536
        if count > limit:
            edges = np.linspace(0, count, limit + 1, dtype=np.int64)
            indices = np.random.default_rng(0).integers(edges[:-1], edges[1:])
            data = data[np.unravel_index(indices, data.shape)]
        samples = scalar_component(data, component)
    finite = samples[np.isfinite(samples)]
    if not finite.size:
        plane = np.asarray(plane)
        finite = plane[np.isfinite(plane)]
        if not finite.size:
            return [0.0, 1.0]
    lo, hi = float(finite.min()), float(finite.max())
    cache[key] = [lo, hi]
    return cache[key]


def series_data(entry, kind, map_name=None):
    if kind == "map":
        return (entry.get("fitted_maps") or {}).get(map_name)
    if kind == "original" and entry.get("original_mr_image") is not None:
        return entry["original_mr_image"]
    if kind in {"image", "original", "processed"}:
        return entry.get("mr_image")
    return None
