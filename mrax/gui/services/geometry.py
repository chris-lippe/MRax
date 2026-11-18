"""Native-grid spatial geometry and rasterization."""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from mrax.analysis import ROI

from ..masks import replace_mask_slice
from ..services.roi import _rotate_points as _rotate_points


def _copy_roi_list(rois: Sequence[ROI]) -> list[ROI]:
    return [
        ROI(
            roi.name,
            np.asarray(roi.mask).copy(),
            dims=None if roi.dims is None else tuple(roi.dims),
            metadata=dict(roi.metadata),
        )
        for roi in rois
    ]


def _regular_polygon_points(
    center: tuple[float, float],
    radius: float,
    sides: int,
    *,
    angle: float = 0.0,
) -> np.ndarray:
    n = max(3, int(sides))
    cx, cy = float(center[0]), float(center[1])
    angles = np.linspace(-np.pi / 2.0, 3.0 * np.pi / 2.0, n, endpoint=False) + np.deg2rad(
        float(angle)
    )
    return np.column_stack(
        [cx + float(radius) * np.cos(angles), cy + float(radius) * np.sin(angles)]
    )


def _brush_mask(
    shape: tuple[int, int], center_yx: tuple[int, int], radius: int | float
) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    y, x = int(center_yx[0]), int(center_yx[1])
    r = max(0.0, float(radius))
    out = np.zeros((h, w), dtype=bool)
    extent = int(np.ceil(r))
    y0, y1 = max(0, y - extent), min(h, y + extent + 1)
    x0, x1 = max(0, x - extent), min(w, x + extent + 1)
    if y1 <= y0 or x1 <= x0:
        return out
    yy, xx = np.ogrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (yy - y) ** 2 + (xx - x) ** 2 <= r**2
    return out


def _brush_radius_from_size(size: int | float) -> float:
    return max(0.0, (float(size) - 1.0) * 0.5)


def _shift_array_zero(arr: np.ndarray, shifts: Mapping[int, int]) -> np.ndarray:
    out = np.zeros_like(arr)
    src: list[slice] = [slice(None)] * arr.ndim
    dst: list[slice] = [slice(None)] * arr.ndim
    for axis, shift in shifts.items():
        axis_i = int(axis)
        amount = int(shift)
        size = int(arr.shape[axis_i])
        if amount == 0:
            continue
        if abs(amount) >= size:
            return out
        if amount > 0:
            src[axis_i] = slice(0, size - amount)
            dst[axis_i] = slice(amount, size)
        else:
            src[axis_i] = slice(-amount, size)
            dst[axis_i] = slice(0, size + amount)
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _delete_mask_slice(
    mask: Any,
    *,
    dims: Sequence[str],
    indices: Mapping[str, int] | None = None,
    plane_dims: Sequence[str] = ("x", "y"),
) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim <= 2:
        return np.zeros(arr.shape, dtype=arr.dtype)
    dim_names = tuple(str(dim) for dim in dims)
    if len(dim_names) != arr.ndim:
        dim_names = tuple(dim_names[: arr.ndim])
    plane = tuple(str(dim) for dim in plane_dims if str(dim) in dim_names)
    if len(plane) != 2:
        plane = tuple(dim_names[:2])
    plane_shape = tuple(arr.shape[dim_names.index(dim)] for dim in plane)
    empty = np.zeros(plane_shape, dtype=arr.dtype)
    return replace_mask_slice(arr, empty, dims=dim_names, indices=indices, plane_dims=plane)


def _sanitize_polygon_points(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] <= 1:
        return arr
    keep = [arr[0]]
    for point in arr[1:]:
        if np.linalg.norm(point - keep[-1]) > 1e-9:
            keep.append(point)
    arr = np.asarray(keep, dtype=float)
    if arr.shape[0] > 2 and np.linalg.norm(arr[0] - arr[-1]) <= 1e-9:
        arr = arr[:-1]
    return arr


def _simplify_polyline(
    points: np.ndarray, *, tolerance: float = 0.0, closed: bool = False
) -> np.ndarray:
    arr = _sanitize_polygon_points(points)
    if arr.shape[0] <= 1:
        return arr
    if arr.shape[0] <= 2 or tolerance <= 0:
        return arr
    simplified = _rdp_simplify(arr, float(tolerance))
    if (
        closed
        and simplified.shape[0] > 1
        and np.linalg.norm(simplified[0] - simplified[-1]) <= float(tolerance)
    ):
        simplified = simplified[:-1]
    return simplified


def _rdp_simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    if points.shape[0] <= 2:
        return points
    start = points[0]
    end = points[-1]
    line = end - start
    norm = float(np.linalg.norm(line))
    if norm <= 1e-12:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        rel = points - start
        distances = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / norm
    idx = int(np.argmax(distances))
    if float(distances[idx]) <= float(tolerance):
        return np.vstack([start, end])
    left = _rdp_simplify(points[: idx + 1], tolerance)
    right = _rdp_simplify(points[idx:], tolerance)
    return np.vstack([left[:-1], right])


def _closed_curve_length(points: Any) -> float:
    pts = _sanitize_polygon_points(points)
    if pts.shape[0] < 2:
        return 0.0
    closed = np.vstack([pts, pts[0]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def _moving_average_closed_curve(points: Any, *, radius: int = 2) -> np.ndarray:
    pts = _sanitize_polygon_points(points)
    if pts.shape[0] < 3:
        return pts
    r = max(1, int(radius))
    smoothed = []
    count = pts.shape[0]
    for idx in range(count):
        window = [pts[(idx + offset) % count] for offset in range(-r, r + 1)]
        smoothed.append(np.mean(window, axis=0))
    return np.asarray(smoothed, dtype=float)


def _resample_closed_polyline(points: Any, count: int) -> np.ndarray:
    pts = _sanitize_polygon_points(points)
    target_count = max(3, int(count))
    if pts.shape[0] < 3:
        return pts
    closed = np.vstack([pts, pts[0]])
    distances = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(distances))
    if total <= 1e-12:
        return pts
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    samples = np.linspace(0.0, total, target_count, endpoint=False)
    out = []
    for value in samples:
        idx = int(np.searchsorted(cumulative, value, side="right") - 1)
        idx = max(0, min(idx, distances.size - 1))
        segment_length = float(distances[idx])
        ratio = (
            0.0
            if segment_length <= 1e-12
            else (float(value) - float(cumulative[idx])) / segment_length
        )
        out.append(closed[idx] + ratio * (closed[idx + 1] - closed[idx]))
    return np.asarray(out, dtype=float)


def _smooth_lasso_support_points(
    points: Any,
    *,
    image: Any | None = None,
    snap: bool = False,
    snap_radius: int = 4,
    snap_strength: float = 0.25,
    support_points: int | None = None,
    spline_kind: str = "cubic",
) -> np.ndarray:
    pts = _sanitize_polygon_points(points)
    if pts.shape[0] < 3:
        return pts
    if snap and image is not None:
        snapped = _snap_polygon_points_to_gradient(
            pts, image, radius=snap_radius, strength=snap_strength
        )
        if snapped.shape[0] >= 3:
            pts = snapped
    length = _closed_curve_length(pts)
    target_count = max(3, int(support_points)) if support_points is not None else None
    sample_count = int(
        max(
            48,
            min(
                1024,
                max(
                    (target_count or 0) * 4,
                    pts.shape[0] * 2,
                    length / 1.5 if length > 0 else pts.shape[0],
                ),
            ),
        )
    )
    kind = str(spline_kind or "cubic").lower()
    if kind == "linear":
        curve = _resample_closed_polyline(pts, sample_count)
    else:
        try:
            from scipy import interpolate

            degree = min(3, pts.shape[0] - 1)
            smoothness = max(0.0, pts.shape[0] * 0.08)
            tck, _u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=smoothness, per=True, k=degree)
            u_new = np.linspace(0.0, 1.0, sample_count, endpoint=False)
            x_new, y_new = interpolate.splev(u_new, tck)
            curve = np.column_stack([x_new, y_new])
        except Exception:
            curve = _moving_average_closed_curve(pts, radius=2)
    if target_count is not None:
        support = _resample_closed_polyline(curve, target_count)
        return support if support.shape[0] >= 3 else pts
    tolerance = max(0.75, min(5.0, length / 180.0 if length > 0 else 1.0))
    closed_curve = np.vstack([curve, curve[0]])
    support = _simplify_polyline(closed_curve, tolerance=tolerance, closed=True)
    if support.shape[0] < 3:
        support = _simplify_polyline(
            closed_curve, tolerance=max(0.35, tolerance * 0.4), closed=True
        )
    return support if support.shape[0] >= 3 else pts


def _roi_plane_image(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        arr = np.asarray(arr.reshape((-1,) + arr.shape[-2:])[0], dtype=float)
    return arr


def _robust_intensity_span(image: Any) -> float:
    arr = _roi_plane_image(image)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    if finite.size >= 16:
        lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    else:
        lo, hi = np.nanmin(finite), np.nanmax(finite)
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        span = float(np.nanmax(finite) - np.nanmin(finite))
    return span if np.isfinite(span) and span > 0 else 0.0


def _otsu_threshold(values: Any, *, bins: int = 256) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return lo
    counts, edges = np.histogram(arr, bins=max(2, int(bins)), range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    if total <= 0:
        return float(np.mean(arr))
    weight_bg = np.cumsum(counts).astype(float)
    weight_fg = float(total) - weight_bg
    sum_bg = np.cumsum(counts * centers)
    sum_total = float(sum_bg[-1])
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return float(np.mean(arr))
    mean_bg = np.zeros_like(centers, dtype=float)
    mean_fg = np.zeros_like(centers, dtype=float)
    mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
    mean_fg[valid] = (sum_total - sum_bg[valid]) / weight_fg[valid]
    score = np.zeros_like(centers, dtype=float)
    score[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
    return float(centers[int(np.argmax(score))])


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage

        return np.asarray(ndimage.binary_fill_holes(arr), dtype=bool)
    except Exception:
        return arr


def _largest_component_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not np.any(arr):
        return np.zeros(arr.shape, dtype=bool)
    try:
        from scipy import ndimage

        labels, count = ndimage.label(arr, structure=np.ones((3, 3), dtype=bool))
        if int(count) == 0:
            return np.zeros(arr.shape, dtype=bool)
        sizes = np.bincount(labels.ravel())
        if sizes.size <= 1:
            return np.zeros(arr.shape, dtype=bool)
        sizes[0] = 0
        return labels == int(np.argmax(sizes))
    except Exception:
        visited = np.zeros(arr.shape, dtype=bool)
        best: list[tuple[int, int]] = []
        for y, x in np.argwhere(arr):
            y_i, x_i = int(y), int(x)
            if visited[y_i, x_i]:
                continue
            component = _component_points(arr, (y_i, x_i), visited=visited)
            if len(component) > len(best):
                best = component
        out = np.zeros(arr.shape, dtype=bool)
        for y, x in best:
            out[y, x] = True
        return out


def _component_points(
    mask: np.ndarray, seed_yx: tuple[int, int], *, visited: np.ndarray | None = None
) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    sy, sx = int(seed_yx[0]), int(seed_yx[1])
    if not (0 <= sy < h and 0 <= sx < w) or not arr[sy, sx]:
        return []
    seen = np.zeros(arr.shape, dtype=bool) if visited is None else visited
    points: list[tuple[int, int]] = []
    queue: deque[tuple[int, int]] = deque([(sy, sx)])
    seen[sy, sx] = True
    while queue:
        y, x = queue.popleft()
        points.append((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and arr[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
    return points


def _connected_component_from_seed(mask: Any, seed_yx: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    out = np.zeros(arr.shape, dtype=bool)
    if arr.ndim != 2:
        return out
    sy, sx = int(seed_yx[0]), int(seed_yx[1])
    if not (0 <= sy < arr.shape[0] and 0 <= sx < arr.shape[1]) or not arr[sy, sx]:
        return out
    try:
        from scipy import ndimage

        labels, _count = ndimage.label(arr, structure=np.ones((3, 3), dtype=bool))
        label = int(labels[sy, sx])
        return labels == label if label else out
    except Exception:
        for y, x in _component_points(arr, (sy, sx)):
            out[y, x] = True
        return out


def _seeded_region_mask(
    image: Any,
    seed_yx: tuple[int, int],
    *,
    tolerance_fraction: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = _roi_plane_image(image)
    if arr.ndim != 2:
        return np.zeros(arr.shape[-2:], dtype=bool), {"auto_method": "seeded_region"}
    sy, sx = int(seed_yx[0]), int(seed_yx[1])
    h, w = arr.shape
    if not (0 <= sy < h and 0 <= sx < w) or not np.isfinite(arr[sy, sx]):
        return np.zeros((h, w), dtype=bool), {"auto_method": "seeded_region"}
    span = _robust_intensity_span(arr)
    tolerance = max(float(tolerance_fraction) * span, 1e-12)
    seed_value = float(arr[sy, sx])
    candidate = np.isfinite(arr) & (np.abs(arr - seed_value) <= tolerance)
    region = _connected_component_from_seed(candidate, (sy, sx))
    return region, {
        "auto_method": "seeded_region",
        "seed": [int(sy), int(sx)],
        "seed_value": seed_value,
        "tolerance_fraction": float(tolerance_fraction),
        "tolerance": float(tolerance),
    }


def _auto_threshold_mask(
    image: Any,
    *,
    bins: int = 256,
    foreground: str = "bright",
    largest_component: bool = True,
    fill_holes: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or not 16 <= bins <= 4096:
        raise ValueError("Histogram bins must be an integer between 16 and 4096")
    if not isinstance(foreground, str) or foreground not in {"bright", "dark"}:
        raise ValueError("Choose foreground above or below the threshold")
    if not isinstance(largest_component, bool) or not isinstance(fill_holes, bool):
        raise ValueError("Region selection and hole filling require boolean values")
    settings = {
        "auto_method": "otsu_largest_component" if largest_component else "otsu",
        "bins": int(bins),
        "foreground": foreground,
        "largest_component": largest_component,
        "fill_holes": fill_holes,
    }
    arr = _roi_plane_image(image)
    if arr.ndim != 2:
        return np.zeros(arr.shape[-2:], dtype=bool), settings
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=bool), settings
    threshold = _otsu_threshold(arr[finite], bins=bins)
    if not np.isfinite(threshold):
        return np.zeros(arr.shape, dtype=bool), settings
    candidate = finite & (
        arr >= float(threshold) if foreground == "bright" else arr < float(threshold)
    )
    component = _largest_component_mask(candidate) if largest_component else candidate
    if fill_holes:
        component = _fill_mask_holes(component)
    component &= finite
    return component, {
        **settings,
        "threshold": float(threshold),
        "voxel_count": int(np.count_nonzero(component)),
    }


def _gradient_magnitude(image: Any) -> np.ndarray:
    arr = _roi_plane_image(image)
    if arr.ndim != 2:
        return np.zeros(arr.shape[-2:], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(arr)
    return np.hypot(gx, gy)


def _snap_polygon_points_to_gradient(
    points: Any, image: Any, *, radius: int = 4, strength: float = 0.35
) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape((-1, 2))
    if pts.shape[0] == 0:
        return pts
    gradient = _gradient_magnitude(image)
    h, w = gradient.shape
    r = max(1, int(radius))
    assist = float(np.clip(strength, 0.0, 1.0))
    snapped = []
    for x, y in pts:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            snapped.append([float(x), float(y)])
            continue
        window = gradient[y0:y1, x0:x1]
        if not np.any(np.isfinite(window)):
            snapped.append([float(x), float(y)])
            continue
        yy, xx = np.unravel_index(int(np.nanargmax(window)), window.shape)
        target_x = float(x0 + xx)
        target_y = float(y0 + yy)
        snapped.append(
            [float(x) + assist * (target_x - float(x)), float(y) + assist * (target_y - float(y))]
        )
    return np.asarray(snapped, dtype=float)


def _ellipse_mask(
    shape: tuple[int, int],
    bounds: list[float] | tuple[float, float, float, float],
    *,
    angle: float = 0.0,
) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = [float(v) for v in bounds]
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=bool)
    xx, yy = np.meshgrid(np.arange(w, dtype=float) + 0.5, np.arange(h, dtype=float) + 0.5)
    coords = np.column_stack([xx.ravel() - x, yy.ravel() - y])
    if abs(float(angle)) > 1e-9:
        coords = _rotate_points(coords, -float(angle), origin=(0.0, 0.0))
    x_eval = coords[:, 0].reshape((h, w))
    y_eval = coords[:, 1].reshape((h, w))
    cx = bw / 2.0
    cy = bh / 2.0
    rx = max(bw / 2.0, 1e-6)
    ry = max(bh / 2.0, 1e-6)
    return ((x_eval - cx) / rx) ** 2 + ((y_eval - cy) / ry) ** 2 <= 1.0


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=bool)
    pts = _simplify_polyline(np.asarray(points, dtype=float), tolerance=0.0, closed=True)
    if pts.shape[0] < 3 or h <= 0 or w <= 0:
        return out

    x0 = max(0, int(np.floor(float(np.min(pts[:, 0])))))
    x1 = min(w, int(np.ceil(float(np.max(pts[:, 0])))) + 1)
    y0 = max(0, int(np.floor(float(np.min(pts[:, 1])))))
    y1 = min(h, int(np.ceil(float(np.max(pts[:, 1])))) + 1)
    if x1 <= x0 or y1 <= y0:
        return out

    x = np.arange(x0, x1, dtype=float) + 0.5
    y = np.arange(y0, y1, dtype=float) + 0.5
    xx, yy = np.meshgrid(x, y)
    inside = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    x_points = pts[:, 0]
    y_points = pts[:, 1]
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = x_points[i], y_points[i]
        xj, yj = x_points[j], y_points[j]
        intersects = ((yi > yy) != (yj > yy)) & (
            xx < (xj - xi) * (yy - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= intersects
        j = i
    out[y0:y1, x0:x1] = inside
    return out
