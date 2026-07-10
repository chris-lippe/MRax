from __future__ import annotations

from typing import Iterable

import jax.numpy as jnp
import numpy as np
from scipy import stats as sp_stats

ArrayLike = Iterable[float] | np.ndarray


def _finite(values: ArrayLike) -> np.ndarray:
    if isinstance(values, (np.ndarray, jnp.ndarray)):
        arr = values
    else:
        arr = list(values)
    xp = jnp if isinstance(arr, jnp.ndarray) else np
    arr = xp.asarray(arr, dtype=float).ravel()
    return arr[xp.isfinite(arr)]


def describe_values(values: ArrayLike, *, ddof: int = 1) -> dict[str, float]:
    """Return descriptive statistics for a set of values."""

    finite = _finite(values)
    xp = jnp if isinstance(finite, jnp.ndarray) else np
    n = int(finite.size)
    nan_count = int(np.asarray(values).size - n)
    if n == 0:
        return {
            "count": 0.0,
            "nan_count": float(nan_count),
            "mean": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }

    mean = float(xp.mean(finite))
    var = float(xp.var(finite, ddof=ddof)) if n > ddof else float("nan")
    std = float(xp.sqrt(var)) if np.isfinite(var) else float("nan")
    sem = float(std / xp.sqrt(n)) if n > 1 and np.isfinite(std) else float("nan")
    median = float(xp.median(finite))
    min_val = float(xp.min(finite))
    max_val = float(xp.max(finite))
    p25 = float(xp.percentile(finite, 25))
    p75 = float(xp.percentile(finite, 75))

    return {
        "count": float(n),
        "nan_count": float(nan_count),
        "mean": mean,
        "variance": var,
        "std": std,
        "sem": sem,
        "median": median,
        "min": min_val,
        "max": max_val,
        "p25": p25,
        "p75": p75,
    }


def cohens_d(a: ArrayLike, b: ArrayLike, *, ddof: int = 1, paired: bool = False) -> float:
    a_vals = _finite(a)
    b_vals = _finite(b)
    xp = jnp if isinstance(a_vals, jnp.ndarray) or isinstance(b_vals, jnp.ndarray) else np
    if paired:
        if a_vals.size != b_vals.size or a_vals.size < 2:
            return float("nan")
        diff = a_vals - b_vals
        denom = xp.std(diff, ddof=ddof)
        if denom == 0:
            return float("nan")
        return float(xp.mean(diff) / denom)

    n1 = int(a_vals.size)
    n2 = int(b_vals.size)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1 = xp.var(a_vals, ddof=ddof)
    var2 = xp.var(b_vals, ddof=ddof)
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / float(n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    return float((xp.mean(a_vals) - xp.mean(b_vals)) / xp.sqrt(pooled))


def hedges_g(d: float, n1: int, n2: int) -> float:
    if not np.isfinite(d):
        return float("nan")
    df = n1 + n2 - 2
    if df <= 0:
        return float("nan")
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d * correction)


def ttest(a: ArrayLike, b: ArrayLike, *, paired: bool = False, equal_var: bool = False):
    a_vals = np.asarray(_finite(a))
    b_vals = np.asarray(_finite(b))
    if paired:
        if a_vals.size != b_vals.size or a_vals.size == 0:
            return float("nan"), float("nan")
        result = sp_stats.ttest_rel(a_vals, b_vals, nan_policy="omit")
    else:
        if a_vals.size == 0 or b_vals.size == 0:
            return float("nan"), float("nan")
        result = sp_stats.ttest_ind(a_vals, b_vals, equal_var=equal_var, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def nonparametric_test(a: ArrayLike, b: ArrayLike, *, paired: bool = False):
    a_vals = np.asarray(_finite(a))
    b_vals = np.asarray(_finite(b))
    if a_vals.size == 0 or b_vals.size == 0:
        return float("nan"), float("nan"), ""
    if paired:
        if a_vals.size != b_vals.size:
            return float("nan"), float("nan"), ""
        result = sp_stats.wilcoxon(a_vals, b_vals)
        return float(result.statistic), float(result.pvalue), "wilcoxon"
    result = sp_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    return float(result.statistic), float(result.pvalue), "mannwhitney"


def fdr_bh(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    xp = jnp if any(isinstance(v, jnp.ndarray) for v in values) else np
    p = xp.asarray(values, dtype=float)
    n = int(p.size)
    if n == 0:
        return []
    order = xp.argsort(p)
    ranks = xp.arange(1, n + 1, dtype=float)
    q = xp.empty_like(p)
    q[order] = p[order] * n / ranks
    q = xp.minimum.accumulate(q[::-1])[::-1]
    q = xp.clip(q, 0.0, 1.0)
    return np.asarray(q).tolist()


__all__ = [
    "cohens_d",
    "describe_values",
    "fdr_bh",
    "hedges_g",
    "nonparametric_test",
    "ttest",
]
