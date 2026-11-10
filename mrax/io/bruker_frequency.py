"""Bruker imaging/reference frequencies in MHz, shared by import and preview."""

from __future__ import annotations

import math
from statistics import median

import numpy as np


def _values(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in np.asarray(value, dtype=object).flat:
            yield from _values(item)
    elif value is not None and not isinstance(value, (bool, np.bool_)):
        try:
            frequency = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(frequency) and 1 <= frequency <= 2000:
            yield frequency


def bruker_imaging_frequency(method, visu=None, acqp=None):
    """Prefer nominal/reference frequency to the scan's offset working frequency."""
    method, visu, acqp = method or {}, visu or {}, acqp or {}
    for key, sources in (
        ("BF1", (acqp, method)),
        ("PVM_FrqRef", (method,)),
        ("VisuAcqImagingFrequency", (visu,)),
        ("SFO1", (acqp, method)),
        ("PVM_FrqWork", (method,)),
        ("ACQ_imaging_frequency", (acqp, method, visu)),
    ):
        for source in sources:
            frequency = next(_values(source.get(key)), None)
            if frequency is not None:
                return frequency
    return None


def consistent_bruker_frequency(values):
    """Use the median when scan frequencies agree within 100 ppm (at least 1 kHz).

    Small scan offsets and header rounding must not clear the study value.
    Distinct field strengths or nuclei still require an explicit user value.
    """
    frequencies = [frequency for value in values for frequency in _values(value)]
    if not frequencies:
        return None
    frequency = float(median(frequencies))
    return (
        frequency if max(frequencies) - min(frequencies) <= max(0.001, frequency * 1e-4) else None
    )
