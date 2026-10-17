"""Isolated compressed-shape decoding for supported Pulseq sequence formats."""
from __future__ import annotations

import copy
import importlib
import types

import numpy as np


def read_sequence(sequence, path):
    """Read with RF-use detection, including pre-1.4 compressed shapes.

    Shape decompression requires NumPy arrays rather than Python lists.
    Clone the reader's globals for this call only;
    never monkey-patch the installed package or its process-wide functions.
    """
    module = importlib.import_module('pypulseq.Sequence.read_seq')
    reader = module.read
    read_shapes = reader.__globals__.get('__read_shapes')
    if read_shapes is None or 'decompress_shape' not in read_shapes.__globals__:
        sequence.read(str(path), detect_rf_use=True)
        return
    original = read_shapes.__globals__['decompress_shape']

    def decompress(shape, *args, **kwargs):
        shape = copy.copy(shape)
        shape.data = np.asarray(shape.data)
        return original(shape, *args, **kwargs)

    shape_globals = {**read_shapes.__globals__, 'decompress_shape': decompress}
    shape_reader = types.FunctionType(read_shapes.__code__, shape_globals, read_shapes.__name__, read_shapes.__defaults__, read_shapes.__closure__)
    local_reader = types.FunctionType(reader.__code__, {**reader.__globals__, '__read_shapes': shape_reader}, reader.__name__, reader.__defaults__, reader.__closure__)
    local_reader(sequence, path=str(path), detect_rf_use=True, remove_duplicates=True)
