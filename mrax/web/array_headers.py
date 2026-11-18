"""Read NumPy shape and dtype without materializing an array."""

import ast
import struct

import numpy as np


def read_array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    if version != (3, 0):
        raise ValueError("Unsupported NumPy array header")
    length = stream.read(4)
    if len(length) != 4:
        raise ValueError("Incomplete NumPy array header")
    size = struct.unpack("<I", length)[0]
    if size > 10000:
        raise ValueError("NumPy array header is too large")
    try:
        header = ast.literal_eval(stream.read(size).decode("utf-8"))
        if set(header) != {"shape", "descr", "fortran_order"}:
            raise ValueError("Invalid NumPy array header")
        shape = header["shape"]
        if not isinstance(shape, tuple) or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("Invalid NumPy array shape")
        if not isinstance(header["fortran_order"], bool):
            raise ValueError("Invalid NumPy array order")
        return shape, header["fortran_order"], np.lib.format.descr_to_dtype(header["descr"])
    except (SyntaxError, TypeError, UnicodeError, KeyError) as exc:
        raise ValueError("Invalid NumPy array header") from exc
