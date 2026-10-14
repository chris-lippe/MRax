"""MRax package."""

from ._version import MRAX_VERSION, MRAX_VERSION_TAG
from .devices import get_available_devices

__version__ = MRAX_VERSION

__all__ = ["MRAX_VERSION", "MRAX_VERSION_TAG", "__version__", "get_available_devices"]
