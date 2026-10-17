from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from ._version import MRAX_VERSION, MRAX_VERSION_TAG


class MRaxVersionError(ValueError):
    """Base error for invalid or unsupported MRax producer-version metadata."""


class MRaxVersionIncompatibilityError(MRaxVersionError):
    """Raised when a file declares a producer version outside the supported line."""


class MRaxVersionTagError(MRaxVersionError):
    """Raised when a producer-version tag exists but cannot be interpreted."""


@dataclass(frozen=True)
class MRaxVersionInfo:
    producer: str | None
    current: str = MRAX_VERSION

    @property
    def tag_missing(self) -> bool:
        return self.producer is None


def _version_tuple(value: object) -> tuple[int, int, int]:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:[-+].*)?", text)
    if match is None:
        raise MRaxVersionTagError(
            f"Invalid MRax producer-version tag {text!r}; expected a semantic version such as '1.2.3'."
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def mrax_versions_are_compatible(producer: object, current: object = MRAX_VERSION) -> bool:
    """Return compatibility for the persisted-data contract, not package equality.

    Stable releases share a data-compatibility line by major version. While MRax is
    pre-1.0, the minor version identifies that line; patch releases remain compatible.
    """

    producer_version = _version_tuple(producer)
    current_version = _version_tuple(current)
    if current_version[0] == 0 or producer_version[0] == 0:
        return producer_version[:2] == current_version[:2]
    return producer_version[0] == current_version[0]


def validate_mrax_file_version(
    payload: Mapping[str, object],
    *,
    file_kind: str,
) -> MRaxVersionInfo:
    """Validate producer versions; callers report missing tags separately."""

    raw_version = payload.get(MRAX_VERSION_TAG)
    if raw_version is None or not str(raw_version).strip():
        return MRaxVersionInfo(None)
    producer = str(raw_version).strip()
    if not mrax_versions_are_compatible(producer):
        current = MRAX_VERSION
        raise MRaxVersionIncompatibilityError(
            f"MRax version incompatibility: {file_kind} was created with MRax {producer}, "
            f"but this installation is MRax {current}. Files are compatible within supported "
            "data-version lines; this file is outside the line supported by the running version."
        )
    return MRaxVersionInfo(producer)


def missing_mrax_version_message(file_kind: str) -> str:
    return (
        f"{file_kind} has no '{MRAX_VERSION_TAG}' tag; "
        "compatibility cannot be verified, so loading will be attempted."
    )


def versioned_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a copy carrying the running MRax producer version."""

    return {**dict(payload), MRAX_VERSION_TAG: MRAX_VERSION}


__all__ = [
    "MRAX_VERSION_TAG",
    "MRAX_VERSION",
    "MRaxVersionError",
    "MRaxVersionIncompatibilityError",
    "MRaxVersionInfo",
    "MRaxVersionTagError",
    "missing_mrax_version_message",
    "mrax_versions_are_compatible",
    "validate_mrax_file_version",
    "versioned_payload",
]
