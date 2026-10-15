"""Small, deterministic profile registry; configuration never executes code."""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Mapping


def _matches(value, expected, *, present):
    if not isinstance(expected, dict):
        return value in expected if isinstance(expected, list) else value == expected
    allowed = {"exists", "equals", "one_of", "prefix", "at_least", "less_than", "version_at_least", "version_before"}
    if not expected or set(expected) - allowed:
        raise ValueError("Unknown or empty import-profile predicate")
    for operation, target in expected.items():
        if operation == "exists":
            if not isinstance(target, bool):
                raise ValueError("Existence predicates require a boolean")
            if present != target:
                return False
        elif not present:
            return False
        elif operation == "equals" and value != target:
            return False
        elif operation == "one_of" and (not isinstance(target, list) or value not in target):
            return False
        elif operation == "prefix" and (not isinstance(value, str) or not isinstance(target, str) or not value.startswith(target)):
            return False
        elif operation in {"at_least", "less_than"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isinstance(target, (int, float)):
                return False
            if not (value >= target if operation == "at_least" else value < target):
                return False
        elif operation in {"version_at_least", "version_before"}:
            source_version = re.search(r"\d+(?:\.\d+)+", str(value))
            target_version = re.fullmatch(r"\d+(?:\.\d+)+", str(target))
            if source_version is None or target_version is None:
                return False
            source_parts = tuple(int(part) for part in source_version[0].split("."))
            target_parts = tuple(int(part) for part in target_version[0].split("."))
            length = max(len(source_parts), len(target_parts))
            source_parts += (0,) * (length - len(source_parts))
            target_parts += (0,) * (length - len(target_parts))
            if not (source_parts >= target_parts if operation == "version_at_least" else source_parts < target_parts):
                return False
    return True


@dataclass(frozen=True)
class ImportProfile:
    name: str
    role: str
    match: Mapping[str, object]
    settings: Mapping[str, object]


class DefinitionLibrary:
    """Select a single matching profile and explain or reject ambiguity."""

    def __init__(self, paths=()):
        documents = [tomllib.loads(files(__package__).joinpath("bruker.toml").read_text())]
        documents.extend(tomllib.loads(Path(path).read_text()) for path in paths)
        profiles = []
        names = set()
        for document in documents:
            if set(document) - {"schema_version", "profiles"} or document.get("schema_version") != 1:
                raise ValueError("Unsupported import-definition document")
            for row in document.get("profiles", []):
                if set(row) != {"name", "role", "match", "settings"}:
                    raise ValueError("Import profiles require name, role, match and settings")
                if not isinstance(row["name"], str) or not isinstance(row["role"], str) or not isinstance(row["match"], dict) or not isinstance(row["settings"], dict):
                    raise ValueError("Profile names and roles must be strings; predicates and settings must be tables")
                if row["name"] in names:
                    raise ValueError("Import profile names must be unique")
                names.add(row["name"])
                profiles.append(ImportProfile(**row))
        self.profiles = tuple(profiles)

    def select(self, role: str, features: Mapping[str, object], *, profile: str | None = None) -> ImportProfile:
        candidates = [p for p in self.profiles if p.role == role and
            (profile is None or p.name == profile) and
            all(_matches(features.get(key), value, present=key in features)
                for key, value in p.match.items())]
        if len(candidates) != 1:
            state = "No supported" if not candidates else "Ambiguous"
            raise ValueError(f"{state} {role} import profile; provide an explicit compatible profile")
        return candidates[0]
