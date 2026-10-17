"""Editable mappings from sequence definitions to normalized observables."""
from __future__ import annotations

import tomllib
from dataclasses import fields
from importlib.resources import files
from pathlib import Path

import numpy as np

from mrax.io.import_definitions import NormalizedMetadata


def sequence_metadata(definitions, frame_count, path, *, definition_paths=()):
    """Translate declared Pulseq values; time aliases in the library use seconds."""
    document = tomllib.loads(files('mrax.io.definitions').joinpath('pulseq.toml').read_text())
    rules = dict(document['observables'])
    for filename in definition_paths:
        extension = tomllib.loads(Path(filename).read_text())
        if extension.get('schema_version') != 1 or set(extension) - {'schema_version', 'observables'}:
            raise ValueError("Unsupported Pulseq definition library schema")
        rules.update(extension.get('observables', {}))
    valid = {f.name for f in fields(NormalizedMetadata)}
    values = {}
    for target, rule in rules.items():
        if target not in valid or not {'aliases', 'scale'} <= set(rule) or set(rule) - {'aliases', 'scale', 'components'}:
            raise ValueError("Pulseq observable definitions require a normalized field, aliases and scale")
        if not isinstance(rule['aliases'], list) or not rule['aliases'] or not all(isinstance(alias, str) for alias in rule['aliases']):
            raise ValueError("Pulseq observable aliases must be a nonempty list of names")
        if not np.isfinite(float(rule['scale'])):
            raise ValueError("Pulseq observable unit scale must be finite")
        for alias in rule['aliases']:
            if alias in definitions:
                array = np.asarray(definitions[alias], dtype=float).reshape(-1)
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite Pulseq observable: {alias}")
                array = array * float(rule['scale'])
                if 'components' in rule:
                    components = int(rule['components'])
                    if components < 1 or array.size % components:
                        raise ValueError("Pulseq vector observable has an incompatible component count")
                    array = array.reshape((-1, components))
                values[target] = array.tolist()
                break
    name = str(definitions.get('Name') or definitions.get('SequenceName') or Path(path).stem)
    if 'SpinLock' in definitions:
        flag = definitions['SpinLock']
        if isinstance(flag, str) and flag.lower() in ('true', 'false'):
            flag = flag.lower() == 'true'
        if flag not in (True, False, 0, 1):
            raise ValueError("SpinLock must be a boolean or 0/1")
        values['spin_lock'] = bool(flag)
    return NormalizedMetadata(method_name=name, sequence_name=name, protocol_name=name,
        frame_count=frame_count, **values)
