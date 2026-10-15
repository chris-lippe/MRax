"""Bruker parameters translation; MRI reconstruction is separate."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np

_BRUKER_WORD_DTYPE = {
    "_8BIT_UNSGN_INT": np.uint8,
    "_8BIT_SGN_INT": np.int8,
    "_16BIT_UNSGN_INT": np.uint16,
    "_16BIT_SGN_INT": np.int16,
    "_32BIT_UNSGN_INT": np.uint32,
    "_32BIT_SGN_INT": np.int32,
    "_32BIT_FLOAT": np.float32,
    "_64BIT_FLOAT": np.float64,
}


def _parse_value(raw: str):
    raw = raw.strip()

    def _parse_scalar(tok: str):
        tok = tok.strip()
        if tok == "":
            return tok
        if tok.startswith("<") and tok.endswith(">"):
            return tok[1:-1]
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", tok):
            return float(tok) if ("." in tok or "e" in tok.lower()) else int(tok)
        return tok

    # Strip Bruker array-shape prefixes like "( 2 )" or "( 2, 30 )" and keep the payload.
    m_shape = re.match(r"^\(\s*([0-9]+(?:\s*,\s*[0-9]+)*)\s*\)\s*(.*)$", raw, re.S)
    if m_shape:
        payload = m_shape.group(2).strip()
        if payload:
            dims = [int(x) for x in re.split(r"\s*,\s*", m_shape.group(1).strip()) if x.strip() != ""]
            if dims and all(d > 0 for d in dims):
                raw = payload

    # Expand Bruker RLE segments like "@30*(0.5) @30*(0.5658)".
    if "@" in raw:
        rle = re.compile(r"@([0-9]+)\*\(([^)]*)\)")
        if rle.search(raw):
            expanded: list[object] = []
            pos = 0
            for m in rle.finditer(raw):
                prefix = raw[pos : m.start()].strip()
                if prefix:
                    prefix_val = _parse_value(prefix)
                    if isinstance(prefix_val, list):
                        expanded.extend(prefix_val)
                    else:
                        expanded.append(prefix_val)

                n = int(m.group(1))
                inner = m.group(2).strip()
                inner_val = _parse_value(inner)
                if isinstance(inner_val, list):
                    expanded.extend(inner_val * n)
                else:
                    expanded.extend([inner_val] * n)
                pos = m.end()

            suffix = raw[pos:].strip()
            if suffix:
                suffix_val = _parse_value(suffix)
                if isinstance(suffix_val, list):
                    expanded.extend(suffix_val)
                else:
                    expanded.append(suffix_val)
            return expanded

    tokens = re.findall(r"<[^>]*>|\([^)]*\)|[^\s]+", raw)
    if not tokens:
        return raw

    parsed: list[object] = []
    for tok in tokens:
        if tok.startswith("(") and tok.endswith(")") and "," in tok:
            parts = [p.strip() for p in tok[1:-1].split(",") if p.strip() != ""]
            parsed.append([_parse_scalar(p) for p in parts])
        else:
            parsed.append(_parse_scalar(tok))

    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _parse_jcamp_file(path: Path, entries: Dict[str, object] | None = None) -> Dict[str, object]:
    entries = dict(entries or {})
    lines = path.read_text(errors="replace").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("##$") or "=" not in line:
            idx += 1
            continue

        key, raw = line[3:].split("=", 1)
        key = key.strip()
        value_lines = [raw.split("$$", 1)[0].strip()]
        idx += 1
        while idx < len(lines) and not lines[idx].startswith("##"):
            cont = lines[idx].strip()
            if cont and not cont.startswith("$$"):
                value_lines.append(cont.split("$$", 1)[0].strip())
            idx += 1

        if key and key != "END":
            raw_value = "\n".join(part for part in value_lines if part != "")
            entries[key] = _parse_value(raw_value)
    return entries


def discover_bruker_experiments(folder: str | Path) -> list[Path]:
    """Find study experiments or a standalone export using parameter-file evidence."""
    root = Path(folder)
    if (root / "method").is_file() or (root / "acqp").is_file():
        return [root]
    return sorted(
        (entry for entry in root.iterdir() if entry.is_dir() and
         (entry.name.isdigit() or (entry / "method").is_file() or (entry / "acqp").is_file())),
        key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else 0, path.name),
    )


def parse_jcamp(path: Path, *, update_with_output_files: bool = True) -> Dict[str, object]:
    """Parse a Bruker JCAMP parameter file.

    ParaVision 360 may write companion ``*.out`` files whose values supersede
    the base file. This mirrors Bruker's MATLAB ``readBrukerParamFile`` behavior.
    """

    path = Path(path)
    entries = _parse_jcamp_file(path)
    output_path = path.with_name(path.name + ".out")
    if update_with_output_files and output_path.exists():
        entries = _parse_jcamp_file(output_path, entries)
    return entries


def bruker_word_dtype(word: str) -> np.dtype:
    key = str(word)
    if key not in _BRUKER_WORD_DTYPE:
        raise ValueError(f"Unknown VisuCoreWordType {word}")
    return _BRUKER_WORD_DTYPE[key]


def _flatten_numeric(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (np.generic, int, float)):
        return [float(value)]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [float(value)]
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    return []


def _normalize_size(value: object, *, dim: int | None = None) -> list[float]:
    vals = _flatten_numeric(value)
    if not vals:
        return []
    vals = [float(v) for v in vals if v is not None]
    if dim is None or dim <= 0:
        return vals
    vals = [v for v in vals if v > 0]
    if not vals:
        return []
    if len(vals) >= dim:
        return vals[-dim:]
    if len(vals) == 1:
        return vals * dim
    return vals + [vals[-1]] * (dim - len(vals))


def _as_reco_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_as_reco_string_list(item))
        return out
    return [str(value)]


def _as_positive_int(value: object, default: int | None = None) -> int | None:
    vals = _flatten_numeric(value)
    if vals:
        resolved = int(round(vals[0]))
    else:
        try:
            resolved = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
    return resolved if resolved > 0 else default


def _stage_node_text(reco: dict) -> str:
    nodes = reco.get("RecoStageNodes")
    if isinstance(nodes, str):
        return nodes
    if not isinstance(nodes, (list, tuple)):
        return ""
    parts: list[str] = []
    for item in nodes:
        if isinstance(item, (list, tuple)):
            parts.append(" ".join(str(part) for part in item))
        else:
            parts.append(str(item))
    return " ".join(parts)


def _first_finite_float(value: object) -> float | None:
    for val in _flatten_numeric(value):
        val_f = float(val)
        if np.isfinite(val_f):
            return val_f
    return None


def _contains_token(value: object, token: str) -> bool:
    needle = token.lower()
    if value is None:
        return False
    if isinstance(value, str):
        return needle in value.lower()
    if isinstance(value, (list, tuple)):
        return any(_contains_token(item, token) for item in value)
    return needle in str(value).lower()


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_as_string_list(item))
        return out
    return [str(value)]
