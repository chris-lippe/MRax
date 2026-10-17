"""Read numerical hints from declared reconstruction graphs without executing them."""
from __future__ import annotations

import re
from collections import defaultdict, deque


def spectral_window_powers(reco, dimensions):
    """Count serial window applications while keeping receiver branches parallel.

    A window can occur before sorting and again in a Fourier pass. Counting all
    filters would also count parallel receivers. Instead, propagate per-axis
    counts through each pass DAG, then compose the declared passes in series.
    Array operations themselves remain in the vendor-independent reconstruction.
    """
    groups = defaultdict(dict)
    for row in reco.get("RecoStageNodes", ()):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        match = re.match(r"(\w+)\s+(\w+)\s*\{(.*)\}", str(row[2]))
        if match:
            groups[(str(row[0]), int(row[1]))][match[2]] = (match[1], match[3])
    edges = defaultdict(list)
    for row in reco.get("RecoStageEdges", ()):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        match = re.fullmatch(r"(\w+)->(\w+)", str(row[2]).replace("\\", ""))
        if match:
            edges[(str(row[0]), int(row[1]))].append((match[1], match[2]))
    total = [0] * dimensions
    for group, nodes in groups.items():
        additions = {}
        for name, (kind, options) in nodes.items():
            if kind != "RecoSineWinMultFilter":
                continue
            axis = re.search(r"winDirection\s*=\s*(\d+)", options)
            if axis is None or int(axis[1]) >= dimensions:
                raise ValueError("Spectral-window graph axis is absent or outside the encoding dimensions")
            additions[name] = int(axis[1])
        if not additions:
            continue
        if not edges[group]:
            raise ValueError("Window multiplicity requires explicit reconstruction graph edges")
        children = defaultdict(list)
        incoming = {name: 0 for name in nodes}
        for first, second in edges[group]:
            incoming.setdefault(first, 0)
            incoming.setdefault(second, 0)
            incoming[second] += 1
            children[first].append(second)
        counts = {name: [0] * dimensions for name in incoming}
        ready = deque(name for name, degree in incoming.items() if degree == 0)
        visited = 0
        while ready:
            name = ready.popleft()
            visited += 1
            if name in additions:
                counts[name][additions[name]] += 1
            for child in children[name]:
                counts[child] = [max(a, b) for a, b in zip(counts[child], counts[name], strict=True)]
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
        if visited != len(incoming):
            raise ValueError("Reconstruction graph contains a cycle")
        total = [old + max(value[axis] for value in counts.values()) for axis, old in enumerate(total)]
    # An absent graph leaves the explicitly declared array window as one pass.
    return tuple(max(1, value) for value in total)
