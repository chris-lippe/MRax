"""Translate declared image products into vendor-independent recipe settings."""
from __future__ import annotations

import re


def product_recipe(method, reco, visu, *, library):
    units = str(visu.get("VisuCoreDataUnits", "")).strip("<>")
    if units != "Hz":
        return {}
    profile = library.select("product", {"image_units": units,
        "paired_echoes": int(method.get("PVM_NEchoImages", 1)) == 2})
    settings = profile.settings
    if set(settings) != {"kind", "echo_spacing_key", "time_unit"} or settings["kind"] != "frequency_hz" or settings["time_unit"] != "ms":
        raise ValueError("Unsupported quantitative image product definition")
    spacing = float(method[settings["echo_spacing_key"]]) * 1e-3
    from .bruker_parameters import _stage_node_text

    stages = _stage_node_text(reco)
    mask_filter = re.search(r"RecoCreateMaskFilter\s+\w+\{([^}]*)\}", stages)
    options = {}
    if mask_filter is not None:
        for source, target in (("snr", "product_mask_snr"), ("signalLevel", "product_signal_fraction"),
                               ("cornerPart", "product_corner_fraction")):
            match = re.search(rf"{source}\s*=\s*([-+\d.eE]+)", mask_filter[1])
            if match is not None:
                options[target] = float(match[1])
    return {"output_product": "frequency_hz", "echo_spacing_seconds": spacing,
        "product_unwrap": "RecoUnwrapFilter" in stages, **options}
