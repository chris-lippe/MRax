"""General workflow starting points with explicit input requirements."""

from importlib.resources import files

from mrax.workflows import loads_workflow


def workflow_presets():
    """Load independent copies of the bundled, versioned workflow documents."""
    from mrax.reco.plans import reconstruction_presets

    presets = []
    for path in sorted(files(__name__).iterdir(), key=lambda p: p.name):
        if path.name.endswith(".mflow"):
            document = loads_workflow(path.read_text(encoding="utf-8"))
            presets.append(
                {
                    "id": path.name.removesuffix(".mflow"),
                    "document": document.model_dump(),
                    "requirements": document.extensions["org.mrax.preset"]["requirements"],
                }
            )
    for preset in reconstruction_presets():
        encoding = preset["plan"]["encoding"]
        requirements = [
            "Prepared k-space with acquisition geometry and coil information.",
            {
                "cartesian": "Cartesian sampling.",
                "noncartesian": "Non-Cartesian sampling with a valid trajectory.",
                "auto": "Sampling supported by the acquisition's reconstruction plan.",
            }.get(encoding, "A sampling trajectory compatible with this reconstruction."),
        ]
        preset["document"]["extensions"]["org.mrax.preset"] = {"requirements": requirements}
        presets.append(
            {
                "id": "reco-" + preset["id"],
                "document": preset["document"],
                "requirements": requirements,
            }
        )
    return presets
