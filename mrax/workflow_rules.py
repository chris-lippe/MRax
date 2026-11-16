"""Ordering and cardinality rules for experiment workflows."""


def validate_pipeline_order(operations):
    for operation, label in (("reconstruct", "reconstruction"), ("fit", "fitting")):
        if operations.count(operation) > 1:
            raise ValueError(f"Use one {label} block per experiment")
    if "reconstruct" in operations and operations[0] != "reconstruct":
        raise ValueError("Reconstruction defines the source and must come first")
