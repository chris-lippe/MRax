#!/usr/bin/env python3
"""Compare Bruker vendor images with MRax's raw Cartesian IFFT reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mrax.io import load_bruker_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", type=Path, help="Bruker scan directory containing method and raw data")
    parser.add_argument("--pdata", default="1", help="Processed-data index (default: 1)")
    parser.add_argument("--output", type=Path, help="Optional comparison figure path")
    return parser


def _comparison_metrics(vendor: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
    left = np.asarray(vendor, dtype=np.float64).reshape((vendor.shape[0], -1))
    right = np.asarray(reconstructed, dtype=np.float64).reshape((reconstructed.shape[0], -1))

    left_centered = left - np.mean(left, axis=1, keepdims=True)
    right_centered = right - np.mean(right, axis=1, keepdims=True)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(right_centered, axis=1)
    correlations = np.sum(left_centered * right_centered, axis=1) / np.maximum(denominator, 1e-12)

    scales = np.sum(left * right, axis=1) / np.maximum(np.sum(right * right, axis=1), 1e-12)
    residual = left - scales[:, None] * right
    nrmse = np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(left, axis=1), 1e-12)
    return float(np.median(correlations)), float(np.median(nrmse))


def _comparison_figure(vendor: np.ndarray, reconstructed: np.ndarray, output: Path) -> None:
    frame_indices = sorted({0, int(vendor.shape[0] // 2), int(vendor.shape[0] - 1)})
    z_index = int(vendor.shape[-1] // 2) if vendor.ndim >= 4 else None
    figure, axes = plt.subplots(3, len(frame_indices), squeeze=False, figsize=(4 * len(frame_indices), 9))
    for column, frame_index in enumerate(frame_indices):
        vendor_image = vendor[frame_index]
        reconstructed_image = reconstructed[frame_index]
        if z_index is not None:
            vendor_image = vendor_image[..., z_index]
            reconstructed_image = reconstructed_image[..., z_index]
        difference = np.abs(vendor_image - reconstructed_image)
        for row, (label, image, cmap) in enumerate(
            (
                ("Vendor", vendor_image, "gray"),
                ("MRax IFFT", reconstructed_image, "gray"),
                ("Absolute difference", difference, "magma"),
            )
        ):
            axes[row, column].imshow(np.asarray(image).T, cmap=cmap, origin="lower")
            axes[row, column].set_title(f"{label}, frame {frame_index}")
            axes[row, column].axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    _, dataset, _, metadata = load_bruker_experiment(
        args.scan,
        pdata_index=args.pdata,
        load_raw_kspace=True,
    )
    reconstructed = (dataset.processed or {}).get("raw_reconstruction")
    if reconstructed is None:
        status = (metadata.get("raw_reconstruction") or {}).get("unsupported_reason", "unavailable")
        raise SystemExit(f"MRax raw reconstruction is unavailable: {status}")

    vendor = np.asarray(dataset.data)
    reconstructed = np.asarray(reconstructed)
    if vendor.shape != reconstructed.shape:
        raise SystemExit(
            f"Shape mismatch: vendor={vendor.shape}, MRax IFFT={reconstructed.shape}"
        )

    correlation, nrmse = _comparison_metrics(vendor, reconstructed)
    print(f"vendor shape: {vendor.shape}")
    print(f"k-space shape: {None if dataset.kspace is None else tuple(dataset.kspace.shape)}")
    print(f"median frame correlation: {correlation:.6f}")
    print(f"median scale-adjusted NRMSE: {nrmse:.6f}")
    if args.output is not None:
        _comparison_figure(vendor, reconstructed, args.output)
        print("comparison figure written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
