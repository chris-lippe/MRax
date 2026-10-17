Saving and exporting
====================

Saving and Exporting
--------------------

``Save...`` defaults to a ``.mrax`` bundle; an editable package folder is optional.
Saving fades the workspace and shows writing/compression progress. It can include:

- the study snapshot and latest fit result,
- GUI state,
- shared ROIs as compressed masks plus JSON metadata.

``Export...`` writes analysis files outside the MRax package layout. Images, raw
k-space, fitted maps, and ROIs can be exported as NumPy arrays, compressed NumPy
files, CSV tables for suitable 1D/2D real arrays, NIfTI frames, derived DICOM image
series, ROI sets, or metadata JSON. Each export folder includes
``export_manifest.json`` with file paths and any conversion warnings. NIfTI export
splits non-spatial frames and removes singleton non-spatial axes; spatial geometry
is retained. See :doc:`storage` for DICOM scaling and Python export.

MRax raw reconstructions can be exported as combined magnitude, combined phase,
and per-coil magnitude/phase arrays. Phase is expressed in radians. The combined
magnitude preserves the vendor-style magnitude combination, while combined phase
uses the coherent sum of the channel-corrected complex coil images. Exported phase
is spatially unwrapped by default; an additional ``*_phase_wrapped`` export retains
the original wrapped values.

When registration transforms are stored, ROI export can resample masks between the
moving and reference image grids. Nearest-neighbor interpolation is used for masks
so binary and integer label values are preserved.
