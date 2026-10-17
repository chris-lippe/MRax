Loading and viewing data
========================

Loading Data
------------

``Load...`` opens:

- ``study_manifest.json`` files written by
  :func:`mrax.data_struct.storage.save_study_results`.
- ``.mrax`` bundles.
- MRax package folders containing ``mrax_package.json``.
- GUI state files written by this release.

GUI state loading is strict about the state format and version. The referenced
study manifest is loaded first, then saved view settings, ROIs, fitted maps, frame
selections, and fit controls are restored. Warnings and handled errors are shown on
the Diagnostics page and can be saved with ``Save Diagnostic Report``.

``Import...`` opens one folder picker and automatically detects Bruker ParaVision,
Pulseq or DICOM data. Pulseq folders must contain sequences and their measurements.
The study overview shows the detected vendor and lets you select measurements, experiment
types and models. Raw-only inputs use the shared reconstruction pipeline.
Sequence pairing must be unambiguous; reader options and metadata sidecars follow
:doc:`pulseq_import`.
DICOM discovery is recursive, groups image objects by series, and creates one
short-named MRax experiment per series. ``Add Experiment`` also accepts an individual
DICOM file or series. Imports are converted into MRax objects immediately; use
``Save...`` to write a bundle or package. Standalone NIfTI and NumPy imports use
``Add Experiment``.

Viewing
-------

The left panel keeps experiment selection and the current view mode visible.
``Image`` displays vendor image data, ``MRax reconstruction`` displays reconstructions
from raw k-space, ``K-space`` displays raw k-space when present, and ``Map`` displays
fitted maps. The MRax reconstruction view can switch between magnitude and phase and
between the combined image and each available receiver coil. Phase unwrapping is
enabled by default and can be disabled in the reconstruction controls.

The viewer supports:

- axial, coronal, and sagittal views for 3D images. Plane selection changes the
  active spatial slider and uses stored voxel spacing for the display aspect ratio;
  fixed ``x/y/z`` array semantics are independent of native acquisition orientation,
- controls for frame, slice, echo, b-value, offset, or other non-displayed axes
  inferred from experiment dimensions,
- k-space display modes ``log_magnitude``, ``magnitude``, ``phase``, ``real``,
  and ``imag``,
- k-space centering modes ``auto``, ``none``, and ``fftshift``,
- fitted-map display with units and ROI overlays,
- per-experiment view state for display-only masks, stored transforms, color
  channels, and opacity.

View-state changes are cached per experiment and can be reset. They do not modify
source arrays, maps, k-space, or ROI masks. The Fit page uses view-state masks or
transforms only when ``Use view-state masks/transforms`` is enabled.
