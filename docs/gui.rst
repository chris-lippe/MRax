Desktop GUI
===========

MRax includes a Qt desktop GUI for inspecting saved MRax studies, drawing ROIs,
running fits, and exporting maps or masks. It uses the same ``Study``,
``Experiment``, ``DataSet``, and ``Model`` objects as the Python API. GUI actions
do not change source Bruker files.

Installation
------------

Install and launch:

.. code-block:: bash

   pip install mrax
   mrax-gui
   mrax-gui path/to/study_manifest.json

The Python package includes the GUI dependencies. On minimal Linux systems, Qt may
still need native libraries such as ``libglib-2.0.so.0`` or ``libGL.so.1``:

.. code-block:: bash

   apt-get update
   apt-get install -y libglib2.0-0 libgl1

Without root access, place extracted native libraries in
``~/.local/mrax-gui-libs`` or set ``MRAX_GUI_LOCAL_LIB_DIR``.

The GUI can also be launched from Python:

.. code-block:: python

   import mrax.gui
   mrax.gui.launch("path/to/study_manifest.json")

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

``Import...`` accepts a Bruker ParaVision study folder or a folder containing DICOM
files. DICOM discovery is recursive, groups image objects by series, and creates one
short-named MRax experiment per series. ``Add Experiment`` also accepts an individual
DICOM file or series. Imports are converted into MRax objects immediately; use
``Save...`` to write a bundle or package. Direct Pulseq, NIfTI, and NumPy import
dialogs are not included; use the Python import APIs first, save a manifest, then open
that manifest in the GUI.

Saving and Exporting
--------------------

``Save...`` writes a ``.mrax`` bundle or an editable package folder. It can include:

- the study snapshot and latest fit result,
- GUI state,
- shared ROIs as compressed masks plus JSON metadata.

``Export...`` writes analysis files outside the MRax package layout. Images, raw
k-space, fitted maps, and ROIs can be exported as NumPy arrays, compressed NumPy
files, CSV tables for suitable 1D/2D real arrays, NIfTI for compatible 2D/3D image
data, ROI sets, or metadata JSON. Each export folder includes
``export_manifest.json`` with file paths and any conversion warnings.

MRax raw reconstructions can be exported as combined magnitude, combined phase,
and per-coil magnitude/phase arrays. Phase is expressed in radians. The combined
magnitude preserves the vendor-style magnitude combination, while combined phase
uses the coherent sum of the channel-corrected complex coil images. Exported phase
is spatially unwrapped by default; an additional ``*_phase_wrapped`` export retains
the original wrapped values.

When registration transforms are stored, ROI export can resample masks between the
moving and reference image grids. Nearest-neighbor interpolation is used for masks
so binary and integer label values are preserved.

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

Registration and Drift
----------------------

The Registration page aligns the active experiment image to a selected reference
experiment. SimpleITK registration exposes the transform, metric, optimizer,
interpolation, initializer, sampling, histogram, iteration, learning-rate, and
multi-resolution settings. ROIs can be used as metric masks.

Registration stores transforms in the GUI session and reports pre/post normalized
correlation, MSE, MAD, optimizer diagnostics, selected mask pixel counts, and
transform parameters. A reference overlay can be shown for visual inspection; it
does not alter the experiment view state.

Manual alignment can also create stored transforms using pixel shifts, rotation,
and X/Y stretch. Stored transforms can be applied to the current view state or to
ROIs.

The same page exposes MRax linear drift correction for frame stacks. It fits one
linear transform-parameter drift model over selected dimensions using the JAX drift
optimizer. Drift results are stored as view-state results, not written over the
experiment image data. The original image snapshot remains available through
``Show original data`` and ``Restore Source Image``.

ROIs and Masks
--------------

ROI tools create rectangles, circles/ellipses, regular polygons, lassos,
brush/erase edits, wand masks, and Otsu-threshold masks on the active image slice.
Masks are stored as :class:`mrax.analysis.ROI` objects.

Geometric ROIs remain editable until they are consolidated. Brush, erase, and wand
edits convert the selected geometric ROI to a raster mask. Label masks are
supported with integer labels; label ``0`` is background.

Slice and volume tools include combine, subtract, shift, copy z, delete z, and
interpolate z. Interpolation uses signed distance fields for linear and spline
strategies.

ROIs are managed as one shared ROI set with per-experiment enablement:

- loading or drawing an ROI adds it to the shared set and enables it for the active
  experiment,
- newly visited experiments keep shared ROIs disabled until selected,
- ``Remove`` deletes an ROI from all experiments,
- ``Disable All`` disables ROIs only for the active experiment,
- ``Apply All`` copies the active experiment's enabled ROI set to every experiment.

Save and load ROI sets with:

.. code-block:: python

   from mrax.gui.masks import load_roi_set, save_roi_set

   save_roi_set(rois, "subject_rois.npz")
   rois = load_roi_set("subject_rois.json")

Raw NumPy masks can be imported with :func:`mrax.gui.masks.import_mask` when their
shape matches the experiment spatial shape.

Fitting
-------

The Fit page uses the same fitting functions as scripted MRax workflows:
:func:`mrax.core.loss_builder.run_fit`,
:func:`mrax.core.loss_builder.make_model_callable`, and the experiment sanity
checks. Frame exclusions are passed as ``frame_mask``. ROI selections are passed
as spatial masks for full-map fits.

Available model choices mirror the built-in model factories in
:mod:`mrax.library.default_models`.

Run ``Fit Sanity Check`` before fitting. It checks the exact GUI fit inputs and
reports missing maps, shape mismatches, invalid bounds, and other conditions that
would prevent ``run_fit`` from starting.

``Preview ROI Fit`` fits the selected ROI timecourse. ``Run Full-Map Fit`` fits
the active experiment and replaces the displayed fitted maps with the result.
``Save Fit`` writes the latest result using the normal MRax manifest layout.

Timecourses
-----------

ROI timecourses are extracted with :func:`mrax.analysis.extract_timecourse` and
:func:`mrax.analysis.extract_timecourse_from_experiment`. Use the same functions
in scripts when you need a reproducible non-GUI analysis path.
