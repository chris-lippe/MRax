ROIs and masks
==============

ROIs and Masks
--------------

ROI tools create rectangles, circles/ellipses, regular polygons, lassos,
brush/erase edits, wand masks, and Otsu-threshold masks on the active image slice.
Masks are stored as :class:`mrax.analysis.ROI` objects.

Draw and edit ROIs in image or map views on the native image grid.

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
