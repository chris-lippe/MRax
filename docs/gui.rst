Using the GUI
=============

Open **MRax.exe**, or start a Python installation with:

.. code-block:: bash

   pip install "mrax[gui]"
   mrax-gui

Basic workflow
--------------

1. **Open or import.** Use **Data & Studies** on the left to open a saved
   ``.mrax`` workspace or import data. Bruker studies open directly from their
   folders. Review the experiments before importing.
2. **Inspect.** Select an experiment, slice package, frame and slice. Use the
   intensity bounds above the image. Frames and slices share a scale by default.
   Choose **Scale per frame** to inspect a low-signal frame, or enter manual bounds.
   Scroll to zoom and drag to pan.
   **View** provides overlays, split comparisons, metadata and signal curves.
3. **Process.** Use **Reconstruct**, **Denoise** or **Register** as needed.
   Expand the processing pipeline to insert, edit, reorder or remove steps,
   then apply or discard the draft. **Original** returns to the imported data.
4. **Fit.** Choose the model, observables and fitting region in **Fit**.
   The default is the whole image. Draw or import optional ROIs in **Segment**.
   Use **Check fit readiness** before fitting. Preview the mean signal or run a
   voxelwise map fit.
5. **Analyze and save.** **Analyze** provides ROI timecourses and map statistics.
   **Save workspace** preserves the session. **Export data** writes selected
   images, masks, curves or reports. Imports remain unchanged.

Signal previews show the whole-image or ROI mean along the selected axis.
Spectroscopy uses the curve viewer. Declared frequency coordinates label its axis.
Dynamic fits produce maps for each time point. Each parameter's **Dimension
settings** can combine spatial and temporal penalties with separate weights, or
share a parameter across time. **Loss RMS scale** controls which image samples
set the data-term scale.

Processing changes retain existing fit maps. Rerun fitting to update them.
**Register → Series alignment** aligns acquisition frames.
**Workflows** saves reusable presets and complete chains as ``.mflow`` files.
See :doc:`workflows`.

**Settings**, at the bottom left, contains appearance, keyboard shortcuts,
accessibility options and a tutorial using a separate synthetic study.
