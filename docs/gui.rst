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
   intensity bounds above the image. Scroll to zoom and drag to pan.
   **View** provides overlays, split comparisons, metadata and signal curves.
3. **Process.** Use **Reconstruct**, **Preprocess** or **Register** as needed.
   Expand the processing pipeline to insert, edit, reorder or remove steps,
   then apply or discard the draft. **Original** returns to the imported data.
4. **Segment and fit.** Draw or import ROIs in **Segment**. In **Fit**, choose
   the model, observables and fitting region. Preview an ROI signal or run a
   voxelwise map fit. Visible overlays and fitting masks are separate choices.
5. **Analyze and save.** **Analyze** provides ROI timecourses and map statistics.
   **Save workspace** preserves the session. **Export data** writes selected
   images, masks, curves or reports. Imports remain unchanged.

Signal previews average the current image until an ROI is available.
Spectroscopy uses the curve viewer. Declared frequency coordinates label its axis.

Processing changes retain existing fit maps. Rerun fitting to update them.
The workflow panels save reusable processing (``.mpip``), reconstruction
(``.mrec``) and fitting (``.mfit``) configurations. See :doc:`workflows`.

**Settings**, at the bottom left, contains appearance, keyboard shortcuts,
accessibility options and a tutorial using a separate synthetic study.
