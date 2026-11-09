Loading and viewing data
========================

Use **Open workspace** for a saved ``.mrax`` file or **Import data** to load
an acquisition. In the import dialog, choose **New study** or add an experiment
to the selected study under **Import destination**. **New study** in **Data & studies**
starts an empty study. **Example studies** provides synthetic datasets.

Imports accept files and study folders. Select a Bruker study folder directly;
no archive is required. Supported
formats include MRax packages, Bruker, Pulseq, DICOM, NIfTI, NumPy and raw arrays.
For Bruker studies, review the experiment list before importing. Select the
experiments and reconstructions to load, check types, models and frequency,
and enable raw k-space only when needed.
For raw arrays, specify the shape, dimension order, data type and byte order.
Edit metadata, observables and study constants through **View → Metadata**.
See :doc:`gui_metadata` for navigation and imports.
Tables show editable values and units; expand nested fields or arrays to inspect
them. **Raw** opens the same draft as JSON for structural edits. If the table
cannot be built, the raw text remains available. Changes enter the workspace
only after **Apply**.

Select experiments in **Data & studies**. Choose the view and plane above
the image. Sliders select frames and slices; available views depend on the data.
**View → Image display** controls complex components, colormap, window centre and
width, contrast and zoom. Hold Alt while dragging to pan. **Reset display**
fits the image to the available space using its physical voxel spacing.
Display changes do not resample the data or masks.

Use **Reconstruct** to load raw k-space or reconstruct images
before selecting those views.
