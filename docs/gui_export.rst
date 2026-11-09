Saving and exporting
====================

**Save workspace** writes a self-contained ``.mrax`` file at the chosen location, containing the
study, images, ROIs, fit settings and results, and registration and drift data.
Use **Open workspace** to resume it later. Save before closing a study or stopping
the GUI.
By default, import and processing results stay in memory until you save or export.
Imported files are read-only inputs. Image processing, ROI and metadata edits
remain in MRax; saving a workspace preserves those edits and the original images.

In **Data controls** on the left, select **Export data** below **Import data**.
In the export dialog, choose an object
and an available format.
Options include MRax packages, NumPy, CSV, NIfTI, DICOM, ROI sets and reports,
depending on the selected object. Choose a destination file or folder; progress appears in
the same dialog. Closing and reopening the dialog retains its export settings.

Files are written only to the destination selected in Save or Export. For
multi-file exports, select a folder or create one with **New folder**. MRax adds
the generated files there and keeps all other contents, including subfolders.
Folders are never replaced or cleared.

If filenames match existing files, review the list before confirming file
replacement. Cancelling writes nothing to the destination. Imported metadata
and unrelated text files remain protected; choose another destination when a
file cannot be replaced. Single metadata and text exports require a new filename.

For automatic recovery, start with
``mrax-gui --state-dir /path/to/recovery``. Restarting with the same folder
restores its workspaces as unsaved studies. Keep saved bundles as durable
copies; recovery does not replace explicit saves.
