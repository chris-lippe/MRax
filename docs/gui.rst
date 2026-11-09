GUI user guide
==============

Use the GUI to load studies, view images, define ROIs, fit quantitative models,
register images and export results.

Starting the GUI
----------------

.. code-block:: bash

   pip install "mrax[gui]"
   mrax-gui

In a standalone Windows distribution, start **MRax.exe**. Its installer includes
the Python and browser runtimes; no Python installation is required.

For a Python installation, the command opens a browser tab. Keep the terminal running while you work;
save your workspace before stopping it with Ctrl+C. To open a saved workspace
at startup, use ``mrax-gui /path/to/study.mrax``. Run ``mrax-gui --help`` for
startup options.

Working with local folders
--------------------------

Use **Open workspace**, **Import data**, **Save workspace** and **Export data**
to choose files or folders. Select a row; double-click a folder or press Enter
to open it. Use Back, Forward, Up or the **Location** address bar to navigate.
Standard folders, drives and mounted folders appear on the left. **Add to favorites**
keeps a folder shortcut there for later use; removing a shortcut leaves its folder intact.
These are folders available to the system running MRax, including container mounts.
Files are read and saved directly at the selected location.
Operating-system permissions still apply. Confirm **Replace** to overwrite an
existing file.

For a restricted shared installation, an administrator can use
``mrax-gui --filesystem-scope shared --read-root /path/to/studies --write-root /path/to/results``.

Workspace
---------

The **Data & studies** panel on the left lists open studies and their experiments.
The buttons under **Data controls** create, open, import, save, export or close studies.
**New study** opens a form for empty or example studies. **View → Metadata**
provides names, constants and data inspection in the main workspace.
**Export data**, below **Import data**,
opens a dialog for array, ROI, package and report exports.
In compact windows, **Data & studies** opens the same panel as a drawer.

Choose a workflow tab in the header. Tabs with a chevron open a task menu;
the selected task's controls appear on the right, or in the main workspace for
the metadata browser. Each tab remembers its
last task and settings. Use arrow keys to move between tabs, Down to open a
menu, and Escape to close it.
Plane selection, intensity controls and frame/slice sliders stay beside the image
throughout processing. The signal preview shows the selected ROI. Drag its bottom
divider down to enlarge it without shrinking the image; arrow keys resize the
focused divider, and Home or double-click restores its default height.
**Settings**, fixed at the bottom of the study panel, controls appearance,
keyboard bindings and accessibility. **System**, the default appearance, follows
the operating system; Light and Dark override it. Accessibility options include
larger text and controls, higher contrast, stronger focus outlines, reduced
motion, and opaque panels.

**Keyboard shortcuts** lists editable bindings and fixed interaction keys.
App bindings use Ctrl+Shift+1–9; image bindings work only with a focused image.
Bindings do not run in text fields or dialogs, and reserved keys cannot be assigned.
Choose **Tutorial → Start tutorial** for a guided workflow on a separate
synthetic study. Existing studies stay open; practice results are retained.

Use the **Workflows** buttons in the processing, reconstruction and fitting
controls to reuse configurations across experiments or save them for a later
session; see :doc:`workflows`.

Drag either panel divider to resize it; related analysis controls use columns when
space permits. Use arrow keys on a focused divider, or double-click to reset
its width.

The panels share the active experiment and ROI. In **View → Image and ROI overlays**, add image, fitted-map
or registered layers and choose which ROIs to display. Edit those masks in
**Segment → Create and edit ROIs**, then select the masks to use in **Fit**. Display visibility and
fitting selections are independent. Each experiment retains its view and layer
choices; saving the workspace retains layers, masks, fits and registrations.

Use **View → Image display** for presentation modes, components and colormaps.
Choose **Overlay** or **Split view**, then select **Image A** and **Image B**.
Both can come from any experiment in the study;
switching modes keeps the pair. **Overlay** blends compatible grids with an opacity
slider. **Split view** offers **Swipe** and **Side by side** with a movable divider.
Different grids are shown side by side; use **Register** before overlaying them.
**Original vs processed**, or **Compare** in the pipeline, selects protected source
data and the current processing output. Frame and slice controls navigate the chosen
pair independently of the active experiment;
**Image options** selects fixed frames, components and palettes.
Expand **View controls** above the image for plane selection, brightness,
contrast and a histogram with draggable value bounds. Collapse the group to
leave more room for the image. Enter exact bounds in **Min** and **Max**, or use **Auto** to reset
the intensity window. Split view can link both windows or adjust each image
independently. These controls change the display without altering image data.
**Image** returns to the current experiment with its saved layers and ROI masks.

**View** adjusts the current working display. **Reconstruct** provides experiment,
MRax, vendor comparison, difference and k-space views beside the image. **Fit**
opens available parameter maps with a color scale and units; switch to **Signal**
to inspect the input. **Segment** opens the editable image and ROI selector.
Each action remembers its viewing choices while you work.
Scroll over an image to zoom and drag to pan. While drawing, use Alt-drag or the
middle mouse button to pan. Double-click or press **0** to fit the image; **+** and
**-** also zoom. The image context menu provides zoom and reset commands.

Open **View → Signal curves**, or the preview's chart button, for detailed signal
inspection. Choose the current slice profile, saved preview-fit observations and
predictions, or whole-ROI curves. Whole-ROI curves share the selections and results
of **Analyze → ROI timecourses**. These scopes can differ.
Inspect exact samples with the pointer, sample slider or arrow keys. Select an
X range, pan or zoom; customize scales, bounds, titles, curves and figure appearance
on the right. **Samples** provides a paginated table; **Statistics** summarizes
included finite samples in the X range, independently of Y display limits.
Saved fits also provide residuals, RMSE, bias and R². **Export data** saves the
selected curves as full-precision CSV or the displayed figure as SVG.
Viewing options do not change image data or fitting settings.

Use **Reconstruct** for image reconstruction and **Preprocess** for denoising and
signal normalization. **Register** aligns images and corrects drift. **Analyze**
groups time-series corrections, ROI timecourses, map statistics and derived
diffusion metrics. **View** changes display settings without processing data.

Processing updates the selected experiment. Its pipeline appears above the
image. **Source** selects the protected **Vendor image** or an available **MRax
reconstruction** as input to the processing steps. Confirmed source changes
reapply later steps and retain existing fits and maps. Reconstruct images first
to enable that source.
Expand the pipeline to draft changes. Insert, copy or remove steps; drag them or
use the arrow buttons to change their order. **Edit** opens inline settings;
**Use action controls** opens the corresponding parameter form. Both edit the
draft. **Apply pipeline** confirms the complete chain; **Discard changes**
restores the confirmed chain. Drafts stay with their experiment while navigating.
Running steps show progress and completion status; **Cancel run** stops a replay
at a safe boundary. Failed or cancelled runs retain the previous images and chain.
Reconstruction defines the source and remains first. Workspace bundles and saved
workflows contain the confirmed pipeline, so apply edits before saving them.
The center workspace scrolls, so expanding settings moves the image down without
reducing its size.
Running another denoising or registration from its action tab replaces the
existing step; explicit copies in the pipeline can be edited independently.
**Original** shows protected source data while retaining the pipeline;
**Processed** returns to its output. **Remove all processing**, followed by
**Apply pipeline**, restores the source and clears the steps. Saving a workspace
retains the confirmed pipeline.

Fits use the current experiment state and selected ROI mask. Existing fits and
maps remain available when processing changes; a notice identifies results from
an earlier state. Rerun the fit to update them.

.. toctree::
   :maxdepth: 1

   gui_loading
   gui_metadata
   gui_rois
   gui_fitting
   gui_registration
   gui_export
