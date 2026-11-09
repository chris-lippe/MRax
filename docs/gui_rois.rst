ROIs and masks
==============

In **Segment → Create and edit ROIs**, choose a rectangle, ellipse, polygon, lasso, brush, intensity wand
or Otsu selection, then draw on the image. Complete a polygon with Enter or a
double click; Escape cancels it. Use coordinate fields and edit handles to
adjust shapes. Drawing is available in native-grid views.

Select an ROI to rename it, enable it for individual experiments, or inspect
its mean signal below the image. **Data table** shows observations and fitted
values; the curve-axis control selects the plotted dimension.

Choose **View → Image and ROI overlays** to show or hide individual masks. Hidden masks
remain editable in **Segment**. Enabling a mask attaches it to an experiment;
its visibility and use in fitting are separate choices.

Select an ROI in **Segment**, or in **View → Image and ROI overlays**, to change
its color and opacity. Indexed masks also offer individual **Label colors**.
Appearance is saved with the experiment and does not change mask values or fits.
**Reset ROI appearance** restores the automatic palette and default opacity.

Use **Convert to pixel mask** beneath the selected ROI to retain its labelled
voxels and continue editing with a brush. **Undo ROI edit** restores the shape.
**Preview export mask** shows the stored native label array used by NIfTI and
other mask exports, with plane/slice selection and exact label inspection using
the pointer or arrow keys. It remains opaque for inspection even when the ROI
overlay is transparent. Registration may display the mask on a different grid;
the export preview shows its original grid.

Use **Drawing mode** to add or subtract successive rounds from the selected ROI.
Drawing stays active between rounds and slice changes; turn it off to finish.
**Auto subtract** removes newly drawn voxels from other enabled ROIs, for shapes
and brush strokes. **Undo ROI edit** restores the entire previous edit.

**Segment → Edit label masks** provides brush painting, erasing, relabelling and
label deletion. Choose a coloured label or enter a new **Label value**.
**Erase only label** protects other labels in the same mask; zero is background.

**Segment → 3D expansion** copies the current slice to selected slices or
interpolates gaps without changing existing labelled voxels. Slice numbers start at 1; change the image plane
to choose the expansion axis. **Grow ROI in 3D** adds background voxels within
the chosen Euclidean distance in voxel units, preserving existing labels.

**Segment → Combine ROIs** creates a new joint binary ROI or indexed label mask
and retains the source ROIs. Indexed labels follow selection order; choose
which source wins at overlaps, or reject overlapping masks.

Use **Segment → Import masks and ROI sets** to load masks; save them from **Export data**.

**Analyze → ROI map statistics** shows mean, standard deviation, median and finite voxel count
for enabled masks across a fitted or derived map, including masks hidden in View.
