Registration and drift
======================

In **Register → Image alignment**, select the fixed and moving experiments and frames.
Choose automatic alignment and review the transform, metric and interpolation
settings, or set shifts, rotation and scale manually. An ROI can constrain
the registration metric.

Inspect the metric summaries, then use **View registered overlay** to select
the reference image. The result is an image layer: adjust its opacity, color
and frame in **View → Image and ROI overlays**, or let it follow the base frame to inspect a sequence
with the stored transform. Plane registrations apply to their stored slice;
volume registrations support other planes. Transfer ROIs with the label-transfer
controls; label resampling uses nearest-neighbour interpolation.

In **Register → Drift correction**, select reference frames, review the correction settings
and run the operation. Progress and cancellation controls are available for
registration and drift. Save the workspace to retain transforms and corrected
data; export transform files and diagnostics from the corresponding controls.

Registration and drift become steps in the experiment's pipeline, above the
image. Edit, disable or remove them there. Subsequent processing and fitting use
the corrected data. Use **Original** to inspect the protected source without
discarding the steps.
