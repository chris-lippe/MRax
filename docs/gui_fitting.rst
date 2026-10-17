Fitting and timecourses
=======================

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
