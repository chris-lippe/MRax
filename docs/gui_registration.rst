Registration and drift
======================

Registration and Drift
----------------------

The Registration page aligns the active experiment image to a selected reference
experiment. SimpleITK registration exposes the transform, metric, optimizer,
interpolation, initializer, sampling, histogram, iteration, learning-rate, and
multi-resolution settings. ROIs can be used as metric masks.

Registration stores transforms in the GUI session and reports pre/post normalized
correlation, MSE, MAD, optimizer diagnostics, selected mask pixel counts, and
transform parameters. A reference overlay supports visual inspection while
preserving the experiment view state.

Manual alignment can also create stored transforms using pixel shifts, rotation,
and X/Y stretch. Stored transforms can be applied to the current view state or to
ROIs.

The same page exposes MRax linear drift correction for frame stacks. It fits one
linear transform-parameter drift model over selected dimensions using the JAX drift
optimizer. Drift results are stored separately in the view state. The original image snapshot remains available through
``Show original data`` and ``Restore Source Image``.
