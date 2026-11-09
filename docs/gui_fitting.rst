Fitting and analysis
====================

In **Fit**:

1. Choose a model, compute device and optimizer. Set the iteration count and
   learning rate, then review initial values, bounds and fixed parameters.
2. Supply the required observables and references. Select frames to include
   and configure regularization or constant dimensions if needed.
3. Select a preview ROI to fit its mean signal. For voxelwise maps, choose
   **Selected ROIs** and check the masks to include, or choose **Whole image**,
   then use **Run full map fit**. Multiple masks form a union; hidden ROIs can
   still be included. Display overlays never become fitting data.

Use the sanity check to inspect missing inputs and shape or observable
mismatches before fitting. The signal panel and fit inspection show predictions,
parameter values, units and residuals. See :doc:`models` for model assumptions.

Progress and cancellation controls remain available during a fit. Changing
fit inputs prevents an earlier running result from replacing those edits.
Save the workspace to retain fit settings and results.

In **Analyze → ROI timecourses**, select enabled ROIs, an acquisition axis and a summary statistic,
then choose **Analyze timecourse**. The chart compares the full ROI volumes;
other acquisition axes follow the current frame. Switch to **Data table** to
inspect values and finite voxel counts. Fit frame exclusions can be applied
explicitly. Download the result from **Export data → Download timecourse CSV**.
ROI map statistics and derived diffusion metrics are also available in Analyze.
