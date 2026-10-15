Post-fit Analysis
=================

The analysis toolbox provides reusable utilities for extracting ROI statistics, computing
ROI timecourses, and summarizing/comparing groups from batch outputs. The API is designed
to stay modular: ROI definitions are independent of how you load data, and group statistics
work on long-form tables so you can extend or replace individual steps.

ROI statistics
--------------

Compute per-map ROI summaries (mean, variance, etc.) from fitted parameter maps:

.. code-block:: python

   import numpy as np

   from mrax.analysis import ROI, collect_roi_stats
   from mrax.data_struct import load_study_results

   data = load_study_results("batch_output/subj_a/subj_a.mrax")
   maps = data["experiments"]["t1_ir"]["fitted_maps"]

   mask = np.zeros((16, 16), dtype=bool)
   mask[4:12, 4:12] = True
   roi = ROI("center", mask)

   rows = collect_roi_stats(maps, [roi], stats=("mean", "variance"))
   for row in rows:
       print(row["map"], row["roi"], row["stat"], row["value"], row.get("unit", ""))

Timecourse extraction
---------------------

Extract an ROI timecourse by reducing across spatial dimensions. If the data array is
``(frames, x, y)`` or ``(non_spatial..., x, y)``, the ROI mask is aligned to the spatial
axes automatically:

.. code-block:: python

   from mrax.analysis import extract_timecourse

   mr_image = data["experiments"]["t1_ir"]["mr_image"]
   timecourse = extract_timecourse(mr_image, roi, stat="mean")
   print(timecourse.shape)

Group summaries and comparisons
-------------------------------

Group analysis utilities build on the batch output layout. Provide group members as
manifest paths or study folder names inside ``output_root``:

.. code-block:: python

   from mrax.analysis import analyze_groups

   groups = {
       "control": ["subj_a", "subj_b"],
       "treated": ["subj_c", "subj_d"],
   }

   results = analyze_groups(
       groups,
       output_root="batch_output",
       experiment_name="t1_ir",
       rois=[roi],
       map_names=["t1"],
       stats=("mean",),
   )

   subject_rows = results["subject_stats"]
   group_summary = results["group_summary"]
   comparisons = results["comparisons"]

Plotting summaries
------------------

The analysis functions return plain Python rows and arrays. Use your plotting library
of choice to visualize the outputs. For example, the group summary table can be
converted to a bar plot with ``matplotlib``:

.. code-block:: python

   import matplotlib.pyplot as plt

   rows = [
       row for row in results["group_summary"]
       if row["map"] == "t1" and row["roi"] == "center" and row["stat"] == "mean"
   ]
   plt.bar([row["group"] for row in rows], [row["mean"] for row in rows])
   plt.ylabel("T1 (ms)")

See the analysis notebook for an end-to-end demo using synthetic data and batch outputs.
