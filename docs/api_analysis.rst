.. _api-analysis:

Post-fit analysis
=================

Utilities for extracting ROI statistics, ROI timecourses, and performing group-level
comparisons on batch outputs.

- ``ROI`` defines an ROI mask with optional dimension names for broadcasting.
- ``collect_roi_stats`` produces a long-form table of ROI summaries (mean/variance/etc.).
- ``extract_timecourse`` reduces an image stack to an ROI timecourse.
- ``analyze_groups`` builds subject-level metrics, group summaries, and pairwise statistics.

.. autoclass:: mrax.analysis.roi.ROI
   :members: name, mask, dims

.. autofunction:: mrax.analysis.roi.collect_roi_stats
.. autofunction:: mrax.analysis.roi.match_spatial_mask
.. autofunction:: mrax.analysis.timecourse.extract_timecourse
.. autofunction:: mrax.analysis.timecourse.extract_timecourse_from_experiment
.. autofunction:: mrax.analysis.group.analyze_groups
.. autofunction:: mrax.analysis.group.collect_subject_roi_stats
.. autofunction:: mrax.analysis.group.summarize_group_stats
.. autofunction:: mrax.analysis.group.compare_group_stats
