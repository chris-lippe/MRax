Batch Runs
==========

Use :func:`mrax.core.batch.run_batch_studies` when you want to process multiple studies
back-to-back without any interdependencies between them. The helper:

- Discovers subfolders under a studies root (or a user-specified subset).
- Loads each folder into a :class:`mrax.data_struct.study.Study` via a ``study_loader``.
- Runs fitting/analysis (planner + executor by default, override with ``analysis_fn``).
- Persists each run in the standard internal layout under a shared output root so the
  manifests can be loaded later for further analysis.

Quickstart example
------------------

Create two synthetic study folders, then batch fit and export them into a common
``batch_output`` directory:

.. code-block:: python

   from pathlib import Path
   import json

   from mrax.core.batch import run_batch_studies
   from mrax.core.synthetic import make_example_suite
   from mrax.data_struct import Quantity, Study, load_study_results

   studies_root = Path("batch_inputs")
   for seed, name in ((0, "subj_a"), (1, "subj_b")):
       study_dir = studies_root / name
       study_dir.mkdir(parents=True, exist_ok=True)
       (study_dir / "config.json").write_text(json.dumps({"seed": seed}))

   def loader(study_dir: Path):
       cfg = json.loads((study_dir / "config.json").read_text())
       examples = make_example_suite(noise_level=0.02, seed=cfg["seed"])[:1]
       study_globals = {"field_strength_mhz": Quantity(400.0, "MHz")}
       return Study.from_examples(examples, study_globals=study_globals), {"config": cfg}

   runs = run_batch_studies(
       studies_root=studies_root,
       study_loader=loader,
       output_root="batch_output",
       analysis_kwargs={"num_steps": 10, "step_size": 0.02},
   )

   for run in runs:
       data = load_study_results(run["manifest_path"])
       print(run["study_name"], "->", data["manifest"]["targets"])

Selective execution
-------------------

- Pass ``selected=["subject_001", "subject_005"]`` to limit processing to specific
  study folders.
- When using ``target_names`` in ``analysis_kwargs``, those targets are
  forwarded to :func:`mrax.data_struct.storage.save_study_results` so only the requested
  experiments are exported.

Custom analysis hooks
---------------------

Provide ``analysis_fn`` to run a custom pipeline per study. The function receives the
``Study`` plus ``study_dir`` and ``metadata`` (returned by the loader):

.. code-block:: python

   def compute_metrics(study, study_dir=None, metadata=None, **kwargs):
       from mrax.core.study_runner import execute_study_plan, plan_study_fits

       plan = plan_study_fits(study, target_names=kwargs.pop("target_names", None))
       results, _ = execute_study_plan(plan, run_fit_kwargs=kwargs)
       return results  # structure consumed by save_study_results

   run_batch_studies(
       studies_root=studies_root,
       study_loader=loader,
       output_root="batch_output",
       analysis_fn=compute_metrics,
       analysis_kwargs={"num_steps": 5, "step_size": 0.02},
       save_kwargs={"file_extension": ".mrd"},
   )

Outputs are saved to ``batch_output/<study_name>/<study_name>.mrax``.
Pass ``save_kwargs={"bundle": False}`` for an editable manifest folder.
