Reusable workflows
==================

Use **Processing workflows** in Preprocess, Register or Analyze,
**Reconstruction workflows** in Reconstruct, or **Fitting workflows** in Fit.
Capture a configuration once, then apply it to other experiments or studies.
The session library is shared by all open studies.

1. Configure and apply your processing steps, or configure reconstruction or fitting.
   Fitting and reconstruction capture the values currently shown in their forms.
2. Open the corresponding workflows panel and choose **Capture current pipeline**
   or **Capture current configuration**. Give it a descriptive name and choose
   **Keep workflow changes**.
3. Select target experiments. Assign any requested ROI, map or reference-experiment
   inputs explicitly; these resources come from the target study.
4. Apply the workflow. For fitting, leave **Run fits after loading configuration**
   unchecked to load settings without replacing existing maps.
5. Choose **Save workflow file** to retain the recipe for another session.
   **Open workflow file** adds a saved recipe to the current session library.
   Removing a session entry does not delete its saved file.

Processing workflows use ``.mpip``; reconstruction uses ``.mrec``; fitting uses
``.mfit``. These are ordinary files opened and saved through the application's
file browser. They contain settings, not measurement arrays, ROI masks,
registration transforms or fitted results.

Applying workflows
------------------

**Replace processing steps** replays the recipe from protected source data,
retaining the target's reconstruction step. **Append workflow steps** adds to
the existing pipeline. Replacing is repeatable; appending intentionally applies
the added operations again. Individual imported steps remain editable in the
processing pipeline.

A reconstruction workflow uses the target's raw data and replaces its
reconstruction step. Existing image processing is then reapplied.
Processing and reconstruction changes retain fitted maps; the GUI marks them
as requiring a new fit when their input state has changed.

A fitting workflow includes the model, solver settings, parameter overrides,
bounds, regularizers, pools, constant dimensions, frame exclusions and fitting
region. It fits the current processing state. Running it replaces that target's
fit maps; loading its configuration alone retains them.

Inputs use logical bindings rather than source study identifiers. An ROI may have
a different name in each target. Reference ROIs and maps belong to the experiment
selected for their reference binding. A ``target`` experiment default means the
experiment being processed. Stored registration transforms cannot be exported as
portable processing steps: capture a fresh registration operation so that it can
be estimated again on the target.

The application checks target compatibility before starting. All selected
experiments within one study succeed together or leave that study unchanged.
Different studies are processed separately; completed studies retain their results
if a later study fails or is cancelled. Progress and errors identify each study.
Edits made by another client prevent stale worker results from overwriting them.

File contract
-------------

All three suffixes use UTF-8 JSON with the same document structure. The suffix
must match ``kind``. For example, a minimal processing definition is::

   {
     "format": "mrax.workflow",
     "format_version": 1,
     "kind": "processing",
     "mrax_version": "0.4.0",
     "name": "Spatial Gaussian",
     "description": "Denoise each image plane",
     "bindings": [],
     "steps": [{
       "id": "denoise",
       "operation": "process",
       "version": 1,
       "enabled": true,
       "settings": {
         "method": "gaussian",
         "input": "image",
         "axes": ["x", "y"],
         "parameters": {"sigma": 1.0}
       }
     }],
     "extensions": {}
   }

Exports record the installed MRax version and resolve processing and alignment
defaults into the saved settings. ``mrax_version`` identifies the producer;
``format_version`` and each step's ``version`` control how the file is interpreted.
Readers reject unsupported versions and unknown executable fields instead of
guessing their meaning. A change in operation semantics requires a new operation
version; a change in the document structure requires a new format version and
an explicit converter.

Version 1 permits ``process``, ``register`` and ``drift`` steps in ``.mpip``,
one ``reconstruct`` step in ``.mrec``, and one ``fit`` step in ``.mfit``.
Step order is significant. IDs are unique within their lists, and at least one
step must be enabled. Bindings are declared with an ID, kind and label; an input
refers to one using ``{"$binding": "input_1"}``. Resource names are assigned
outside the file when the workflow is applied.

JSON comments, duplicate keys, non-finite numbers and implicit display-state
inputs are rejected. Files are limited to 1 MiB and 64 steps.
Optional metadata belongs in ``extensions`` under namespaced keys such as
``org.example.notes``. Extension metadata is preserved but never executed.
Use **Edit definition** to inspect or edit the JSON; invalid edits leave the
previous session entry intact.

Python access
-------------

``mrax.workflows`` provides ``load_workflow``, ``save_workflow``,
``loads_workflow`` and ``dumps_workflow``. ``WorkflowDocument.model_json_schema()``
returns the validation schema for editors and tooling. File writes are atomic
and require ``overwrite=True`` to replace an existing file.
