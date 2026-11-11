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
   inputs explicitly. These resources come from the target study.
4. Apply the workflow. For fitting, leave **Run fits after loading configuration**
   unchecked to load settings without replacing existing maps.
5. Choose **Save workflow file** to retain the recipe for another session.
   **Open workflow file** adds a saved recipe to the current session library.
   Removing a session entry does not delete its saved file.

Processing workflows use ``.mpip``. Reconstruction uses ``.mrec``. Fitting uses
``.mfit``. These are ordinary files opened and saved through the application's
file browser. They store settings and input bindings. Images, ROI masks,
registration transforms and fitted results remain in the studies.

Applying workflows
------------------

**Replace processing steps** replaces the chain with the saved recipe.
Without a reconstruction block, its source is imported image data.
**Append workflow steps** adds and runs the saved steps after the existing chain.
All imported steps remain editable in the processing pipeline.

Use **Insert step** to include a reconstruction (``.mrec``) or fitting (``.mfit``)
workflow. Choose a session workflow or open its file, then assign its inputs.
Reconstruction requires valid acquired k-space and can occur only once, first.
Incompatible acquisitions are rejected. Fitting uses the image state at its
position in the chain and stores separately named maps without replacing images.
Removing a fitting block retains its maps. Confirm or discard edits with the
pipeline controls.

A reconstruction workflow uses the target's raw data and replaces its
reconstruction step. Existing image processing is then reapplied.
In **Reconstruct**, choose a ``.mrec`` preset, insert or remove stages, and expand
each stage to edit its settings. Fourier, gridded Fourier, NUFFT adjoint and
variational presets are included. **Apply reconstruction** confirms the changes.
**Discard changes** restores the confirmed plan. **Open .mrec** loads a plan
without running it. **Reconstruction workflows** saves or shares the draft.

Processing and reconstruction changes retain fitted maps. The GUI marks them
as requiring a new fit when their input state has changed.

A fitting workflow includes the model, solver settings, parameter overrides,
bounds, regularizers, pools, constant dimensions, frame exclusions and fitting
region. It fits the current processing state. Running it replaces that target's
fit maps. Loading its configuration alone retains them.

Inputs use logical bindings resolved within each target study. An ROI may have
a different name in each target. Reference ROIs and maps belong to the experiment
selected for their reference binding. A ``target`` experiment default means the
experiment being processed. Stored registration transforms cannot be exported as
portable processing steps: capture a fresh registration operation so that it can
be estimated again on the target.

The application checks target compatibility before starting. All selected
experiments within one study succeed together or leave that study unchanged.
Different studies are processed separately. Completed studies retain their results
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
defaults into the saved settings. ``mrax_version`` identifies the producer.
``format_version`` and each step's ``version`` control how the file is interpreted.
Readers reject unsupported versions and unknown executable fields. A change in operation semantics requires a new operation
version. A change in the document structure requires a new format version and
an explicit converter.

Version 1 permits ``process``, ``register`` and ``drift`` steps in ``.mpip``,
one ``reconstruct`` step in ``.mrec``, and one ``fit`` step in ``.mfit``.

Processing files containing reconstruction or fitting blocks use format version 2.
Each block's ``settings.workflow`` embeds the complete version 1 ``.mrec`` or
``.mfit`` document. ``settings.bindings`` maps its input IDs to bindings declared
by the enclosing ``.mpip``. Embedded documents keep their own binding namespace
and cannot contain further workflow blocks. No external workflow paths are saved.
Version 1 files remain readable. Readers that do not support version 2 reject it.

Reconstruction settings contain a ``pipeline`` with ``schema_version: 1``, a
``name``, an ``encoding`` (``any``, ``cartesian`` or ``sampled``), and ordered
``stages``. Each stage has ``kind``, ``options``, optional ``enabled`` and ``when``.
Empty options inherit the target acquisition's settings. ``when`` is ``always``,
``cartesian``, ``sampled`` or ``configured``. The latter runs an acquisition
correction only when its metadata or explicit options configure it. Disabled or
removed stages do not run. Exactly one solver, output geometry and coil
combination are required. Measured samples are retained.

Step order is significant. IDs are unique within their lists, and at least one
step must be enabled. Bindings are declared with an ID, kind and label. An input
refers to one using ``{"$binding": "input_1"}``. Resource names are assigned
outside the file when the workflow is applied.

JSON comments, duplicate keys, non-finite numbers and implicit display-state
inputs are rejected. Files are limited to 1 MiB and 64 steps.
Optional metadata belongs in ``extensions`` under namespaced keys such as
``org.example.notes``. Extension metadata is preserved but never executed.
Use **Edit definition** to inspect or edit the JSON. Invalid edits leave the
previous session entry intact.

Python access
-------------

``mrax.workflows`` provides ``load_workflow``, ``save_workflow``,
``loads_workflow`` and ``dumps_workflow``. ``WorkflowDocument.model_json_schema()``
returns the validation schema for editors and tooling. File writes are atomic
and require ``overwrite=True`` to replace an existing file.
