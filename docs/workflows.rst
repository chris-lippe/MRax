Reusable workflows
==================

Open **Workflows** from Reconstruct, Denoise, Register, Fit or Analyze.
The session library is shared by all open studies. Use its category filter to
find a preset or a combined processing chain.

**Built-in presets** cover reconstruction, spatial filtering, registration and
basic T1/T2/T2* fitting. Select one to review its input requirements, then choose
**Use preset** to create an editable session copy. **Check inputs** checks the
selected experiments before running. Review registration overlays and fit quality.

1. Choose **Capture current configuration** to save the current form.
   **Capture current pipeline** includes the experiment's confirmed chain.
2. Give the workflow a name. Review its blocks and choose **Keep workflow changes**.
3. Select target experiments and assign the requested ROIs, maps and reference
   experiments. These inputs come from the target study.
4. Apply the workflow, or load fitting and reconstruction settings for a later
   run. Loading and rerunning refresh the same block. A single fitting block uses
   the current image state.
5. Choose **Save workflow file** to keep the recipe for another session.
   **Open workflow file** adds a saved file to the session library.

All workflows use ``.mflow`` files. A file can hold a single preset or a complete
chain. Files contain settings and logical input bindings. Images, masks,
estimated transforms and fitted results remain in the workspace.

Applying workflows
------------------

**Replace processing steps** starts the chain from the imported image data.
**Append workflow steps** adds new blocks and refreshes matching workflow blocks.
Reconstruction requires acquired k-space and must come first on its target.
The application checks acquisition compatibility before reconstruction.

Fit blocks use the image state at their position in the chain. Their named maps
are stored alongside the images. Rerunning the same workflow updates its fitting
blocks. Other fit results remain available.
Changing an earlier processing step reruns the fit when the pipeline is applied.

Use **Copy** in the pipeline editor to repeat image processing in sequence.
**Edit** opens the block's settings in the action panel. Both panels share the
same draft. **Apply** runs the edited chain. **Discard changes** restores its settings.

The pipeline selector belongs to the selected experiment. Use **New pipeline**
to start from its source data, or **Duplicate pipeline** to copy its current state.
Switching restores cached images, maps and fit settings. **Compare pipeline results**
opens an image or map comparison. Workspace bundles retain all pipelines and results.

All selected experiments within a study are applied together. An error or
cancellation leaves that study unchanged. Studies are processed separately.
Completed studies retain their results if a later study fails.

File contract
-------------

Files use UTF-8 JSON. A minimal definition is::

   {
     "format": "mrax.workflow",
     "kind": "denoising",
     "mrax_version": "0.4.0",
     "name": "Spatial Gaussian",
     "bindings": [],
     "steps": [{
       "id": "smooth",
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

Blocks use ``process``, ``register``, ``drift``, ``reconstruct`` or ``fit``.
Each block contains its settings directly. Reconstruction settings contain an
ordered reconstruction plan. Fitting settings contain the model, solver,
parameters, pools, bounds, regularization and fitting region.
The ``kind`` field is a category derived from the blocks.
Each pipeline can contain one reconstruction block and one fitting block.
Reconstruction comes first. Other processing blocks run in sequence and may repeat.

A resource input uses ``{"$binding": "input_id"}``.
Bindings describe experiments, images, maps or ROIs. Map and ROI bindings can
name an experiment binding as their owner. Fitting blocks list auxiliary
resources under ``settings.inputs``, keyed by their role in the selected model.
Role names and model settings are operation content. The fitting code checks
them when the workflow is applied. The file schema has no list of models,
experiment types or model-specific inputs.
An optional block ``target`` uses an experiment binding.
Without a target binding, the block runs on the experiment being processed.

Image outputs use ``{"$output": "block_id", "port": "image"}``.
Map outputs add a field name:
``{"$output": "fit_block", "port": "map", "name": "t2"}``.
Output references must point to earlier enabled blocks. Unknown maps and
incompatible inputs cause an error. Forward references and cycles are rejected.
