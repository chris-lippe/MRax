import type { Workflow, WorkflowTask } from './WorkflowNav';

export type TutorialStepId =
  | 'study'
  | 'navigate'
  | 'metadata'
  | 'reconstruct'
  | 'denoise'
  | 'pipeline'
  | 'compare'
  | 'register'
  | 'roi'
  | 'mask'
  | 'fit'
  | 'statistics'
  | 'signal'
  | 'workflows'
  | 'files';

type TutorialStep = {
  id: TutorialStepId;
  title: string;
  text: string;
  action: string;
  context?: { section: Workflow; task: WorkflowTask };
  optional?: boolean;
};

export const tutorialSteps: readonly TutorialStep[] = [
  {
    id: 'study',
    title: 'Your practice study',
    action: '',
    text: 'This synthetic T2 study contains eight echoes, three slices and prepared Cartesian k-space. Expand or collapse studies in Data & Studies to browse experiments. The tutorial works in this separate study; your other studies remain open.',
  },
  {
    id: 'navigate',
    title: 'Navigate images',
    action: 'Open image controls',
    context: { section: 'image', task: 'display' },
    text: 'Move Echo or Slice below the image; arrow keys also adjust a focused slider. View controls contain brightness, contrast and value bounds. Use the mouse wheel to zoom and drag the image to pan.',
  },
  {
    id: 'metadata',
    title: 'Inspect metadata',
    action: 'Open Metadata',
    context: { section: 'image', task: 'metadata' },
    text: 'View → Metadata contains study settings, experiment metadata and acquisition headers. Search fields, edit or add values, or import a metadata file. Edits stay in the MRax workspace; original import files are unchanged.',
  },
  {
    id: 'reconstruct',
    title: 'Reconstruct from k-space',
    action: 'Open Reconstruct',
    context: { section: 'reconstruct', task: 'reconstruction' },
    text: 'Choose the Cartesian Fourier preset and Apply reconstruction. The editor lets you insert, edit or remove reconstruction stages. The reconstruction becomes the first processing block; its input is the synthetic k-space.',
  },
  {
    id: 'denoise',
    title: 'Denoise the current images',
    action: 'Open Denoising',
    context: { section: 'preprocess', task: 'denoising' },
    text: 'In Preprocess → Denoising, keep the Gaussian denoiser and Apply denoising. It acts on the reconstruction at the current pipeline position. Signal normalization is also under Preprocess.',
  },
  {
    id: 'pipeline',
    title: 'Edit the processing pipeline',
    action: 'Show pipeline controls',
    context: { section: 'preprocess', task: 'denoising' },
    text: 'Expand Processing pipeline above the image. Edit the Gaussian block and change a setting, then Apply pipeline or Discard changes. You can also insert, copy, move or remove blocks. Draft edits do not affect images until confirmed.',
  },
  {
    id: 'compare',
    title: 'Compare images',
    action: 'Open comparison controls',
    context: { section: 'image', task: 'display' },
    text: 'Choose Original vs processed, then move the split divider with the mouse or arrow keys. View also offers Overlay and independent image sources, including other experiments with compatible geometry.',
  },
  {
    id: 'register',
    title: 'Align acquisitions',
    action: 'Open Register',
    optional: true,
    context: { section: 'registration', task: 'alignment' },
    text: 'Register → Image alignment selects reference and moving images, transform and scope. Applied registration becomes an editable processing block. This phantom is already aligned, so applying a transform is optional. Drift correction is in the same menu.',
  },
  {
    id: 'roi',
    title: 'Create an ROI',
    action: 'Open Segment',
    context: { section: 'rois', task: 'roi-edit' },
    text: 'Activate Draw rectangle and drag over the phantom. Segment supports multiple ROIs, drawing rounds, 3D expansion and mask combinations. The same ROIs are available in View overlays, fitting and analysis.',
  },
  {
    id: 'mask',
    title: 'Edit a pixel mask',
    action: 'Open mask editing',
    context: { section: 'rois', task: 'label-edit' },
    text: 'For the selected ROI, choose Convert to pixel mask. You can then use the brush or eraser, adjust ROI color and opacity, and Preview export mask to inspect the voxels that will be exported. Undo restores earlier edits.',
  },
  {
    id: 'fit',
    title: 'Fit parameter maps',
    action: 'Open Fit',
    context: { section: 'fit', task: 'model-fit' },
    text: 'Use the T2 spin-echo model, keep your ROI selected and Run full map fit. Fitting uses the current pipeline output. Maps remain available when processing changes; rerun the fit to update them. The synthetic reference T2 is 80 ms.',
  },
  {
    id: 'statistics',
    title: 'Analyze fitted maps',
    action: 'Open map statistics',
    context: { section: 'analyze', task: 'statistics' },
    text: 'Choose t2 in the image Map control and in Statistics map. Analyze → ROI map statistics summarizes fitted values inside your ROIs. The Analyze menu also contains timecourses, time-series corrections and derived maps.',
  },
  {
    id: 'signal',
    title: 'Inspect scientific signal curves',
    action: 'Open signal curves',
    context: { section: 'image', task: 'signal' },
    text: 'View → Signal curves provides exact sample values, adjustable axes, range statistics and fit diagnostics. Select your ROI, then explore Plot, Samples and Statistics. Collapse this guide while inspecting the plot. The compact preview below the image can also be resized.',
  },
  {
    id: 'workflows',
    title: 'Reuse a workflow',
    action: 'Show processing workflows',
    optional: true,
    context: { section: 'preprocess', task: 'denoising' },
    text: 'Processing workflows captures the image chain as .mpip. Reconstruction (.mrec) and fitting (.mfit) plans can be inserted as blocks; insert a fitting block to include a fit in the chain. Reuse plans in this session or save them for later. Saving is optional here.',
  },
  {
    id: 'files',
    title: 'Save and export',
    action: 'Show data controls',
    optional: true,
    text: 'Save workspace (.mrax) retains data, pipelines, ROIs and results. Export data writes selected images, masks, maps or tables to a chosen file or folder. Saving is optional. Finish keeps the practice results and returns to any previously open study.',
  },
];
