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
    text: 'This synthetic T2 study has eight echoes, three slices and Cartesian k-space. Expand a study in Data & Studies to browse its experiments. Other studies stay open.',
  },
  {
    id: 'navigate',
    title: 'Navigate images',
    action: 'Open image controls',
    context: { section: 'image', task: 'display' },
    text: 'Move Echo or Slice below the image. Arrow keys adjust a focused slider. Set intensity in View controls. Image-border arrows show increasing axis directions. Scroll to zoom and drag to pan.',
  },
  {
    id: 'metadata',
    title: 'Inspect metadata',
    action: 'Open Metadata',
    context: { section: 'image', task: 'metadata' },
    text: 'Open View → Metadata to search, edit, add or import metadata. Changes stay in the workspace. Source files remain unchanged.',
  },
  {
    id: 'reconstruct',
    title: 'Reconstruct from k-space',
    action: 'Open Reconstruct',
    context: { section: 'reconstruct', task: 'reconstruction' },
    text: 'Choose Cartesian Fourier and Apply reconstruction. Insert, edit or remove stages in the editor. Reconstruction is the first pipeline block and uses the synthetic k-space.',
  },
  {
    id: 'denoise',
    title: 'Denoise the current images',
    action: 'Open Denoising',
    context: { section: 'denoise', task: 'denoising' },
    text: 'Choose Gaussian in Denoise, then Apply denoising. It uses the current pipeline images.',
  },
  {
    id: 'pipeline',
    title: 'Edit the processing pipeline',
    action: 'Show pipeline controls',
    context: { section: 'denoise', task: 'denoising' },
    text: 'Expand Processing pipeline. Edit the Gaussian block, change a setting, then Apply pipeline or Discard changes. Insert, copy, move or remove blocks as needed. Images update when you apply the draft.',
  },
  {
    id: 'compare',
    title: 'Compare images',
    action: 'Open comparison controls',
    context: { section: 'image', task: 'display' },
    text: 'Choose Original vs processed. Move the split divider with the mouse or arrow keys. View also offers overlays and comparisons across experiments with compatible geometry.',
  },
  {
    id: 'register',
    title: 'Align acquisitions',
    action: 'Open Register',
    optional: true,
    context: { section: 'registration', task: 'alignment' },
    text: 'In Register → Image alignment, choose reference and moving images, transform and scope. Registration adds an editable pipeline block. This phantom is already aligned, so registration is optional. The menu also contains drift correction.',
  },
  {
    id: 'roi',
    title: 'Create an ROI',
    action: 'Open Segment',
    context: { section: 'rois', task: 'roi-edit' },
    text: 'Choose Rectangle and drag over the phantom. Click Rectangle again to pan. Use the same ROIs in overlays, fitting and analysis.',
  },
  {
    id: 'mask',
    title: 'Edit a pixel mask',
    action: 'Open mask editing',
    context: { section: 'rois', task: 'roi-edit' },
    text: 'Choose Convert to pixel mask for the selected ROI. Edit with the brush or eraser. Adjust color and opacity. Use Preview export mask to inspect export voxels. Undo restores earlier edits.',
  },
  {
    id: 'fit',
    title: 'Fit parameter maps',
    action: 'Open Fit',
    context: { section: 'fit', task: 'model-fit' },
    text: 'Select T2 spin echo and your ROI. Click Check fit readiness, then Run full map fit. Fitting uses the current pipeline output. Rerun fitting after processing changes to update the retained maps. The reference T2 is 80 ms.',
  },
  {
    id: 'statistics',
    title: 'Analyze fitted maps',
    action: 'Open map statistics',
    context: { section: 'analyze', task: 'statistics' },
    text: 'Choose t2 in Map and Statistics map. Analyze → Maps and statistics summarizes fitted values in your ROIs. Analyze also contains timecourses, time-series corrections and derived maps.',
  },
  {
    id: 'signal',
    title: 'Inspect scientific signal curves',
    action: 'Open signal curves',
    context: { section: 'image', task: 'signal' },
    text: 'Open View → Signal curves. Select your ROI and explore Plot, Samples and Statistics. Adjust axes and inspect sample values or fit diagnostics. Collapse this guide for more space. Resize the compact preview below the image as needed.',
  },
  {
    id: 'workflows',
    title: 'Reuse a workflow',
    action: 'Show processing workflows',
    optional: true,
    context: { section: 'denoise', task: 'denoising' },
    text: 'Workflows captures the pipeline as .mflow. Insert reconstruction (.mflow) and fitting (.mflow) plans as blocks. Reuse workflows in this session or save them for later. Saving is optional.',
  },
  {
    id: 'files',
    title: 'Save and export',
    action: 'Show data controls',
    optional: true,
    text: 'Save workspace (.mrax) retains data, pipelines, ROIs and results. Export data writes selected images, masks, maps or tables. Saving is optional. Finish keeps these results and returns to the previous study.',
  },
];
