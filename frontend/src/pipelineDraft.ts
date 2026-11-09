import type { Job, Options, PipelineStep, Workspace } from './api';
import type { DataDraft } from './Controls';
import type { components } from './api.generated';
import { parseData, printData } from './structuredData';

export type PipelineCatalog = components['schemas']['ProcessingCatalog'];
export type DraftStep = PipelineStep & {
  source_id: string | null;
  source_version: string | null;
  editor?: DataDraft;
};
export type PipelineDraft = {
  generation: number;
  steps: DraftStep[];
  baseline: string;
  submission?: string;
};
const uid = () => crypto.randomUUID();

export function stepOptions(step: DraftStep): Options {
  if (!step.editor) return step.options;
  const raw = step.editor.raw || !step.editor.node ? step.editor.text : printData(step.editor.node);
  const value = parseData(raw);
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw Error('Step settings must be a JSON object.');
  return value as Options;
}
export function stepSpec(step: DraftStep) {
  return {
    id: step.id,
    source_id: step.source_id,
    source_version: step.source_version,
    kind: step.kind,
    enabled: step.enabled,
    options: stepOptions(step),
  };
}
export function newPipelineDraft(w: Workspace): PipelineDraft {
  const steps = w.pipeline.steps.map((step) => ({
    ...step,
    options: structuredClone(step.options),
    source_id: step.id,
    source_version: step.version,
  }));
  return {
    generation: w.pipeline.generation,
    steps,
    baseline: JSON.stringify(steps.map(stepSpec)),
  };
}
export function draftProblem(draft: PipelineDraft) {
  try {
    if (draft.steps.length > 64) return 'Use at most 64 steps.';
    for (const [index, step] of draft.steps.entries()) {
      try {
        stepOptions(step);
      } catch (failure) {
        return 'Step ' + (index + 1) + ': ' + String(failure);
      }
      if (step.kind === 'reconstruct' && index !== 0)
        return 'Reconstruction defines the source and stays first.';
    }
    return '';
  } catch (failure) {
    return String(failure);
  }
}
export function draftChanged(draft: PipelineDraft) {
  try {
    return JSON.stringify(draft.steps.map(stepSpec)) !== draft.baseline;
  } catch {
    return true;
  }
}
export function draftRequest(draft: PipelineDraft): Options {
  const problem = draftProblem(draft);
  if (problem) throw Error(problem);
  return { action: 'replace', base_generation: draft.generation, steps: draft.steps.map(stepSpec) };
}
export function copiedStep(step: DraftStep): DraftStep {
  return {
    ...step,
    id: uid(),
    version: uid(),
    options: structuredClone(stepOptions(step)),
    editor: undefined,
  };
}
export function insertedStep(
  kind: string,
  options: Options,
  label: string,
  family = kind,
): DraftStep {
  return {
    id: uid(),
    version: uid(),
    source_id: null,
    source_version: null,
    kind,
    method: kind === 'process' ? String(options.method) : kind,
    family,
    label,
    enabled: true,
    options,
    plane_dims: [],
  };
}
export function movedSteps(steps: DraftStep[], id: string, destination: number) {
  const index = steps.findIndex((step) => step.id === id);
  if (index < 0 || steps[index].kind === 'reconstruct') return steps;
  const minimum = steps[0]?.kind === 'reconstruct' ? 1 : 0;
  const target = Math.max(minimum, Math.min(steps.length - 1, destination));
  if (target === index) return steps;
  const next = [...steps],
    [step] = next.splice(index, 1);
  next.splice(target, 0, step);
  return next;
}
export function processDraftStep(w: Workspace, catalog: PipelineCatalog, methodId: string) {
  const method = catalog.methods.find((method) => method.id === methodId);
  if (!method || method.id === 'dti' || method.unavailable)
    throw Error('Choose an available image processing method.');
  const family =
    method.group === 'Denoise'
      ? 'denoise'
      : ['normalize', 'dilution'].includes(method.id)
        ? 'normalize'
        : method.id;
  return insertedStep(
    'process',
    {
      method: method.id,
      input: 'image',
      component: 'auto',
      ...(method.axes === 'filter' ? { axes: catalog.spatial_axes } : {}),
      ...(method.axes === 'series' ? { axis: catalog.series_axes[0] ?? 'frame' } : {}),
      ...(method.reference ? { reference_experiment: w.experiment, reference_frame: 0 } : {}),
      parameters: Object.fromEntries(method.fields.map((field) => [field.key, field.default])),
    },
    method.label,
    family,
  );
}
export function activePipelineJob(job: Job | null) {
  return Boolean(job && ['queued', 'running', 'cancelling'].includes(job.status));
}
