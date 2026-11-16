import type { Job, Options, PipelineStep, Workspace } from './api';
import type { components } from './api.generated';

export type PipelineCatalog = components['schemas']['ProcessingCatalog'];
export type DraftStep = PipelineStep & {
  source_id: string | null;
  source_version: string | null;
  problem?: string;
};
export type PipelineDraft = {
  pipeline_id: string;
  generation: number;
  steps: DraftStep[];
  baseline: string;
  submission?: string;
};
const uid = () => crypto.randomUUID();

export function stepSpec(step: DraftStep) {
  return {
    id: step.id,
    source_id: step.source_id,
    source_version: step.source_version,
    kind: step.kind,
    enabled: step.enabled,
    options: step.options,
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
    pipeline_id: w.pipeline.id,
    generation: w.pipeline.generation,
    steps,
    baseline: JSON.stringify(steps.map(stepSpec)),
  };
}
export function draftProblem(draft: PipelineDraft, workspace?: Workspace) {
  try {
    if (draft.steps.length > 64) return 'Use at most 64 steps.';
    for (const [index, step] of draft.steps.entries()) {
      if (step.problem) return step.problem;
      try {
        const options = step.options;
        if (step.enabled && ['reconstruct', 'fit'].includes(step.kind)) {
          const document = options.workflow as
            | {
                name?: string;
                kind?: string;
                bindings?: { id: string }[];
                steps?: { settings: Options }[];
              }
            | undefined;
          if (document) {
            if (!document.name?.trim()) return 'Give each workflow block a name.';
            if (document.kind !== (step.kind === 'reconstruct' ? 'reconstruction' : 'fitting'))
              return 'Choose the matching workflow for each block.';
            const bindings = options.bindings as Record<string, string> | undefined;
            if (document.bindings?.some((binding) => !bindings?.[binding.id]))
              return 'Assign every workflow block input.';
          }
          if (step.kind === 'reconstruct' && workspace) {
            if (!workspace.can_reconstruct)
              return (
                workspace.reconstruction_unavailable ?? 'This experiment has no valid k-space data.'
              );
            const pipeline = (document?.steps?.[0]?.settings.pipeline ?? options.pipeline) as
              { encoding?: string; stages?: { kind: string; enabled?: boolean }[] } | undefined;
            if (
              pipeline?.encoding &&
              !['any', workspace.reconstruction_encoding].includes(pipeline.encoding)
            )
              return 'The reconstruction workflow is incompatible with this acquisition encoding.';
            if (
              workspace.reconstruction_encoding === 'cartesian' &&
              pipeline?.stages?.some((stage) => stage.enabled !== false && stage.kind === 'nufft')
            )
              return 'NUFFT reconstruction requires non-Cartesian k-space.';
          }
        }
      } catch (failure) {
        return 'Step ' + (index + 1) + ': ' + String(failure);
      }
      if (
        step.kind === 'reconstruct' &&
        draft.steps.slice(0, index).some((row) => row.kind !== 'reconstruct')
      )
        return 'Reconstruction defines the source and stays first.';
    }
    for (const [kind, label] of [
      ['reconstruct', 'reconstruction'],
      ['fit', 'fitting'],
    ]) {
      if (draft.steps.filter((step) => step.kind === kind).length > 1)
        return 'Use one ' + label + ' block per experiment.';
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
  return {
    action: 'replace',
    pipeline_id: draft.pipeline_id,
    base_generation: draft.generation,
    steps: draft.steps.map(stepSpec),
  };
}
export function copiedStep(step: DraftStep): DraftStep {
  if (['reconstruct', 'fit'].includes(step.kind))
    throw Error('This operation supports one block per experiment.');
  return {
    ...step,
    id: uid(),
    version: uid(),
    options: structuredClone(step.options),
    pending: true,
    result_maps: [],
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
    result_maps: [],
    pending: true,
  };
}
export function movedSteps(steps: DraftStep[], id: string, destination: number) {
  const index = steps.findIndex((step) => step.id === id);
  if (index < 0 || steps[index].kind === 'reconstruct') return steps;
  const minimum = steps[0]?.kind === 'reconstruct' ? 1 : 0;
  const target = Math.max(minimum, Math.min(steps.length - 1, destination));
  if (target === index) return steps;
  const next = [...steps];
  const [step] = next.splice(index, 1);
  next.splice(target, 0, step);
  return next;
}

export function droppedSteps(steps: DraftStep[], id: string, boundary: number) {
  const source = steps.findIndex((step) => step.id === id);
  return movedSteps(steps, id, boundary > source ? boundary - 1 : boundary);
}

export function processDraftStep(catalog: PipelineCatalog, methodId: string) {
  const method = catalog.methods.find((method) => method.id === methodId);
  if (!method || method.id === 'dti' || method.unavailable)
    throw Error('Choose an available image processing method.');
  const family = method.group === 'Denoise' ? 'denoise' : method.id;
  return insertedStep(
    'process',
    {
      method: method.id,
      input: 'image',
      component: 'auto',
      ...(method.axes === 'filter' ? { axes: catalog.spatial_axes } : {}),
      ...(method.axes === 'series' ? { axis: catalog.series_axes[0] ?? 'frame' } : {}),
      parameters: Object.fromEntries(method.fields.map((field) => [field.key, field.default])),
    },
    method.label,
    family,
  );
}
export function activePipelineJob(job: Job | null) {
  return Boolean(job && ['queued', 'running', 'cancelling'].includes(job.status));
}
