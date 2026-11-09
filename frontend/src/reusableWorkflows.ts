import { api, type Job, type Options } from './api';
import type { components } from './api.generated';

type WireDocument = components['schemas']['WorkflowDocument'];
export type WorkflowKind = WireDocument['kind'];
export type WorkflowDocument = Omit<
  WireDocument,
  'description' | 'bindings' | 'steps' | 'extensions'
> & {
  description: string;
  bindings: NonNullable<WireDocument['bindings']>;
  steps: (Omit<WireDocument['steps'][number], 'enabled'> & { enabled: boolean })[];
  extensions: Options;
};
export type WorkflowEntry = {
  id: string;
  revision: number;
  document: WorkflowDocument;
  origin: { workspace_id: string; experiment: string } | null;
  suggested_bindings: Record<string, string>;
};
export type WorkflowTarget = {
  workspace_id: string;
  label: string;
  revision: number;
  experiments: {
    experiment: string;
    dims: Record<string, number>;
    resources: Record<string, { value: string; label: string }[]>;
  }[];
};
export const workflowExtensions = {
  processing: '.mpip',
  reconstruction: '.mrec',
  fitting: '.mfit',
};
export const targetKey = (workspace: string, experiment: string) =>
  JSON.stringify([workspace, experiment]);
export const printWorkflow = (document: WorkflowDocument) =>
  JSON.stringify(document, null, 2) + '\n';
export const workflowActive = (job: Job) =>
  ['queued', 'running', 'cancelling'].includes(job.status);

export async function waitWorkflow(job: Job, update: (job: Job) => void) {
  update(job);
  while (workflowActive(job)) {
    await new Promise((resolve) => setTimeout(resolve, 150));
    job = await api<Job>('/api/jobs/' + job.id);
    update(job);
  }
  if (job.status !== 'completed') throw new Error(job.detail);
  return job;
}
