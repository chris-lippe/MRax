import { api, type Job } from './api';
import type { AnalysisProps } from './analysisTypes';

export function isFileJob(job: Job | null) {
  return Boolean(job && ['export', 'save'].includes(job.operation));
}

export function JobStatus({
  workspace,
  job,
  changeView,
  studyLabel,
}: Pick<AnalysisProps, 'workspace' | 'job' | 'changeView'> & { studyLabel?: string }) {
  const running = job && ['queued', 'running', 'cancelling'].includes(job.status);
  const currentJob = job?.workspace_id === workspace?.id;
  if (!job || (!currentJob && !running)) return null;
  return (
    <div className="job-status" aria-live="polite">
      {!currentJob && (
        <span className="job-context">Background task: {studyLabel ?? 'another study'}</span>
      )}
      <strong>{job.detail}</strong>
      {running && (
        <>
          <progress max={1} value={job.progress ?? undefined} />
          <button
            className="secondary"
            disabled={job.status === 'cancelling'}
            onClick={() => void api(`/api/jobs/${job.id}/cancel`, 'POST')}
          >
            Cancel operation
          </button>
        </>
      )}
      {job.status === 'completed' &&
        job.summary?.saved === true &&
        typeof job.summary?.path === 'string' && (
          <span className="file-source">Saved to {job.summary.path}</span>
        )}
      {currentJob &&
        job.status === 'completed' &&
        typeof job.summary?.experiment === 'string' &&
        workspace?.experiment !== job.summary.experiment && (
          <button
            className="secondary full"
            onClick={() => void changeView({ experiment: job.summary!.experiment })}
          >
            Open result
          </button>
        )}
    </div>
  );
}
