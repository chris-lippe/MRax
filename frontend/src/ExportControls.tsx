import { useEffect, useState } from 'react';
import { api, type Job, type TimecourseResult } from './api';
import { timecourseCsv } from './timecourse';
import { useFileSystem } from './FileSystemDialog';
import { defaultExportDraft, type ExportDraft } from './exportDraft';
import type { AnalysisProps } from './analysisTypes';
import type { Inspection } from './metadata';

export function ExportControls({
  workspace: w,
  busy,
  run,
  timecourse,
  signalExport,
  exportDraft,
  onExportDraftChange,
  job,
  onJobChange,
  scope = 'data',
  active = true,
}: Pick<AnalysisProps, 'workspace' | 'busy' | 'run' | 'job'> & {
  timecourse?: TimecourseResult | null;
  signalExport?: { csv: () => string; svg: () => string } | null;
  exportDraft?: ExportDraft;
  onExportDraftChange?: (draft: ExportDraft) => void;
  onJobChange?: (job: Job) => void;
  scope?: 'data' | 'rois';
  active?: boolean;
}) {
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const choose = useFileSystem();
  const [selecting, setSelecting] = useState(false);
  const [error, setError] = useState('');
  const draft = exportDraft ?? defaultExportDraft;
  const { selected, format } = draft;
  const updateExport = (patch: Partial<ExportDraft>) =>
    onExportDraftChange?.({ ...draft, ...patch });
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    if (w)
      void api<Inspection>('/api/workspaces/' + w.id + '/inspect')
        .then((value) => {
          if (!cancelled) setInspection(value);
        })
        .catch((failure) => {
          if (!cancelled) setError(String(failure));
        });
    else setInspection(null);
    return () => {
      cancelled = true;
    };
  }, [active, w?.id, w?.revision, w?.experiment]);
  const objects =
    inspection?.objects.filter((item) =>
      scope === 'rois' ? item.kind === 'roi' : item.kind !== 'roi',
    ) ?? [];
  const object = objects.find((item) => item.id === selected);
  const pending =
    job?.workspace_id === w?.id &&
    job?.status === 'completed' &&
    typeof job.summary?.pending_export === 'string'
      ? job.summary
      : null;
  const conflicts = (Array.isArray(pending?.conflicts) ? pending.conflicts : []) as {
    path: string;
    replaceable: boolean;
    reason: string;
  }[];
  async function cancelPrepared() {
    if (!job) return;
    try {
      const cancelled = await api<Job>('/api/jobs/' + job.id + '/cancel', 'POST');
      onJobChange?.(cancelled);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not cancel the export');
    }
  }
  async function saveExport(kind: string) {
    if (!w) return;
    setSelecting(true);
    setError('');
    try {
      const folder =
        ['package', 'state', 'rois'].includes(kind) ||
        (kind === 'object' && ['nifti', 'dicom', 'roi_set'].includes(format));
      const suffix = kind === 'object' ? '.' + format : kind === 'timecourse' ? '.csv' : '.json';
      const destination = await choose({
        title:
          kind === 'timecourse'
            ? 'Save timecourse CSV'
            : kind === 'rois'
              ? 'Save ROI set'
              : scope === 'rois'
                ? 'Export ROI mask'
                : 'Export data',
        mode: folder ? 'folder' : 'save-file',
        writable: true,
        allowOverwrite:
          ['package', 'state', 'rois'].includes(kind) ||
          (kind === 'object' && !['json', 'csv'].includes(format)),
        extensions: folder ? undefined : [suffix],
        suggestedName: folder
          ? undefined
          : kind === 'object'
            ? 'mrax-export' + suffix
            : 'mrax-' + kind + suffix,
      });
      if (!destination) return;
      const target = {
        root: destination.root,
        path: destination.path,
        overwrite: destination.overwrite ?? false,
      };
      if (kind === 'timecourse' && timecourse) {
        await api('/api/files/text', 'POST', {
          ...target,
          workspace_id: w.id,
          revision: w.revision,
          content: timecourseCsv(timecourse),
        });
        setError('');
      } else await run('export', { kind, object: selected, format, ...target });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not save the export');
    } finally {
      setSelecting(false);
    }
  }
  async function saveSignal(format: 'csv' | 'svg') {
    if (!w || !signalExport) return;
    setSelecting(true);
    setError('');
    try {
      const content = signalExport[format]();
      const destination = await choose({
        title: format === 'svg' ? 'Save signal figure' : 'Save signal source values',
        mode: 'save-file',
        writable: true,
        allowOverwrite: false,
        extensions: ['.' + format],
        suggestedName: 'mrax-signal.' + format,
      });
      if (!destination) return;
      await api('/api/files/text', 'POST', {
        root: destination.root,
        path: destination.path,
        workspace_id: w.id,
        revision: w.revision,
        content,
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not save the signal export');
    } finally {
      setSelecting(false);
    }
  }
  return (
    <div className="tool-section">
      {error && <p role="alert">{error}</p>}
      {pending && (
        <section className="export-conflict-review" role="alert" aria-label="Existing output files">
          <h3>Existing output files</h3>
          <p>
            {conflicts.length} matching{' '}
            {conflicts.length === 1 ? 'file already exists' : 'files already exist'}. Only the
            listed files will be overwritten. Other files and all folders will be retained.
          </p>
          <ul className="export-conflict-list">
            {conflicts.map((file) => (
              <li key={file.path}>
                <span>{file.path}</span>
                {!file.replaceable && <small>{file.reason}</small>}
              </li>
            ))}
          </ul>
          {conflicts.some((file) => !file.replaceable) && (
            <p>Choose another destination to preserve protected files.</p>
          )}
          <div className="button-row">
            <button className="secondary" disabled={busy} onClick={() => void cancelPrepared()}>
              Cancel export
            </button>
            <button
              className="primary"
              disabled={busy || conflicts.some((file) => !file.replaceable)}
              onClick={() =>
                void run('export', {
                  prepared_export: pending.pending_export,
                  replace_files: conflicts.map((file) => file.path),
                })
              }
            >
              Replace listed files
            </button>
          </div>
        </section>
      )}
      {scope === 'data' && <h3>Export an object</h3>}
      {scope === 'data' && signalExport && (
        <section className="signal-export" aria-label="Signal curve exports">
          <h4>Signal curves</h4>
          <div className="button-row">
            <button
              className="secondary"
              disabled={busy || selecting}
              onClick={() => void saveSignal('csv')}
            >
              Save signal CSV
            </button>
            <button
              className="secondary"
              disabled={busy || selecting}
              onClick={() => void saveSignal('svg')}
            >
              Save signal figure SVG
            </button>
          </div>
          <p className="control-note">
            CSV includes all source samples and inclusion flags for the selected curves. SVG uses
            the current plot appearance.
          </p>
        </section>
      )}
      {scope === 'data' && (
        <button
          className="secondary full"
          disabled={!timecourse || busy || selecting}
          onClick={() => {
            void saveExport('timecourse');
          }}
        >
          Save timecourse CSV
        </button>
      )}
      <label>
        {scope === 'rois' ? 'ROI mask' : 'Export object'}
        <select
          aria-label={scope === 'rois' ? 'Export ROI' : 'Export object'}
          value={selected}
          onChange={(e) => {
            updateExport({
              selected: e.target.value,
              format: scope === 'rois' ? 'roi_set' : 'npz',
            });
          }}
        >
          <option value="">{scope === 'rois' ? 'Choose ROI mask' : 'Choose array or map'}</option>
          {objects.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
              {o.unit ? ` [${o.unit}]` : ''}
            </option>
          ))}
        </select>
      </label>
      <label>
        Export format
        <select
          aria-label="Export format"
          value={format}
          onChange={(e) => updateExport({ format: e.target.value })}
        >
          {object?.formats.map((f) => (
            <option key={f.id} value={f.id}>
              {f.label}
            </option>
          ))}
        </select>
      </label>
      <button
        className="primary full"
        disabled={!object || busy || selecting}
        onClick={() => void saveExport('object')}
      >
        {scope === 'rois' ? 'Export selected mask' : 'Export selected object'}
      </button>
      {scope === 'rois' ? (
        <button
          className="secondary full"
          disabled={!w?.rois.length || busy || selecting}
          onClick={() => void saveExport('rois')}
        >
          Save ROI set
        </button>
      ) : (
        <details>
          <summary>Packages and reports</summary>
          <div className="button-stack">
            {[
              ['package', 'Export workspace files'],
              ['state', 'Export portable study and view state'],
              ['fit', 'Save full-fit results'],
              ['diagnostics', 'Save diagnostics report'],
            ].map(([kind, label]) => (
              <button
                key={kind}
                className="secondary full"
                disabled={!w?.loaded || busy || selecting}
                onClick={() => void saveExport(kind)}
              >
                {label}
              </button>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
