import { useEffect, useId, useState } from 'react';
import { api, type Job, type Options, type PipelineStep, type Workspace } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';
import { DataEditor, dataController, dataDraft } from './Controls';
import { PipelineWorkflowBlock, type BlockOptions } from './PipelineWorkflowBlock';
import type { WorkflowDocument } from './reusableWorkflows';
import {
  activePipelineJob,
  copiedStep,
  draftChanged,
  draftProblem,
  insertedStep,
  movedSteps,
  newPipelineDraft,
  processDraftStep,
  stepOptions,
  type DraftStep,
  type PipelineCatalog,
  type PipelineDraft,
} from './pipelineDraft';

type Props = {
  workspace: Workspace;
  busy: boolean;
  job: Job | null;
  pipelineJob: Job | null;
  draft: PipelineDraft | null;
  onDraft: (value: PipelineDraft | null) => void;
  apply: (draft: PipelineDraft) => Promise<void>;
  cancel: (id: string) => Promise<void>;
  run: (operation: string, options?: Options) => Promise<void>;
  edit: (step: PipelineStep) => void;
  editing: string | null;
  cancelEdit: () => void;
  compare: () => void;
};
const statusLabels: Record<string, string> = {
  pending: 'Waiting',
  running: 'Running',
  completed: 'Completed',
  skipped: 'Disabled',
  unchanged: 'Unchanged',
  failed: 'Failed',
  cancelled: 'Cancelled',
};
function icon(step: PipelineStep) {
  return step.kind === 'reconstruct'
    ? 'reconstruct'
    : step.kind === 'fit'
      ? 'fit'
      : ['register', 'stored_registration'].includes(step.kind)
        ? 'registration'
        : step.kind === 'drift'
          ? 'drift'
          : step.family === 'denoise'
            ? 'denoise'
            : 'series';
}

export function Pipeline({
  workspace: w,
  busy,
  job,
  pipelineJob,
  draft,
  onDraft,
  apply,
  cancel,
  run,
  edit,
  editing,
  cancelEdit,
  compare,
}: Props) {
  const pipeline = w.pipeline;
  const [expanded, setExpanded] = useState(false);
  const [settings, setSettings] = useState<string | null>(null);
  const [insertAt, setInsertAt] = useState<number | null>(null);
  const [insertion, setInsertion] = useState('gaussian');
  const [block, setBlock] = useState<BlockOptions | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [loaded, setLoaded] = useState<{ key: string; catalog: PipelineCatalog } | null>(null);
  const contentId = useId();
  const context = JSON.stringify([w.id, w.experiment]);
  const catalogKey = JSON.stringify([w.id, w.experiment, w.revision]);
  useEffect(() => {
    if (!expanded) return;
    let cancelled = false;
    void api<PipelineCatalog>('/api/workspaces/' + w.id + '/processing')
      .then((catalog) => {
        if (!cancelled) setLoaded({ key: catalogKey, catalog });
      })
      .catch((failure) => {
        if (!cancelled) setError(String(failure));
      });
    return () => {
      cancelled = true;
    };
  }, [expanded, catalogKey, w.id]);
  useEffect(() => {
    setSettings(null);
    setInsertAt(null);
    setError('');
    setNotice('');
    setBlock(null);
  }, [context]);
  const catalog = loaded?.key === catalogKey ? loaded.catalog : null;
  const current = draft ?? newPipelineDraft(w);
  const changed = draft ? draftChanged(draft) : false;
  const issue = draft ? draftProblem(draft, w) : '';
  const stale = Boolean(draft && draft.generation !== pipeline.generation);
  const blocked = busy || activePipelineJob(job);
  const ownRun = draft?.submission && draft.submission === job?.id ? job : pipelineJob;
  const signature = ownRun?.summary?.pipeline as
    { signature?: string; generation?: number } | undefined;
  const awaitingCommit =
    ownRun?.status === 'completed' &&
    ((signature?.generation ?? 0) > pipeline.generation || draft?.submission === ownRun.id);
  const matchingRun =
    ownRun &&
    (activePipelineJob(ownRun) ||
      draft?.submission === ownRun.id ||
      (!changed && ownRun.status === 'completed' && signature?.signature === pipeline.signature))
      ? ownRun
      : null;
  const activity = matchingRun?.pipeline;
  const updatingWorkspace =
    activePipelineJob(job) &&
    job?.workspace_id === w.id &&
    (!job.pipeline || job.pipeline.experiment === w.experiment);
  const locked =
    busy ||
    updatingWorkspace ||
    submitting ||
    awaitingCommit ||
    Boolean(
      draft?.submission && draft.submission === matchingRun?.id && activePipelineJob(matchingRun),
    ) ||
    Boolean(!draft && activity && activePipelineJob(matchingRun));
  const steps =
    !draft && activity && activePipelineJob(matchingRun)
      ? activity.steps.map(
          (row) =>
            current.steps.find((step) => step.id === row.id) ?? {
              ...insertedStep(row.kind, { method: row.method }, row.label, row.family),
              id: row.id,
              enabled: row.status !== 'skipped',
            },
        )
      : current.steps;
  const reconstruction = steps.find((step) => step.kind === 'reconstruct');
  const sourceKind = reconstruction?.enabled ? 'kspace' : 'image';
  const source = pipeline.sources?.[sourceKind];
  const methods = catalog?.methods.filter((method) => method.id !== 'dti') ?? [];
  function label(step: DraftStep) {
    try {
      const options = stepOptions(step);
      const method = options.method;
      if (options.workflow) return (options.workflow as WorkflowDocument).name;
      return step.kind === 'process'
        ? (catalog?.methods.find((item) => item.id === method)?.label ?? step.label)
        : step.label;
    } catch {
      return step.label;
    }
  }
  function update(next: PipelineDraft) {
    onDraft({ ...next, submission: undefined });
    setError('');
  }
  function changeStep(id: string, patch: Partial<DraftStep>) {
    update({
      ...current,
      steps: current.steps.map((step) => (step.id === id ? { ...step, ...patch } : step)),
    });
  }
  function move(id: string, target: number) {
    const next = movedSteps(current.steps, id, target);
    update({ ...current, steps: next });
    setNotice(
      'Step moved to position ' +
        (next.findIndex((step) => step.id === id) + 1) +
        '. Changes are unapplied.',
    );
  }
  function discard() {
    onDraft(null);
    setSettings(null);
    setInsertAt(null);
    setError('');
    setNotice('Pipeline edits discarded.');
  }
  function add() {
    if (insertAt === null || !catalog) return;
    try {
      const step = ['reconstruct', 'fit'].includes(insertion)
        ? (() => {
            if (!block) throw Error('Choose a workflow for the block.');
            return insertedStep(insertion, structuredClone(block), block.workflow.name);
          })()
        : insertion === 'register'
          ? insertedStep(
              'register',
              {
                reference_experiment:
                  w.experiments.find((name) => name !== w.experiment) ?? w.experiment,
                moving_experiment: w.experiment,
                reference_frame: 0,
                moving_frame: w.frame,
                scope: 'plane',
                manual: false,
                options: {},
              },
              'Registration',
              'register',
            )
          : insertion === 'drift'
            ? insertedStep('drift', { settings: {}, roi_name: w.selected_roi }, 'Drift correction')
            : processDraftStep(catalog, insertion);
      const next = [...current.steps];
      const minimum = next[0]?.kind === 'reconstruct' ? 1 : 0;
      next.splice(
        insertion === 'reconstruct' ? 0 : Math.max(minimum, Math.min(insertAt, next.length)),
        0,
        step,
      );
      const problem = draftProblem({ ...current, steps: next }, w);
      if (problem) throw Error(problem);
      update({ ...current, steps: next });
      setSettings(step.id);
      setInsertAt(null);
      setBlock(null);
      setNotice('Step inserted. Configure its settings before applying the pipeline.');
    } catch (failure) {
      setError(String(failure));
    }
  }
  async function confirm() {
    if (!draft || !changed || blocked || locked || stale || issue) return;
    setSubmitting(true);
    setError('');
    try {
      await apply(draft);
      setSettings(null);
      setInsertAt(null);
      setNotice('');
    } catch (failure) {
      setError(String(failure));
    } finally {
      setSubmitting(false);
    }
  }
  const selected = steps.find((step) => step.id === settings);
  const editor =
    selected?.editor ?? (selected ? dataDraft(JSON.stringify(selected.options, null, 2)) : null);
  const controller =
    editor && selected
      ? dataController(editor, (next) => {
          const value = typeof next === 'function' ? next(editor) : next;
          changeStep(selected.id, { editor: value });
        })
      : null;
  const currentStep = activity?.steps.find((step) => step.status === 'running');
  const runText =
    matchingRun?.status === 'completed'
      ? 'Pipeline applied'
      : matchingRun?.status === 'failed'
        ? 'Pipeline failed. Previous data kept'
        : matchingRun?.status === 'cancelled'
          ? 'Pipeline cancelled. Previous data kept'
          : matchingRun?.status === 'stale'
            ? 'Inputs changed. Pipeline was not applied'
            : matchingRun?.status === 'cancelling'
              ? 'Cancelling pipeline'
              : currentStep
                ? 'Running ' + currentStep.label
                : matchingRun
                  ? 'Preparing pipeline'
                  : '';

  return (
    <section className="pipeline-panel" aria-label="Processing pipeline">
      <header>
        <h2>
          <button
            className="pipeline-toggle"
            aria-label={expanded ? 'Hide processing steps' : 'Show processing steps'}
            aria-expanded={expanded}
            aria-controls={contentId}
            onClick={() => setExpanded(!expanded)}
          >
            <WorkbenchIcon name="chevron" /> Processing pipeline{' '}
            {steps.length > 0 && <small>{steps.length}</small>}
          </button>
        </h2>
        {!expanded && (
          <span className="pipeline-output-state">
            {activePipelineJob(matchingRun) ? (
              <>
                <i className="pipeline-spinner" aria-hidden="true" /> Running
              </>
            ) : changed ? (
              'Draft'
            ) : pipeline.steps.length > 0 ? (
              pipeline.enabled ? (
                'Processed'
              ) : (
                'Original'
              )
            ) : (
              ''
            )}
          </span>
        )}
      </header>
      <div id={contentId} hidden={!expanded}>
        <div className="pipeline-toolbar">
          <label className="pipeline-input">
            Source
            <select
              aria-label="Pipeline source"
              disabled={locked}
              value={reconstruction?.enabled ? 'mrax' : 'vendor'}
              onChange={(event) => {
                const enabled = event.target.value === 'mrax';
                if (reconstruction) changeStep(reconstruction.id, { enabled });
                else if (enabled)
                  update({
                    ...current,
                    steps: [
                      insertedStep(
                        'reconstruct',
                        { view: 'mrax_reconstruction', component: 'magnitude', device: 'cpu' },
                        w.reconstruction_label ?? 'Reconstruction',
                      ),
                      ...current.steps,
                    ],
                  });
              }}
            >
              <option value="vendor">Image data</option>
              <option
                value="mrax"
                disabled={!w.can_reconstruct || (!reconstruction && steps.length >= 64)}
                title={
                  !w.can_reconstruct
                    ? (w.reconstruction_unavailable ?? 'No valid k-space data.')
                    : undefined
                }
              >
                {reconstruction ? label(reconstruction) : 'Reconstruction (.mrec)'}
              </option>
            </select>
          </label>
          <div className="pipeline-actions">
            {pipeline.steps.length > 0 && (
              <div className="segmented-control" role="group" aria-label="Pipeline output">
                <button
                  aria-pressed={pipeline.enabled}
                  disabled={blocked || changed || submitting}
                  onClick={() => void run('pipeline', { action: 'enabled', enabled: true })}
                >
                  Processed
                </button>
                <button
                  aria-pressed={!pipeline.enabled}
                  disabled={blocked || changed || submitting}
                  onClick={() => void run('pipeline', { action: 'enabled', enabled: false })}
                >
                  Original
                </button>
              </div>
            )}
            {pipeline.steps.length > 0 && (
              <button className="secondary" disabled={blocked || submitting} onClick={compare}>
                <WorkbenchIcon name="split" />
                Compare
              </button>
            )}
            <button
              className="secondary"
              disabled={locked || steps.length >= 64}
              onClick={() => {
                setInsertAt(steps.length);
                setSettings(null);
              }}
            >
              {' '}
              <WorkbenchIcon name="add" />
              Insert step
            </button>
            <button
              className="secondary"
              disabled={locked || !steps.length}
              onClick={() => {
                update({ ...current, steps: [] });
                setSettings(null);
              }}
            >
              Remove all processing
            </button>
          </div>
        </div>
        <div className="pipeline-draft-toolbar">
          <span>{changed ? 'Unapplied changes' : 'Confirmed pipeline'}</span>
          {draft && (
            <button
              className="secondary"
              data-tutorial="pipeline-discard"
              disabled={locked}
              onClick={discard}
            >
              Discard changes
            </button>
          )}
          <button
            className="primary"
            disabled={!changed || Boolean(issue) || stale || blocked || locked}
            onClick={() => void confirm()}
          >
            {locked ? 'Applying pipeline…' : 'Apply pipeline'}
          </button>
        </div>
        {stale && (
          <p role="alert">
            The confirmed pipeline changed while this draft was being edited. Discard this draft to
            load its current steps.
          </p>
        )}
        {issue && <p role="alert">{issue}</p>}
        {error && <p role="alert">{error}</p>}
        <span className="visually-hidden" role="status" aria-label="Pipeline edit status">
          {notice}
        </span>
        {matchingRun && (
          <div className="pipeline-run-status" data-state={matchingRun.status}>
            <span role="status" aria-label="Pipeline run status">
              {runText}
            </span>
            {activePipelineJob(matchingRun) && (
              <progress
                aria-label="Pipeline progress"
                max={1}
                value={matchingRun.progress ?? undefined}
              />
            )}
            {activePipelineJob(matchingRun) && (
              <button
                className="secondary"
                disabled={matchingRun.status === 'cancelling'}
                onClick={() => {
                  void cancel(matchingRun.id).catch((failure) => setError(String(failure)));
                }}
              >
                {matchingRun.status === 'cancelling' ? 'Cancelling…' : 'Cancel run'}
              </button>
            )}
            {['failed', 'stale'].includes(matchingRun.status) && <p>{matchingRun.detail}</p>}
          </div>
        )}
        <ol className="pipeline-steps" aria-busy={Boolean(activePipelineJob(matchingRun))}>
          <li
            className="pipeline-source"
            aria-label="Pipeline input"
            title={source?.path ?? undefined}
          >
            <WorkbenchIcon name={sourceKind === 'kspace' ? 'reconstruct' : 'image'} />
            <strong>{sourceKind === 'kspace' ? 'Acquired k-space' : 'Image data'}</strong>
            {source && (
              <>
                <small>{source.format}</small>
                <small className="pipeline-source-location">{source.location}</small>
              </>
            )}
          </li>
          {steps.map((step, index) => {
            const progress = activity?.steps.find(
              (row) => row.client_id === step.id || row.id === step.id,
            );
            const state = progress?.status ?? '';
            const text =
              state === 'pending' && matchingRun && !activePipelineJob(matchingRun)
                ? 'Not run'
                : statusLabels[state];
            return (
              <li
                key={step.id}
                className={'pipeline-step' + (!step.enabled ? ' disabled' : '')}
                aria-label={label(step)}
                data-step-id={step.id}
                data-state={state}
                data-editing={editing === step.id || settings === step.id}
                draggable={!locked && !blocked && step.kind !== 'reconstruct'}
                onDragStart={(event) => {
                  setDragging(step.id);
                  event.dataTransfer.setData('application/x-mrax-step', step.id);
                  event.dataTransfer.effectAllowed = 'move';
                }}
                onDragEnd={() => setDragging(null)}
                onDragOver={(event) => {
                  if (dragging && step.kind !== 'reconstruct') {
                    event.preventDefault();
                    event.dataTransfer.dropEffect = 'move';
                  }
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  const id = event.dataTransfer.getData('application/x-mrax-step');
                  if (id === dragging && !locked && !blocked) move(id, index);
                  setDragging(null);
                }}
              >
                <div className="pipeline-step-heading">
                  <span className="pipeline-order">{index + 1}</span>
                  <label className="check-label">
                    <input
                      type="checkbox"
                      aria-label={'Enable ' + label(step)}
                      checked={step.enabled}
                      disabled={locked}
                      onChange={(event) => changeStep(step.id, { enabled: event.target.checked })}
                    />
                    <WorkbenchIcon name={icon(step)} />
                    <strong>{label(step)}</strong>
                  </label>
                  <span
                    className="pipeline-drag-handle"
                    title={
                      step.kind === 'reconstruct'
                        ? 'Reconstruction defines the source and stays first.'
                        : 'Drag to move this step'
                    }
                  >
                    <WorkbenchIcon name={step.kind === 'reconstruct' ? 'lock' : 'drag'} />
                  </span>
                </div>
                {progress && (
                  <span className="pipeline-step-status">
                    {state === 'running' && <i className="pipeline-spinner" aria-hidden="true" />}
                    {text}
                  </span>
                )}
                {state === 'running' && (
                  <progress
                    aria-label={label(step) + ' progress'}
                    max={1}
                    value={progress?.progress || undefined}
                  />
                )}
                {!['image', 'view', undefined].includes(
                  step.options.input as string | undefined,
                ) && <p className="control-note">Input: {String(step.options.input)}</p>}
                {step.kind === 'fit' && (
                  <small className="control-note">
                    Fitting workflow (.mfit) · {step.result_maps?.length ?? 0} maps stored alongside
                    image data
                  </small>
                )}
                <div className="pipeline-step-actions">
                  <button
                    className="secondary"
                    data-tutorial={step.family === 'denoise' ? 'pipeline-denoise-edit' : undefined}
                    disabled={locked}
                    onClick={() => {
                      setSettings(settings === step.id ? null : step.id);
                      setInsertAt(null);
                      if (!draft) onDraft(current);
                    }}
                  >
                    Edit
                  </button>
                  <button
                    className="secondary"
                    aria-label={'Move ' + label(step) + ' earlier'}
                    title="Move earlier"
                    disabled={
                      locked || step.kind === 'reconstruct' || index <= (reconstruction ? 1 : 0)
                    }
                    onClick={() => move(step.id, index - 1)}
                  >
                    <WorkbenchIcon name="back" />
                  </button>
                  <button
                    className="secondary"
                    aria-label={'Move ' + label(step) + ' later'}
                    title="Move later"
                    disabled={locked || step.kind === 'reconstruct' || index === steps.length - 1}
                    onClick={() => move(step.id, index + 1)}
                  >
                    <WorkbenchIcon name="forward" />
                  </button>
                  <button
                    className="secondary"
                    aria-label={'Copy ' + label(step)}
                    title={
                      step.kind === 'reconstruct'
                        ? 'Reconstruction is the single image source.'
                        : 'Copy step'
                    }
                    disabled={locked || step.kind === 'reconstruct' || steps.length >= 64}
                    onClick={() => {
                      try {
                        const next = [...current.steps];
                        next.splice(index + 1, 0, copiedStep(step));
                        update({ ...current, steps: next });
                      } catch (failure) {
                        setError(String(failure));
                      }
                    }}
                  >
                    <WorkbenchIcon name="copy" />
                  </button>
                  <button
                    className="secondary"
                    aria-label={'Insert after ' + label(step)}
                    title="Insert after"
                    disabled={locked || steps.length >= 64}
                    onClick={() => {
                      setInsertAt(index + 1);
                      setSettings(null);
                    }}
                  >
                    <WorkbenchIcon name="add" />
                  </button>
                  <button
                    className="secondary"
                    disabled={locked}
                    onClick={() => {
                      update({
                        ...current,
                        steps: current.steps.filter((row) => row.id !== step.id),
                      });
                      if (settings === step.id) setSettings(null);
                    }}
                  >
                    Remove
                  </button>
                </div>
                {step.kind !== 'reconstruct' && index === (reconstruction ? 1 : 0) && (
                  <button
                    className="pipeline-insert-before secondary"
                    disabled={locked || steps.length >= 64}
                    onClick={() => {
                      setInsertAt(index);
                      setSettings(null);
                    }}
                  >
                    Insert before
                  </button>
                )}
              </li>
            );
          })}
        </ol>
        {insertAt !== null && (
          <div className="pipeline-settings" role="form" aria-label="Insert pipeline step">
            <h3>Insert at position {insertAt + 1}</h3>
            <label>
              Processing step
              <select
                aria-label="New pipeline step"
                value={insertion}
                disabled={locked || !catalog}
                onChange={(event) => {
                  setInsertion(event.target.value);
                  setBlock(null);
                }}
              >
                {methods.map((method) => (
                  <option key={method.id} value={method.id} disabled={Boolean(method.unavailable)}>
                    {method.label}
                    {method.unavailable ? ' (unavailable)' : ''}
                  </option>
                ))}
                <option value="register">Registration</option>
                <option value="drift" disabled={w.frame_count < 2}>
                  Drift correction
                </option>
                <option
                  value="reconstruct"
                  disabled={!w.can_reconstruct || Boolean(reconstruction)}
                >
                  Reconstruction workflow (.mrec)
                </option>
                <option value="fit">Fitting workflow (.mfit)</option>
              </select>
            </label>
            {insertion === 'reconstruct' && (
              <p className="control-note">
                Reconstruction uses acquired k-space and is inserted first.
              </p>
            )}
            {['reconstruct', 'fit'].includes(insertion) && (
              <PipelineWorkflowBlock
                workspace={w}
                kind={insertion as 'reconstruct' | 'fit'}
                value={block}
                onChange={setBlock}
                disabled={locked}
              />
            )}
            <div className="button-row">
              <button
                type="button"
                onClick={add}
                className="secondary"
                disabled={
                  locked || !catalog || (['reconstruct', 'fit'].includes(insertion) && !block)
                }
              >
                Insert into draft
              </button>
              <button type="button" className="secondary" onClick={() => setInsertAt(null)}>
                Cancel insertion
              </button>
            </div>
          </div>
        )}
        {selected && controller && (
          <div className="pipeline-settings">
            {['reconstruct', 'fit'].includes(selected.kind) ? (
              <PipelineWorkflowBlock
                key={selected.id}
                workspace={w}
                kind={selected.kind as 'reconstruct' | 'fit'}
                value={selected.options}
                onChange={(options) =>
                  changeStep(selected.id, {
                    options,
                    editor: undefined,
                    version: crypto.randomUUID(),
                  })
                }
                disabled={locked}
              />
            ) : (
              <>
                <DataEditor
                  key={selected.id}
                  label="Pipeline step settings"
                  controller={controller}
                  disabled={locked}
                  applyLabel="Validate draft settings"
                  onApply={async (options) =>
                    changeStep(selected.id, {
                      options,
                      editor: undefined,
                      version: crypto.randomUUID(),
                    })
                  }
                />
              </>
            )}
            <p className="control-note">
              Settings remain in this draft until Apply pipeline. Use action controls for parameter
              descriptions and dedicated forms.
            </p>
            <div className="button-row">
              {selected.kind !== 'fit' && (
                <button
                  className="secondary"
                  disabled={locked || Boolean(issue)}
                  onClick={() => {
                    try {
                      const options = stepOptions(selected);
                      changeStep(selected.id, { options, editor: undefined });
                      setSettings(null);
                      edit({ ...selected, options });
                    } catch (failure) {
                      setError(String(failure));
                    }
                  }}
                >
                  Use action controls
                </button>
              )}
              <button className="secondary" onClick={() => setSettings(null)}>
                Close settings
              </button>
            </div>
          </div>
        )}
        {!['Image', 'Map', 'Registration overlay'].includes(w.view) && (
          <p className="control-note">
            Source view. Select Image to view the current pipeline output.
          </p>
        )}
        {!pipeline.enabled && pipeline.steps.length > 0 && (
          <p className="control-note">Showing original data. Processing steps are retained.</p>
        )}
        {editing && (
          <div className="pipeline-edit-status">
            <span>
              Editing draft step settings in the action controls. Apply the pipeline after updating
              the draft.
            </span>
            <button className="secondary" onClick={cancelEdit}>
              Stop editing step
            </button>
          </div>
        )}
        {pipeline.fits_stale && (
          <p className="pipeline-fit-status" role="status">
            Fit results and maps are retained from an earlier pipeline state. Rerun the fit to
            update them.
          </p>
        )}
      </div>
    </section>
  );
}
