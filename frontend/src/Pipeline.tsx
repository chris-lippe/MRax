import { Fragment, useEffect, useId, useRef, useState } from 'react';
import {
  api,
  type ComparisonSettings,
  type Job,
  type Options,
  type PipelineStep,
  type Workspace,
} from './api';
import { PipelineSelector } from './PipelineSelector';
import { PipelineComparison } from './PipelineComparison';
import { WorkbenchIcon } from './WorkbenchIcon';
import type { WorkflowDocument } from './reusableWorkflows';
import {
  activePipelineJob,
  copiedStep,
  draftChanged,
  draftProblem,
  droppedSteps,
  insertedStep,
  movedSteps,
  newPipelineDraft,
  processDraftStep,
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
  compare: (settings?: ComparisonSettings) => void;
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

function pointerBoundary(element: HTMLOListElement, x: number) {
  const cards = Array.from(element.querySelectorAll<HTMLElement>('.pipeline-step'));
  const next = cards.findIndex((card) => {
    const box = card.getBoundingClientRect();
    return x < box.left + box.width / 2;
  });
  return Math.max(Number(element.dataset.minimum ?? 0), next < 0 ? cards.length : next);
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
  const [insertAt, setInsertAt] = useState<number | null>(null);
  const [insertion, setInsertion] = useState('gaussian');
  const [dragging, setDragging] = useState<string | null>(null);
  const [dropAt, setDropAt] = useState<number | null>(null);
  const list = useRef<HTMLOListElement>(null);
  const dragged = useRef<string | null>(null);
  const pointerX = useRef<number | null>(null);
  const dragOrigin = useRef<{ x: number; y: number } | null>(null);
  const dropPosition = useRef<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [loaded, setLoaded] = useState<{ key: string; catalog: PipelineCatalog } | null>(null);
  const contentId = useId();
  const context = JSON.stringify([w.id, w.experiment, w.pipeline.id]);
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
    setInsertAt(null);
    setError('');
    setNotice('');
  }, [context]);
  useEffect(() => {
    if (!dragging) return;
    let frame = 0;
    function scroll() {
      const element = list.current;
      if (element && pointerX.current !== null) {
        const box = element.getBoundingClientRect(),
          x = pointerX.current;
        const speed =
          x < box.left + 40
            ? -Math.min(14, (box.left + 40 - x) / 3)
            : x > box.right - 40
              ? Math.min(14, (x - box.right + 40) / 3)
              : 0;
        const before = element.scrollLeft;
        element.scrollLeft += speed;
        if (element.scrollLeft !== before) indicateDrop(pointerBoundary(element, x));
      }
      frame = requestAnimationFrame(scroll);
    }
    frame = requestAnimationFrame(scroll);
    return () => cancelAnimationFrame(frame);
  }, [dragging]);
  useEffect(() => {
    if (!dragging) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        finishDrag();
      }
    };
    window.addEventListener('keydown', cancel);
    return () => window.removeEventListener('keydown', cancel);
  }, [dragging]);
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
    ownRun.pipeline?.pipeline_id === w.pipeline.id &&
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
  const reconstructionCount = steps.filter((step) => step.kind === 'reconstruct').length;
  const sourceKind = reconstruction?.enabled ? 'kspace' : 'image';
  const source = pipeline.sources?.[sourceKind];
  const methods = catalog?.methods.filter((method) => method.id !== 'dti') ?? [];
  function label(step: DraftStep) {
    try {
      const options = step.options;
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
  function openInsert(position: number) {
    setInsertion('gaussian');
    setInsertAt(position);
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
    cancelEdit();
    setInsertAt(null);
    setError('');
    setNotice('Pipeline edits discarded.');
  }
  async function add() {
    if (insertAt === null || !catalog || locked) return;
    setSubmitting(true);
    try {
      let step: DraftStep;
      if (['reconstruct', 'fit'].includes(insertion)) {
        if (current.steps.some((row) => row.kind === insertion))
          throw Error('This operation supports one block per experiment.');
        const choices = await api<{ defaults: Record<string, Options> }>(
          '/api/workspaces/' + w.id + '/pipeline-blocks',
        );
        step = insertedStep(
          insertion,
          choices.defaults[insertion],
          insertion === 'fit' ? 'Fitting' : 'Reconstruction',
        );
      } else if (insertion === 'register') {
        step = insertedStep(
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
        );
      } else if (insertion === 'drift') {
        step = insertedStep(
          'drift',
          { settings: {}, roi_name: w.selected_roi },
          'Drift correction',
        );
      } else step = processDraftStep(catalog, insertion);
      const next = [...current.steps];
      next.splice(insertion === 'reconstruct' ? 0 : insertAt, 0, step);
      const problem = draftProblem({ ...current, steps: next }, w);
      if (problem) throw Error(problem);
      update({ ...current, steps: next });
      setInsertAt(null);
      edit(step);
      setNotice('Step inserted. Edit its settings in the action controls.');
    } catch (failure) {
      setError(String(failure));
    } finally {
      setSubmitting(false);
    }
  }
  async function confirm() {
    if (!draft || !changed || blocked || locked || stale || issue) return;
    setSubmitting(true);
    setError('');
    try {
      await apply(draft);
      setInsertAt(null);
      setNotice('');
    } catch (failure) {
      setError(String(failure));
    } finally {
      setSubmitting(false);
    }
  }
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

  function finishDrag() {
    dragged.current = null;
    pointerX.current = null;
    setDragging(null);
    dragOrigin.current = null;
    dropPosition.current = null;
    setDropAt(null);
  }
  function indicateDrop(position: number) {
    dropPosition.current = position;
    setDropAt(position);
  }
  function dropSlot(position: number) {
    const available = position >= reconstructionCount && !locked;
    return (
      <li
        className="pipeline-drop-slot"
        role="presentation"
        data-position={position}
        data-drop-active={dragging && dropAt === position ? 'true' : undefined}
      >
        <button
          className="pipeline-slot-button"
          aria-label={'Insert at position ' + (position + 1)}
          disabled={!available || steps.length >= 64}
          onClick={() => openInsert(position)}
        >
          <WorkbenchIcon name="add" />
        </button>
        {dragging && dropAt === position && (
          <span className="pipeline-drop-caption">Drop here</span>
        )}
      </li>
    );
  }
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
        <PipelineSelector key={context} workspace={w} disabled={locked || blocked} run={run} />
        {!expanded && (
          <span className="pipeline-output-state">
            {activePipelineJob(matchingRun) ? (
              <>
                <i className="pipeline-spinner" aria-hidden="true" /> Running
              </>
            ) : changed ? (
              'Draft'
            ) : pipeline.steps.some((step) => step.pending) ? (
              'Settings pending'
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
      <PipelineComparison workspace={w} disabled={locked || blocked} compare={compare} />
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
                if (reconstruction)
                  update({
                    ...current,
                    steps: current.steps.map((step) =>
                      step.kind === 'reconstruct' ? { ...step, enabled } : step,
                    ),
                  });
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
                {reconstruction ? label(reconstruction) : 'Reconstruction'}
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
              <button
                className="secondary"
                disabled={blocked || submitting}
                onClick={() => compare()}
              >
                <WorkbenchIcon name="split" />
                Compare original
              </button>
            )}
            <button
              className="secondary"
              disabled={locked || steps.length >= 64}
              onClick={() => {
                openInsert(steps.length);
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
                cancelEdit();
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
        <ol
          className="pipeline-steps"
          ref={list}
          data-minimum={reconstructionCount}
          data-dragging={Boolean(dragging)}
          aria-busy={Boolean(activePipelineJob(matchingRun))}
        >
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
              <Fragment key={step.id}>
                {dropSlot(index)}
                <li
                  className={'pipeline-step' + (!step.enabled ? ' disabled' : '')}
                  aria-label={label(step)}
                  data-step-id={step.id}
                  data-state={state}
                  data-editing={editing === step.id}
                  data-dragging={dragging === step.id}
                  data-movable={!locked && !blocked && step.kind !== 'reconstruct'}
                  onPointerDown={(event) => {
                    if (
                      locked ||
                      blocked ||
                      step.kind === 'reconstruct' ||
                      event.button !== 0 ||
                      (event.target as Element).closest('button, input, select, textarea, a, label')
                    )
                      return;
                    event.preventDefault();
                    event.currentTarget.setPointerCapture(event.pointerId);
                    dragged.current = step.id;
                    dragOrigin.current = { x: event.clientX, y: event.clientY };
                  }}
                  onPointerMove={(event) => {
                    const origin = dragOrigin.current;
                    if (dragged.current !== step.id || !origin || !list.current) return;
                    if (Math.hypot(event.clientX - origin.x, event.clientY - origin.y) < 4) return;
                    const box = list.current.getBoundingClientRect();
                    if (event.clientY < box.top || event.clientY > box.bottom) {
                      pointerX.current = null;
                      dropPosition.current = null;
                      setDropAt(null);
                      return;
                    }
                    pointerX.current = event.clientX;
                    setDragging(step.id);
                    indicateDrop(pointerBoundary(list.current, event.clientX));
                  }}
                  onPointerUp={(event) => {
                    if (
                      dragged.current === step.id &&
                      dropPosition.current !== null &&
                      list.current &&
                      event.clientX >= list.current.getBoundingClientRect().left &&
                      event.clientX <= list.current.getBoundingClientRect().right &&
                      !locked &&
                      !blocked
                    ) {
                      update({
                        ...current,
                        steps: droppedSteps(current.steps, step.id, dropPosition.current),
                      });
                      setNotice('Step moved. Changes are unapplied.');
                    }
                    finishDrag();
                    if (event.currentTarget.hasPointerCapture(event.pointerId))
                      event.currentTarget.releasePointerCapture(event.pointerId);
                  }}
                  onPointerCancel={finishDrag}
                  onLostPointerCapture={finishDrag}
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
                  {step.pending && (
                    <span className="pipeline-step-status">Configured · awaiting run</span>
                  )}
                  {progress && !step.pending && (
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
                      {step.result_maps?.length ?? 0} maps stored alongside image data
                    </small>
                  )}
                  <div className="pipeline-step-actions">
                    <button
                      className="secondary"
                      data-tutorial={
                        step.family === 'denoise' ? 'pipeline-denoise-edit' : undefined
                      }
                      disabled={locked}
                      onClick={() => {
                        setInsertAt(null);
                        edit(step);
                      }}
                    >
                      Edit
                    </button>
                    <button
                      className="secondary"
                      aria-label={'Move ' + label(step) + ' earlier'}
                      title="Move earlier"
                      disabled={
                        locked || step.kind === 'reconstruct' || index <= reconstructionCount
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
                        ['fit', 'reconstruct'].includes(step.kind)
                          ? 'One block per experiment'
                          : 'Copy step'
                      }
                      disabled={
                        locked || ['fit', 'reconstruct'].includes(step.kind) || steps.length >= 64
                      }
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
                      disabled={locked}
                      onClick={() => {
                        if (editing === step.id) cancelEdit();
                        update({
                          ...current,
                          steps: current.steps.filter((row) => row.id !== step.id),
                        });
                      }}
                    >
                      Remove
                    </button>
                  </div>
                  <div className="pipeline-step-actions pipeline-insert-actions">
                    <button
                      className="secondary"
                      aria-label={'Insert before ' + label(step)}
                      disabled={locked || index < reconstructionCount || steps.length >= 64}
                      onClick={() => {
                        openInsert(index);
                      }}
                    >
                      <WorkbenchIcon name="add" />
                      Insert before
                    </button>
                    <button
                      className="secondary"
                      aria-label={'Insert after ' + label(step)}
                      disabled={locked || steps.length >= 64}
                      onClick={() => {
                        openInsert(index + 1);
                      }}
                    >
                      <WorkbenchIcon name="add" />
                      Insert after
                    </button>
                  </div>
                </li>
              </Fragment>
            );
          })}
          {dropSlot(steps.length)}
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
                  disabled={!w.can_reconstruct || Boolean(reconstruction) || insertAt !== 0}
                >
                  Reconstruction
                </option>
                <option value="fit" disabled={steps.some((step) => step.kind === 'fit')}>
                  Fitting
                </option>
              </select>
            </label>
            {insertion === 'reconstruct' && (
              <p className="control-note">
                Reconstruction uses acquired k-space and is inserted first.
              </p>
            )}
            <div className="button-row">
              <button
                type="button"
                onClick={() => void add()}
                className="secondary"
                disabled={locked || !catalog}
              >
                Insert into draft
              </button>
              <button type="button" className="secondary" onClick={() => setInsertAt(null)}>
                Cancel insertion
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
            <span>Editing settings in the action controls.</span>
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
