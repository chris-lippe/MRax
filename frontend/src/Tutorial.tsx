import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type { Options, Tool, Workspace } from './api';
import type { ViewingContext } from './viewing';
import type { Workflow, WorkflowTasks } from './WorkflowNav';
import { WorkbenchIcon } from './WorkbenchIcon';
import { findTutorialTarget, measureTutorialTarget, type ScrollCue } from './tutorialGeometry';
import { tutorialSteps, type TutorialStepId } from './tutorialSteps';

export type TutorialSession = {
  studyId: string;
  previousId: string | null;
  previousContext: ViewingContext;
  previousWorkflow: Workflow;
  previousTasks: WorkflowTasks;
  previousTool: Tool;
  previousBrushSize: number;
  previousLabel: number;
  previousDrawOptions: Options;
  previousControlsOpen: boolean;
  initialFrame: number;
  initialSlice: number;
  step: number;
  completed: TutorialStepId[];
};

export function Tutorial({
  session,
  workspace,
  context,
  drawing,
  pipelineChanged,
  blocked,
  canResume,
  onStep,
  onOpen,
  onResume,
  onEnd,
}: {
  session: TutorialSession;
  workspace: Workspace | null;
  context: ViewingContext;
  drawing: boolean;
  pipelineChanged: boolean;
  blocked: boolean;
  canResume: boolean;
  onStep: (step: number, completed?: boolean) => void;
  onOpen: () => void;
  onResume: () => void;
  onEnd: () => void;
}) {
  const card = useRef<HTMLElement>(null);
  const guidanceId = useId();
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => setCollapsed(false), [session.step]);
  const step = tutorialSteps[session.step];
  const owned = workspace?.id === session.studyId;
  const comparison = workspace?.scene.comparison;
  const [editedPipeline, setEditedPipeline] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  useEffect(() => {
    if (owned && step.id === 'pipeline' && pipelineChanged) setEditedPipeline(true);
  }, [owned, step.id, pipelineChanged]);
  const inContext =
    step.context?.section === context.section && step.context?.task === context.task;
  const complete =
    owned &&
    (session.completed.includes(step.id) ||
      step.id === 'study' ||
      (step.id === 'navigate' &&
        (workspace.frame !== session.initialFrame || workspace.z !== session.initialSlice)) ||
      (step.id === 'metadata' && inContext) ||
      (step.id === 'reconstruct' &&
        workspace.has_reconstruction &&
        workspace.pipeline.steps.some((s) => s.kind === 'reconstruct' && s.enabled)) ||
      (step.id === 'denoise' &&
        workspace.pipeline.steps.some((s) => s.family === 'denoise' && s.enabled)) ||
      (step.id === 'pipeline' && editedPipeline && !pipelineChanged) ||
      (step.id === 'compare' && comparison?.enabled && comparison.position !== 0.5) ||
      (step.id === 'register' && inContext) ||
      (step.id === 'roi' && workspace.rois.some((r) => r.enabled && r.pixels > 0)) ||
      (step.id === 'mask' &&
        workspace.rois.some(
          (r) =>
            r.enabled && r.pixels > 0 && (r.metadata.consolidated || r.metadata.kind === 'mask'),
        )) ||
      (step.id === 'fit' &&
        workspace.maps.some((map) => map.name === 't2') &&
        !workspace.pipeline.fits_stale) ||
      (step.id === 'statistics' && inContext && workspace.active_map === 't2') ||
      (step.id === 'signal' && inContext && Boolean(workspace.timecourse)) ||
      (step.id === 'workflows' && inContext) ||
      step.id === 'files');
  let selector = '';
  if (owned) {
    if (step.id === 'study') selector = '.study-item[aria-current="true"]';
    if (step.id === 'navigate') selector = '.dimension-controls:not([hidden])';
    if (step.id === 'metadata')
      selector = inContext ? '[aria-label="Metadata browser"]' : '#workflow-image';
    if (step.id === 'reconstruct')
      selector = inContext ? '[data-tutorial="reconstruct-apply"]' : '#workflow-reconstruct';
    if (step.id === 'denoise')
      selector = inContext ? '[data-tutorial="denoise-apply"]' : '#workflow-denoise';
    if (step.id === 'pipeline')
      selector = pipelineChanged
        ? '[data-tutorial="pipeline-discard"]'
        : '[aria-label="Show processing steps"], [data-tutorial="pipeline-denoise-edit"]';
    if (step.id === 'compare')
      selector = inContext
        ? comparison?.enabled
          ? '.comparison-divider'
          : '[data-tutorial="view-compare-original"]'
        : '#workflow-image';
    if (step.id === 'register')
      selector = inContext
        ? '#analysis-controls [aria-label="Reference image"]'
        : '#workflow-registration';
    if (step.id === 'roi')
      selector = inContext
        ? drawing
          ? '[data-testid="roi-surface"]'
          : '[data-tutorial="roi-start"]'
        : '#workflow-rois';
    if (step.id === 'mask')
      selector = inContext ? '[data-tutorial="roi-pixel-mask"]' : '#workflow-rois';
    if (step.id === 'fit') selector = inContext ? '[data-tutorial="fit-apply"]' : '#workflow-fit';
    if (step.id === 'statistics')
      selector = inContext
        ? workspace.active_map === 't2'
          ? '#analysis-controls [aria-label="Statistics"]'
          : '.viewer-controls [aria-label="Map"]'
        : '#workflow-analyze';
    if (step.id === 'signal')
      selector = inContext ? '[aria-label="Scientific signal viewer"]' : '#workflow-image';
    if (step.id === 'workflows')
      selector = inContext ? '[data-tutorial="processing-workflows"]' : '#workflow-denoise';
    if (step.id === 'files') selector = '[aria-label="Workspace files"]';
  }
  const [geometry, setGeometry] = useState<{
    target: { left: number; top: number; width: number; height: number } | null;
    cue: ScrollCue | null;
    left: number;
    top: number;
    width: number;
  } | null>(null);
  useLayoutEffect(() => {
    let frame = 0;
    let observedTarget: HTMLElement | null = null;
    const schedule = () => {
      if (!frame)
        frame = requestAnimationFrame(() => {
          frame = 0;
          position();
        });
    };
    const observer = new ResizeObserver(schedule);
    if (card.current) observer.observe(card.current);
    const position = () => {
      const modal = [...document.querySelectorAll<HTMLElement>('[role="dialog"]')].some(
        (element) => element.getClientRects().length,
      );
      setDialogOpen(modal);
      const target = findTutorialTarget(selector);
      if (target !== observedTarget) {
        observer.disconnect();
        if (card.current) observer.observe(card.current);
        for (let element = target; element; element = element.parentElement)
          observer.observe(element);
        observedTarget = target;
      }
      const measured = measureTutorialTarget(target);
      const width = Math.min(380, window.innerWidth - 32);
      const height = card.current?.offsetHeight ?? 240;
      const rect = measured.anchor;
      let left = window.innerWidth - width - 16;
      let top = Math.max(76, window.innerHeight - height - 16);
      if (rect) {
        const right = rect.left + rect.width;
        const bottom = rect.top + rect.height;
        if (window.innerWidth - right > width + 24) left = right + 16;
        else if (rect.left > width + 24) left = rect.left - width - 16;
        else left = Math.min(Math.max(16, rect.left), window.innerWidth - width - 16);
        top = bottom + height + 24 < window.innerHeight ? bottom + 12 : rect.top - height - 12;
      }
      const next = {
        target: measured.target,
        cue: measured.cue,
        left: Math.max(16, left),
        top: Math.max(16, Math.min(top, window.innerHeight - height - 16)),
        width,
      };
      setGeometry((old) => (JSON.stringify(old) === JSON.stringify(next) ? old : next));
    };
    position();
    const mutations = new MutationObserver((records) => {
      if (
        records.some(
          (record) =>
            !(
              record.target instanceof Element ? record.target : record.target.parentElement
            )?.closest('[data-tutorial-overlay]'),
        )
      )
        schedule();
    });
    mutations.observe(document.body, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['open', 'hidden', 'class', 'style'],
    });
    window.addEventListener('resize', schedule);
    window.addEventListener('scroll', schedule, true);
    return () => {
      observer.disconnect();
      mutations.disconnect();
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', schedule);
      window.removeEventListener('scroll', schedule, true);
    };
  }, [selector, session.step, dialogOpen]);
  useEffect(() => {
    if (!dialogOpen) card.current?.focus({ preventScroll: true });
  }, [session.step, dialogOpen]);
  if (dialogOpen) return null;
  return (
    <>
      {owned && geometry?.target && (
        <div
          className="tutorial-spotlight"
          data-tutorial-overlay
          aria-hidden="true"
          style={geometry.target}
        />
      )}
      {owned && geometry?.cue && (
        <button
          className="tutorial-scroll-cue"
          data-tutorial-overlay
          data-direction={geometry.cue.direction}
          style={{ left: geometry.cue.left, top: geometry.cue.top }}
          aria-label={`Scroll ${geometry.cue.direction} to the tutorial control`}
          title={`Scroll ${geometry.cue.direction} to the tutorial control`}
          onClick={() =>
            findTutorialTarget(selector)?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
          }
        >
          <WorkbenchIcon name="up" />
        </button>
      )}
      <aside
        ref={card}
        className="tutorial-card"
        data-tutorial-overlay
        role="region"
        aria-label="Workflow tutorial"
        tabIndex={-1}
        style={{
          left: geometry?.left ?? 16,
          top: geometry?.top ?? 76,
          width: geometry?.width ?? 380,
        }}
      >
        <header>
          <span>
            {collapsed ? step.title : `Step ${session.step + 1} of ${tutorialSteps.length}`}
          </span>
          <div className="tutorial-window-actions">
            <button
              className="icon-button tutorial-collapse"
              aria-label={collapsed ? 'Expand tutorial guidance' : 'Collapse tutorial guidance'}
              aria-expanded={!collapsed}
              aria-controls={guidanceId}
              onClick={() => setCollapsed(!collapsed)}
            >
              <WorkbenchIcon name="chevron" />
            </button>
            <button className="icon-button" aria-label="End tutorial" onClick={onEnd}>
              <WorkbenchIcon name="close" />
            </button>
          </div>
        </header>
        <div id={guidanceId} hidden={collapsed}>
          <h2>{step.title}</h2>
          <p>{owned ? step.text : 'Tutorial paused. Return to its synthetic study to continue.'}</p>
          {owned && geometry?.cue && (
            <p className="tutorial-scroll-hint" role="status">
              Scroll {geometry.cue.direction} in the indicated panel to reach the control. You can
              also use the arrow button.
            </p>
          )}
          {!owned && !canResume && (
            <p>The practice study has been closed. Start another tutorial from Settings.</p>
          )}
          {!owned && canResume && (
            <button className="secondary" disabled={blocked} onClick={onResume}>
              Return to tutorial study
            </button>
          )}
          {owned && step.action && (
            <button className="secondary" disabled={blocked} onClick={onOpen}>
              {step.action}
            </button>
          )}
          <p className="tutorial-progress" role="status" aria-live="polite">
            {owned && complete
              ? 'Step complete.'
              : owned
                ? step.optional
                  ? 'Optional. Open the controls or skip this step.'
                  : 'Complete the action above, or skip this step.'
                : 'Paused.'}
          </p>
          <div className="tutorial-actions">
            <button
              className="secondary"
              disabled={blocked || !owned || session.step === 0}
              onClick={() => onStep(session.step - 1)}
            >
              Back
            </button>
            {session.step < tutorialSteps.length - 1 ? (
              <>
                <button
                  className="secondary"
                  disabled={blocked || !owned}
                  onClick={() => onStep(session.step + 1)}
                >
                  Skip step
                </button>
                <button
                  className="primary"
                  disabled={blocked || !complete}
                  onClick={() => onStep(session.step + 1, true)}
                >
                  Next
                </button>
              </>
            ) : (
              <button className="primary" disabled={blocked || !owned} onClick={onEnd}>
                Finish tutorial
              </button>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
