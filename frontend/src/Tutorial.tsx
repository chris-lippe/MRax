import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { Options, Tool, Workspace } from './api';
import type { ViewingContext } from './viewing';
import type { Workflow, WorkflowTasks } from './WorkflowNav';
import { WorkbenchIcon } from './WorkbenchIcon';

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
};

export const tutorialSteps = [
  {
    title: 'Your practice study',
    text: 'This synthetic T2 study has eight echoes and three slices. The study browser selects studies and experiments. The tutorial uses this study throughout; your other studies remain open.',
    action: '',
  },
  {
    title: 'Inspect the image',
    text: 'Move the Echo or Slice slider. Arrow keys also adjust a focused slider. Use the mouse wheel to zoom and drag to pan; the View controls expose brightness and contrast.',
    action: 'Open image controls',
  },
  {
    title: 'Apply denoising',
    text: 'Open Denoising under Preprocess, keep the Gaussian denoiser, and apply it. The processing pipeline retains the original image.',
    action: 'Open Denoising',
  },
  {
    title: 'Compare with the original',
    text: 'Choose Compare with original above the image, then move the split divider. The divider can also be focused and adjusted with arrow keys.',
    action: 'Show comparison controls',
  },
  {
    title: 'Draw an ROI',
    text: 'Open Segment, activate Draw rectangle, and drag over part of the phantom. The ROI is shared with viewing, fitting and analysis.',
    action: 'Open Segment',
  },
  {
    title: 'Fit a parameter map',
    text: 'Open Fit and run the full map fit with the T2 model. You can limit fitting to your ROI. The synthetic reference T2 is 80 ms.',
    action: 'Open Fit',
  },
  {
    title: 'Review the result',
    text: 'Open ROI map statistics under Analyze. Select t2 in the Map control above the image and in Statistics map to view the fitted T2 values inside your ROI.',
    action: 'Open map statistics',
  },
  {
    title: 'Save or export',
    text: 'Save workspace keeps the study, pipeline, masks and fit results together. Export data writes selected images, masks or maps. Saving a file is optional in this tutorial; the practice study stays open when you finish.',
    action: 'Show data controls',
  },
] as const;

export function Tutorial({
  session,
  workspace,
  context,
  drawing,
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
  blocked: boolean;
  canResume: boolean;
  onStep: (step: number) => void;
  onOpen: () => void;
  onResume: () => void;
  onEnd: () => void;
}) {
  const card = useRef<HTMLElement>(null);
  const step = tutorialSteps[session.step];
  const owned = workspace?.id === session.studyId;
  const comparison = workspace?.scene.comparison;
  const complete =
    owned &&
    (session.step === 0 ||
      (session.step === 1 &&
        (workspace.frame !== session.initialFrame || workspace.z !== session.initialSlice)) ||
      (session.step === 2 &&
        workspace.pipeline.steps.some((s) => s.family === 'denoise' && s.enabled)) ||
      (session.step === 3 && comparison?.enabled && comparison.position !== 0.5) ||
      (session.step === 4 && workspace.rois.some((r) => r.enabled && r.pixels > 0)) ||
      (session.step === 5 &&
        workspace.maps.some((map) => map.name === 't2') &&
        !workspace.pipeline.fits_stale) ||
      (session.step === 6 &&
        context.section === 'analyze' &&
        context.task === 'statistics' &&
        workspace.active_map === 't2') ||
      session.step === 7);
  let selector = '';
  if (owned) {
    if (session.step === 0) selector = '.study-item[aria-current="true"]';
    if (session.step === 1) selector = '.dimension-controls';
    if (session.step === 2)
      selector =
        context.section === 'preprocess'
          ? '[data-tutorial="denoise-apply"]'
          : '#workflow-preprocess';
    if (session.step === 3)
      selector = comparison?.enabled ? '.comparison-divider' : '[data-tutorial="compare-original"]';
    if (session.step === 4)
      selector =
        context.section === 'rois'
          ? drawing
            ? '[data-testid="roi-surface"]'
            : '[data-tutorial="roi-start"]'
          : '#workflow-rois';
    if (session.step === 5)
      selector = context.section === 'fit' ? '[data-tutorial="fit-apply"]' : '#workflow-fit';
    if (session.step === 6)
      selector =
        context.section === 'analyze'
          ? workspace.active_map === 't2'
            ? '#analysis-controls [aria-label="Statistics"]'
            : '.viewer-controls [aria-label="Map"]'
          : '#workflow-analyze';
    if (session.step === 7) selector = '[aria-label="Workspace files"]';
  }
  const [geometry, setGeometry] = useState<{
    target: { left: number; top: number; width: number; height: number } | null;
    left: number;
    top: number;
    width: number;
  } | null>(null);
  useLayoutEffect(() => {
    const target = selector
      ? [...document.querySelectorAll<HTMLElement>(selector)].find(
          (el) => el.getClientRects().length,
        )
      : undefined;
    target?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    const position = () => {
      const width = Math.min(380, window.innerWidth - 32);
      const height = card.current?.offsetHeight ?? 240;
      const rect = target?.getBoundingClientRect();
      let left = window.innerWidth - width - 16;
      let top = Math.max(76, window.innerHeight - height - 16);
      if (rect) {
        if (window.innerWidth - rect.right > width + 24) left = rect.right + 16;
        else if (rect.left > width + 24) left = rect.left - width - 16;
        else left = Math.min(Math.max(16, rect.left), window.innerWidth - width - 16);
        top =
          rect.bottom + height + 24 < window.innerHeight
            ? rect.bottom + 12
            : rect.top - height - 12;
      }
      const next = {
        target:
          rect && rect.width && rect.height
            ? {
                left: rect.left - 3,
                top: rect.top - 3,
                width: rect.width + 6,
                height: rect.height + 6,
              }
            : null,
        left: Math.max(16, left),
        top: Math.max(16, Math.min(top, window.innerHeight - height - 16)),
        width,
      };
      setGeometry((old) => (JSON.stringify(old) === JSON.stringify(next) ? old : next));
    };
    position();
    const observer = new ResizeObserver(position);
    if (target) observer.observe(target);
    if (card.current) observer.observe(card.current);
    window.addEventListener('resize', position);
    window.addEventListener('scroll', position, true);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', position);
      window.removeEventListener('scroll', position, true);
    };
  }, [selector, session.step]);
  useEffect(() => {
    card.current?.focus({ preventScroll: true });
  }, [session.step]);
  return (
    <>
      {owned && geometry?.target && (
        <div className="tutorial-spotlight" aria-hidden="true" style={geometry.target} />
      )}
      <aside
        ref={card}
        className="tutorial-card"
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
            Step {session.step + 1} of {tutorialSteps.length}
          </span>
          <button className="icon-button" aria-label="End tutorial" onClick={onEnd}>
            <WorkbenchIcon name="close" />
          </button>
        </header>
        <h2>{step.title}</h2>
        <p>{owned ? step.text : 'Tutorial paused. Return to its synthetic study to continue.'}</p>
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
              ? 'Complete the action above to continue.'
              : 'Paused.'}
        </p>
        <div className="tutorial-actions">
          <button
            className="secondary"
            disabled={blocked || session.step === 0}
            onClick={() => onStep(session.step - 1)}
          >
            Back
          </button>
          {session.step < tutorialSteps.length - 1 ? (
            <button
              className="primary"
              disabled={blocked || !complete}
              onClick={() => onStep(session.step + 1)}
            >
              Next
            </button>
          ) : (
            <button className="primary" disabled={blocked || !owned} onClick={onEnd}>
              Finish tutorial
            </button>
          )}
        </div>
      </aside>
    </>
  );
}
