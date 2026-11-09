import { useEffect, useRef, type ReactNode } from 'react';
import type { AnalysisProps, DrawingProps } from './analysisTypes';
import { RoiControls } from './RoiControls';
import { FitControls } from './FitControls';
import { RegistrationControls } from './RegistrationControls';
import { ProcessControls } from './ProcessControls';
import { ReconstructionControls } from './ReconstructionControls';
import { RoiStatistics } from './RoiStatistics';
import { JobStatus, isFileJob } from './JobStatus';
import { MaskImportControls } from './MaskImportControls';
import { TimecourseControls } from './TimecourseControls';
import type { TimecourseResult } from './api';
import { workflows, type Workflow, type WorkflowTask } from './WorkflowNav';
import { WorkbenchIcon } from './WorkbenchIcon';

export function AnalysisPanel(
  props: AnalysisProps &
    DrawingProps & {
      section: Workflow;
      task: WorkflowTask;
      displayControls: ReactNode;
      signalSourceControls: ReactNode;
      signalDisplayControls: ReactNode;
      signalComparison: boolean;
      jobStudyLabel?: string;
      onTimecourseResult: (result: TimecourseResult | null) => void;
    },
) {
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => {
    body.current?.scrollTo({ top: 0 });
  }, [props.section, props.task]);
  const running = props.job && ['queued', 'running', 'cancelling'].includes(props.job.status);
  const blocked = { ...props, busy: props.busy || Boolean(running) };
  const section = props.section;
  const task = props.task;
  const processingSection =
    task === 'denoising'
      ? 'denoise'
      : task === 'series-processing'
        ? 'series'
        : task === 'normalization'
          ? 'normalize'
          : task === 'derived-maps'
            ? 'analyze'
            : null;
  const workflow = workflows.find((item) => item.id === section)!;
  return (
    <aside
      id="analysis-controls"
      className="tools-panel"
      aria-label="Analysis controls"
      hidden={task === 'metadata'}
    >
      <header className="inspector-heading">
        <h2 id="workflow-task-heading">
          <WorkbenchIcon name={workflow.tasks.find((item) => item.id === task)!.icon} />
          {workflow.tasks.find((item) => item.id === task)!.label}
        </h2>
        {props.workspace?.loaded && (
          <span className="inspector-context">{props.workspace.experiment}</span>
        )}
        {['preprocess', 'registration', 'analyze'].includes(section) && (
          <button
            className="secondary small workflow-library-trigger"
            disabled={blocked.busy}
            onClick={() => props.openWorkflows?.('processing')}
          >
            <WorkbenchIcon name="pipeline" />
            Processing workflows
          </button>
        )}
      </header>
      <div
        ref={body}
        className="panel-body"
        id="workflow-panel"
        role="tabpanel"
        aria-labelledby={`workflow-${section} workflow-task-heading`}
        tabIndex={0}
      >
        <div hidden={section !== 'image' || task === 'signal'}>{props.displayControls}</div>
        <div hidden={task !== 'signal'}>{props.signalSourceControls}</div>
        <div hidden={task !== 'timecourse' && !props.signalComparison}>
          <TimecourseControls
            workspace={props.workspace}
            busy={props.busy}
            onResult={props.onTimecourseResult}
            viewer={task === 'signal'}
          />
        </div>
        <div hidden={task !== 'signal'}>{props.signalDisplayControls}</div>
        <div hidden={task !== 'statistics'}>
          <RoiStatistics workspace={props.workspace} active={task === 'statistics'} />
        </div>
        <div hidden={!processingSection}>
          <ProcessControls
            {...blocked}
            section={processingSection ?? 'denoise'}
            active={Boolean(processingSection)}
          />
        </div>
        <div hidden={!['roi-edit', 'label-edit', 'roi-volume', 'roi-combine'].includes(task)}>
          <RoiControls {...blocked} task={task} />
        </div>
        <div hidden={task !== 'reconstruction'}>
          <ReconstructionControls {...blocked} />
        </div>
        <div hidden={task !== 'mask-import'}>
          <MaskImportControls {...blocked} />
        </div>
        <div hidden={section !== 'fit'}>
          <FitControls {...blocked} />
        </div>
        <div hidden={task !== 'alignment' && task !== 'drift'}>
          <RegistrationControls {...blocked} task={props.task} />
        </div>
      </div>
      {!isFileJob(props.job) && (
        <JobStatus
          workspace={props.workspace}
          job={props.job}
          changeView={props.changeView}
          studyLabel={props.jobStudyLabel}
        />
      )}
    </aside>
  );
}
