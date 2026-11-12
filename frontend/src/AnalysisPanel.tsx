import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { AnalysisProps, DrawingProps } from './analysisTypes';
import { RoiControls } from './RoiControls';
import { FitControls } from './FitControls';
import { RegistrationControls } from './RegistrationControls';
import { ProcessControls } from './ProcessControls';
import { ReconstructionControls } from './ReconstructionControls';
import { RoiStatistics } from './RoiStatistics';
import { JobStatus, isFileJob } from './JobStatus';
import { RoiFileControls } from './RoiFileControls';
import { TimecourseControls } from './TimecourseControls';
import type { Job, TimecourseResult } from './api';
import { workflows, type Workflow, type WorkflowTask } from './WorkflowNav';
import { WorkbenchIcon } from './WorkbenchIcon';
import { InspectorGroup } from './InspectorSection';
import { useControlLayout } from './useControlLayout';

export function AnalysisPanel(
  props: AnalysisProps &
    DrawingProps & {
      section: Workflow;
      task: WorkflowTask;
      displayControls: ReactNode;
      signalSourceControls: ReactNode;
      signalDisplayControls: ReactNode;
      signalComparison: boolean;
      signalPrimary: boolean;
      jobStudyLabel?: string;
      onJobChange: (job: Job) => void;
      onTimecourseResult: (result: TimecourseResult | null) => void;
    },
) {
  const body = useRef<HTMLDivElement>(null);
  useControlLayout(body);
  const [analysisProcessingOpen, setAnalysisProcessingOpen] = useState(false);
  useEffect(() => {
    body.current?.scrollTo({ top: 0 });
  }, [props.section, props.task]);
  const running = props.job && ['queued', 'running', 'cancelling'].includes(props.job.status);
  const blocked = { ...props, busy: props.busy || Boolean(running) };
  const section = props.section;
  const task = props.task;
  const signalTools =
    task === 'signal' || (section === 'image' && task === 'display' && props.signalPrimary);
  const processingSection =
    task === 'denoising'
      ? 'denoise'
      : task === 'timecourse'
        ? 'series'
        : task === 'normalization'
          ? 'normalize'
          : task === 'statistics'
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
          <WorkbenchIcon
            name={
              signalTools ? 'timecourse' : workflow.tasks.find((item) => item.id === task)!.icon
            }
          />
          {signalTools ? 'Signal display' : workflow.tasks.find((item) => item.id === task)!.label}
        </h2>
        {props.workspace?.loaded && (
          <span className="inspector-context">{props.workspace.experiment}</span>
        )}
        {['preprocess', 'registration', 'analyze'].includes(section) && (
          <button
            className="secondary small workflow-library-trigger"
            data-tutorial="processing-workflows"
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
        <div hidden={section !== 'image' || signalTools}>{props.displayControls}</div>
        <div hidden={!signalTools}>{props.signalSourceControls}</div>
        <div hidden={task !== 'timecourse' && !props.signalComparison}>
          <TimecourseControls
            workspace={props.workspace}
            busy={props.busy}
            onResult={props.onTimecourseResult}
            viewer={signalTools}
          />
        </div>
        <div hidden={!signalTools}>{props.signalDisplayControls}</div>
        <div hidden={task !== 'statistics'}>
          <RoiStatistics workspace={props.workspace} active={task === 'statistics'} />
        </div>
        <div hidden={!processingSection}>
          {section === 'analyze' ? (
            <InspectorGroup
              title={task === 'timecourse' ? 'Time-series corrections' : 'Derived maps'}
              icon={task === 'timecourse' ? 'series' : 'maps'}
              revealKey={props.editingStep?.id}
              onOpenChange={setAnalysisProcessingOpen}
            >
              <ProcessControls
                {...blocked}
                section={processingSection ?? 'series'}
                active={analysisProcessingOpen && Boolean(processingSection)}
              />
            </InspectorGroup>
          ) : (
            <ProcessControls
              {...blocked}
              section={processingSection ?? 'denoise'}
              active={Boolean(processingSection)}
            />
          )}
        </div>
        <div hidden={task !== 'roi-edit'}>
          <RoiControls {...blocked} />
        </div>
        <div hidden={task !== 'roi-files'}>
          <RoiFileControls
            {...blocked}
            active={task === 'roi-files'}
            onJobChange={props.onJobChange}
          />
        </div>
        <div hidden={task !== 'reconstruction'}>
          <ReconstructionControls {...blocked} active={task === 'reconstruction'} />
        </div>
        <div hidden={section !== 'fit'}>
          <FitControls {...blocked} />
        </div>
        <div hidden={task !== 'alignment' && task !== 'drift'}>
          <RegistrationControls {...blocked} task={props.task} />
        </div>
      </div>
      {(!isFileJob(props.job) || task === 'roi-files') && (
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
