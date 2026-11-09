import { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import type { AnalysisProps } from './analysisTypes';
import type { Job, TimecourseResult } from './api';
import { ExportControls } from './ExportControls';
import { defaultExportDraft, type ExportDraft } from './exportDraft';
import { JobStatus, isFileJob } from './JobStatus';

export function ExportDialog({
  open,
  onClose,
  onAfterClose,
  timecourse,
  signalExport,
  error,
  jobStudyLabel,
  onJobChange,
  ...props
}: AnalysisProps & {
  open: boolean;
  onClose: () => void;
  onAfterClose: () => void;
  timecourse: TimecourseResult | null;
  signalExport?: { csv: () => string; svg: () => string } | null;
  error: string;
  jobStudyLabel?: string;
  onJobChange?: (job: Job) => void;
}) {
  const [drafts, setDrafts] = useState<Record<string, ExportDraft>>({});
  const context = props.workspace?.id ?? '';
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content
          className="export-dialog"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            onAfterClose();
          }}
        >
          <Dialog.Title>Export data</Dialog.Title>
          <Dialog.Description>
            {props.workspace?.label}
            {props.workspace?.experiment ? ` / ${props.workspace.experiment}` : ''}
          </Dialog.Description>
          <Dialog.Close className="drawer-close" aria-label="Close export dialog">
            ×
          </Dialog.Close>
          <section aria-label="Export controls">
            <ExportControls
              {...props}
              onJobChange={onJobChange}
              timecourse={timecourse}
              signalExport={signalExport}
              exportDraft={drafts[context] ?? defaultExportDraft}
              onExportDraftChange={(draft) =>
                setDrafts((previous) => ({ ...previous, [context]: draft }))
              }
            />
            {isFileJob(props.job) && <JobStatus {...props} studyLabel={jobStudyLabel} />}
            {error && <p role="alert">{error}</p>}
          </section>
          <Dialog.Close asChild>
            <button className="secondary full">Close export</button>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
