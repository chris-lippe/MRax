import { useState } from 'react';
import type { AnalysisProps } from './analysisTypes';
import { useFileSystem } from './FileSystemDialog';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import type { Job } from './api';
import { ExportControls } from './ExportControls';
import { type ExportDraft } from './exportDraft';

export function RoiFileControls({
  active,
  onJobChange,
  ...props
}: AnalysisProps & { active: boolean; onJobChange: (job: Job) => void }) {
  const { workspace: w, busy, action } = props;
  const choose = useFileSystem();
  const [error, setError] = useState('');
  const [reading, setReading] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, ExportDraft>>({});
  const context = w?.id ?? '';
  const selectedRoi = w?.selected_roi ?? w?.rois[0]?.name;
  return (
    <div className="control-layout">
      <InspectorSection
        title="Import masks and ROI sets"
        icon="import"
        help="Open an MRax ROI set, NIfTI label mask or NumPy mask on the displayed experiment grid."
      >
        <button
          className="primary full"
          disabled={!w?.loaded || busy || reading}
          onClick={() => {
            setReading(true);
            setError('');
            void (async () => {
              try {
                const source = await choose({
                  title: 'Open ROI set or label mask',
                  mode: 'open',
                  extensions: ['.npy', '.npz', '.nii', '.nii.gz', '.json'],
                });
                if (!source) return;
                await action('roi.load', {
                  root: source.root,
                  path: source.path,
                  mask:
                    !source.directory &&
                    !source.name.endsWith('.json') &&
                    !source.name.endsWith('.npz'),
                  name: source.name.replace(/\.(nii\.gz|npy|npz|nii)$/i, ''),
                });
              } catch (failure) {
                setError(failure instanceof Error ? failure.message : 'Could not open ROI data');
              } finally {
                setReading(false);
              }
            })();
          }}
        >
          <WorkbenchIcon name="open" />
          Open ROI set or mask…
        </button>
        {error && <p role="alert">{error}</p>}
      </InspectorSection>
      <InspectorSection
        title="Export ROIs and masks"
        icon="export"
        help="Save all ROIs as an MRax ROI set, or export an individual mask as NIfTI, NumPy or another supported format."
      >
        <ExportControls
          {...props}
          active={active}
          scope="rois"
          onJobChange={onJobChange}
          exportDraft={
            drafts[context] ?? {
              selected: selectedRoi ? 'roi/' + selectedRoi : '',
              format: 'roi_set',
            }
          }
          onExportDraftChange={(draft) =>
            setDrafts((previous) => ({ ...previous, [context]: draft }))
          }
        />
      </InspectorSection>
    </div>
  );
}
