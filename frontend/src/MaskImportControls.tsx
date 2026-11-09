import { useState } from 'react';
import type { AnalysisProps } from './analysisTypes';
import { useFileSystem } from './FileSystemDialog';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';

export function MaskImportControls({ workspace: w, busy, action }: AnalysisProps) {
  const choose = useFileSystem();
  const [error, setError] = useState('');
  const [reading, setReading] = useState(false);
  return (
    <div className="control-layout mask-import">
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
    </div>
  );
}
