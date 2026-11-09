import { useEffect, useState } from 'react';
import type { AnalysisProps } from './analysisTypes';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { ReconstructionDisplayOptions } from './ReconstructionDisplayOptions';

export function ReconstructionControls({
  workspace: w,
  busy,
  run,
  changeView,
  editingStep,
  openWorkflows,
}: Pick<
  AnalysisProps,
  'workspace' | 'busy' | 'run' | 'changeView' | 'editingStep' | 'openWorkflows'
>) {
  const editing = editingStep?.kind === 'reconstruct';
  const editComponent = editing ? String(editingStep.options.component ?? 'magnitude') : '';
  const [component, setComponent] = useState('magnitude');
  useEffect(() => {
    if (editComponent) setComponent(editComponent);
  }, [editComponent]);
  return (
    <div className="control-layout">
      <button
        className="secondary control-wide workflow-library-trigger"
        disabled={busy}
        onClick={() => openWorkflows?.('reconstruction', { component })}
      >
        <WorkbenchIcon name="pipeline" />
        Reconstruction workflows
      </button>
      <InspectorSection
        title="Raw data"
        icon="reconstruct"
        help="Raw samples can be loaded separately for inspection, or loaded automatically when reconstructing."
      >
        <div className="control-status">
          <WorkbenchIcon name="reconstruct" />
          <span>k-space</span>
          <strong>
            {w?.has_kspace ? 'Loaded' : w?.can_load_raw ? 'Available on disk' : 'Unavailable'}
          </strong>
        </div>
        <button
          className="secondary full"
          disabled={busy || !w?.loaded || !w.can_load_raw || w.has_kspace}
          onClick={() => void run('reconstruct', { view: 'k_space' })}
        >
          <WorkbenchIcon name="open" />
          Load raw k-space
        </button>
      </InspectorSection>
      <InspectorSection
        title="Output"
        icon="image"
        help="Reconstruction becomes the first pipeline step. Existing image processing is then reapplied to its output."
      >
        <label>
          Reconstruction data
          <select
            aria-label="Reconstruction data"
            value={component}
            onChange={(event) => setComponent(event.target.value)}
          >
            <option value="magnitude">Magnitude</option>
            <option value="phase">Phase</option>
          </select>
        </label>
        <button
          className="primary full"
          disabled={
            busy || !w?.loaded || (!w.can_reconstruct && !w.can_load_raw && !w.has_reconstruction)
          }
          onClick={() =>
            void run('reconstruct', {
              view: 'mrax_reconstruction',
              component,
              step_id: editing ? editingStep?.id : undefined,
            })
          }
        >
          <WorkbenchIcon name="reconstruct" />
          {editing ? 'Stage reconstruction settings' : 'Reconstruct images'}
        </button>
      </InspectorSection>
      <ReconstructionDisplayOptions workspace={w} busy={busy} changeView={changeView} />
    </div>
  );
}
