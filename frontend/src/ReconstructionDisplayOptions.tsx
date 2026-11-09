import type { Options, Workspace } from './api';
import { CheckField } from './Controls';
import { InspectorSection } from './InspectorSection';

export function ReconstructionDisplayOptions({
  workspace,
  busy,
  changeView,
}: {
  workspace: Workspace | null;
  busy: boolean;
  changeView: (command: Options) => Promise<void>;
}) {
  if (
    !workspace ||
    workspace.scene.comparison.enabled ||
    !['MRax reconstruction', 'K-space'].includes(workspace.view)
  )
    return null;
  return (
    <InspectorSection title="Display options" icon="image" className="control-wide">
      <fieldset className="field-grid" disabled={busy}>
        {workspace.view === 'MRax reconstruction' ? (
          <>
            <label>
              Reconstruction component
              <select
                aria-label="Reconstruction component"
                value={workspace.reconstruction_component}
                onChange={(event) =>
                  void changeView({ reconstruction_component: event.target.value })
                }
              >
                <option value="magnitude">Magnitude</option>
                <option value="phase">Phase</option>
              </select>
            </label>
            <label>
              Coil
              <select
                aria-label="Coil"
                value={workspace.reconstruction_coil ?? ''}
                onChange={(event) =>
                  void changeView({
                    coil: event.target.value === '' ? null : Number(event.target.value),
                  })
                }
              >
                <option value="">Combined</option>
                {Array.from({ length: workspace.coil_count }, (_, index) => (
                  <option key={index} value={index}>
                    Coil {index + 1}
                  </option>
                ))}
              </select>
            </label>
            <CheckField
              label="Unwrap phase"
              checked={workspace.unwrap_phase}
              disabled={busy}
              onChange={(value) => changeView({ unwrap_phase: value })}
            />
          </>
        ) : (
          <>
            <label>
              K-space component
              <select
                aria-label="K-space component"
                value={workspace.kspace_mode}
                onChange={(event) => void changeView({ kspace_mode: event.target.value })}
              >
                {['log_magnitude', 'magnitude', 'phase', 'real', 'imag'].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
            <label>
              Centering
              <select
                aria-label="Centering"
                value={workspace.kspace_centering}
                onChange={(event) => void changeView({ kspace_centering: event.target.value })}
              >
                {['auto', 'none', 'fftshift'].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
          </>
        )}
      </fieldset>
    </InspectorSection>
  );
}
