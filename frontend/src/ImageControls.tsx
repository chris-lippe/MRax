import type { Options, Workspace } from './api';
import { JsonEditor } from './Controls';
import { SceneControls } from './SceneControls';
import { ViewingModes } from './ViewingModes';
import type { WorkflowTask } from './WorkflowNav';
import { InspectorSection } from './InspectorSection';
import { viewLabel } from './viewing';

export function ImageControls({
  workspace,
  task,
  busy,
  changeView,
  comparisonControlsRef,
  action,
  previewOpacity,
  previewRoiOpacity,
}: {
  workspace: Workspace | null;
  task: WorkflowTask;
  busy: boolean;
  changeView: (command: Options) => Promise<void>;
  comparisonControlsRef: (element: HTMLDivElement | null) => void;
  action: (name: string, options?: Options) => Promise<void>;
  previewOpacity: (id: string | null, value: number) => void;
  previewRoiOpacity?: (name: string, value: number) => void;
}) {
  const reconstruction = workspace?.view === 'MRax reconstruction';
  const kspace = workspace?.view === 'K-space';
  const component = reconstruction
    ? workspace!.reconstruction_component
    : kspace
      ? workspace!.kspace_mode
      : String(workspace?.display.component ?? 'auto');
  const components = reconstruction
    ? ['magnitude', 'phase']
    : kspace
      ? ['log_magnitude', 'magnitude', 'phase', 'real', 'imag']
      : ['auto', 'magnitude', 'phase', 'real', 'imag'];
  return (
    <div className="image-controls">
      <div hidden={task !== 'overlays'}>
        {workspace?.loaded && (
          <SceneControls
            key={`${workspace.id}-${workspace.experiment}`}
            workspace={workspace}
            busy={busy}
            changeView={changeView}
            previewOpacity={previewOpacity}
            previewRoiOpacity={previewRoiOpacity}
          />
        )}
      </div>
      <div className="control-layout" hidden={task !== 'display'}>
        <InspectorSection title="Presentation" icon="layers" className="control-wide">
          {workspace?.loaded && (
            <>
              <ViewingModes workspace={workspace} busy={busy} changeView={changeView} />
              <label hidden={workspace.scene.comparison.enabled}>
                View
                <select
                  aria-label="View"
                  disabled={busy}
                  value={workspace.view}
                  onChange={(event) => void changeView({ view: event.target.value })}
                >
                  {[
                    'Image',
                    'MRax reconstruction',
                    'Vendor vs MRax',
                    'Vendor MRax Diff',
                    'K-space',
                    'Map',
                    'Registration overlay',
                  ].map((view) => (
                    <option
                      key={view}
                      value={view}
                      disabled={
                        view === 'Map'
                          ? !workspace.maps.length
                          : view === 'Registration overlay'
                            ? !workspace.registrations.length
                            : view === 'K-space'
                              ? !workspace.has_kspace
                              : !['Image', 'Map', 'Registration overlay'].includes(view)
                                ? !workspace.has_reconstruction
                                : false
                      }
                    >
                      {viewLabel(workspace, view)}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
          <div ref={comparisonControlsRef} />
        </InspectorSection>
        <InspectorSection title="Appearance" icon="image">
          <fieldset className="control-form" disabled={busy || !workspace?.loaded}>
            <div className="field-grid">
              {workspace && (
                <>
                  <label
                    hidden={
                      workspace.scene.comparison.enabled ||
                      !['Image', 'Map', 'MRax reconstruction', 'K-space'].includes(workspace.view)
                    }
                  >
                    Component
                    <select
                      aria-label="Component"
                      value={component}
                      onChange={(e) =>
                        void changeView({
                          [reconstruction
                            ? 'reconstruction_component'
                            : kspace
                              ? 'kspace_mode'
                              : 'component']: e.target.value,
                        })
                      }
                    >
                      {components.map((v) => (
                        <option key={v}>{v}</option>
                      ))}
                    </select>
                  </label>
                  <label hidden={workspace.scene.comparison.enabled}>
                    Colormap
                    <select
                      aria-label="Colormap"
                      value={String(workspace.display.colormap ?? 'gray')}
                      onChange={(e) => void changeView({ colormap: e.target.value })}
                    >
                      {['gray', 'viridis', 'diverging', 'hot'].map((v) => (
                        <option key={v}>{v}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Curve axis
                    <select
                      aria-label="Curve axis"
                      value={String(workspace.display.plot_axis ?? 'auto')}
                      onChange={(e) => void changeView({ plot_axis: e.target.value })}
                    >
                      <option value="auto">Automatic</option>
                      {Object.keys(workspace.dimensions).map((v) => (
                        <option key={v}>{v}</option>
                      ))}
                    </select>
                  </label>
                </>
              )}
            </div>
          </fieldset>
        </InspectorSection>
        <details className="control-advanced control-wide">
          <summary>Advanced display settings</summary>
          <JsonEditor
            label="Visual settings"
            value={Object.fromEntries(
              Object.entries(workspace?.visual_state ?? {}).filter(
                ([key]) => key.endsWith('_channel') || key.endsWith('_alpha'),
              ),
            )}
            disabled={!workspace || busy}
            onApply={(value) =>
              action(
                'visual.update',
                Object.fromEntries(
                  Object.entries(value).filter(
                    ([key]) => key.endsWith('_channel') || key.endsWith('_alpha'),
                  ),
                ),
              )
            }
          />
        </details>
      </div>
    </div>
  );
}
