import type { ReactNode } from 'react';
import { comparisonNavigation, originalComparison, type Options, type Workspace } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';
import { reconstructionComparison, type ViewingContext } from './viewing';

export function ViewerControls({
  workspace,
  context,
  busy,
  change,
  expanded,
  onExpandedChange,
  children,
}: {
  workspace: Workspace;
  context: ViewingContext;
  busy: boolean;
  change: (command: Options) => Promise<void>;
  expanded: boolean;
  onExpandedChange: (value: boolean) => void;
  children: ReactNode;
}) {
  const comparison = workspace.scene.comparison;
  const navigation = comparison.enabled ? comparisonNavigation(workspace) : null;
  const singleView = (view: string) =>
    void change({
      view,
      ...(context.section === 'reconstruct' && navigation
        ? {
            frame: Math.min(navigation.frame, workspace.frame_count - 1),
            plane: navigation.plane,
            indices: Object.fromEntries(
              Object.entries(navigation.indices)
                .filter(([dim]) => dim in workspace.dimensions)
                .map(([dim, value]) => [dim, Math.min(value, workspace.dimensions[dim] - 1)]),
            ),
          }
        : {}),
      scene: { comparison: { ...comparison, enabled: false } },
    });
  return (
    <details
      className="viewer-controls"
      open={expanded}
      onToggle={(event) => onExpandedChange(event.currentTarget.open)}
    >
      <summary>
        <WorkbenchIcon name="settings" />
        <span>View controls</span>
        <WorkbenchIcon name="chevron" />
      </summary>
      <fieldset className="view-selectors" disabled={busy}>
        {context.section === 'reconstruct' && (
          <div
            className="segmented-control context-view-modes"
            role="group"
            aria-label="Reconstruction display"
          >
            {(
              [
                ['Image', 'Experiment'],
                ['MRax reconstruction', workspace.reconstruction_label],
                ['comparison', 'Vendor comparison'],
                ['Vendor MRax Diff', 'Difference'],
                ['K-space', 'K-space'],
              ] as const
            ).map(([view, label]) => (
              <button
                key={view}
                disabled={
                  busy ||
                  (view === 'K-space'
                    ? !workspace.has_kspace
                    : view !== 'Image' && !workspace.has_reconstruction)
                }
                aria-pressed={
                  view === 'comparison'
                    ? comparison.enabled
                    : !comparison.enabled && workspace.view === view
                }
                onClick={() =>
                  view === 'comparison'
                    ? void change({ scene: { comparison: reconstructionComparison(workspace) } })
                    : singleView(view)
                }
              >
                {label}
              </button>
            ))}
          </div>
        )}
        {context.section === 'fit' && (
          <div
            className="segmented-control context-view-modes"
            role="group"
            aria-label="Fit display"
          >
            <button aria-pressed={workspace.view !== 'Map'} onClick={() => singleView('Image')}>
              Signal
            </button>
            <button
              disabled={!workspace.maps.length}
              aria-pressed={workspace.view === 'Map'}
              onClick={() => singleView('Map')}
            >
              Parameter map
            </button>
          </div>
        )}
        {context.section === 'denoise' && (
          <button
            className="secondary"
            aria-pressed={comparison.enabled}
            data-tutorial="compare-original"
            onClick={() =>
              comparison.enabled
                ? singleView('Image')
                : void change({ scene: { comparison: originalComparison(workspace) } })
            }
          >
            Compare with original
          </button>
        )}
        <label>
          Plane
          <select
            aria-label="Plane"
            value={navigation?.plane ?? workspace.plane}
            onChange={(event) =>
              void change(
                navigation
                  ? {
                      scene: {
                        comparison: {
                          ...comparison,
                          plane: event.target.value,
                          frame: navigation.frame,
                          indices: navigation.indices,
                        },
                      },
                    }
                  : { plane: event.target.value },
              )
            }
          >
            {['axial', 'coronal', 'sagittal'].map((plane) => (
              <option
                key={plane}
                disabled={
                  plane !== (navigation?.plane ?? workspace.plane) &&
                  !(navigation?.volume ?? workspace.can_change_plane)
                }
              >
                {plane}
              </option>
            ))}
          </select>
        </label>
        {workspace.view === 'Map' && !workspace.scene.comparison.enabled && (
          <label>
            Map
            <select
              aria-label="Map"
              value={workspace.active_map ?? ''}
              onChange={(event) => void change({ map_name: event.target.value })}
            >
              {workspace.maps.map((map) => (
                <option key={String(map.name)} value={String(map.name)}>
                  {String(map.name)} ({String(map.unit)})
                </option>
              ))}
            </select>
          </label>
        )}
      </fieldset>
      {children}
    </details>
  );
}
