import { originalComparison, type Options, type Workspace } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';

export function ViewingModes({
  workspace,
  busy,
  changeView,
}: {
  workspace: Workspace;
  busy: boolean;
  changeView: (command: Options) => Promise<void>;
}) {
  const comparison = workspace.scene.comparison;
  return (
    <div className="viewing-modes">
      <div className="segmented-control" role="group" aria-label="Viewing mode">
        {(
          [
            ['image', 'Image', 'image'],
            ['overlay', 'Overlay', 'layers'],
            ['split', 'Split view', 'split'],
          ] as const
        ).map(([mode, label, icon]) => (
          <button
            key={mode}
            disabled={busy}
            aria-pressed={
              mode === 'image'
                ? !comparison.enabled
                : comparison.enabled && comparison.mode === mode
            }
            onClick={() =>
              void changeView({
                scene: {
                  comparison: {
                    ...comparison,
                    enabled: mode !== 'image',
                    ...(mode !== 'image' ? { mode } : {}),
                  },
                },
              })
            }
          >
            <WorkbenchIcon name={icon} />
            {label}
          </button>
        ))}
      </div>
      <button
        className="secondary full"
        disabled={busy}
        onClick={() => void changeView({ scene: { comparison: originalComparison(workspace) } })}
      >
        <WorkbenchIcon name="split" />
        Original vs processed
      </button>
    </div>
  );
}
