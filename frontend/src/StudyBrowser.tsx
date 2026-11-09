import type { Ref } from 'react';
import type { Workspace } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';

export function StudyBrowser({
  workspace,
  workspaces,
  busy,
  connected,
  running,
  importButton,
  exportButton,
  settingsButton,
  newStudyButton,
  onSettings,
  onNew,
  onOpen,
  onImport,
  onExport,
  onSave,
  onStudy,
  onExperiment,
  onClose,
}: {
  workspace: Workspace | null;
  workspaces: Workspace[];
  busy: boolean;
  connected: boolean;
  running: boolean;
  importButton: Ref<HTMLButtonElement>;
  exportButton: Ref<HTMLButtonElement>;
  settingsButton: Ref<HTMLButtonElement>;
  newStudyButton: Ref<HTMLButtonElement>;
  onSettings: () => void;
  onNew: () => void;
  onOpen: () => void;
  onImport: () => void;
  onExport: () => void;
  onSave: () => void;
  onStudy: (id: string) => void;
  onExperiment: (name: string) => void;
  onClose: (study: Workspace, trigger: HTMLButtonElement) => void;
}) {
  return (
    <aside id="study-controls" className="study-panel" aria-label="Data & studies">
      <section className="data-controls" aria-labelledby="data-controls-heading">
        <h2 id="data-controls-heading">Data controls</h2>
        <div
          className="study-files"
          role="toolbar"
          aria-label="Workspace files"
          aria-orientation="vertical"
          onKeyDown={(event) => {
            const step = event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0;
            if (!step && !['Home', 'End'].includes(event.key)) return;
            const buttons = Array.from(
              event.currentTarget.querySelectorAll<HTMLButtonElement>('button:enabled'),
            ).filter((button) => button.getClientRects().length > 0);
            if (!buttons.length) return;
            const index = buttons.findIndex((button) => button === document.activeElement);
            const next =
              event.key === 'Home'
                ? 0
                : event.key === 'End'
                  ? buttons.length - 1
                  : (index + step + buttons.length) % buttons.length;
            event.preventDefault();
            buttons[next].focus();
          }}
        >
          <button
            className="secondary"
            ref={newStudyButton}
            aria-haspopup="dialog"
            disabled={busy || !connected}
            onClick={onNew}
          >
            <WorkbenchIcon name="new" />
            <span>New study</span>
          </button>
          <button className="secondary" disabled={busy || !connected} onClick={onOpen}>
            <WorkbenchIcon name="open" />
            <span>Open workspace</span>
          </button>
          <button
            className="secondary"
            ref={importButton}
            disabled={busy || !connected}
            onClick={onImport}
          >
            <WorkbenchIcon name="import" />
            <span>Import data</span>
          </button>
          <button
            className="secondary"
            ref={exportButton}
            disabled={!workspace?.loaded || busy || !connected}
            aria-haspopup="dialog"
            onClick={onExport}
          >
            <WorkbenchIcon name="export" />
            <span>Export data</span>
          </button>
          <button
            className="secondary"
            disabled={!workspace?.loaded || busy || running}
            onClick={onSave}
          >
            <WorkbenchIcon name="save" />
            <span>Save workspace</span>
          </button>
        </div>
      </section>
      <div className="study-browser-body">
        <div className="panel-title">
          <h3>Open studies</h3>
          <span>{workspaces.length} open</span>
        </div>
        <nav className="study-list" aria-label="Study and experiment browser">
          {!workspaces.length && <p className="muted">No studies open.</p>}
          {workspaces.map((item) => {
            const selected = workspace?.id === item.id;
            return (
              <div className="study-group" key={item.id}>
                <div className="study-row" data-study-id={item.id}>
                  <button
                    className={`study-item ${selected ? 'selected' : ''}`}
                    aria-current={selected ? 'true' : undefined}
                    disabled={busy}
                    onClick={() => onStudy(item.id)}
                  >
                    <svg className="study-symbol" viewBox="0 0 20 20" aria-hidden="true">
                      <path d="M2.5 5.5h5l2-2h8v13h-15z" />
                    </svg>
                    <span>
                      <strong>
                        {item.label}
                        {item.dirty && (
                          <span className="modified-indicator" title="Unsaved changes">
                            {' '}
                            ●
                          </span>
                        )}
                      </strong>
                      <small>
                        {item.experiments.length}{' '}
                        {item.experiments.length === 1 ? 'experiment' : 'experiments'}
                      </small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className="study-close"
                    aria-label={'Close study: ' + item.label}
                    title={'Close study: ' + item.label}
                    aria-haspopup={item.dirty ? 'dialog' : undefined}
                    disabled={busy || running}
                    onClick={(event) => {
                      event.stopPropagation();
                      onClose(item, event.currentTarget);
                    }}
                  >
                    <WorkbenchIcon name="close" />
                  </button>
                </div>
                {selected && workspace.loaded && (
                  <section className="experiment-list" aria-label="Experiments">
                    <h3 className="visually-hidden">Experiments</h3>
                    {workspace.experiments.map((name) => (
                      <button
                        key={name}
                        type="button"
                        className={workspace.experiment === name ? 'selected' : ''}
                        aria-current={workspace.experiment === name ? 'true' : undefined}
                        disabled={busy}
                        onClick={() => onExperiment(name)}
                      >
                        <svg className="study-symbol" viewBox="0 0 20 20" aria-hidden="true">
                          <path d="M4 3.5h12v13H4zM7 7h6M7 10h6M7 13h4" />
                        </svg>
                        <span>{name}</span>
                      </button>
                    ))}
                  </section>
                )}
              </div>
            );
          })}
        </nav>
      </div>
      <div className="study-panel-footer">
        <button
          className="study-settings"
          ref={settingsButton}
          aria-haspopup="dialog"
          onClick={onSettings}
        >
          <WorkbenchIcon name="gear" />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  );
}
