import { useEffect, useState } from 'react';
import { api, type Workspace } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';

type FramePage = {
  revision: number;
  frame_count: number;
  frames: { index: number; label: string }[];
};

const pageSize = 32;

export function FitFrameSelector({
  workspace,
  excluded,
  disabled,
  onChange,
  onView,
}: {
  workspace: Workspace;
  excluded: number[];
  disabled: boolean;
  onChange: (values: number[]) => void;
  onView: (frame: number) => void;
}) {
  const context = workspace.id + ':' + workspace.experiment;
  const [paging, setPaging] = useState({ context, offset: 0 });
  const offset = Math.min(
    paging.context === context ? paging.offset : 0,
    Math.max(0, Math.floor((workspace.frame_count - 1) / pageSize) * pageSize),
  );
  const key = JSON.stringify([workspace.id, workspace.revision, offset]);
  const [loaded, setLoaded] = useState<{ key: string; data?: FramePage; error?: string } | null>(
    null,
  );
  const page = loaded?.key === key ? loaded : null;
  useEffect(() => {
    const controller = new AbortController();
    void api<FramePage>(
      '/api/workspaces/' + workspace.id + '/fit-frames?offset=' + offset + '&limit=' + pageSize,
      'GET',
      undefined,
      controller.signal,
    )
      .then((data) => {
        if (!controller.signal.aborted && data.revision === workspace.revision)
          setLoaded({ key, data });
      })
      .catch((error) => {
        if (!controller.signal.aborted) setLoaded({ key, error: String(error) });
      });
    return () => controller.abort();
  }, [key, offset, workspace.id, workspace.revision]);
  const selected = new Set(excluded);
  const includedCount = workspace.frame_count - selected.size;
  const goTo = (value: number) => setPaging({ context, offset: value });
  return (
    <div className="fit-frame-picker">
      <p className="control-note" aria-live="polite">
        {includedCount} of {workspace.frame_count} frames included
      </p>
      <div className="button-row">
        <button
          type="button"
          className="secondary small"
          disabled={disabled || !excluded.length}
          onClick={() => onChange([])}
        >
          Include all
        </button>
        <button
          type="button"
          className="secondary small"
          disabled={disabled}
          onClick={() => goTo(Math.floor(workspace.frame / pageSize) * pageSize)}
        >
          Show current frame
        </button>
        <button
          type="button"
          className="secondary small"
          disabled={disabled || selected.has(workspace.frame) || includedCount <= 1}
          onClick={() => onChange([...excluded, workspace.frame].sort((a, b) => a - b))}
        >
          Exclude current frame
        </button>
      </div>
      {page?.error && <p role="alert">{page.error}</p>}
      {!page && <p role="status">Loading frames…</p>}
      <ul className="fit-frame-list" aria-label="Frames for fitting">
        {page?.data?.frames.map((frame) => (
          <li key={frame.index} aria-current={frame.index === workspace.frame ? 'true' : undefined}>
            <label className="check-label">
              <input
                type="checkbox"
                aria-label={'Include frame ' + (frame.index + 1)}
                checked={!selected.has(frame.index)}
                disabled={disabled || (includedCount <= 1 && !selected.has(frame.index))}
                onChange={(event) =>
                  onChange(
                    event.target.checked
                      ? excluded.filter((index) => index !== frame.index)
                      : [...excluded, frame.index].sort((a, b) => a - b),
                  )
                }
              />
              <span>
                Frame {frame.index + 1}
                <small>{frame.label}</small>
              </span>
            </label>
            <button
              type="button"
              className="secondary small"
              disabled={disabled}
              aria-label={'View frame ' + (frame.index + 1)}
              title="View frame"
              onClick={() => onView(frame.index)}
            >
              <WorkbenchIcon name="image" />
            </button>
          </li>
        ))}
      </ul>
      {workspace.frame_count > pageSize && (
        <div className="fit-frame-pagination">
          <button
            type="button"
            className="secondary small"
            disabled={disabled || offset === 0}
            onClick={() => goTo(offset - pageSize)}
          >
            Previous
          </button>
          <span>
            {offset + 1} to {Math.min(offset + pageSize, workspace.frame_count)} of{' '}
            {workspace.frame_count}
          </span>
          <button
            type="button"
            className="secondary small"
            disabled={disabled || offset + pageSize >= workspace.frame_count}
            onClick={() => goTo(offset + pageSize)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
