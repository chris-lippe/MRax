import { useRef, useState, type PointerEvent, type ReactNode } from 'react';

const minimum = 140,
  maximum = 800;
const defaultHeight = () => Math.round(Math.max(160, Math.min(200, window.innerHeight * 0.18)));
const clamp = (value: number) => Math.round(Math.max(minimum, Math.min(maximum, value)));

export function SignalPreview({ hidden, children }: { hidden: boolean; children: ReactNode }) {
  const [height, setHeight] = useState(() => {
    try {
      const saved = Number(localStorage.getItem('mrax.signalHeight'));
      if (Number.isFinite(saved) && saved >= minimum && saved <= maximum) return saved;
    } catch {
      /* Storage is optional. */
    }
    return defaultHeight();
  });
  const drag = useRef<{
    id: number;
    y: number;
    height: number;
    latest: number;
    scroll: number;
  } | null>(null);
  const commit = (value: number) => {
    const next = clamp(value);
    setHeight(next);
    try {
      localStorage.setItem('mrax.signalHeight', String(next));
    } catch {
      /* Storage is optional. */
    }
  };
  function finish(event: PointerEvent<HTMLDivElement>, cancel = false) {
    const state = drag.current;
    if (!state || state.id !== event.pointerId) return;
    drag.current = null;
    if (cancel) setHeight(state.height);
    else commit(state.latest);
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  }
  return (
    <div className="signal-preview" hidden={hidden} style={{ height }}>
      <div id="signal-preview-body" className="signal-preview-body">
        {children}
      </div>
      <div
        className="signal-resize"
        role="separator"
        aria-label="Resize signal preview"
        aria-orientation="horizontal"
        aria-controls="signal-preview-body"
        aria-valuemin={minimum}
        aria-valuemax={maximum}
        aria-valuenow={height}
        aria-valuetext={height + ' pixels'}
        tabIndex={0}
        title="Drag down to enlarge. Arrow keys resize; Home or double-click resets."
        onDoubleClick={() => commit(defaultHeight())}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.preventDefault();
          event.currentTarget.focus({ preventScroll: true });
          const main = event.currentTarget.closest('main');
          drag.current = {
            id: event.pointerId,
            y: event.clientY,
            height,
            latest: height,
            scroll: main?.scrollTop ?? 0,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const state = drag.current;
          if (!state || state.id !== event.pointerId) return;
          const main = event.currentTarget.closest('main');
          if (main && event.clientY > main.getBoundingClientRect().bottom - 24)
            main.scrollTop +=
              Math.max(0, event.clientY - main.getBoundingClientRect().bottom + 24) / 2;
          state.latest = clamp(
            state.height + event.clientY - state.y + (main?.scrollTop ?? 0) - state.scroll,
          );
          setHeight(state.latest);
        }}
        onPointerUp={(event) => finish(event)}
        onPointerCancel={(event) => finish(event, true)}
        onLostPointerCapture={(event) => finish(event)}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && drag.current) {
            const state = drag.current;
            drag.current = null;
            setHeight(state.height);
            if (event.currentTarget.hasPointerCapture(state.id))
              event.currentTarget.releasePointerCapture(state.id);
            event.preventDefault();
            return;
          }
          const step = event.shiftKey ? 64 : 16;
          const next =
            event.key === 'ArrowDown'
              ? height + step
              : event.key === 'ArrowUp'
                ? height - step
                : event.key === 'Home'
                  ? defaultHeight()
                  : event.key === 'End'
                    ? maximum
                    : null;
          if (next !== null) {
            event.preventDefault();
            commit(next);
          }
        }}
      >
        <span />
      </div>
    </div>
  );
}
