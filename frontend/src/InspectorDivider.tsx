import { useEffect, useRef, useState, type PointerEvent } from 'react';

function usePanelWidth(
  storageKey: string,
  minimum: number,
  defaultWidth: number,
  limit: number,
  available: number,
) {
  const maximum = Math.max(minimum, Math.min(limit, available));
  const clamp = (value: number) => Math.round(Math.max(minimum, Math.min(maximum, value)));
  const [preferred, setPreferred] = useState(() => {
    try {
      const saved = Number(localStorage.getItem(storageKey));
      return Number.isFinite(saved) && saved >= minimum && saved <= limit ? saved : defaultWidth;
    } catch {
      return defaultWidth;
    }
  });
  const [dragging, setDragging] = useState(false);
  const resize = (value: number) => setPreferred(clamp(value));
  const save = (next: number) => {
    setPreferred(next);
    try {
      localStorage.setItem(storageKey, String(next));
    } catch {
      /* Storage is optional. */
    }
  };
  return {
    width: clamp(preferred),
    minimum,
    maximum,
    defaultWidth,
    dragging,
    setDragging,
    resize,
    commit: (value: number) => save(clamp(value)),
    reset: () => save(defaultWidth),
  };
}

export function useWorkspacePanels(inspectorVisible: boolean) {
  const [viewport, setViewport] = useState(() => window.innerWidth);
  useEffect(() => {
    const update = () => setViewport(window.innerWidth);
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);
  const browser = usePanelWidth(
    'mrax.studyWidth',
    208,
    248,
    720,
    viewport - 408 - (inspectorVisible ? 288 + 16 : 8),
  );
  const inspector = usePanelWidth(
    'mrax.inspectorWidth',
    288,
    360,
    900,
    viewport - browser.width - 408 - 16,
  );
  return { browser, inspector };
}

type PanelWidth = ReturnType<typeof usePanelWidth>;

export function InspectorDivider({ layout }: { layout: PanelWidth }) {
  return (
    <ResizeDivider
      layout={layout}
      label="Resize analysis controls"
      controls="analysis-controls"
      direction={-1}
    />
  );
}

export function StudyDivider({ layout }: { layout: PanelWidth }) {
  return (
    <ResizeDivider
      layout={layout}
      label="Resize study browser"
      controls="study-controls"
      direction={1}
    />
  );
}

function ResizeDivider({
  layout,
  label,
  controls,
  direction,
}: {
  layout: PanelWidth;
  label: string;
  controls: string;
  direction: 1 | -1;
}) {
  const drag = useRef<{ id: number; x: number; width: number; latest: number } | null>(null);
  const setDragging = layout.setDragging;
  useEffect(() => () => setDragging(false), [setDragging]);
  const finish = (event: PointerEvent<HTMLDivElement>, cancel = false) => {
    const current = drag.current;
    if (!current || current.id !== event.pointerId) return;
    drag.current = null;
    layout.setDragging(false);
    if (cancel) layout.resize(current.width);
    else layout.commit(current.latest);
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return (
    <div
      className="inspector-divider"
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      aria-controls={controls}
      aria-valuemin={layout.minimum}
      aria-valuemax={layout.maximum}
      aria-valuenow={layout.width}
      aria-valuetext={layout.width + ' pixels'}
      tabIndex={0}
      title="Drag to resize. Arrow keys adjust width. Double-click to reset."
      onDoubleClick={layout.reset}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.currentTarget.focus({ preventScroll: true });
        drag.current = {
          id: event.pointerId,
          x: event.clientX,
          width: layout.width,
          latest: layout.width,
        };
        event.currentTarget.setPointerCapture(event.pointerId);
        layout.setDragging(true);
      }}
      onPointerMove={(event) => {
        const current = drag.current;
        if (!current || event.pointerId !== current.id) return;
        current.latest = Math.round(
          Math.max(
            layout.minimum,
            Math.min(layout.maximum, current.width + (event.clientX - current.x) * direction),
          ),
        );
        layout.resize(current.latest);
      }}
      onPointerUp={(event) => finish(event)}
      onPointerCancel={(event) => finish(event, true)}
      onLostPointerCapture={(event) => finish(event)}
      onKeyDown={(event) => {
        if (event.key === 'Escape' && drag.current) {
          event.preventDefault();
          const current = drag.current;
          drag.current = null;
          layout.resize(current.width);
          layout.setDragging(false);
          if (event.currentTarget.hasPointerCapture(current.id))
            event.currentTarget.releasePointerCapture(current.id);
          return;
        }
        const step = event.shiftKey ? 64 : 16;
        const next =
          event.key === 'ArrowLeft'
            ? layout.width - step * direction
            : event.key === 'ArrowRight'
              ? layout.width + step * direction
              : event.key === 'Home'
                ? layout.defaultWidth
                : event.key === 'End'
                  ? layout.maximum
                  : null;
        if (next !== null) {
          event.preventDefault();
          layout.commit(next);
        }
      }}
    >
      <span />
    </div>
  );
}
