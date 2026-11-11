import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react';

export function ResizablePreview({
  className,
  label,
  bodyId,
  storageKey,
  minimum,
  maximum,
  defaultHeight,
  growWithContent = false,
  hidden,
  children,
}: {
  className: string;
  label: string;
  bodyId: string;
  storageKey: string;
  minimum: number;
  maximum: number;
  defaultHeight?: () => number;
  growWithContent?: boolean;
  hidden: boolean;
  children: ReactNode;
}) {
  const clamp = (value: number) => Math.round(Math.max(minimum, Math.min(maximum, value)));
  const [height, setHeight] = useState<number | undefined>(() => {
    try {
      const saved = Number(localStorage.getItem(storageKey));
      if (Number.isFinite(saved) && saved >= minimum && saved <= maximum) return saved;
    } catch {
      /* Storage is optional. */
    }
    return defaultHeight?.();
  });
  const container = useRef<HTMLDivElement>(null);
  const [measured, setMeasured] = useState(minimum);
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.height > 0) setMeasured(Math.round(entry.contentRect.height));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const drag = useRef<{
    id: number;
    y: number;
    height: number;
    before: number | undefined;
    latest: number;
    scroll: number;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const commit = (value: number | undefined) => {
    const next = value === undefined ? undefined : clamp(value);
    setHeight(next);
    try {
      if (next === undefined) localStorage.removeItem(storageKey);
      else localStorage.setItem(storageKey, String(next));
    } catch {
      /* Storage is optional. */
    }
  };
  function finish(event: PointerEvent<HTMLDivElement>, cancel = false) {
    const state = drag.current;
    if (!state || state.id !== event.pointerId) return;
    drag.current = null;
    setDragging(false);
    if (cancel) setHeight(state.before);
    else commit(state.latest);
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  }
  const currentHeight = growWithContent ? measured : (height ?? measured);
  return (
    <div
      ref={container}
      className={className}
      hidden={hidden}
      style={{
        minHeight: growWithContent ? (height ?? minimum) : minimum,
        height: growWithContent ? undefined : height,
        flex: height === undefined ? undefined : '0 0 auto',
      }}
    >
      {children}
      <div
        className="preview-resize"
        data-dragging={dragging || undefined}
        role="separator"
        aria-label={label}
        aria-orientation="horizontal"
        aria-controls={bodyId}
        aria-valuemin={minimum}
        aria-valuemax={maximum}
        aria-valuenow={currentHeight}
        aria-valuetext={currentHeight + ' pixels'}
        tabIndex={0}
        title="Drag down to enlarge. Arrow keys resize. Home or double-click resets."
        onDoubleClick={() => commit(defaultHeight?.())}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.preventDefault();
          event.currentTarget.focus({ preventScroll: true });
          const main = event.currentTarget.closest('main');
          const actual = container.current?.getBoundingClientRect().height ?? currentHeight;
          drag.current = {
            id: event.pointerId,
            y: event.clientY,
            height: actual,
            before: height,
            latest: actual,
            scroll: main?.scrollTop ?? 0,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
          setDragging(true);
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
            setDragging(false);
            setHeight(state.before);
            if (event.currentTarget.hasPointerCapture(state.id))
              event.currentTarget.releasePointerCapture(state.id);
            event.preventDefault();
            event.stopPropagation();
            return;
          }
          if (event.key === 'Home') {
            event.preventDefault();
            event.stopPropagation();
            commit(defaultHeight?.());
            return;
          }
          const step = event.shiftKey ? 64 : 16;
          const actual = Math.round(
            container.current?.getBoundingClientRect().height ?? currentHeight,
          );
          const next =
            event.key === 'ArrowDown'
              ? actual + step
              : event.key === 'ArrowUp'
                ? actual - step
                : event.key === 'End'
                  ? maximum
                  : null;
          if (next !== null) {
            event.preventDefault();
            event.stopPropagation();
            commit(next);
          }
        }}
      >
        <span />
      </div>
    </div>
  );
}
