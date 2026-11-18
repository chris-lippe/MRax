import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { KeyboardEvent, MouseEvent } from 'react';

type Action = { label: string; run: () => void; disabled?: boolean };
type Position = { x: number; y: number };

export function useContextMenu(label: string, actions: Action[]) {
  const [position, setPosition] = useState<Position | null>(null);
  const trigger = useRef<HTMLElement | SVGElement | null>(null);
  const close = useCallback((restoreFocus = true) => {
    setPosition(null);
    if (restoreFocus) trigger.current?.focus({ preventScroll: true });
  }, []);
  const open = (target: HTMLElement | SVGElement, x: number, y: number) => {
    trigger.current = target;
    setPosition({ x, y });
  };
  return {
    close,
    onContextMenu: (event: MouseEvent<HTMLElement | SVGElement>) => {
      event.preventDefault();
      event.stopPropagation();
      open(event.currentTarget, event.clientX, event.clientY);
    },
    onKeyDown: (event: KeyboardEvent<HTMLElement | SVGElement>) => {
      if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) {
        event.preventDefault();
        const bounds = event.currentTarget.getBoundingClientRect();
        open(event.currentTarget, bounds.left + 24, bounds.top + 24);
      }
    },
    menu: position && (
      <ContextMenu label={label} actions={actions} position={position} onClose={close} />
    ),
  };
}

function ContextMenu({
  label,
  actions,
  position,
  onClose,
}: {
  label: string;
  actions: Action[];
  position: Position;
  onClose: (restoreFocus?: boolean) => void;
}) {
  const menu = useRef<HTMLDivElement>(null);
  const [placed, setPlaced] = useState(position);
  useLayoutEffect(() => {
    const element = menu.current!;
    const bounds = element.getBoundingClientRect();
    setPlaced({
      x: Math.max(8, Math.min(position.x, window.innerWidth - bounds.width - 8)),
      y: Math.max(8, Math.min(position.y, window.innerHeight - bounds.height - 8)),
    });
    (element.querySelector<HTMLButtonElement>('button:enabled') ?? element).focus({
      preventScroll: true,
    });
  }, [position]);
  useEffect(() => {
    const outside = (event: PointerEvent) => {
      if (!menu.current?.contains(event.target as Node)) onClose(false);
    };
    const dismiss = () => onClose(false);
    document.addEventListener('pointerdown', outside, true);
    window.addEventListener('resize', dismiss);
    // Layout and focus can scroll without user input while the menu is opening.
    window.addEventListener('wheel', dismiss, { capture: true, passive: true });
    window.addEventListener('touchmove', dismiss, { capture: true, passive: true });
    window.addEventListener('blur', dismiss);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      window.removeEventListener('resize', dismiss);
      window.removeEventListener('wheel', dismiss, true);
      window.removeEventListener('touchmove', dismiss, true);
      window.removeEventListener('blur', dismiss);
    };
  }, [onClose]);
  return createPortal(
    <div
      ref={menu}
      role="menu"
      aria-label={label}
      tabIndex={-1}
      className="context-menu"
      style={{ left: placed.x, top: placed.y }}
      onContextMenu={(event) => event.preventDefault()}
      onKeyDown={(event) => {
        if (event.key === 'PageUp' || event.key === 'PageDown') {
          onClose(false);
          return;
        }
        if (event.key === 'Escape' || event.key === 'Tab') {
          if (event.key === 'Escape') event.preventDefault();
          onClose();
          return;
        }
        const items = [...menu.current!.querySelectorAll<HTMLButtonElement>('button:enabled')];
        const index = items.indexOf(document.activeElement as HTMLButtonElement);
        const next =
          event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? items.length - 1
              : event.key === 'ArrowDown'
                ? (index + 1) % items.length
                : event.key === 'ArrowUp'
                  ? (index + items.length - 1) % items.length
                  : null;
        if (next !== null) {
          event.preventDefault();
          items[next]?.focus();
        }
      }}
    >
      {actions.map((action) => (
        <button
          key={action.label}
          type="button"
          role="menuitem"
          disabled={action.disabled}
          onClick={() => {
            onClose();
            action.run();
          }}
        >
          {action.label}
        </button>
      ))}
    </div>,
    document.body,
  );
}
