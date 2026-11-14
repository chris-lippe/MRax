import { useEffect, useRef, useState } from 'react';

export const shortcutDefinitions = [
  { id: 'view', label: 'View', context: 'app', binding: 'Ctrl+Shift+1' },
  { id: 'reconstruct', label: 'Reconstruct', context: 'app', binding: 'Ctrl+Shift+2' },
  { id: 'denoise', label: 'Denoise', context: 'app', binding: 'Ctrl+Shift+3' },
  { id: 'register', label: 'Register', context: 'app', binding: 'Ctrl+Shift+4' },
  { id: 'segment', label: 'Segment', context: 'app', binding: 'Ctrl+Shift+5' },
  { id: 'fit', label: 'Fit', context: 'app', binding: 'Ctrl+Shift+6' },
  { id: 'analyze', label: 'Analyze', context: 'app', binding: 'Ctrl+Shift+7' },
  { id: 'settings', label: 'Open Settings', context: 'app', binding: 'Ctrl+Shift+8' },
  { id: 'drawing', label: 'Toggle ROI drawing', context: 'segment', binding: 'Ctrl+Shift+9' },
  { id: 'resetImage', label: 'Fit image to view', context: 'image', binding: '0' },
  { id: 'zoomIn', label: 'Zoom in', context: 'image', binding: '+' },
  { id: 'zoomOut', label: 'Zoom out', context: 'image', binding: '-' },
] as const;
export type ShortcutAction = (typeof shortcutDefinitions)[number]['id'];
export type Bindings = Record<ShortcutAction, string>;
const imageKeys = ['0', '+', '-', 'F', 'I', 'O', '[', ']'];
const appKeys = Array.from({ length: 9 }, (_, index) => 'Ctrl+Shift+' + (index + 1));

export function shortcutChoices(id: ShortcutAction) {
  return shortcutDefinitions.find((item) => item.id === id)!.context === 'image'
    ? imageKeys
    : appKeys;
}

export function defaultBindings(): Bindings {
  return Object.fromEntries(shortcutDefinitions.map((item) => [item.id, item.binding])) as Bindings;
}

export function bindingError(bindings: Bindings, id: ShortcutAction, value: string) {
  if (value && !shortcutChoices(id).includes(value)) return 'This key combination is reserved.';
  const duplicate = shortcutDefinitions.find(
    (item) => item.id !== id && value && bindings[item.id] === value,
  );
  return duplicate ? 'Already assigned to ' + duplicate.label + '.' : null;
}

function readBindings(): Bindings {
  let saved: Record<string, unknown> = {};
  try {
    const value = JSON.parse(localStorage.getItem('mrax.shortcuts') ?? '{}');
    if (value && typeof value === 'object' && !Array.isArray(value)) saved = value;
  } catch {
    // Use safe defaults if stored bindings cannot be read.
  }
  const result = {} as Bindings;
  for (const definition of shortcutDefinitions) {
    const value = saved[definition.id];
    const candidate =
      typeof value === 'string' && (!value || shortcutChoices(definition.id).includes(value))
        ? value
        : definition.binding;
    result[definition.id] = Object.values(result).includes(candidate) && candidate ? '' : candidate;
  }
  return result;
}

export function useBindings() {
  const [bindings, setBindings] = useState<Bindings>(readBindings);
  useEffect(() => {
    try {
      localStorage.setItem('mrax.shortcuts', JSON.stringify(bindings));
    } catch {
      // Bindings still work for the current session without persistent storage.
    }
  }, [bindings]);
  return { bindings, setBindings };
}

export function useShortcuts(bindings: Bindings, dispatch: (action: ShortcutAction) => boolean) {
  const current = useRef(dispatch);
  useEffect(() => {
    current.current = dispatch;
  });
  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.repeat ||
        event.isComposing ||
        event.altKey ||
        event.metaKey
      )
        return;
      const target = event.target instanceof Element ? event.target : null;
      if (
        target?.closest(
          'input, textarea, select, [contenteditable="true"], [role="textbox"], [role="menu"]',
        )
      )
        return;
      if (document.querySelector('[role="dialog"], [role="alertdialog"]')) return;
      const image = Boolean(target?.closest('.image-viewport, .comparison-canvas-viewport'));
      const appBinding =
        event.ctrlKey && event.shiftKey && /^Digit[1-9]$/.test(event.code)
          ? 'Ctrl+Shift+' + event.code.slice(-1)
          : '';
      const imageBinding =
        !event.ctrlKey && image && (!event.shiftKey || event.key === '+')
          ? event.key === '='
            ? '+'
            : event.key.toUpperCase()
          : '';
      const definition = shortcutDefinitions.find((item) => {
        const binding = item.context === 'image' ? imageBinding : appBinding;
        return binding && bindings[item.id] === binding;
      });
      if (definition && current.current(definition.id)) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    window.addEventListener('keydown', keyboard);
    return () => window.removeEventListener('keydown', keyboard);
  }, [bindings]);
}
