import { useMemo, useState, type SetStateAction } from 'react';
import type { Pixels } from './api';

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;

type DisplaySettings = {
  zoom: number;
  pan: [number, number];
  window: { center: number; width: number } | null;
};
const initial: DisplaySettings = { zoom: MIN_ZOOM, pan: [0, 0], window: null };

export function useImageDisplay(key: string | undefined, values: Pixels | null, rgb = false) {
  const [settings, setSettings] = useState<Record<string, DisplaySettings>>({});
  const id = key ?? '';
  const { zoom, pan, window } = settings[id] ?? initial;
  function setField<K extends keyof DisplaySettings>(
    field: K,
    value: SetStateAction<DisplaySettings[K]>,
  ) {
    setSettings((previous) => {
      const before = previous[id] ?? initial;
      const next =
        typeof value === 'function'
          ? (value as (old: DisplaySettings[K]) => DisplaySettings[K])(before[field])
          : value;
      return { ...previous, [id]: { ...before, [field]: next } };
    });
  }
  const setZoom = (value: SetStateAction<number>) => setField('zoom', value);
  const setPan = (value: SetStateAction<[number, number]>) => setField('pan', value);
  const setWindow = (value: SetStateAction<DisplaySettings['window']>) => setField('window', value);
  const range = useMemo(() => {
    if (rgb) return [0, 1] as const;
    let minimum = Infinity,
      maximum = -Infinity;
    if (values)
      for (const value of values)
        if (Number.isFinite(value)) {
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        }
    return Number.isFinite(minimum) ? ([minimum, maximum] as const) : ([0, 1] as const);
  }, [values, rgb]);
  function resetIntensity() {
    setWindow(null);
  }
  return {
    zoom,
    setZoom,
    zoomIn: () => setZoom((value) => Math.min(MAX_ZOOM, Number((value * 1.25).toFixed(1)))),
    zoomOut: () => setZoom((value) => Math.max(MIN_ZOOM, Number((value / 1.25).toFixed(1)))),
    pan,
    setPan,
    setWindow,
    range,
    resetIntensity,
    resetPosition: () => {
      setZoom(1);
      setPan([0, 0]);
    },
    level: window ?? { center: (range[0] + range[1]) / 2, width: range[1] - range[0] || 1 },
    reset: () => {
      resetIntensity();
      setZoom(1);
      setPan([0, 0]);
    },
  };
}

export type ImageDisplay = ReturnType<typeof useImageDisplay>;
