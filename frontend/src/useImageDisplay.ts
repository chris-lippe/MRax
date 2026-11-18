import { useMemo, useState, type SetStateAction } from 'react';
import type { Pixels } from './api';

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;

type DisplaySettings = {
  perFrame: boolean;
  zoom: number;
  pan: [number, number];
  window: { center: number; width: number } | null;
};
const initial: DisplaySettings = { zoom: MIN_ZOOM, pan: [0, 0], window: null, perFrame: false };

export function useImageDisplay(
  key: string | undefined,
  values: Pixels | null,
  fixedRange: readonly number[] | undefined,
  plane: string | undefined,
  rgb = false,
) {
  const [settings, setSettings] = useState<Record<string, DisplaySettings>>({});
  const id = key ?? '';
  const positionId = JSON.stringify([key, plane]);
  const { zoom, pan } = settings[positionId] ?? initial;
  const { window, perFrame } = settings[id] ?? initial;
  function setField<K extends keyof DisplaySettings>(
    field: K,
    value: SetStateAction<DisplaySettings[K]>,
  ) {
    const target = field === 'zoom' || field === 'pan' ? positionId : id;
    setSettings((previous) => {
      const before = previous[target] ?? initial;
      const next =
        typeof value === 'function'
          ? (value as (old: DisplaySettings[K]) => DisplaySettings[K])(before[field])
          : value;
      return { ...previous, [target]: { ...before, [field]: next } };
    });
  }
  const setZoom = (value: SetStateAction<number>) => setField('zoom', value);
  const setPan = (value: SetStateAction<[number, number]>) => setField('pan', value);
  const setWindow = (value: SetStateAction<DisplaySettings['window']>) => setField('window', value);
  const range = useMemo((): readonly [number, number] => {
    if (rgb) return [0, 1] as const;
    if (!perFrame && fixedRange) return [fixedRange[0], fixedRange[1]];
    let minimum = Infinity,
      maximum = -Infinity;
    if (values)
      for (const value of values)
        if (Number.isFinite(value)) {
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        }
    return Number.isFinite(minimum) ? ([minimum, maximum] as const) : ([0, 1] as const);
  }, [values, rgb, fixedRange, perFrame]);
  function resetIntensity() {
    setWindow(null);
    setField('perFrame', false);
  }
  return {
    zoom,
    setZoom,
    zoomIn: () => setZoom((value) => Math.min(MAX_ZOOM, Number((value * 1.25).toFixed(1)))),
    zoomOut: () => setZoom((value) => Math.max(MIN_ZOOM, Number((value / 1.25).toFixed(1)))),
    pan,
    setPan,
    setWindow,
    perFrame,
    manual: window !== null,
    setPerFrame: (next: boolean) => {
      setField('perFrame', next);
      setWindow(null);
    },
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
