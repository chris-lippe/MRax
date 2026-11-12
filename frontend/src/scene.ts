import type { components } from './api.generated';
import type { Workspace } from './api';

export type Layer = components['schemas']['OverlayLayer'];
export type LayerInfo = components['schemas']['OverlayInfo'];
export type RoiAppearance = components['schemas']['RoiAppearance'];
export type ScenePreview = {
  layers: Record<string, number>;
  roiOpacity?: number;
  rois?: Record<string, number>;
  labels?: Record<string, Record<string, number>>;
};

const palette = ['#2dd4bf', '#fbbf24', '#f472b6', '#a78bfa', '#38bdf8', '#fb923c'];
export function nextRoiName(rois: readonly { name: string }[]) {
  const names = new Set(rois.map((roi) => roi.name));
  let number = rois.length + 1;
  while (names.has('ROI ' + number)) number++;
  return 'ROI ' + number;
}

export function roiColor(name: string, label = 1, appearance?: RoiAppearance) {
  const custom = appearance?.label_colors?.[String(label)] ?? appearance?.color;
  if (custom) return custom;
  let hash = 0;
  for (const character of name) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  const index = (hash + label - 1) % palette.length;
  const cycle = Math.floor((label - 1) / palette.length);
  if (!cycle) return palette[index];
  const hue = (index * 60 + cycle * 137.508) % 360;
  const channel = (offset: number) => {
    const k = (offset + hue / 30) % 12;
    return Math.round(255 * (0.6 - 0.272 * Math.max(-1, Math.min(k - 3, 9 - k, 1))))
      .toString(16)
      .padStart(2, '0');
  };
  return '#' + channel(0) + channel(8) + channel(4);
}

function maskBoundary(rows: Map<number, [number, number][]>) {
  const edges: string[] = [];
  for (const spans of rows.values()) spans.sort((a, b) => a[0] - b[0]);
  const horizontal = (y: number, start: number, end: number, neighbours: [number, number][]) => {
    let cursor = start;
    for (const [left, right] of neighbours) {
      if (right <= cursor) continue;
      if (left >= end) break;
      if (left > cursor) edges.push(`M${cursor} ${y}H${Math.min(left, end)}`);
      cursor = Math.max(cursor, right);
      if (cursor >= end) return;
    }
    if (cursor < end) edges.push(`M${cursor} ${y}H${end}`);
  };
  for (const [y, spans] of rows) {
    for (const [start, end] of spans) {
      edges.push(`M${start} ${y}v1M${end} ${y}v1`);
      horizontal(y, start, end, rows.get(y - 1) ?? []);
      horizontal(y + 1, start, end, rows.get(y + 1) ?? []);
    }
  }
  return edges.join('');
}

export function roiMaskPaths(runs: number[][], outlines = false) {
  const labels = new Map<number, string[]>();
  const boundaries = outlines ? new Map<number, Map<number, [number, number][]>>() : null;
  for (const [y, x, length, value] of runs) {
    let paths = labels.get(value);
    if (!paths) {
      paths = [];
      labels.set(value, paths);
    }
    paths.push(`M${x} ${y}h${length}v1H${x}Z`);
    if (boundaries) {
      let rows = boundaries.get(value);
      if (!rows) {
        rows = new Map();
        boundaries.set(value, rows);
      }
      const spans = rows.get(y) ?? [];
      spans.push([x, x + length]);
      rows.set(y, spans);
    }
  }
  return [...labels].map(([value, paths]) => ({
    value,
    path: paths.join(''),
    outline: boundaries ? maskBoundary(boundaries.get(value)!) : undefined,
  }));
}

export function layerInput(layer: LayerInfo): Layer {
  return {
    id: layer.id,
    experiment: layer.experiment,
    kind: layer.kind,
    map_name: layer.map_name,
    registration_key: layer.registration_key,
    frame: layer.frame,
    component: layer.component,
    colormap: layer.colormap,
    opacity: layer.opacity,
    visible: layer.visible,
    window: layer.window,
  };
}

export function fitRois(workspace: Workspace) {
  return workspace.rois
    .filter(
      (r) =>
        r.enabled &&
        r.pixels > 0 &&
        (workspace.scene.fit_rois === null || workspace.scene.fit_rois.includes(r.name)),
    )
    .map((r) => r.name);
}
