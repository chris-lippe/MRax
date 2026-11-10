import {
  api,
  comparisonNavigation,
  rememberImages,
  type ComparisonImages,
  type Pixels,
  type Workspace,
} from './api';

export type NavigationFrame = {
  workspace: Workspace;
  values: Pixels | null;
  layers: Record<string, Pixels>;
  comparison?: ComparisonImages;
};

export function navigationContext(w: Workspace) {
  const otherDisplay = { ...w.display };
  delete otherDisplay.scene;
  delete otherDisplay.base_view;
  const { component, colormap, plot_axis } = otherDisplay;
  delete otherDisplay.component;
  delete otherDisplay.colormap;
  delete otherDisplay.plot_axis;
  const display = {
    component: component ?? 'auto',
    colormap: colormap ?? 'gray',
    plot_axis: plot_axis ?? 'auto',
    ...otherDisplay,
  };
  const comparison = w.scene.comparison.enabled
    ? { ...w.scene.comparison, frame: null, indices: null, plane: comparisonNavigation(w).plane }
    : w.scene.comparison;
  return JSON.stringify([
    w.id,
    w.revision,
    w.experiment,
    w.view,
    w.plane,
    w.active_map,
    display,
    w.visual_state,
    w.reconstruction_component,
    w.reconstruction_coil,
    w.unwrap_phase,
    w.kspace_mode,
    w.kspace_centering,
    w.selected_roi,
    w.fit_settings,
    { ...w.scene, comparison },
  ]);
}

export function navigationIndices(w: Workspace, changes: Record<string, number> = {}) {
  if (w.scene.comparison.enabled) {
    const cursor = comparisonNavigation(w);
    return { ...cursor.indices, frame: cursor.frame, ...changes };
  }
  const indices = { ...w.indices, ...changes };
  if (w.dimension_controls.some((control) => control.key === 't')) {
    if (changes.frame !== undefined) indices.t = changes.frame;
    indices.frame = indices.t ?? 0;
  }
  return indices;
}

export function navigationKey(w: Workspace, changes: Record<string, number> = {}) {
  return JSON.stringify([
    navigationContext(w),
    Object.entries(navigationIndices(w, changes))
      .filter(([, value]) => value !== 0)
      .sort(([a], [b]) => a.localeCompare(b)),
  ]);
}

export class NavigationCache {
  private readonly entries = new Map<string, { frame: NavigationFrame; bytes: number }>();
  private used = 0;
  constructor(private readonly limit = 64 * 1024 * 1024) {}
  get bytes() {
    return this.used;
  }
  get(key: string) {
    const value = this.entries.get(key);
    if (value) {
      this.entries.delete(key);
      this.entries.set(key, value);
    }
    return value?.frame;
  }
  set(frame: NavigationFrame) {
    const key = navigationKey(frame.workspace);
    const old = this.entries.get(key);
    if (old) {
      this.entries.delete(key);
      this.used -= old.bytes;
    }
    const arrays = [
      frame.values,
      ...Object.values(frame.layers),
      ...(frame.comparison?.values ?? []),
    ];
    const bytes =
      JSON.stringify(frame.workspace).length * 2 +
      arrays.reduce((total, array) => total + (array?.byteLength ?? 0), 0);
    if (bytes > this.limit) return;
    this.entries.set(key, { frame, bytes });
    this.used += bytes;
    while (this.used > this.limit || this.entries.size > 128) {
      const first = this.entries.keys().next().value!;
      this.used -= this.entries.get(first)!.bytes;
      this.entries.delete(first);
    }
  }
  clear(workspaceId: string) {
    for (const [key, value] of this.entries)
      if (value.frame.workspace.id === workspaceId) {
        this.entries.delete(key);
        this.used -= value.bytes;
      }
  }
}

export const navigationCache = new NavigationCache();

export function decodeNavigation(buffer: ArrayBuffer, cached?: NavigationFrame): NavigationFrame {
  if (buffer.byteLength < 8) throw new Error('Incomplete navigation frame.');
  const prefix = new DataView(buffer);
  const size = prefix.getUint32(4, true);
  if (prefix.getUint32(0, true) !== 1 || size > buffer.byteLength - 8)
    throw new Error('Invalid navigation frame.');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, size))) as {
    workspace: Workspace;
    planes: { id: string; dtype: string; shape: number[]; length: number }[];
    comparison: Omit<ComparisonImages, 'values'> | null;
    images_included?: boolean;
  };
  let offset = 8 + Math.ceil(size / 8) * 8;
  const planes: Record<string, Pixels> = {};
  for (const plane of header.planes) {
    const width = plane.dtype === 'float64-le' ? 8 : plane.dtype === 'float32-le' ? 4 : 0;
    if (
      !width ||
      !Number.isSafeInteger(plane.length) ||
      plane.length < 0 ||
      plane.shape.some((size) => !Number.isSafeInteger(size) || size < 0) ||
      plane.shape.reduce((count, size) => count * size, 1) !== plane.length ||
      offset + plane.length * width > buffer.byteLength ||
      plane.id in planes
    )
      throw new Error('Invalid navigation plane.');
    planes[plane.id] =
      width === 8
        ? new Float64Array(buffer, offset, plane.length)
        : new Float32Array(buffer, offset, plane.length);
    offset += Math.ceil((plane.length * width) / 8) * 8;
  }
  if (offset !== buffer.byteLength) throw new Error('Incomplete navigation planes.');
  if (header.images_included === false) {
    if (
      !cached ||
      header.planes.length ||
      navigationKey(header.workspace) !== navigationKey(cached.workspace) ||
      JSON.stringify(header.workspace.shape) !== JSON.stringify(cached.workspace.shape)
    )
      throw new Error('Cached navigation plane is no longer available.');
    return { ...cached, workspace: header.workspace };
  }
  const { image: values, left, right } = planes;
  const layers = Object.fromEntries(
    Object.entries(planes)
      .filter(([key]) => key.startsWith('layer:'))
      .map(([key, values]) => [key.slice(6), values]),
  );
  if (header.comparison && (!left || !right)) throw new Error('Incomplete comparison frame.');
  if (!header.comparison && !values) throw new Error('Missing image plane.');
  return {
    workspace: header.workspace,
    values: values ?? null,
    layers,
    ...(header.comparison
      ? { comparison: { ...header.comparison, values: [left, right] } as ComparisonImages }
      : {}),
  };
}

export function rememberNavigation(frame: NavigationFrame) {
  navigationCache.set(frame);
  rememberImages(frame.workspace, frame.values, frame.layers, frame.comparison);
}

export async function fetchNavigation(
  w: Workspace,
  changes: Record<string, number>,
  commit: boolean,
  signal?: AbortSignal,
) {
  const cached = commit ? navigationCache.get(navigationKey(w, changes)) : undefined;
  const response = await api<Response>(
    `/api/workspaces/${w.id}/navigate`,
    'POST',
    {
      revision: w.revision,
      view_revision: w.view_revision,
      indices: navigationIndices(w, changes),
      commit,
      include_images: !cached,
    },
    signal,
  );
  const frame = decodeNavigation(await response.arrayBuffer(), cached);
  if (frame.workspace.id !== w.id || frame.workspace.revision !== w.revision)
    throw new Error('The study changed while loading the image.');
  if (!signal?.aborted) rememberNavigation(frame);
  return frame;
}

export function nearbyIndices(w: Workspace, preferred?: { key: string; direction: number }) {
  const comparison = w.scene.comparison.enabled ? comparisonNavigation(w) : null;
  const controls = comparison
    ? [
        { key: 'frame', size: comparison.frameCount },
        ...Object.entries(comparison.dimensions)
          .filter(
            ([key]) =>
              !(
                comparison.plane === 'axial'
                  ? ['x', 'y']
                  : comparison.plane === 'coronal'
                    ? ['x', 'z']
                    : ['y', 'z']
              ).includes(key),
          )
          .map(([key, size]) => ({ key, size })),
      ]
    : w.dimension_controls.map((control) => ({
        key: String(control.key),
        size: Number(control.size),
      }));
  const cursor = navigationIndices(w);
  controls.sort((a, b) => Number(b.key === preferred?.key) - Number(a.key === preferred?.key));
  return controls
    .flatMap(({ key, size }) => {
      const direction = key === preferred?.key && (preferred?.direction ?? 1) < 0 ? -1 : 1;
      return [direction, -direction].flatMap((delta) => {
        const value = (cursor[key] ?? 0) + delta;
        return value >= 0 && value < size ? [{ [key]: value }] : [];
      });
    })
    .slice(0, 4);
}
