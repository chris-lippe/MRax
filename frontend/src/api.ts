import type { components } from './api.generated';
import { ImagePlaneCache } from './imageCache';

export type Workspace = components['schemas']['WorkspaceInfo'];
export type PipelineStep = components['schemas']['PipelineStep'];
export type ComparisonSettings = components['schemas']['ComparisonSettings'];
export type ComparisonSource = components['schemas']['ComparisonSource'];
export type Job = components['schemas']['JobInfo'];
export type TimecourseResult = components['schemas']['TimecourseResult'];
export type Bounds = [number, number, number, number];
export type Pixels = Float32Array | Float64Array;
export type Options = Record<string, unknown>;
export type Tool =
  | 'rect'
  | 'ellipse'
  | 'regular_polygon'
  | 'polygon'
  | 'lasso'
  | 'brush'
  | 'erase'
  | 'wand'
  | 'otsu'
  | 'pan';

export function originalComparison(workspace: Workspace): ComparisonSettings {
  return {
    ...workspace.scene.comparison,
    enabled: true,
    frame: null,
    plane: null,
    indices: null,
    position: 0.5,
    linked_window: true,
    left: {
      experiment: workspace.experiment!,
      kind: 'original',
      component: 'auto',
      colormap: 'gray',
      frame: null,
    },
    right: {
      experiment: workspace.experiment!,
      kind: 'processed',
      component: 'auto',
      colormap: 'gray',
      frame: null,
    },
  };
}

export function comparisonNavigation(workspace: Workspace, settings = workspace.scene.comparison) {
  const sources = [settings.left, settings.right];
  const choices = sources.map((source) =>
    workspace.scene.comparison_sources.find((choice) =>
      ['experiment', 'pipeline_id', 'kind', 'map_name', 'registration_key'].every(
        (key) =>
          (choice[key as keyof typeof choice] ?? null) ===
          (source[key as keyof typeof source] ?? null),
      ),
    ),
  );
  const dimensions = Object.fromEntries(
    [...new Set(choices.flatMap((choice) => Object.keys(choice?.dimensions ?? {})))].map((dim) => [
      dim,
      Math.min(...choices.map((choice) => choice?.dimensions[dim] ?? 1)),
    ]),
  );
  const following = choices.flatMap((choice, index) =>
    sources[index].frame === null && (choice?.frames ?? 1) > 1 ? [choice!.frames] : [],
  );
  const frameCount = following.length ? Math.min(...following) : 1;
  const volume = choices.every((choice) => choice && 'z' in choice.dimensions);
  return {
    dimensions,
    frameCount,
    frame: Math.min(settings.frame ?? workspace.frame, frameCount - 1),
    plane: (volume ? (settings.plane ?? workspace.plane) : 'axial') as NonNullable<
      ComparisonSettings['plane']
    >,
    volume,
    indices: Object.fromEntries(
      Object.entries(dimensions).map(([dim, size]) => [
        dim,
        Math.min((settings.indices ?? workspace.indices)[dim] ?? 0, size - 1),
      ]),
    ),
  };
}

let token: string | undefined;
export async function api<T>(
  path: string,
  method = 'GET',
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  if (method !== 'GET' && !token) {
    token = (await api<{ token: string }>('/api/session')).token;
  }
  const response = await fetch(path, {
    method,
    signal,
    headers: {
      ...(token ? { 'x-mrax-token': token } : {}),
      'Content-Type': 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: 'MRax could not complete this operation.' }));
    const validation = Array.isArray(error.detail)
      ? error.detail
          .slice(0, 4)
          .map((item: { msg?: unknown }) =>
            typeof item.msg === 'string' ? item.msg.slice(0, 500) : '',
          )
          .filter(Boolean)
          .join('. ')
      : '';
    throw new Error(
      typeof error.detail === 'string'
        ? error.detail
        : validation || 'Check the input values and try again.',
    );
  }
  if (response.headers.get('content-type')?.includes('application/json')) return response.json();
  return response as T;
}

const imageCache = new ImagePlaneCache();
type ComparisonPlane = {
  shape: [number, number];
  spacing: [number, number];
  plane_dims: [string, string];
  plane_axis_labels?: [string, string];
  plane_axis_directions?: [number, number];
  range: [number, number];
  frame: number;
  unit: string;
};
type ComparisonMetadata = {
  planes: [ComparisonPlane, ComparisonPlane];
  geometry_compatible: boolean;
  geometry_message: string | null;
  same_units: boolean;
};
export type ComparisonImages = ComparisonMetadata & { values: [Pixels, Pixels] };
const comparisonMetadata = new Map<string, ComparisonMetadata>();

export function comparisonKey(workspace: Workspace) {
  const { left, right } = workspace.scene.comparison;
  const navigation = comparisonNavigation(workspace);
  return JSON.stringify([
    workspace.id,
    'comparison',
    workspace.revision,
    workspace.experiment,
    navigation.plane,
    navigation.indices,
    navigation.frame,
    ...[left, right].map((source) => ({ ...source, colormap: undefined, window: undefined })),
  ]);
}

export async function comparisonImages(
  workspace: Workspace,
  signal: AbortSignal,
): Promise<ComparisonImages> {
  const key = comparisonKey(workspace);
  const cached = imageCache.get(key);
  const metadata = comparisonMetadata.get(key);
  let values: Pixels;
  let info: ComparisonMetadata;
  if (cached && metadata) {
    values = cached;
    info = metadata;
  } else {
    const response = await api<Response>(
      `/api/workspaces/${workspace.id}/comparison`,
      'GET',
      undefined,
      signal,
    );
    if (Number(response.headers.get('x-view-revision')) !== workspace.view_revision)
      throw new Error('The comparison changed while loading.');
    info = JSON.parse(response.headers.get('x-comparison') ?? '{}') as ComparisonMetadata;
    if (
      !Array.isArray(info.planes) ||
      info.planes.length !== 2 ||
      info.planes.some(
        (plane) =>
          plane.shape.length !== 2 ||
          plane.shape.some((size) => !Number.isSafeInteger(size) || size < 1),
      )
    )
      throw new Error('Invalid comparison dimensions.');
    values = new Float64Array(await response.arrayBuffer());
    if (
      info.planes.reduce((count, plane) => count + plane.shape[0] * plane.shape[1], 0) !==
      values.length
    )
      throw new Error('Incomplete comparison image.');
    if (!signal.aborted) {
      imageCache.set(key, values);
      comparisonMetadata.set(key, info);
      while (comparisonMetadata.size > 128)
        comparisonMetadata.delete(comparisonMetadata.keys().next().value!);
    }
  }
  const length = info.planes[0].shape[0] * info.planes[0].shape[1];
  return { ...info, values: [values.subarray(0, length), values.subarray(length)] };
}

export function clearImageCache(workspaceId: string) {
  imageCache.clearWorkspace(workspaceId);
  for (const key of comparisonMetadata.keys())
    if (JSON.parse(key)[0] === workspaceId) comparisonMetadata.delete(key);
}

export function imageKey(workspace: Workspace, layer?: string, withRevision = true) {
  const selectedLayer = layer ? workspace.scene.layers.find((entry) => entry.id === layer) : null;
  return JSON.stringify([
    workspace.id,
    withRevision ? workspace.revision : null,
    workspace.experiment,
    workspace.view,
    workspace.plane,
    Object.entries(workspace.indices)
      .filter(([, value]) => value !== 0)
      .sort(([a], [b]) => a.localeCompare(b)),
    workspace.frame,
    workspace.z,
    workspace.active_map,
    workspace.display.component ?? 'auto',
    workspace.visual_state,
    workspace.reconstruction_component,
    workspace.reconstruction_coil,
    workspace.unwrap_phase,
    workspace.kspace_mode,
    workspace.kspace_centering,
    layer,
    selectedLayer
      ? {
          ...selectedLayer,
          opacity: undefined,
          window: undefined,
          colormap: undefined,
          visible: undefined,
          label: undefined,
        }
      : null,
  ]);
}

export function retainImagePlanes(previous: Workspace, next: Workspace, roisChanged = false) {
  for (const layer of [undefined, ...next.scene.layers.map((entry) => entry.id)]) {
    if (!layer && roisChanged && next.visual_state.mask_enabled) continue;
    if (imageKey(previous, layer, false) !== imageKey(next, layer, false)) continue;
    const values = imageCache.get(imageKey(previous, layer));
    if (values) imageCache.set(imageKey(next, layer), values);
  }
}

export function rememberImages(
  workspace: Workspace,
  values: Pixels | null,
  layers: Record<string, Pixels>,
  comparison?: ComparisonImages,
) {
  if (values) imageCache.set(imageKey(workspace), values);
  for (const [id, data] of Object.entries(layers)) imageCache.set(imageKey(workspace, id), data);
  if (comparison) {
    const key = comparisonKey(workspace);
    if (imageCache.get(key) && comparisonMetadata.has(key)) return;
    const joined = new Float64Array(comparison.values[0].length + comparison.values[1].length);
    joined.set(comparison.values[0]);
    joined.set(comparison.values[1], comparison.values[0].length);
    imageCache.set(key, joined);
    comparisonMetadata.set(key, {
      planes: comparison.planes,
      geometry_compatible: comparison.geometry_compatible,
      geometry_message: comparison.geometry_message,
      same_units: comparison.same_units,
    });
    while (comparisonMetadata.size > 128)
      comparisonMetadata.delete(comparisonMetadata.keys().next().value!);
  }
}

export async function image(
  workspace: Workspace,
  layer?: string,
  signal?: AbortSignal,
): Promise<Pixels> {
  const key = imageKey(workspace, layer);
  const cached = imageCache.get(key);
  if (cached) {
    return cached;
  }
  const response = await api<Response>(
    '/api/workspaces/' +
      workspace.id +
      '/slice' +
      (layer ? '?layer=' + encodeURIComponent(layer) : ''),
    'GET',
    undefined,
    signal,
  );
  if (Number(response.headers.get('x-view-revision')) !== workspace.view_revision)
    throw new Error('The view changed. Select the frame again.');
  const buffer = await response.arrayBuffer();
  const values =
    response.headers.get('x-dtype') === 'float64-le'
      ? new Float64Array(buffer)
      : new Float32Array(buffer);
  if (values.length !== workspace.shape[0] * workspace.shape[1] * (layer ? 1 : workspace.channels))
    throw new Error('The image shape does not match the view.');
  if (!signal?.aborted) imageCache.set(key, values);
  return values;
}

export function nativePoint(
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>,
  shape: number[],
): [number, number] {
  return [
    Math.max(0, Math.min(shape[1], ((clientX - rect.left) / rect.width) * shape[1])),
    Math.max(0, Math.min(shape[0], ((clientY - rect.top) / rect.height) * shape[0])),
  ];
}

export function rectangle(start: [number, number], end: [number, number]): Bounds {
  return [
    Math.min(start[0], end[0]),
    Math.min(start[1], end[1]),
    Math.abs(start[0] - end[0]),
    Math.abs(start[1] - end[1]),
  ];
}
