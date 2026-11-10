import { useEffect, useMemo, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { WorkbenchIcon } from './WorkbenchIcon';
import { api, type Workspace } from './api';
import type { components } from './api.generated';
import { roiColor, roiMaskPaths } from './scene';

type Preview = components['schemas']['MaskPreview'];
export function MaskPreviewDialog({
  workspace: w,
  name,
  onClose,
  onEdit,
  busy,
}: {
  workspace: Workspace;
  name: string;
  onClose: () => void;
  onEdit: () => void;
  busy: boolean;
}) {
  const [axes, setAxes] = useState('');
  const [slice, setSlice] = useState(0);
  const [data, setData] = useState<Preview | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [cursor, setCursor] = useState([0, 0]);
  const [announce, setAnnounce] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError('');
    const query = new URLSearchParams({ name, slice_index: String(slice) });
    if (axes) query.set('axes', axes);
    void api<Preview>('/api/workspaces/' + w.id + '/roi-mask?' + query)
      .then((value) => {
        if (!cancelled) {
          setData(value);
          setCursor([0, 0]);
        }
      })
      .catch((failure) => {
        if (!cancelled) setError(String(failure));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [w.id, w.revision, name, axes, slice]);
  const paths = useMemo(() => roiMaskPaths(data?.runs ?? []), [data]);
  const pixels = useMemo(() => {
    if (!data) return new Uint16Array();
    const values = new Uint16Array(data.plane_shape[0] * data.plane_shape[1]);
    for (const [row, column, length, label] of data.runs)
      values.fill(
        label,
        row * data.plane_shape[1] + column,
        row * data.plane_shape[1] + column + length,
      );
    return values;
  }, [data]);
  const style = w.scene.roi_styles[name];
  const planes =
    data?.dims.flatMap((dim, i) => data.dims.slice(i + 1).map((other) => [dim, other])) ?? [];
  return (
    <Dialog.Root
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content className="export-dialog mask-preview-dialog" aria-busy={loading}>
          <Dialog.Title>Export mask preview</Dialog.Title>
          <Dialog.Description>
            {name} · Native voxels used for mask exports, including NIfTI. Colors are display
            settings; 0 is background.
          </Dialog.Description>
          <Dialog.Close className="drawer-close" aria-label="Close mask preview">
            <WorkbenchIcon name="close" />
          </Dialog.Close>
          {error && <p role="alert">{error}</p>}
          {data && !error && (
            <>
              <div className="mask-preview-controls">
                <label>
                  Plane
                  <select
                    aria-label="Mask preview plane"
                    value={axes || data.plane_dims.join(',')}
                    onChange={(event) => {
                      setAxes(event.target.value);
                      setSlice(0);
                    }}
                  >
                    {planes.map((pair) => (
                      <option key={pair.join(',')} value={pair.join(',')}>
                        {pair.join(' × ')}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  {data.slice_dim ? data.slice_dim + ' slice' : 'Slice'}{' '}
                  <output>
                    {slice + 1} / {data.slice_count}
                  </output>
                  <input
                    type="range"
                    aria-label="Mask preview slice"
                    min={0}
                    max={data.slice_count - 1}
                    value={slice}
                    disabled={data.slice_count <= 1}
                    onChange={(event) => setSlice(Number(event.target.value))}
                  />
                </label>
                <span>
                  {data.shape.join(' × ')} voxels · {data.dtype}
                </span>
              </div>
              <div className="mask-preview-viewport">
                <svg
                  role="img"
                  aria-label={'Native export mask for ' + name}
                  tabIndex={0}
                  aria-describedby="mask-preview-help"
                  viewBox={'0 0 ' + data.plane_shape[1] + ' ' + data.plane_shape[0]}
                  onKeyDown={(event) => {
                    const directions: Record<string, number[]> = {
                      ArrowUp: [-1, 0],
                      ArrowDown: [1, 0],
                      ArrowLeft: [0, -1],
                      ArrowRight: [0, 1],
                    };
                    const delta = directions[event.key];
                    if (!delta || loading) return;
                    event.preventDefault();
                    setAnnounce(true);
                    setCursor((previous) => [
                      Math.max(0, Math.min(data.plane_shape[0] - 1, previous[0] + delta[0])),
                      Math.max(0, Math.min(data.plane_shape[1] - 1, previous[1] + delta[1])),
                    ]);
                  }}
                  onPointerMove={(event) => {
                    if (loading) return;
                    const matrix = event.currentTarget.getScreenCTM()?.inverse();
                    if (!matrix) return;
                    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(
                      matrix,
                    );
                    if (
                      point.x < 0 ||
                      point.y < 0 ||
                      point.x >= data.plane_shape[1] ||
                      point.y >= data.plane_shape[0]
                    )
                      return;
                    setAnnounce(false);
                    setCursor([Math.floor(point.y), Math.floor(point.x)]);
                  }}
                >
                  <title>{name} · native label mask</title>
                  <desc>
                    Integer voxel boundaries without smoothing or resampling. {data.pixels} labelled
                    voxels in this slice.
                  </desc>
                  <rect width={data.plane_shape[1]} height={data.plane_shape[0]} fill="#111318" />
                  {paths.map(({ value, path }) => (
                    <path key={value} d={path} fill={roiColor(name, value, style)}>
                      <title>Label {value}</title>
                    </path>
                  ))}
                  <rect
                    x={cursor[1]}
                    y={cursor[0]}
                    width={1}
                    height={1}
                    fill="none"
                    stroke="#ffffff"
                    strokeWidth={1}
                    vectorEffect="non-scaling-stroke"
                    pointerEvents="none"
                    aria-hidden="true"
                  />
                </svg>
              </div>
              <div
                className="mask-preview-readout"
                role="status"
                aria-label="Mask label value"
                aria-live={announce ? 'polite' : 'off'}
              >
                Row {cursor[0]}, column {cursor[1]} · Label{' '}
                {pixels[cursor[0] * data.plane_shape[1] + cursor[1]] ?? 0}
              </div>
              <p id="mask-preview-help" className="control-note">
                Point to a voxel or focus the mask and use arrow keys to inspect label values.
              </p>
              <div className="roi-label-choices" aria-label="Preview labels">
                <span>Background: 0</span>
                {data.labels.map((value) => (
                  <span key={value}>
                    <i
                      className="roi-label-swatch"
                      style={{ background: roiColor(name, value, style) }}
                    />
                    Label {value}
                  </span>
                ))}
                <span>{data.pixels} labelled voxels in this slice</span>
              </div>
              <p className="control-note">
                This is the stored mask on its original grid. Registration may display it on a
                different grid in the image viewer.
              </p>
            </>
          )}
          {loading && <p role="status">Loading mask slice…</p>}
          <div className="button-row">
            <button
              className="secondary"
              disabled={busy || !w.editable || !w.rois.find((r) => r.name === name)?.enabled}
              onClick={onEdit}
            >
              Edit with brush
            </button>
            <Dialog.Close className="secondary">Close preview</Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
