import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  nativePoint,
  rectangle,
  type Bounds,
  type Options,
  type Pixels,
  type Tool,
  type Workspace,
} from './api';

import { MIN_ZOOM, MAX_ZOOM, type ImageDisplay } from './useImageDisplay';
import { useContextMenu } from './ContextMenu';
import { roiColor, roiMaskPaths, type ScenePreview } from './scene';
import { palettes, writeColor } from './imageColors';
import { viewLabel } from './viewing';
import { IntensityControls } from './IntensityControls';
import { imageNavigationHelp, useImageNavigation } from './useImageNavigation';

type Point = [number, number];
export function Viewer({
  workspace,
  values,
  layerValues,
  scenePreview,
  drawing,
  tool,
  busy,
  brushSize,
  label,
  onRectangle,
  onShape,
  display,
  editingEnabled,
  controls,
}: {
  workspace: Workspace;
  display: ImageDisplay;
  editingEnabled: boolean;
  controls: (children: ReactNode) => ReactNode;
  values: Pixels;
  layerValues: Record<string, Pixels>;
  scenePreview: ScenePreview;
  drawing: boolean;
  tool: Tool;
  busy: boolean;
  brushSize: number;
  label: number;
  onRectangle: (bounds: Bounds) => void;
  onShape: (options: Options) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const masks = useMemo(
    () => workspace.rois.map((roi) => ({ roi, paths: roiMaskPaths(roi.runs) })),
    [workspace.rois],
  );
  useImageNavigation(viewport, display);
  const [available, setAvailable] = useState({ width: 680, height: 400 });
  const start = useRef<Point | null>(null),
    points = useRef<Point[]>([]),
    panStart = useRef<Point | null>(null);
  const [draft, setDraft] = useState<Point[]>([]),
    [pixel, setPixel] = useState('Pixel value: —');
  const { contrast, zoom, pan, setPan, level } = display;
  const menu = useContextMenu('Image actions', [
    { label: 'Zoom in', run: display.zoomIn, disabled: zoom >= MAX_ZOOM },
    { label: 'Zoom out', run: display.zoomOut, disabled: zoom <= MIN_ZOOM },
    { label: 'Reset display', run: display.reset },
  ]);
  const closeMenu = menu.close;
  useEffect(() => closeMenu(false), [workspace.id, workspace.view_revision, closeMenu]);
  const [rows, columns] = workspace.shape;
  const mapUnit = String(
    workspace.maps.find((map) => map.name === workspace.active_map)?.unit ?? workspace.unit ?? '',
  );
  const aspect = (columns * workspace.spacing[1]) / (rows * workspace.spacing[0]);
  const imageWidth = Math.min(available.width, available.height * aspect);
  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setAvailable({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const selectedGeometry = workspace.rois.find((roi) => roi.name === workspace.selected_roi);
  const geometricKind = String(selectedGeometry?.metadata.kind ?? '');
  const vertexMode = ['polygon', 'lasso'].includes(geometricKind);
  const nativeBounds = (selectedGeometry?.metadata.bounds ?? []) as number[];
  const nodes: Point[] = vertexMode
    ? ((selectedGeometry?.metadata.points ?? []) as Point[])
    : nativeBounds.length === 4
      ? [
          [nativeBounds[0], nativeBounds[1]],
          [nativeBounds[0] + nativeBounds[2], nativeBounds[1] + nativeBounds[3]],
        ]
      : [];
  const handles =
    workspace.editable &&
    editingEnabled &&
    !drawing &&
    !busy &&
    selectedGeometry?.runs.length &&
    !selectedGeometry.metadata.consolidated &&
    ['rect', 'ellipse', 'polygon', 'lasso', 'regular_polygon'].includes(geometricKind) &&
    JSON.stringify(selectedGeometry.metadata.plane_dims) === JSON.stringify(workspace.plane_dims) &&
    (vertexMode || Number(selectedGeometry.metadata.angle ?? 0) === 0)
      ? nodes
      : [];
  const editing = useRef<{ index: number; points: Point[] } | null>(null);
  const [editNodes, setEditNodes] = useState<Point[] | null>(null);
  useEffect(() => {
    start.current = null;
    points.current = [];
    setDraft([]);
    editing.current = null;
    setEditNodes(null);
  }, [workspace.id, workspace.view_revision, drawing, tool]);
  const canvasPixels = useRef<ImageData | null>(null);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const pixels =
      canvasPixels.current?.width === columns && canvasPixels.current.height === rows
        ? canvasPixels.current
        : context.createImageData(columns, rows);
    canvasPixels.current = pixels;
    const palette = palettes[String(workspace.display.colormap)] ?? palettes.gray;
    const scale = contrast / Math.max(level.width, Number.EPSILON);
    for (let index = 0; index < rows * columns; index++) {
      const offset = index * 4;
      if (workspace.channels > 1) {
        for (let channel = 0; channel < 3; channel++)
          pixels.data[offset + channel] = Math.round(
            Math.max(
              0,
              Math.min(
                1,
                (values[index * workspace.channels + channel] - level.center) * scale + 0.5,
              ),
            ) * 255,
          );
      } else if (Number.isFinite(values[index]))
        writeColor(pixels.data, offset, (values[index] - level.center) * scale + 0.5, palette);
      else {
        pixels.data[offset] = 192;
        pixels.data[offset + 1] = 64;
        pixels.data[offset + 2] = 192;
      }
      pixels.data[offset + 3] = 255;
    }
    for (const layer of workspace.scene.layers) {
      const data = layerValues[layer.id];
      if (!layer.visible || !data || layer.unavailable) continue;
      let minimum = Infinity,
        maximum = -Infinity;
      if (!layer.window)
        for (const value of data)
          if (Number.isFinite(value)) {
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
          }
      const window = layer.window ?? {
        center: (minimum + maximum) / 2,
        width: maximum - minimum || 1,
      };
      const alpha = scenePreview.layers[layer.id] ?? layer.opacity;
      const overlayPalette = palettes[layer.colormap] ?? palettes.gray;
      const scale = 1 / window.width;
      for (let index = 0; index < data.length; index++) {
        if (!Number.isFinite(data[index])) continue;
        writeColor(
          pixels.data,
          index * 4,
          (data[index] - window.center) * scale + 0.5,
          overlayPalette,
          alpha,
        );
      }
    }
    context.putImageData(pixels, 0, 0);
  }, [
    values,
    contrast,
    rows,
    columns,
    workspace.channels,
    workspace.display.colormap,
    level.center,
    level.width,
    layerValues,
    workspace.scene.layers,
    scenePreview.layers,
  ]);
  const finishPolygon = () => {
    if (points.current.length >= 3) onShape({ tool, points: points.current, label });
    points.current = [];
    setDraft([]);
  };
  return (
    <>
      {controls(
        <IntensityControls
          values={values}
          range={display.range}
          level={{ center: level.center, width: level.width / contrast }}
          contrast={contrast}
          onChange={(next, gain) => {
            if (gain !== undefined) display.setContrast(gain);
            else display.setWindow({ ...next, width: next.width * contrast });
          }}
          reset={display.resetIntensity}
        />,
      )}
      <div className="image-surround">
        <div className="image-label">
          <span>{workspace.experiment}</span>
          <span>
            {viewLabel(workspace, workspace.view)} · {workspace.plane}
          </span>
        </div>
        {workspace.view === 'Map' && (
          <div className="map-scale" role="group" aria-label="Map scale">
            <span>
              {workspace.active_map} {mapUnit && `(${mapUnit})`}
            </span>
            <span>{(level.center - level.width / contrast / 2).toPrecision(4)}</span>
            <span
              className="map-scale-gradient"
              aria-hidden="true"
              style={{
                background: `linear-gradient(to right, ${(palettes[String(workspace.display.colormap ?? 'gray')] ?? palettes.gray).map((color) => `rgb(${color.join(',')})`).join(',')})`,
              }}
            />
            <span>{(level.center + level.width / contrast / 2).toPrecision(4)}</span>
          </div>
        )}
        <div
          className="image-viewport"
          ref={viewport}
          tabIndex={0}
          aria-label="Image display"
          aria-description={imageNavigationHelp}
          title={imageNavigationHelp}
          onContextMenu={menu.onContextMenu}
          onKeyDown={menu.onKeyDown}
        >
          <div
            className={`image-surface ${drawing ? 'drawing' : ''}`}
            style={{
              aspectRatio: String(aspect),
              width: `${imageWidth}px`,
              height: `${imageWidth / aspect}px`,
              transform: `translate(${pan[0]}px, ${pan[1]}px) scale(${zoom})`,
            }}
          >
            <canvas
              ref={canvas}
              width={columns}
              height={rows}
              role="img"
              aria-label={`${rows} by ${columns} quantitative image, frame ${workspace.frame + 1}, slice ${workspace.z + 1}`}
            />
            <svg
              className="image-overlay"
              data-testid="roi-surface"
              viewBox={`0 0 ${columns} ${rows}`}
              preserveAspectRatio="none"
              aria-label="ROI drawing surface"
              tabIndex={drawing && tool === 'polygon' ? 0 : -1}
              onKeyDown={(e) => {
                if (e.key === 'Enter') finishPolygon();
                if (e.key === 'Escape') {
                  points.current = [];
                  setDraft([]);
                }
              }}
              onDoubleClick={() => {
                if (tool === 'polygon' && drawing) finishPolygon();
                else if (!drawing) display.resetPosition();
              }}
              onPointerDown={(e) => {
                if (busy || (e.button !== 0 && e.button !== 1)) return;
                if (!drawing || tool === 'pan' || e.altKey || e.button === 1) {
                  e.preventDefault();
                  panStart.current = [e.clientX - pan[0], e.clientY - pan[1]];
                  e.currentTarget.setPointerCapture(e.pointerId);
                  return;
                }
                if (!drawing || !workspace.editable) return;
                const point = nativePoint(
                  e.clientX,
                  e.clientY,
                  e.currentTarget.getBoundingClientRect(),
                  workspace.shape,
                );
                if (tool === 'polygon') {
                  points.current.push(point);
                  setDraft([...points.current]);
                  e.currentTarget.focus();
                  return;
                }
                if (tool === 'wand') {
                  onShape({ tool, seed: point, label });
                  return;
                }
                e.currentTarget.setPointerCapture(e.pointerId);
                start.current = point;
                points.current = [point];
                setDraft([point]);
              }}
              onPointerMove={(e) => {
                if (panStart.current) {
                  setPan([e.clientX - panStart.current[0], e.clientY - panStart.current[1]]);
                  return;
                }
                const point = nativePoint(
                  e.clientX,
                  e.clientY,
                  e.currentTarget.getBoundingClientRect(),
                  workspace.shape,
                );
                if (editing.current) {
                  const next = [...editing.current.points];
                  next[editing.current.index] = point;
                  editing.current.points = next;
                  setEditNodes(next);
                  return;
                }
                const column = Math.min(columns - 1, Math.floor(point[0])),
                  row = Math.min(rows - 1, Math.floor(point[1]));
                const value = values[(row * columns + column) * workspace.channels];
                setPixel(
                  `Row ${row}, column ${column} · ${Number.isFinite(value) ? value.toPrecision(6) : 'Missing / nonfinite'}${workspace.unit ? ` ${workspace.unit}` : ''}`,
                );
                if (start.current) {
                  if (['lasso', 'brush', 'erase'].includes(tool)) {
                    if (points.current.length < 4096) points.current.push(point);
                    setDraft([...points.current]);
                  } else setDraft([start.current, point]);
                }
              }}
              onPointerCancel={() => {
                start.current = null;
                panStart.current = null;
                points.current = [];
                setDraft([]);
                editing.current = null;
                setEditNodes(null);
              }}
              onPointerUp={(e) => {
                panStart.current = null;
                if (editing.current && selectedGeometry) {
                  const updated = editing.current.points;
                  editing.current = null;
                  setEditNodes(null);
                  onShape({
                    edit: true,
                    name: selectedGeometry.name,
                    tool: geometricKind,
                    ...(vertexMode
                      ? { points: updated }
                      : { bounds: rectangle(updated[0], updated[1]) }),
                    angle: Number(selectedGeometry.metadata.angle ?? 0),
                    sides: ((selectedGeometry.metadata.points ?? []) as Point[]).length || 6,
                    label: selectedGeometry.labels[0] ?? 1,
                  });
                  return;
                }
                if (!start.current) return;
                const end = nativePoint(
                    e.clientX,
                    e.clientY,
                    e.currentTarget.getBoundingClientRect(),
                    workspace.shape,
                  ),
                  bounds = rectangle(start.current, end);
                const stroke = [...points.current, end];
                start.current = null;
                points.current = [];
                setDraft([]);
                if (tool === 'brush' || tool === 'erase')
                  onShape({
                    tool: 'brush',
                    points: stroke,
                    size: brushSize,
                    label,
                    erase: tool === 'erase',
                  });
                else if (tool === 'lasso') {
                  if (stroke.length >= 3) onShape({ tool, points: stroke, label });
                } else if (bounds[2] >= 0.5 && bounds[3] >= 0.5) {
                  if (tool === 'rect' && label === 1) onRectangle(bounds);
                  else onShape({ tool, bounds, label });
                }
              }}
            >
              {masks
                .filter(({ roi }) => roi.enabled && (editingEnabled || roi.visible))
                .map(({ roi, paths }) => (
                  <g
                    key={roi.name}
                    className="roi-mask"
                    style={{
                      fill: roiColor(roi.name, 1, workspace.scene.roi_styles[roi.name]),
                      opacity:
                        (scenePreview.rois?.[roi.name] ??
                          workspace.scene.roi_styles[roi.name]?.opacity ??
                          scenePreview.roiOpacity ??
                          workspace.scene.roi_opacity) *
                        (editingEnabled && !roi.visible ? 0.55 : 1),
                    }}
                  >
                    {paths.map(({ value, path }) => (
                      <path
                        key={value}
                        d={path}
                        role="img"
                        aria-label={`${roi.name} · label ${value}`}
                        style={{
                          fill: roiColor(roi.name, value, workspace.scene.roi_styles[roi.name]),
                        }}
                      >
                        <title>
                          {roi.name} · label {value}
                        </title>
                      </path>
                    ))}
                  </g>
                ))}
              {(editNodes ?? handles).map((node, index) => (
                <circle
                  key={`handle-${index}`}
                  className="roi-handle"
                  data-testid={`roi-handle-${index}`}
                  cx={node[0]}
                  cy={node[1]}
                  r={0.55}
                  vectorEffect="non-scaling-stroke"
                  aria-label={`ROI ${vertexMode ? 'vertex' : 'corner'} ${index + 1}`}
                  onPointerDown={(event) => {
                    if (busy || event.button !== 0) return;
                    event.stopPropagation();
                    event.currentTarget.setPointerCapture(event.pointerId);
                    editing.current = { index, points: nodes.map((p) => [...p] as Point) };
                    setEditNodes(nodes);
                  }}
                >
                  <title>Drag to edit {selectedGeometry?.name}</title>
                </circle>
              ))}
              {draft.length > 1 &&
                (['rect', 'ellipse', 'regular_polygon'].includes(tool) ? (
                  (() => {
                    const b = rectangle(draft[0], draft[draft.length - 1]);
                    return tool === 'ellipse' ? (
                      <ellipse
                        className="roi-draft"
                        cx={b[0] + b[2] / 2}
                        cy={b[1] + b[3] / 2}
                        rx={b[2] / 2}
                        ry={b[3] / 2}
                        vectorEffect="non-scaling-stroke"
                      />
                    ) : (
                      <rect
                        className="roi-draft"
                        x={b[0]}
                        y={b[1]}
                        width={b[2]}
                        height={b[3]}
                        vectorEffect="non-scaling-stroke"
                      />
                    );
                  })()
                ) : (
                  <polyline
                    className="roi-draft"
                    points={draft.map((p) => p.join(',')).join(' ')}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
            </svg>
            <span className="orientation top">
              {workspace.plane_axis_labels?.[1] ?? workspace.plane_dims[1]}
            </span>
            <span className="orientation left">
              {workspace.plane_axis_labels?.[0] ?? workspace.plane_dims[0]}
            </span>
          </div>
        </div>
        <div className="image-readout">
          <span>{pixel}</span>
          <span>
            {workspace.spacing.join(' × ')} {workspace.spacing_unit} / voxel
          </span>
        </div>
      </div>
      {menu.menu}
      {drawing && tool === 'polygon' && (
        <p className="drawing-hint">
          Click vertices, then press Enter or double click to close. Escape cancels.
        </p>
      )}
    </>
  );
}
