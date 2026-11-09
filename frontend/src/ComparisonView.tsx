import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import {
  comparisonImages,
  comparisonKey,
  comparisonNavigation,
  type ComparisonImages,
  type ComparisonSettings,
  type ComparisonSource,
  type Options,
  type Pixels,
  type Workspace,
} from './api';
import { palettes, writeColor } from './imageColors';
import { WorkbenchIcon } from './WorkbenchIcon';
import { useContextMenu } from './ContextMenu';
import { MAX_ZOOM, MIN_ZOOM, type ImageDisplay } from './useImageDisplay';
import { IntensityControls, type IntensityWindow } from './IntensityControls';
import { OpacityControl } from './OpacityControl';
import { imageNavigationHelp, useImageNavigation } from './useImageNavigation';

const sourceKey = (
  source: Pick<ComparisonSource, 'experiment' | 'kind' | 'map_name' | 'registration_key'>,
) =>
  JSON.stringify([
    source.experiment,
    source.kind,
    source.map_name ?? null,
    source.registration_key ?? null,
  ]);

function ComparisonControls({
  workspace,
  busy,
  change,
  overlayAvailable,
  previewOpacity,
  commitOpacity,
}: {
  workspace: Workspace;
  busy: boolean;
  change: (command: Options) => Promise<void>;
  overlayAvailable: boolean;
  previewOpacity: (value: number) => void;
  commitOpacity: (value: number) => Promise<void>;
}) {
  const settings = workspace.scene.comparison;
  const update = (patch: Partial<ComparisonSettings>) => {
    const next = { ...settings, ...patch };
    const navigation = comparisonNavigation(workspace, next);
    void change({
      scene: {
        comparison: {
          ...next,
          frame: navigation.frame,
          plane: navigation.plane,
          indices: navigation.indices,
        },
      },
    });
  };
  const sources = workspace.scene.comparison_sources;
  function sourceControl(side: 'left' | 'right') {
    const source = settings[side];
    return (
      <label>
        {side === 'left' ? 'Image A' : 'Image B'}
        <select
          aria-label={side === 'left' ? 'Image A' : 'Image B'}
          value={sourceKey(source)}
          onChange={(event) => {
            const selected = sources.find((item) => sourceKey(item) === event.target.value)!;
            update({
              [side]: {
                experiment: selected.experiment,
                kind: selected.kind,
                map_name: selected.map_name,
                registration_key: selected.registration_key,
                component: source.component,
                colormap: source.colormap,
                frame: null,
              },
            });
          }}
        >
          {[...new Set(sources.map((item) => item.experiment))].map((experiment) => (
            <optgroup key={experiment} label={experiment}>
              {sources
                .filter((item) => item.experiment === experiment)
                .map((item) => (
                  <option key={sourceKey(item)} value={sourceKey(item)}>
                    {item.label}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
      </label>
    );
  }
  return (
    <fieldset className="comparison-controls" disabled={busy} aria-label="Image pair settings">
      <div className="comparison-sources">
        {sourceControl('left')}
        <button
          className="icon-button comparison-swap"
          title="Swap images"
          aria-label="Swap images"
          onClick={() => update({ left: settings.right, right: settings.left })}
        >
          <WorkbenchIcon name="swap" />
        </button>
        {sourceControl('right')}
      </div>
      <div className="comparison-options">
        {settings.mode === 'split' && (
          <div className="segmented-control" role="group" aria-label="Split layout">
            {(['swipe', 'side-by-side'] as const).map((layout) => (
              <button
                key={layout}
                aria-pressed={settings.layout === layout}
                onClick={() => update({ layout })}
              >
                {layout === 'swipe' ? 'Swipe' : 'Side by side'}
              </button>
            ))}
          </div>
        )}
        {settings.mode === 'overlay' && (
          <div className="comparison-overlay-opacity">
            <OpacityControl
              label="Overlay opacity"
              value={settings.opacity}
              disabled={busy || !overlayAvailable}
              preview={previewOpacity}
              commit={commitOpacity}
            />
          </div>
        )}
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={settings.linked_window}
            onChange={(event) => update({ linked_window: event.target.checked })}
          />
          Link intensity windows
        </label>
        <details className="comparison-details">
          <summary>Image options</summary>
          <div>
            {(['left', 'right'] as const).map((side) => {
              const source = settings[side];
              const choice = sources.find((item) => sourceKey(item) === sourceKey(source));
              return (
                <fieldset key={side}>
                  <legend>{side === 'left' ? 'Image A' : 'Image B'}</legend>
                  <label>
                    Frame
                    <select
                      aria-label={`Image ${side === 'left' ? 'A' : 'B'} frame`}
                      value={source.frame ?? 'follow'}
                      onChange={(event) =>
                        update({
                          [side]: {
                            ...source,
                            frame:
                              event.target.value === 'follow' ? null : Number(event.target.value),
                          },
                        })
                      }
                    >
                      <option value="follow">Follow viewer</option>
                      {Array.from({ length: choice?.frames ?? 0 }, (_, index) => (
                        <option key={index} value={index}>
                          {index + 1}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Component
                    <select
                      aria-label={`Image ${side === 'left' ? 'A' : 'B'} component`}
                      value={source.component}
                      onChange={(event) =>
                        update({ [side]: { ...source, component: event.target.value } })
                      }
                    >
                      {['auto', 'magnitude', 'phase', 'real', 'imag'].map((component) => (
                        <option key={component} value={component}>
                          {component === 'imag'
                            ? 'Imaginary'
                            : component[0].toUpperCase() + component.slice(1)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Palette
                    <select
                      aria-label={`Image ${side === 'left' ? 'A' : 'B'} palette`}
                      value={source.colormap}
                      onChange={(event) =>
                        update({ [side]: { ...source, colormap: event.target.value } })
                      }
                    >
                      {Object.keys(palettes).map((palette) => (
                        <option key={palette} value={palette}>
                          {palette[0].toUpperCase() + palette.slice(1)}
                        </option>
                      ))}
                    </select>
                  </label>
                </fieldset>
              );
            })}
          </div>
        </details>
      </div>
    </fieldset>
  );
}

function ComparisonCanvas({
  values,
  shape,
  spacing,
  level,
  contrast,
  palette,
  display,
  label,
  probe,
  onProbe,
}: {
  values: Pixels;
  shape: [number, number];
  spacing: [number, number];
  level: { center: number; width: number };
  contrast: number;
  palette: string;
  display: ImageDisplay;
  label: string;
  probe?: [number, number] | null;
  onProbe?: (value: [number, number] | null) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  useImageNavigation(viewport, display);
  const [size, setSize] = useState({ width: 600, height: 400 });
  const panStart = useRef<{ x: number; y: number; pan: [number, number] } | null>(null);
  const [pixel, setPixel] = useState('');
  const sampled =
    probe && probe[0] >= 0 && probe[0] < shape[1] && probe[1] >= 0 && probe[1] < shape[0]
      ? `(${probe[0]}, ${probe[1]}) ${values[probe[1] * shape[1] + probe[0]].toPrecision(5)}`
      : '';
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(viewport.current!);
    return () => observer.disconnect();
  }, []);
  const [rows, columns] = shape;
  const aspect = (columns * spacing[1]) / (rows * spacing[0]);
  const width = Math.min(size.width, size.height * aspect);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    const output = context.createImageData(columns, rows);
    const scale = contrast / Math.max(level.width, Number.EPSILON);
    for (let index = 0; index < values.length; index++) {
      if (Number.isFinite(values[index]))
        writeColor(
          output.data,
          index * 4,
          (values[index] - level.center) * scale + 0.5,
          palettes[palette] ?? palettes.gray,
        );
      else {
        output.data[index * 4] = 192;
        output.data[index * 4 + 1] = 64;
        output.data[index * 4 + 2] = 192;
      }
      output.data[index * 4 + 3] = 255;
    }
    context.putImageData(output, 0, 0);
  }, [values, rows, columns, level.center, level.width, contrast, palette]);
  return (
    <div
      className="comparison-canvas-viewport"
      ref={viewport}
      tabIndex={0}
      aria-label="Comparison image navigation"
      aria-description={imageNavigationHelp}
      title={imageNavigationHelp}
      onDoubleClick={display.resetPosition}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        panStart.current = { x: event.clientX, y: event.clientY, pan: display.pan };
      }}
      onPointerMove={(event) => {
        const start = panStart.current;
        if (start)
          display.setPan([
            start.pan[0] + event.clientX - start.x,
            start.pan[1] + event.clientY - start.y,
          ]);
        const box = canvas.current!.getBoundingClientRect();
        const x = Math.floor(((event.clientX - box.left) * columns) / box.width),
          y = Math.floor(((event.clientY - box.top) * rows) / box.height);
        setPixel(
          x >= 0 && x < columns && y >= 0 && y < rows
            ? `(${x}, ${y}) ${values[y * columns + x].toPrecision(5)}`
            : '',
        );
        onProbe?.([x, y]);
      }}
      onPointerLeave={() => {
        setPixel('');
        onProbe?.(null);
      }}
      onPointerUp={() => {
        panStart.current = null;
      }}
      onPointerCancel={() => {
        panStart.current = null;
      }}
    >
      <canvas
        ref={canvas}
        width={columns}
        height={rows}
        role="img"
        aria-label={label}
        style={{
          width,
          height: width / aspect,
          transform: `translate(${display.pan[0]}px, ${display.pan[1]}px) scale(${display.zoom})`,
        }}
      />
      <span className="comparison-pixel">{onProbe ? sampled : pixel}</span>
    </div>
  );
}

export function ComparisonView({
  workspace,
  display,
  change,
  busy,
  processing,
  pendingIndices,
  navigateIndex,
  controls,
  advancedControls,
}: {
  workspace: Workspace;
  display: ImageDisplay;
  change: (command: Options) => Promise<void>;
  busy: boolean;
  processing: boolean;
  pendingIndices: Record<string, number>;
  navigateIndex: (key: string, value: number) => Promise<void>;
  controls: (children: ReactNode) => ReactNode;
  advancedControls: (children: ReactNode) => ReactNode;
}) {
  const key = comparisonKey(workspace);
  const navigation = comparisonNavigation(workspace);
  const settings = workspace.scene.comparison;
  const navigationContext = JSON.stringify([
    workspace.id,
    workspace.revision,
    workspace.experiment,
    navigation.plane,
    ...[settings.left, settings.right].map((source) => [sourceKey(source), source.component]),
  ]);
  const [loaded, setLoaded] = useState<{
    key: string;
    context: string;
    images?: ComparisonImages;
    error?: string;
  } | null>(null);
  const [opacityDraft, setOpacityDraft] = useState<number | null>(null);
  const [probe, setProbe] = useState<[number, number] | null>(null);
  const [intensitySide, setIntensitySide] = useState<'left' | 'right'>('left');
  const [intensityDraft, setIntensityDraft] = useState<{
    context: string;
    side: 'left' | 'right';
    level: IntensityWindow | null;
  } | null>(null);
  const intensityVersion = useRef(0);
  const [draft, setDraft] = useState<number | null>(null);
  const drag = useRef<number | null>(null);
  const surface = useRef<HTMLDivElement>(null);
  const position = draft ?? settings.position ?? 0.5;
  const menu = useContextMenu('Comparison actions', [
    { label: 'Zoom in', run: display.zoomIn, disabled: display.zoom >= MAX_ZOOM },
    { label: 'Zoom out', run: display.zoomOut, disabled: display.zoom <= MIN_ZOOM },
    { label: 'Reset display', run: display.reset },
  ]);
  useEffect(() => {
    if (processing) return;
    const controller = new AbortController();
    void comparisonImages(workspace, controller.signal)
      .then((images) => {
        if (!controller.signal.aborted) setLoaded({ key, context: navigationContext, images });
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted)
          setLoaded({
            key,
            context: navigationContext,
            error: error instanceof Error ? error.message : 'Comparison unavailable.',
          });
      });
    return () => controller.abort();
    // Pixel requests depend only on source and cursor changes, never the divider or palette.
  }, [key, workspace.view_revision, processing, navigationContext]);
  // Keep the last plane and slider mounted while the next cursor position loads.
  const images = loaded?.context === navigationContext ? loaded.images : undefined;
  const error = loaded?.key === key ? loaded.error : undefined;
  const planeDims = images?.planes[0].plane_dims;
  const matchingPlanes = planeDims?.every(
    (dim, index) => dim === images?.planes[1].plane_dims[index],
  );
  const sliceDim = matchingPlanes
    ? Object.keys(navigation.dimensions).find((dim) => !planeDims!.includes(dim))
    : undefined;
  const navigationControls = [
    { key: 'frame', label: 'Frame', size: navigation.frameCount, value: navigation.frame },
    ...(sliceDim
      ? [
          {
            key: sliceDim,
            label: 'Slice',
            size: navigation.dimensions[sliceDim],
            value: navigation.indices[sliceDim],
          },
        ]
      : []),
  ].filter((control) => control.size > 1);
  const linked = Boolean(
    settings.linked_window &&
    images?.same_units &&
    settings.left.component === settings.right.component,
  );
  const intensityContext = JSON.stringify([
    workspace.id,
    sourceKey(settings.left),
    sourceKey(settings.right),
    settings.left.component,
    settings.right.component,
    linked,
  ]);
  const targetSide = linked ? 'left' : intensitySide;
  const sharedRange = images
    ? ([
        Math.min(...images.planes.map((plane) => plane.range[0])),
        Math.max(...images.planes.map((plane) => plane.range[1])),
      ] as const)
    : ([0, 1] as const);
  function rangeFor(side: 'left' | 'right') {
    return linked
      ? sharedRange
      : (images?.planes[side === 'left' ? 0 : 1].range ?? ([0, 1] as const));
  }
  function windowFor(side: 'left' | 'right') {
    const target = linked ? 'left' : side;
    const range = rangeFor(side);
    const preview = intensityDraft?.context === intensityContext && intensityDraft.side === target;
    return (
      (preview ? intensityDraft.level : settings[target].window) ?? {
        center: (range[0] + range[1]) / 2,
        width: range[1] - range[0] || 1,
      }
    );
  }
  function previewIntensity(level: IntensityWindow) {
    intensityVersion.current++;
    setIntensityDraft({ context: intensityContext, side: targetSide, level });
  }
  async function commitIntensity(level: IntensityWindow | null) {
    const targets = linked ? (['left', 'right'] as const) : [targetSide];
    if (
      targets.every((side) => {
        const current = settings[side].window;
        return level === null
          ? !current
          : current?.center === level.center && current.width === level.width;
      })
    ) {
      setIntensityDraft(null);
      return;
    }
    const version = ++intensityVersion.current;
    setIntensityDraft({ context: intensityContext, side: targetSide, level });
    try {
      await change({
        scene: {
          comparison: {
            ...settings,
            [targetSide]: { ...settings[targetSide], window: level },
            ...(linked ? { right: { ...settings.right, window: level } } : {}),
          },
        },
      });
    } finally {
      if (version === intensityVersion.current) setIntensityDraft(null);
    }
  }
  const overlay = settings.mode === 'overlay';
  const compatible = images?.geometry_compatible !== false;
  const layout = !compatible ? 'side-by-side' : overlay ? 'overlay' : settings.layout;
  async function commitOpacity(value: number) {
    try {
      await change({ scene: { comparison: { ...settings, opacity: value } } });
    } finally {
      setOpacityDraft((current) => (current === value ? null : current));
    }
  }
  const commit = (value: number) => {
    drag.current = null;
    setDraft(value);
    void change({ scene: { comparison: { ...settings, position: value } } }).finally(() =>
      setDraft(null),
    );
  };
  const move = (x: number) => {
    const box = surface.current!.getBoundingClientRect();
    return Math.max(0.05, Math.min(0.95, (x - box.left) / box.width));
  };
  function pane(index: 0 | 1) {
    if (!images) return null;
    const side = index === 0 ? 'left' : 'right';
    const plane = images.planes[index];
    const source = settings[side];
    const label =
      workspace.scene.comparison_sources.find((item) => sourceKey(item) === sourceKey(source))
        ?.label ?? 'Unavailable image';
    const level = windowFor(side);
    return (
      <div className={`comparison-pane comparison-${side}`} key={side}>
        <ComparisonCanvas
          values={images.values[index]}
          shape={plane.shape}
          spacing={plane.spacing}
          level={level}
          contrast={1}
          palette={source.colormap ?? 'gray'}
          display={display}
          probe={layout === 'overlay' ? probe : undefined}
          onProbe={layout === 'overlay' ? setProbe : undefined}
          label={`Image ${side === 'left' ? 'A' : 'B'}: ${label}, frame ${plane.frame + 1}`}
        />
        <div className="comparison-label">
          <strong>{side === 'left' ? 'A' : 'B'}</strong>
          <span>
            {label}
            <small>
              Frame {plane.frame + 1}
              {plane.unit && ` · ${plane.unit}`} · {plane.shape.join(' × ')}
            </small>
          </span>
        </div>
      </div>
    );
  }
  return (
    <div className="comparison-view" aria-label="Image comparison">
      {advancedControls(
        <ComparisonControls
          workspace={workspace}
          busy={busy}
          change={change}
          overlayAvailable={Boolean(images?.geometry_compatible)}
          previewOpacity={setOpacityDraft}
          commitOpacity={commitOpacity}
        />,
      )}
      {controls(
        <>
          {images && (
            <IntensityControls
              key={intensityContext + targetSide}
              values={images.values[targetSide === 'left' ? 0 : 1]}
              range={rangeFor(targetSide)}
              level={windowFor(targetSide)}
              contrast={
                (rangeFor(targetSide)[1] - rangeFor(targetSide)[0] || 1) /
                windowFor(targetSide).width
              }
              onChange={previewIntensity}
              onCommit={(next) => void commitIntensity(next)}
              reset={() => void commitIntensity(null)}
              target={
                <label>
                  Adjust
                  <select
                    aria-label="Adjust comparison intensity"
                    value={linked ? 'both' : intensitySide}
                    disabled={linked}
                    onChange={(event) => setIntensitySide(event.target.value as 'left' | 'right')}
                  >
                    {linked ? (
                      <option value="both">Both images</option>
                    ) : (
                      <>
                        <option value="left">
                          {overlay ? 'A · Base image' : 'A · Left image'}
                        </option>
                        <option value="right">
                          {overlay ? 'B · Overlay image' : 'B · Right image'}
                        </option>
                      </>
                    )}
                  </select>
                </label>
              }
            />
          )}
        </>,
      )}
      {images && !compatible && (
        <p className="hint" role="status">
          {overlay ? 'Overlay unavailable. ' : ''}Different image grids: shown side by side.{' '}
          {images.geometry_message}
        </p>
      )}
      {images &&
        settings.linked_window &&
        (!images.same_units || settings.left.component !== settings.right.component) && (
          <p className="hint">
            Intensity windows are independent for different units or components.
          </p>
        )}
      <div
        className={`comparison-surface ${layout}`}
        ref={surface}
        tabIndex={0}
        style={
          {
            '--split-position': `${position * 100}%`,
            '--overlay-opacity': opacityDraft ?? settings.opacity,
          } as CSSProperties
        }
        onContextMenu={menu.onContextMenu}
        onKeyDown={menu.onKeyDown}
        aria-busy={loaded?.key !== key && !error}
      >
        {images ? (
          <>
            {pane(0)}
            {pane(1)}
            {layout !== 'overlay' && (
              <div
                role="separator"
                aria-label="Move comparison divider"
                aria-orientation="vertical"
                tabIndex={0}
                aria-valuemin={5}
                aria-valuemax={95}
                aria-valuenow={Math.round(position * 100)}
                aria-valuetext={`${Math.round(position * 100)} percent left image`}
                className="comparison-divider"
                onPointerDown={(event) => {
                  if (event.button !== 0) return;
                  event.preventDefault();
                  event.currentTarget.focus({ preventScroll: true });
                  event.currentTarget.setPointerCapture(event.pointerId);
                  drag.current = position;
                  setDraft(move(event.clientX));
                }}
                onPointerMove={(event) => {
                  if (drag.current !== null) setDraft(move(event.clientX));
                }}
                onPointerUp={(event) => {
                  if (drag.current !== null) commit(move(event.clientX));
                }}
                onPointerCancel={() => {
                  drag.current = null;
                  setDraft(null);
                }}
                onLostPointerCapture={() => {
                  if (drag.current !== null) {
                    drag.current = null;
                    setDraft(null);
                  }
                }}
                onDoubleClick={() => commit(0.5)}
                onKeyDown={(event) => {
                  if (event.key === 'Escape') {
                    drag.current = null;
                    setDraft(null);
                    return;
                  }
                  const values: Record<string, number> = {
                    ArrowLeft: position - (event.shiftKey ? 0.1 : 0.02),
                    ArrowRight: position + (event.shiftKey ? 0.1 : 0.02),
                    Home: 0.05,
                    End: 0.95,
                  };
                  if (event.key in values) {
                    event.preventDefault();
                    commit(Math.max(0.05, Math.min(0.95, values[event.key])));
                  }
                }}
              >
                <span>
                  <WorkbenchIcon name="split" />
                </span>
              </div>
            )}
          </>
        ) : (
          <p className="empty-message" role="status">
            {error ?? 'Loading comparison…'}
          </p>
        )}
      </div>
      {navigationControls.length > 0 && (
        <div className="dimension-controls">
          {navigationControls.map((control) => {
            const value = pendingIndices[control.key] ?? control.value;
            return (
              <label key={control.key}>
                {control.label}{' '}
                <strong>
                  {value + 1} / {control.size}
                </strong>
                <input
                  aria-label={control.label}
                  type="range"
                  min={0}
                  max={control.size - 1}
                  value={value}
                  disabled={processing}
                  onChange={(event) => void navigateIndex(control.key, Number(event.target.value))}
                />
              </label>
            );
          })}
        </div>
      )}
      {menu.menu}
    </div>
  );
}
