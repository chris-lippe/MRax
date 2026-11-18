import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type RefObject,
  type PointerEvent,
} from 'react';
import {
  finiteSignal,
  signalNumber,
  signalPath,
  signalScale,
  signalTick,
  signalTicks,
  type SignalData,
  type SignalDomain,
  type SignalOptions,
  type SignalTrace,
} from './signal';

type Props = {
  data: SignalData;
  traces: SignalTrace[];
  options: SignalOptions;
  xDomain: SignalDomain;
  yDomain: SignalDomain;
  svgRef: RefObject<SVGSVGElement | null>;
  update: (patch: Partial<SignalOptions>) => void;
};
const light = [
  '#0072b2',
  '#b64d00',
  '#007a59',
  '#a14689',
  '#805800',
  '#6750a4',
  '#a73239',
  '#276a70',
];
const dark = [
  '#85baff',
  '#ffae80',
  '#72dfb0',
  '#e8a4da',
  '#ecd074',
  '#c7b6ff',
  '#ff989f',
  '#87d9df',
];
export function signalColour(index: number, theme: SignalOptions['theme'] = 'auto') {
  const i = index % light.length;
  const fallback =
    theme === 'auto'
      ? `light-dark(${light[i]}, ${dark[i]})`
      : theme === 'dark'
        ? dark[i]
        : light[i];
  return `var(--signal-colour-${i}, ${fallback})`;
}

export function SignalPlot({ data, traces, options, xDomain, yDomain, svgRef, update }: Props) {
  const clip = useId();
  const [size, setSize] = useState({ width: 680, height: 420 });
  const [cursor, setCursor] = useState(0);
  const [announce, setAnnounce] = useState(false);
  const [tool, setTool] = useState<'inspect' | 'range' | 'pan'>('inspect');
  const [selection, setSelection] = useState<[number, number] | null>(null);
  const drag = useRef<{ id: number; start: number; end: number } | null>(null);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setSize({ width: entry.contentRect.width, height: 420 });
    });
    observer.observe(svg);
    return () => observer.disconnect();
  }, [svgRef]);
  const index = Math.min(cursor, data.x.length - 1);
  const font = options.textSize;
  const legendColumns = Math.max(1, Math.floor((size.width - 36) / 220));
  const legendRows = options.legend ? Math.ceil(traces.length / legendColumns) : 0;
  const legendSpacing = Math.max(24, font * 1.6);
  const legendCharacters = Math.max(
    8,
    Math.floor(((size.width - 36) / legendColumns - 36) / (font * 0.57)),
  );
  const top = 24 + (options.title ? 28 : 0) + legendRows * legendSpacing;
  const height = Math.max(420, top + 300);
  const left = Math.min(
    132,
    Math.max(
      88,
      Math.max(...signalTicks(yDomain, options.yScale).map((v) => signalTick(v).length), 3) *
        font *
        0.57 +
        36,
    ),
  );
  const right = Math.max(left + 40, size.width - 28),
    bottom = height - 58;
  const xMap = signalScale(xDomain, options.xScale, left, right);
  const yMap = signalScale(yDomain, options.yScale, bottom, top);
  const valid = (trace: SignalTrace, i: number) =>
    finiteSignal(data.x[i]) &&
    finiteSignal(trace.values[i]) &&
    (options.xScale !== 'log' || data.x[i]! > 0) &&
    (options.yScale !== 'log' || trace.values[i]! > 0) &&
    (options.excluded || data.included[i] !== false || trace.fitted === true);
  const geometry = useMemo(
    () =>
      traces.map((trace) => {
        const px = signalScale(xDomain, options.xScale, left, right).to;
        const py = signalScale(yDomain, options.yScale, bottom, top).to;
        const allowed = (i: number) =>
          finiteSignal(data.x[i]) &&
          finiteSignal(trace.values[i]) &&
          (options.xScale !== 'log' || data.x[i]! > 0) &&
          (options.yScale !== 'log' || trace.values[i]! > 0) &&
          (options.excluded || data.included[i] !== false || trace.fitted === true);
        const markerStride = Math.max(
          1,
          Math.ceil((data.x.length * Math.max(1, traces.length)) / 1600),
        );
        const markers = trace.fitted
          ? data.x.length === 1 && allowed(0)
            ? [0]
            : []
          : data.x.flatMap((_, i) => (i % markerStride === 0 && allowed(i) ? [i] : []));
        return {
          trace,
          path: signalPath(data.x, trace.values, allowed, px, py, 1600, true),
          markers,
          colour: signalColour(data.traces.findIndex((t) => t.id === trace.id)),
        };
      }),
    [
      data,
      traces,
      options.xScale,
      options.yScale,
      options.excluded,
      xDomain,
      yDomain,
      left,
      right,
      top,
      bottom,
    ],
  );
  const ordered = useMemo(
    () =>
      data.x
        .flatMap((x, i) =>
          finiteSignal(x) && (options.xScale !== 'log' || x > 0) ? [{ x, i }] : [],
        )
        .sort((a, b) => a.x - b.x),
    [data.x, options.xScale],
  );
  function inspect(x: number) {
    const target = xMap.from(x);
    let lo = 0,
      hi = ordered.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (ordered[mid].x < target) lo = mid + 1;
      else hi = mid;
    }
    const a = ordered[Math.max(0, lo - 1)],
      b = ordered[Math.min(ordered.length - 1, lo)];
    if (a && b) setCursor(Math.abs(a.x - target) <= Math.abs(b.x - target) ? a.i : b.i);
    setAnnounce(false);
  }
  function position(event: PointerEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return Math.max(
      left,
      Math.min(right, ((event.clientX - bounds.left) * size.width) / bounds.width),
    );
  }
  function finish(
    event: Pick<PointerEvent<SVGSVGElement>, 'pointerId' | 'currentTarget'>,
    cancel = false,
  ) {
    const current = drag.current;
    if (!current || current.id !== event.pointerId) return;
    drag.current = null;
    setSelection(null);
    if (!cancel && Math.abs(current.end - current.start) > 6) {
      if (tool === 'range')
        update({
          xMin: String(xMap.from(Math.min(current.start, current.end))),
          xMax: String(xMap.from(Math.max(current.start, current.end))),
        });
      else {
        const delta = current.end - current.start;
        const lo = xMap.from(left - delta),
          hi = xMap.from(right - delta);
        if (Number.isFinite(lo) && Number.isFinite(hi) && lo < hi)
          update({ xMin: String(lo), xMax: String(hi) });
      }
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  }
  function zoom(factor: number) {
    const center = (left + right) / 2,
      half = ((right - left) * factor) / 2;
    const lo = xMap.from(center - half),
      hi = xMap.from(center + half);
    if (Number.isFinite(lo) && Number.isFinite(hi) && lo < hi)
      update({ xMin: String(lo), xMax: String(hi) });
  }
  const variables = Object.fromEntries(
    light.flatMap((colour, i) => [
      [`--signal-light-${i}`, colour],
      [`--signal-dark-${i}`, dark[i]],
    ]),
  );
  const omitted = traces.reduce(
    (sum, t) =>
      sum +
      t.values.reduce<number>(
        (n, y, i) =>
          n +
          (finiteSignal(y) &&
          finiteSignal(data.x[i]) &&
          ((options.xScale === 'log' && data.x[i]! <= 0) || (options.yScale === 'log' && y <= 0))
            ? 1
            : 0),
        0,
      ),
    0,
  );
  return (
    <div className="scientific-plot">
      <div className="signal-plot-toolbar" role="toolbar" aria-label="Signal plot tools">
        <div className="signal-tools" role="group" aria-label="Signal pointer tool">
          {(['inspect', 'range', 'pan'] as const).map((value) => (
            <button key={value} aria-pressed={tool === value} onClick={() => setTool(value)}>
              {value === 'inspect' ? 'Inspect' : value === 'range' ? 'Select X range' : 'Pan X'}
            </button>
          ))}
        </div>
        <button onClick={() => zoom(0.5)} aria-label="Zoom in signal X axis">
          +
        </button>
        <button onClick={() => zoom(2)} aria-label="Zoom out signal X axis">
          −
        </button>
        <button onClick={() => update({ xMin: '', xMax: '', yMin: '', yMax: '' })}>
          Reset axes
        </button>
      </div>
      <svg
        ref={svgRef}
        className="scientific-chart"
        data-plot-theme={options.theme}
        data-tool={tool}
        style={{ ...variables, fontSize: font, height }}
        viewBox={`0 0 ${size.width} ${height}`}
        role="img"
        aria-label={
          options.residuals && data.fit
            ? 'Scientific signal residual plot'
            : 'Scientific signal curve plot'
        }
        aria-describedby="signal-plot-help"
        tabIndex={0}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          const x = position(event);
          inspect(x);
          event.currentTarget.focus({ preventScroll: true });
          if (tool === 'inspect') return;
          event.preventDefault();
          drag.current = { id: event.pointerId, start: x, end: x };
          setSelection([x, x]);
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const x = position(event),
            current = drag.current;
          if (current && current.id === event.pointerId) {
            current.end = x;
            setSelection([current.start, x]);
          } else inspect(x);
        }}
        onPointerUp={(event) => finish(event)}
        onPointerCancel={(event) => finish(event, true)}
        onLostPointerCapture={(event) => finish(event)}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && drag.current) {
            event.preventDefault();
            finish({ currentTarget: event.currentTarget, pointerId: drag.current.id }, true);
            return;
          }
          const next =
            event.key === 'ArrowRight'
              ? index + 1
              : event.key === 'ArrowLeft'
                ? index - 1
                : event.key === 'Home'
                  ? 0
                  : event.key === 'End'
                    ? data.x.length - 1
                    : null;
          if (next !== null) {
            event.preventDefault();
            setAnnounce(true);
            setCursor(Math.max(0, Math.min(data.x.length - 1, next)));
          }
        }}
      >
        <title>{options.title || 'Signal curves'}</title>
        <desc>
          {data.description}. {traces.map((t) => t.name).join(', ')}. Hollow points mark samples
          excluded from fitting.
        </desc>
        <defs>
          <clipPath id={clip}>
            <rect x={left} y={top} width={right - left} height={bottom - top} />
          </clipPath>
        </defs>
        {options.title && (
          <text x={size.width / 2} y={24} textAnchor="middle" className="signal-figure-title">
            {options.title}
          </text>
        )}
        {options.legend &&
          geometry.map(({ trace, colour }, i) => {
            const x = 18 + ((i % legendColumns) * (size.width - 36)) / legendColumns;
            const y = (options.title ? 52 : 24) + Math.floor(i / legendColumns) * legendSpacing;
            return (
              <g key={trace.id} className="signal-legend-item">
                <line
                  x1={x}
                  x2={x + 22}
                  y1={y - 5}
                  y2={y - 5}
                  stroke={colour}
                  strokeWidth={options.lineWidth}
                  strokeDasharray={trace.fitted ? '7 4' : undefined}
                />
                <text x={x + 30} y={y}>
                  {trace.name.length > legendCharacters
                    ? trace.name.slice(0, legendCharacters - 1) + '…'
                    : trace.name}
                  <title>{trace.name}</title>
                </text>
              </g>
            );
          })}
        {signalTicks(xDomain, options.xScale, Math.max(2, Math.floor((right - left) / 120))).map(
          (v) => (
            <g key={v}>
              {options.grid && (
                <line
                  className="signal-grid"
                  x1={xMap.to(v)}
                  x2={xMap.to(v)}
                  y1={top}
                  y2={bottom}
                />
              )}
              <line
                className="signal-axis"
                x1={xMap.to(v)}
                x2={xMap.to(v)}
                y1={bottom}
                y2={bottom + 5}
              />
              <text x={xMap.to(v)} y={bottom + 23} textAnchor="middle">
                {signalTick(v)}
              </text>
            </g>
          ),
        )}
        {signalTicks(yDomain, options.yScale).map((v) => (
          <g key={v}>
            {options.grid && (
              <line className="signal-grid" x1={left} x2={right} y1={yMap.to(v)} y2={yMap.to(v)} />
            )}
            <line className="signal-axis" x1={left - 5} x2={left} y1={yMap.to(v)} y2={yMap.to(v)} />
            <text x={left - 10} y={yMap.to(v) + font * 0.32} textAnchor="end">
              {signalTick(v)}
            </text>
          </g>
        ))}
        <path className="signal-axis" fill="none" d={`M${left},${top}V${bottom}H${right}`} />
        <text
          className="signal-axis-title"
          x={(left + right) / 2}
          y={height - 10}
          textAnchor="middle"
        >
          {options.xLabel || data.xLabel}
        </text>
        <text
          className="signal-axis-title"
          transform={`translate(20 ${(top + bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {options.yLabel || data.yLabel}
        </text>
        <g clipPath={`url(#${clip})`}>
          {options.residuals &&
            data.fit &&
            options.yScale === 'linear' &&
            yDomain[0] <= 0 &&
            yDomain[1] >= 0 && (
              <line className="signal-zero" x1={left} x2={right} y1={yMap.to(0)} y2={yMap.to(0)} />
            )}
          <g
            transform={
              tool === 'pan' && selection
                ? `translate(${selection[1] - selection[0]} 0)`
                : undefined
            }
          >
            {geometry.map(({ trace, path, markers, colour }) => (
              <g key={trace.id} data-signal-trace={trace.id}>
                {(options.lines || trace.fitted) && (
                  <path
                    d={path}
                    fill="none"
                    stroke={colour}
                    strokeWidth={options.lineWidth}
                    strokeDasharray={trace.fitted ? '7 4' : undefined}
                  />
                )}
                {options.markers &&
                  markers.map((i) => (
                    <circle
                      key={i}
                      cx={xMap.to(data.x[i]!)}
                      cy={yMap.to(trace.values[i]!)}
                      r={3.5}
                      fill={data.included[i] === false ? 'var(--plot-bg)' : colour}
                      stroke={colour}
                      strokeWidth={1.6}
                    />
                  ))}
              </g>
            ))}
          </g>
          <g data-plot-inspection="">
            {finiteSignal(data.x[index]) && (options.xScale !== 'log' || data.x[index]! > 0) && (
              <line
                className="signal-cursor"
                x1={xMap.to(data.x[index]!)}
                x2={xMap.to(data.x[index]!)}
                y1={top}
                y2={bottom}
              />
            )}
            {traces.map((trace) =>
              valid(trace, index) ? (
                <circle
                  key={trace.id}
                  cx={xMap.to(data.x[index]!)}
                  cy={yMap.to(trace.values[index]!)}
                  r={5}
                  stroke={signalColour(data.traces.findIndex((t) => t.id === trace.id))}
                  fill="var(--plot-bg)"
                  strokeWidth={2}
                >
                  <title>{signalNumber(trace.values[index])}</title>
                </circle>
              ) : null,
            )}
            {selection && tool === 'range' && (
              <rect
                className="signal-range"
                x={Math.min(...selection)}
                y={top}
                width={Math.abs(selection[1] - selection[0])}
                height={bottom - top}
              />
            )}
          </g>
        </g>
      </svg>
      <p id="signal-plot-help" className="control-note">
        Inspect with the pointer or focus the plot and use Left/Right, Home/End. Select X range by
        dragging. Escape cancels. Hollow points are excluded from fitting.
      </p>
      {omitted > 0 && (
        <p className="control-note" role="status">
          {omitted} non-positive values are outside logarithmic axes and are not plotted.
        </p>
      )}
      {data.x.length > 800 && (
        <p className="control-note">
          Dense traces retain extrema and gaps. Markers are thinned. Inspection and CSV retain every
          sample.
        </p>
      )}
      <div className="signal-sample-inspector">
        <label>
          Sample{' '}
          <input
            type="range"
            aria-label="Inspect signal sample"
            min={0}
            max={data.x.length - 1}
            value={index}
            onChange={(e) => {
              setAnnounce(true);
              setCursor(Number(e.target.value));
            }}
          />
        </label>
        <div
          role="status"
          aria-live={announce ? 'polite' : 'off'}
          aria-label="Signal sample values"
        >
          <strong>
            Sample {index + 1} / {data.x.length}
          </strong>
          <span>
            {data.xLabel}: {signalNumber(data.x[index], 12)}
          </span>
          <span>
            {data.included[index] === false ? 'Excluded from fitting' : 'Included in fit'}
          </span>
          {traces.map((trace) => (
            <span key={trace.id}>
              {trace.name}: {signalNumber(trace.values[index], 12)}
              {trace.counts ? ` · n = ${trace.counts[index]}` : ''}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
