import { useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  finiteSignal,
  signalBounds,
  signalNumber,
  signalPath,
  signalScale,
  signalTick,
  signalTicks,
} from './signal';

export function CompactSignalPlot({
  x,
  traces,
  included,
  axisLabel,
  label,
}: {
  x: (number | null)[];
  traces: {
    name: string;
    values: (number | null)[];
    colour: string;
    fitted?: boolean;
    points?: boolean;
  }[];
  included?: boolean[] | null;
  axisLabel: string;
  label: string;
}) {
  const svg = useRef<SVGSVGElement>(null),
    clip = useId();
  const [size, setSize] = useState({ width: 640, height: 100 });
  const hasData = x.some(finiteSignal) && traces.some((trace) => trace.values.some(finiteSignal));
  useEffect(() => {
    const element = svg.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0 && entry.contentRect.height > 0)
        setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [hasData]);
  const [loX, hiX] = signalBounds(x, 'linear').domain;
  const ys = traces.flatMap((t) => t.values);
  const [loY, hiY] = signalBounds(ys, 'linear').domain;
  const left = 76,
    right = Math.max(92, size.width - 16),
    top = 14,
    bottom = Math.max(25, size.height - 39);
  const px = signalScale([loX, hiX], 'linear', left, right).to,
    py = signalScale([loY, hiY], 'linear', bottom, top).to;
  const paths = useMemo(() => {
    const px = signalScale([loX, hiX], 'linear', 76, Math.max(92, size.width - 16)).to;
    const py = signalScale([loY, hiY], 'linear', Math.max(25, size.height - 39), 14).to;
    return traces.map((trace) => ({
      ...trace,
      path: signalPath(
        x,
        trace.values,
        (i) => finiteSignal(x[i]) && finiteSignal(trace.values[i]),
        px,
        py,
      ),
    }));
  }, [x, traces, loX, hiX, loY, hiY, size.width, size.height]);
  if (!hasData)
    return <div className="curve-empty">No finite signal samples in this selection.</div>;
  const stride = Math.max(1, Math.ceil(x.length / 800));
  return (
    <svg
      ref={svg}
      className="compact-signal-chart"
      viewBox={`0 0 ${size.width} ${size.height}`}
      role="img"
      aria-label={label}
    >
      <desc>
        {x.length} samples. {axisLabel}. {traces.map((t) => t.name).join('; ')}. Hollow points are
        excluded from fitting. Use Data table for exact values or open Signal curves for detailed
        inspection.
      </desc>
      <defs>
        <clipPath id={clip}>
          <rect x={left - 5} y={top - 5} width={right - left + 10} height={bottom - top + 10} />
        </clipPath>
      </defs>
      {signalTicks([loY, hiY], 'linear', size.height < 120 ? 2 : 3).map((value) => (
        <g key={value}>
          <line className="chart-grid" x1={left} x2={right} y1={py(value)} y2={py(value)} />
          <text className="axis-text" x={left - 8} y={py(value) + 4} textAnchor="end">
            {signalTick(value)}
          </text>
        </g>
      ))}
      {signalTicks([loX, hiX], 'linear', Math.max(2, Math.floor((right - left) / 125))).map(
        (value) => (
          <g key={value}>
            <line
              className="compact-x-tick"
              x1={px(value)}
              x2={px(value)}
              y1={bottom}
              y2={bottom + 4}
            />
            <text className="axis-text" x={px(value)} y={bottom + 18} textAnchor="middle">
              {signalTick(value)}
            </text>
          </g>
        ),
      )}
      <text
        className="compact-axis-title"
        x={(left + right) / 2}
        y={size.height - 3}
        textAnchor="middle"
      >
        {axisLabel}
      </text>
      <g clipPath={`url(#${clip})`}>
        {paths.map((trace) => (
          <g key={trace.name}>
            {!trace.points && (
              <path
                d={trace.path}
                fill="none"
                stroke={trace.colour}
                strokeWidth={2}
                strokeDasharray={trace.fitted ? '6 3' : undefined}
              />
            )}
            {trace.points &&
              x.map((value, i) =>
                i % stride === 0 && finiteSignal(value) && finiteSignal(trace.values[i]) ? (
                  <circle
                    key={i}
                    className={
                      'observed-point' + (included?.[i] === false ? ' excluded-point' : '')
                    }
                    cx={px(value)}
                    cy={py(trace.values[i]!)}
                    r={4}
                    style={{
                      fill: included?.[i] === false ? 'var(--surface)' : trace.colour,
                      stroke: trace.colour,
                      strokeWidth: 1.5,
                    }}
                  >
                    <title>
                      Sample {i + 1}: {axisLabel} {signalNumber(value)}; {trace.name}{' '}
                      {signalNumber(trace.values[i])};{' '}
                      {included?.[i] === false ? 'Excluded frame' : 'Included frame'}
                    </title>
                  </circle>
                ) : null,
              )}
          </g>
        ))}
      </g>
    </svg>
  );
}
