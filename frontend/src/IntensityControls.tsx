import { useMemo, useRef, type CSSProperties, type ReactNode } from 'react';
import type { Pixels } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';

export type IntensityWindow = { center: number; width: number };
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));
const precise = (value: number) => Number(value.toPrecision(7));

function BoundNumber({
  label,
  value,
  commit,
}: {
  label: string;
  value: number;
  commit: (value: number) => void;
}) {
  return (
    <input
      key={value}
      aria-label={label}
      type="number"
      step="any"
      defaultValue={value}
      title="Apply on Enter or when leaving the field"
      onBlur={(event) => {
        const next = event.currentTarget.valueAsNumber;
        if (Number.isFinite(next)) commit(next);
        else event.currentTarget.value = String(value);
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') event.currentTarget.value = String(value);
        if (event.key === 'Enter' || event.key === 'Escape') {
          event.preventDefault();
          event.currentTarget.blur();
        }
      }}
    />
  );
}

export function IntensityControls({
  values,
  range,
  level,
  contrast,
  onChange,
  onCommit,
  reset,
  target,
}: {
  values: Pixels;
  range: readonly [number, number];
  level: IntensityWindow;
  contrast: number;
  onChange: (level: IntensityWindow, contrast?: number) => void;
  onCommit?: (level: IntensityWindow) => void;
  reset: () => void;
  target?: ReactNode;
}) {
  const span = range[1] - range[0] || 1;
  const minimum = range[1] > range[0] ? range[0] : range[0] - span / 2;
  const maximum = minimum + span;
  const midpoint = minimum + span / 2;
  const gap = Math.max(span * 1e-6, Number.EPSILON * Math.max(span, Math.abs(level.center)));
  const lower = level.center - level.width / 2;
  const upper = level.center + level.width / 2;
  const brightness = ((midpoint - level.center) / span) * 100;
  const pending = useRef<IntensityWindow | null>(null);
  const histogram = useMemo(() => {
    const bins = new Array<number>(96).fill(0);
    const stride = Math.max(1, Math.ceil(values.length / 8192));
    for (let i = 0; i < values.length; i += stride)
      if (Number.isFinite(values[i]))
        bins[clamp(Math.floor(((values[i] - minimum) / span) * bins.length), 0, bins.length - 1)]++;
    const peak = Math.log1p(Math.max(...bins));
    return (
      bins
        .map(
          (count, index) =>
            `${index ? 'L' : 'M'}${index},${28 - (peak ? (Math.log1p(count) / peak) * 26 : 0)}`,
        )
        .join(' ') + 'L95,28L0,28Z'
    );
  }, [values, minimum, span]);
  function change(next: IntensityWindow, gain?: number) {
    pending.current = next;
    onChange(next, gain);
  }
  function finish() {
    const next = pending.current;
    pending.current = null;
    if (next) onCommit?.(next);
  }
  function bound(side: 'lower' | 'upper', value: number, commit = false) {
    const low = side === 'lower' ? Math.min(value, upper - gap) : lower;
    const high = side === 'upper' ? Math.max(value, lower + gap) : upper;
    if ((side === 'lower' && low === lower) || (side === 'upper' && high === upper)) return;
    change({ center: low + (high - low) / 2, width: high - low });
    if (commit) finish();
  }
  return (
    <section className="intensity-controls" aria-label="Image intensity">
      {target && <div className="intensity-target">{target}</div>}
      <div className="intensity-adjustments">
        <label>
          <span>
            Brightness{' '}
            <output>
              {precise(brightness).toLocaleString('en', {
                maximumFractionDigits: 0,
                signDisplay: 'exceptZero',
              })}
              %
            </output>
          </span>
          <input
            aria-label="Brightness"
            type="range"
            min="-100"
            max="100"
            step="1"
            value={clamp(brightness, -100, 100)}
            onChange={(event) =>
              change({ ...level, center: midpoint - (Number(event.target.value) / 100) * span })
            }
            onPointerUp={finish}
            onKeyUp={finish}
            onBlur={finish}
          />
        </label>
        <label>
          <span>
            Contrast <output>{contrast.toFixed(2)}×</output>
          </span>
          <input
            aria-label="Display contrast"
            type="range"
            min="0.5"
            max="2"
            step="0.05"
            value={clamp(contrast, 0.5, 2)}
            onChange={(event) => {
              const gain = Number(event.target.value);
              change({ ...level, width: (level.width * contrast) / gain }, gain);
            }}
            onPointerUp={finish}
            onKeyUp={finish}
            onBlur={finish}
          />
        </label>
      </div>
      <div className="intensity-bounds">
        <div className="intensity-range-title">
          <span>Value range</span>
          <button
            className="intensity-auto"
            aria-label="Auto intensity"
            title="Reset brightness, contrast and value range"
            onClick={() => {
              pending.current = null;
              reset();
            }}
          >
            <WorkbenchIcon name="refresh" />
            Auto
          </button>
        </div>
        <div
          className="intensity-histogram"
          role="group"
          aria-label="Display value range"
          style={
            {
              '--bound-start': `${clamp((lower - minimum) / span, 0, 1) * 100}%`,
              '--bound-end': `${clamp((upper - minimum) / span, 0, 1) * 100}%`,
            } as CSSProperties
          }
        >
          <svg viewBox="0 0 96 28" preserveAspectRatio="none" aria-hidden="true">
            <path d={histogram} />
          </svg>
          <div className="intensity-range-selection" />
          <input
            type="range"
            aria-label="Lower display bound"
            min={minimum}
            max={maximum}
            step={Math.max(Number.MIN_VALUE, span / 1000)}
            value={clamp(lower, minimum, maximum)}
            onChange={(event) => bound('lower', Number(event.target.value))}
            onPointerUp={finish}
            onKeyUp={finish}
            onBlur={finish}
          />
          <input
            type="range"
            aria-label="Upper display bound"
            min={minimum}
            max={maximum}
            step={Math.max(Number.MIN_VALUE, span / 1000)}
            value={clamp(upper, minimum, maximum)}
            onChange={(event) => bound('upper', Number(event.target.value))}
            onPointerUp={finish}
            onKeyUp={finish}
            onBlur={finish}
          />
        </div>
        <div className="intensity-bound-fields">
          <label>
            Min
            <BoundNumber
              label="Minimum display value"
              value={lower}
              commit={(value) => bound('lower', value, true)}
            />
          </label>
          <label>
            Max
            <BoundNumber
              label="Maximum display value"
              value={upper}
              commit={(value) => bound('upper', value, true)}
            />
          </label>
        </div>
      </div>
    </section>
  );
}
