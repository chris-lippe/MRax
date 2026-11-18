import { useMemo, useRef, type CSSProperties, type ReactNode } from 'react';
import type { Pixels } from './api';

export type IntensityWindow = { center: number; width: number };
export type ScaleMode = 'fixed' | 'frame' | 'manual';
export function ScaleSelector({
  label,
  mode,
  change,
  disabled,
}: {
  label: string;
  mode: ScaleMode;
  change: (mode: ScaleMode) => void;
  disabled?: boolean;
}) {
  return (
    <select
      aria-label={label}
      value={mode}
      disabled={disabled}
      onChange={(event) => change(event.target.value as ScaleMode)}
    >
      <option value="fixed">Fixed across frames</option>
      <option value="frame">Scale per frame</option>
      <option value="manual">Manual bounds</option>
    </select>
  );
}
const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

function BoundNumber({
  label,
  value,
  commit,
}: {
  label: string;
  value: number;
  commit: (value: number) => void;
}) {
  const edited = useRef(false);
  const shown = Number(value.toPrecision(7));
  return (
    <input
      key={value}
      aria-label={label}
      type="number"
      step="any"
      defaultValue={shown}
      onChange={() => {
        edited.current = true;
      }}
      title={`Current value: ${value}. Enter to apply, Escape to cancel.`}
      onBlur={(event) => {
        const next = event.currentTarget.valueAsNumber;
        event.currentTarget.value = String(shown);
        if (edited.current && Number.isFinite(next) && next !== value) commit(next);
        edited.current = false;
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape') edited.current = false;
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
  onChange,
  onCommit,
  mode,
  changeMode,
  target,
}: {
  values: Pixels;
  range: readonly [number, number];
  level: IntensityWindow;
  onChange: (level: IntensityWindow) => void;
  onCommit?: (level: IntensityWindow) => void;
  mode: ScaleMode;
  changeMode: (mode: ScaleMode) => void;
  target?: ReactNode;
}) {
  const lower = level.center - level.width / 2;
  const upper = level.center + level.width / 2;
  const minimum = Math.min(range[0], lower);
  const maximum = Math.max(range[1], upper);
  const span = maximum - minimum || 1;
  const gap = Math.max(span * 1e-6, Number.EPSILON * Math.max(span, Math.abs(level.center)));
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
  function change(next: IntensityWindow) {
    pending.current = next;
    onChange(next);
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
      <div className="intensity-bounds">
        <div className="intensity-range-title">
          <span>Value range</span>
          <ScaleSelector
            label="Intensity scaling"
            mode={mode}
            change={(next) => {
              pending.current = null;
              changeMode(next);
            }}
          />
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
