import type { TimecourseResult, Workspace } from './api';

export type SignalSource = 'profile' | 'fit' | 'volume';
export type SignalTrace = {
  id: string;
  name: string;
  values: (number | null)[];
  counts?: number[];
  fitted?: boolean;
};
export type SignalData = {
  x: (number | null)[];
  xLabel: string;
  yLabel: string;
  included: boolean[];
  traces: SignalTrace[];
  description: string;
  fit?: { observed: (number | null)[]; predicted: (number | null)[] };
};
export type SignalOptions = {
  source: SignalSource;
  residuals: boolean;
  xScale: 'linear' | 'log';
  yScale: 'linear' | 'log';
  xMin: string;
  xMax: string;
  yMin: string;
  yMax: string;
  title: string;
  xLabel: string;
  yLabel: string;
  grid: boolean;
  markers: boolean;
  lines: boolean;
  legend: boolean;
  excluded: boolean;
  textSize: number;
  lineWidth: number;
  theme: 'auto' | 'light' | 'dark';
  hidden: string[];
};
export const defaultSignalOptions: SignalOptions = {
  source: 'profile',
  residuals: false,
  xScale: 'linear',
  yScale: 'linear',
  xMin: '',
  xMax: '',
  yMin: '',
  yMax: '',
  title: '',
  xLabel: '',
  yLabel: '',
  grid: true,
  markers: true,
  lines: true,
  legend: true,
  excluded: true,
  textSize: 16,
  lineWidth: 2,
  theme: 'auto',
  hidden: [],
};
export const automaticSignalAxes = { xMin: '', xMax: '', yMin: '', yMax: '' };
export const finiteSignal = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value);
export const signalNumber = (value: number | null | undefined, digits = 7) =>
  finiteSignal(value) ? Number(value.toPrecision(digits)).toString() : '—';
export const signalTick = (value: number) => {
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  return magnitude >= 100000 || magnitude < 0.001
    ? value.toExponential(2)
    : Number(value.toPrecision(5)).toString();
};
export function signalAxisLabel(name: string, unit?: string | null) {
  return unit && unit !== '1' && !name.includes(`(${unit})`) ? `${name} (${unit})` : name;
}
export function signalData(
  w: Workspace | null,
  volume: TimecourseResult | null,
  options: Pick<SignalOptions, 'source' | 'residuals'>,
): SignalData | null {
  if (!w?.loaded) return null;
  if (options.source === 'volume') {
    if (!volume) return null;
    return {
      x: volume.x,
      xLabel: volume.x_name,
      yLabel: signalAxisLabel(
        volume.statistic === 'mean' ? 'Mean signal' : volume.statistic,
        volume.unit,
      ),
      included: volume.included,
      traces: volume.series.map((row) => ({
        id: row.roi,
        name: row.roi,
        values: row.values,
        counts: row.counts,
      })),
      description: `Whole ROI volumes · ${volume.input} · ${volume.component} · ${volume.statistic}`,
    };
  }
  const curve = options.source === 'fit' ? w.preview : w.timecourse;
  if (!curve) return null;
  const name =
    options.source === 'fit'
      ? (w.preview?.roi_name ?? w.selected_roi ?? 'ROI')
      : (w.selected_roi ?? 'ROI');
  const result: SignalData = {
    x: curve.x,
    xLabel: signalAxisLabel(curve.x_name, curve.x_unit),
    yLabel: 'Signal',
    included: curve.x.map((_, i) => curve.included?.[i] !== false),
    traces: [{ id: 'observed', name: name + ' · Observed', values: curve.observed }],
    description:
      options.source === 'fit'
        ? `Saved fit input · ${name} · ${w.preview!.model}`
        : `Current slice profile · ${name} · ${w.plane} · slice ${w.z + 1}`,
  };
  if (options.source === 'fit' && w.preview) {
    result.fit = { observed: w.preview.observed, predicted: w.preview.predicted };
    if (options.residuals) {
      result.yLabel = 'Observed − fitted';
      result.traces = [
        {
          id: 'residual',
          name: name + ' · Residual',
          values: curve.x.map((_, i) =>
            finiteSignal(curve.observed[i]) && finiteSignal(w.preview!.predicted[i])
              ? curve.observed[i]! - w.preview!.predicted[i]!
              : null,
          ),
        },
      ];
    } else {
      result.traces.push({
        id: 'fitted',
        name: name + ' · Fitted',
        values: w.preview.predicted,
        fitted: true,
      });
    }
  }
  return result;
}

export type SignalDomain = [number, number];
export function signalBounds(
  values: (number | null)[],
  scale: 'linear' | 'log',
  lower = '',
  upper = '',
) {
  const valid = values.filter((v): v is number => finiteSignal(v) && (scale !== 'log' || v > 0));
  let lo = valid.reduce((a, b) => Math.min(a, b), Infinity);
  let hi = valid.reduce((a, b) => Math.max(a, b), -Infinity);
  const manualLo = lower.trim() ? Number(lower) : null;
  const manualHi = upper.trim() ? Number(upper) : null;
  let error = '';
  if (!valid.length) {
    lo = scale === 'log' ? 1 : 0;
    hi = scale === 'log' ? 10 : 1;
    error =
      scale === 'log'
        ? 'No positive finite samples for this logarithmic axis.'
        : 'No finite samples for this axis.';
  } else if (lo === hi) {
    const pad = Math.max(Math.abs(lo) * 0.05, lo === 0 ? 0.5 : Number.MIN_VALUE);
    lo = scale === 'log' ? lo / 1.2 : Math.max(-Number.MAX_VALUE, lo - pad);
    hi =
      scale === 'log' ? Math.min(Number.MAX_VALUE, hi * 1.2) : Math.min(Number.MAX_VALUE, hi + pad);
    if (lo === hi) lo = hi / 2;
  } else {
    if (scale === 'log') {
      const margin = (Math.log10(hi) - Math.log10(lo)) * 0.04;
      lo = Math.max(Number.MIN_VALUE, 10 ** (Math.log10(lo) - margin));
      hi = Math.min(Number.MAX_VALUE, 10 ** (Math.log10(hi) + margin));
    } else {
      const margin = (hi / 2 - lo / 2) * 0.08;
      lo = Math.max(-Number.MAX_VALUE, lo - margin);
      hi = Math.min(Number.MAX_VALUE, hi + margin);
    }
  }
  if (manualLo !== null) lo = manualLo;
  if (manualHi !== null) hi = manualHi;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo >= hi)
    error = 'Enter finite bounds with minimum smaller than maximum.';
  else if (scale === 'log' && lo <= 0) error = 'Logarithmic bounds must be positive.';
  return { domain: [lo, hi] as SignalDomain, error };
}
export function signalScale(
  domain: SignalDomain,
  scale: 'linear' | 'log',
  start: number,
  end: number,
) {
  const transform = (v: number) => (scale === 'log' ? Math.log10(v) : v);
  const inverse = (v: number) => (scale === 'log' ? 10 ** v : v);
  const a = transform(domain[0]),
    b = transform(domain[1]);
  return {
    to: (v: number) => start + ((transform(v) / 2 - a / 2) / (b / 2 - a / 2)) * (end - start),
    from: (p: number) => inverse(a + ((p - start) / (end - start)) * (b - a)),
  };
}
export function signalTicks(domain: SignalDomain, scale: 'linear' | 'log', count = 5) {
  if (scale === 'log') {
    const lo = Math.log10(domain[0]),
      hi = Math.log10(domain[1]);
    const step = Math.max(1, Math.ceil((hi - lo) / count));
    const ticks = [];
    for (let e = Math.ceil(lo); e <= hi && ticks.length < 20; e += step) ticks.push(10 ** e);
    return ticks.length >= 2 ? ticks : [...domain];
  }
  const raw = (domain[1] / 2 - domain[0] / 2) * (2 / count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const ratio = raw / power;
  const step = (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * power;
  if (!Number.isFinite(step) || step <= 0) return [...domain];
  const ticks = [];
  for (let i = Math.ceil(domain[0] / step); ticks.length < 20; i++) {
    const value = i * step;
    if (value > domain[1] + step * 1e-9) break;
    ticks.push(Object.is(value, -0) ? 0 : value);
  }
  return ticks.length >= 2 ? ticks : [...domain];
}

export function signalSummary(
  values: (number | null)[],
  x: (number | null)[],
  included: boolean[],
  domain: SignalDomain,
) {
  const selected = values.filter(
    (v, i): v is number =>
      finiteSignal(v) &&
      finiteSignal(x[i]) &&
      included[i] !== false &&
      x[i]! >= domain[0] &&
      x[i]! <= domain[1],
  );
  let mean = 0,
    m2 = 0,
    n = 0;
  for (const value of selected) {
    n++;
    const delta = value - mean;
    mean += delta / n;
    m2 += delta * (value - mean);
  }
  selected.sort((a, b) => a - b);
  return {
    n,
    mean: n ? mean : null,
    sd: n > 1 ? Math.sqrt(m2 / (n - 1)) : null,
    min: selected[0] ?? null,
    max: selected[n - 1] ?? null,
    median: n ? selected[Math.floor((n - 1) / 2)] / 2 + selected[Math.floor(n / 2)] / 2 : null,
  };
}
export function signalFitSummary(data: SignalData, domain: SignalDomain) {
  if (!data.fit) return null;
  const observed: number[] = [],
    residuals: number[] = [];
  data.x.forEach((x, i) => {
    const o = data.fit!.observed[i],
      p = data.fit!.predicted[i];
    if (
      finiteSignal(x) &&
      x >= domain[0] &&
      x <= domain[1] &&
      data.included[i] !== false &&
      finiteSignal(o) &&
      finiteSignal(p)
    ) {
      observed.push(o);
      residuals.push(o - p);
    }
  });
  const summary = signalSummary(
    residuals,
    residuals.map((_, i) => i),
    residuals.map(() => true),
    [0, residuals.length],
  );
  const mean = observed.reduce((sum, v) => sum + v / observed.length, 0);
  const sse = residuals.reduce((sum, v) => sum + v * v, 0);
  const sst = observed.reduce((sum, v) => sum + (v - mean) ** 2, 0);
  return {
    n: observed.length,
    bias: summary.mean,
    rmse: observed.length ? Math.sqrt(sse / observed.length) : null,
    r2: observed.length > 1 && sst > 0 ? 1 - sse / sst : null,
  };
}

// Keep extrema and every gap; rendering dense curves must not invent continuity.
export function signalPath(
  x: (number | null)[],
  y: (number | null)[],
  valid: (index: number) => boolean,
  px: (v: number) => number,
  py: (v: number) => number,
  budget = 1600,
) {
  let result = '',
    start = 0;
  const point = (i: number, first: boolean) =>
    `${first ? 'M' : 'L'}${px(x[i]!).toFixed(2)},${py(y[i]!).toFixed(2)}`;
  while (start < x.length) {
    if (!valid(start)) {
      start++;
      continue;
    }
    let end = start + 1;
    while (end < x.length && valid(end)) end++;
    const stride = Math.max(1, Math.ceil(x.length / budget));
    const indices = new Set<number>([start, end - 1]);
    for (let i = start; i < end; i += stride) {
      let min = i,
        max = i;
      for (let j = i + 1; j < Math.min(end, i + stride); j++) {
        if (y[j]! < y[min]!) min = j;
        if (y[j]! > y[max]!) max = j;
      }
      indices.add(min);
      indices.add(max);
    }
    result += [...indices]
      .sort((a, b) => a - b)
      .map((i, j) => point(i, j === 0))
      .join(' ');
    start = end;
  }
  return result;
}
export function signalCsv(data: SignalData, traces: SignalTrace[]) {
  const cell = (v: unknown) => '"' + String(v ?? '').replaceAll('"', '""') + '"';
  return (
    [
      [
        'Sample',
        data.xLabel,
        'Included in fit',
        ...traces.flatMap((t) => (t.counts ? [t.name, t.name + ' · Valid n'] : [t.name])),
      ],
      ...data.x.map((x, i) => [
        i + 1,
        x,
        data.included[i] !== false ? 'Yes' : 'No',
        ...traces.flatMap((t) => (t.counts ? [t.values[i], t.counts[i]] : [t.values[i]])),
      ]),
    ]
      .map((row) => row.map(cell).join(','))
      .join('\r\n') + '\r\n'
  );
}
