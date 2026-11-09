import { test, expect } from '@playwright/test';
import type { Workspace } from '../src/api';
import {
  defaultSignalOptions,
  finiteSignal,
  signalBounds,
  signalCsv,
  signalData,
  signalFitSummary,
  signalPath,
  signalScale,
  signalSummary,
  signalTicks,
  type SignalData,
} from '../src/signal';

test('scientific bounds cover negative, constant, tiny and missing signals without invalid ticks', () => {
  for (const values of [
    [-8, -4, -2],
    [4, 4, 4],
    [0, 0],
    [1e-12, 2e-12],
    [1e-300, 2e-300],
  ]) {
    const { domain, error } = signalBounds(values, 'linear');
    expect(error).toBe('');
    expect(domain[0]).toBeLessThan(Math.min(...values));
    expect(domain[1]).toBeGreaterThan(Math.max(...values));
    const ticks = signalTicks(domain, 'linear');
    expect(ticks.every(Number.isFinite)).toBe(true);
    expect(new Set(ticks).size).toBe(ticks.length);
    const scale = signalScale(domain, 'linear', 10, 510);
    expect(scale.from(scale.to(values[0]))).toBeCloseTo(values[0], 12);
  }
  expect(signalBounds([null, null], 'linear').error).toContain('No finite');
  expect(signalBounds([-2, 0, null], 'log').error).toContain('No positive');
  expect(signalBounds([1, 10], 'log', '0').error).toContain('positive');
  expect(signalBounds([1, 10], 'linear', '9', '2').error).toContain('minimum smaller');
  expect(signalBounds([1, 10], 'linear', 'nonsense').error).toContain('finite bounds');
  const log = signalBounds([0, -3, 1, 10, 100], 'log');
  expect(log.error).toBe('');
  expect(signalTicks(log.domain, 'log')).toContain(10);
});

test('range statistics and fit diagnostics exclude missing and excluded samples', () => {
  const result = signalSummary([2, 4, null, 100], [0, 1, 2, 3], [true, true, true, false], [0, 3]);
  expect(result).toMatchObject({ n: 2, mean: 3, min: 2, max: 4, median: 3 });
  expect(result.sd).toBeCloseTo(Math.sqrt(2), 12);
  expect(signalSummary([2], [0], [true], [2, 4]).n).toBe(0);
  const data: SignalData = {
    x: [0, 1, 2, 3],
    xLabel: 'TE (ms)',
    yLabel: 'Signal',
    included: [true, true, false, true],
    traces: [],
    description: 'Independent test',
    fit: { observed: [2, 4, 100, null], predicted: [3, 4, 0, 1] },
  };
  expect(signalFitSummary(data, [0, 3])).toMatchObject({ n: 2, bias: -0.5, r2: 0.5 });
  expect(signalFitSummary(data, [0, 3])!.rmse).toBeCloseTo(Math.sqrt(0.5), 12);
  data.fit = { observed: [4, 4, 4, null], predicted: [4, 4, 4, 1] };
  expect(signalFitSummary(data, [0, 3])!.r2).toBeNull();
});

test('dense trace rendering retains extrema and never joins across missing intervals', () => {
  const x = Array.from({ length: 20000 }, (_, i) => i);
  const y: (number | null)[] = x.map(() => 2);
  y[1234] = -9999;
  y[16555] = 8888;
  y[9000] = null;
  const path = signalPath(
    x,
    y,
    (i) => finiteSignal(y[i]),
    (v) => v,
    (v) => v,
    80,
  );
  expect(path).toContain('1234.00,-9999.00');
  expect(path).toContain('16555.00,8888.00');
  expect(path.match(/M/g)).toHaveLength(2);
  expect(path.match(/[ML]/g)!.length).toBeLessThan(400);
  expect(path).not.toMatch(/NaN|Infinity/);
});

test('CSV preserves source precision, inclusion flags, gaps and quoted labels', () => {
  const data: SignalData = {
    x: [10, 20],
    xLabel: 'TE (ms)',
    yLabel: 'Signal',
    included: [true, false],
    description: 'Independent test',
    traces: [{ id: 'a', name: 'ROI, "quoted"', values: [Math.PI, null], counts: [4, 0] }],
  };
  const csv = signalCsv(data, data.traces);
  expect(csv).toContain('"ROI, ""quoted"""');
  expect(csv).toContain('"3.141592653589793"');
  expect(csv).toContain('"2","20","No","","0"');
  expect(defaultSignalOptions.source).toBe('profile');
});

test('slice profiles and saved fit inputs retain their own observations and sample coordinates', () => {
  const workspace = {
    loaded: true,
    selected_roi: 'Current ROI',
    plane: 'axial',
    z: 1,
    timecourse: {
      x: [10, 20],
      x_name: 'TE',
      x_unit: 'ms',
      observed: [5, 3],
      included: [true, false],
    },
    preview: {
      x: [12, 24],
      x_name: 'TE',
      x_unit: 'ms',
      observed: [100, 60],
      predicted: [98, 59],
      included: [true, true],
      roi_name: 'Recorded ROI',
      model: 'T2',
    },
  } as unknown as Workspace;
  const profile = signalData(workspace, null, { source: 'profile', residuals: false })!;
  expect(profile.x).toEqual([10, 20]);
  expect(profile.traces).toHaveLength(1);
  expect(profile.traces[0].values).toEqual([5, 3]);
  expect(profile.included).toEqual([true, false]);
  expect(profile.description).toContain('slice 2');
  const fit = signalData(workspace, null, { source: 'fit', residuals: false })!;
  expect(fit.x).toEqual([12, 24]);
  expect(fit.traces.map((t) => t.values)).toEqual([
    [100, 60],
    [98, 59],
  ]);
  expect(fit.description).toContain('Recorded ROI');
  expect(signalData(workspace, null, { source: 'fit', residuals: true })!.traces[0].values).toEqual(
    [2, 1],
  );
  expect(workspace.preview!.observed).toEqual([100, 60]);
});
