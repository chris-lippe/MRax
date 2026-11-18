import type { TimecourseResult } from './api';

export function timecourseCsv(result: TimecourseResult) {
  const cell = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const rows: unknown[][] = [
    [
      'Experiment',
      'Input',
      'Component',
      'Statistic',
      result.x_name,
      'Included in fit',
      ...result.series.flatMap((row) => [
        `${row.roi}${result.unit ? ` (${result.unit})` : ''}`,
        `${row.roi} valid n`,
      ]),
    ],
    ...result.x.map((x, index) => [
      result.experiment,
      result.input,
      result.component,
      result.statistic,
      x,
      result.included[index] ? 'Yes' : 'No',
      ...result.series.flatMap((row) => [row.values[index], row.counts[index]]),
    ]),
  ];
  return rows.map((row) => row.map(cell).join(',')).join('\r\n') + '\r\n';
}
