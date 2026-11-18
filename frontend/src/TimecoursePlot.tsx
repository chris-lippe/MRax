import { Fragment, memo, useEffect, useState } from 'react';
import type { TimecourseResult } from './api';
import { useContextMenu } from './ContextMenu';
import { roiColor, type RoiAppearance } from './scene';
import { CompactSignalPlot } from './CompactSignalPlot';
import { WorkbenchIcon } from './WorkbenchIcon';

const number = (value: number | null | undefined) =>
  value == null ? 'Missing' : Number(value.toPrecision(7)).toString();
const statistics: Record<string, string> = {
  mean: 'Mean',
  median: 'Median',
  std: 'Standard deviation',
  variance: 'Variance',
  sum: 'Sum',
};

export const TimecoursePlot = memo(function TimecoursePlot({
  result,
  hidden = false,
  onOpen,
  styles = {},
}: {
  result: TimecourseResult | null;
  hidden?: boolean;
  onOpen?: () => void;
  styles?: Record<string, RoiAppearance>;
}) {
  const [table, setTable] = useState(false);
  const menu = useContextMenu('Timecourse actions', [
    {
      label: table ? 'Show chart' : 'Show data table',
      run: () => setTable(!table),
      disabled: !result,
    },
  ]);
  const closeMenu = menu.close;
  useEffect(() => closeMenu(false), [result, hidden, closeMenu]);
  if (hidden)
    return (
      <section className="curve-panel timecourse-plot" hidden aria-label="Timecourse results" />
    );
  return (
    <section
      className="curve-panel timecourse-plot"
      hidden={hidden}
      aria-label="Timecourse results"
      tabIndex={0}
      onContextMenu={menu.onContextMenu}
      onKeyDown={menu.onKeyDown}
    >
      <div className="section-heading">
        <h2>
          Timecourse{result ? ` · ${statistics[result.statistic] ?? result.statistic}` : ''}
          {result?.unit ? ` (${result.unit})` : ''}
        </h2>
        <div className="curve-key">
          <button className="secondary" disabled={!result} onClick={() => setTable(!table)}>
            {table ? 'Chart' : 'Data table'}
          </button>
          <button
            className="secondary signal-open"
            disabled={!result}
            onClick={onOpen}
            aria-label="Open signal curves"
            title="Open detailed signal viewer"
          >
            <WorkbenchIcon name="timecourse" />
          </button>
        </div>
      </div>
      {!result ? (
        <div className="curve-empty">
          Select ROI masks and run a timecourse analysis in Analyze.
        </div>
      ) : table ? (
        <div className="table-scroll">
          <table aria-label="Timecourse values">
            <thead>
              <tr>
                <th>{result.x_name}</th>
                <th>Included in fit</th>
                {result.series.map((row) => (
                  <th colSpan={2} key={row.roi}>
                    {row.roi}
                    {result.unit ? ` (${result.unit})` : ''}
                  </th>
                ))}
              </tr>
              <tr>
                <th />
                <th />
                {result.series.map((row) => (
                  <Fragment key={row.roi}>
                    <th>Value</th>
                    <th>Valid n</th>
                  </Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.x.map((x, index) => (
                <tr key={index}>
                  <td>{number(x)}</td>
                  <td>{result.included[index] ? 'Yes' : 'No'}</td>
                  {result.series.map((row) => (
                    <TimecourseCells
                      key={row.roi}
                      value={row.values[index]}
                      count={row.counts[index]}
                    />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <div className="timecourse-legend">
            {result.series.map((row) => (
              <span key={row.roi}>
                <i
                  className="roi-swatch"
                  style={{ background: roiColor(row.roi, 1, styles[row.roi]) }}
                />
                {row.roi}
              </span>
            ))}
          </div>
          <CompactSignalPlot
            x={result.x}
            included={result.included}
            axisLabel={result.x_name}
            label="ROI timecourse comparison"
            traces={result.series.map((row) => ({
              name: row.roi,
              values: row.values,
              colour: roiColor(row.roi, 1, styles[row.roi]),
            }))}
          />
        </>
      )}
      {menu.menu}
    </section>
  );
});

function TimecourseCells({ value, count }: { value: number | null; count: number }) {
  return (
    <>
      <td>{number(value)}</td>
      <td>{count}</td>
    </>
  );
}
