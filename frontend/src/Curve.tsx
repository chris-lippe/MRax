import { memo, useEffect, useState } from 'react';
import type { Workspace } from './api';
import { useContextMenu } from './ContextMenu';
import { CompactSignalPlot } from './CompactSignalPlot';
import { signalAxisLabel } from './signal';
import { WorkbenchIcon } from './WorkbenchIcon';

export const Curve = memo(function Curve({
  workspace,
  hidden = false,
  onOpen,
  onAxisChange,
  onRoiChange,
  busy = false,
}: {
  workspace: Workspace | null;
  hidden?: boolean;
  onOpen?: () => void;
  onAxisChange: (axis: string) => void;
  onRoiChange: (roi: string) => void;
  busy?: boolean;
}) {
  const [table, setTable] = useState(false);
  const curve = workspace?.preview ?? workspace?.timecourse;
  const menu = useContextMenu('Signal actions', [
    {
      label: table ? 'Show chart' : 'Show data table',
      run: () => setTable(!table),
      disabled: !curve,
    },
    { label: 'Open signal curves', run: () => onOpen?.(), disabled: !curve },
  ]);
  const closeMenu = menu.close;
  useEffect(() => closeMenu(false), [workspace?.id, workspace?.view_revision, hidden, closeMenu]);
  if (hidden) return <section className="curve-panel" hidden aria-label="ROI signal" />;
  const predicted = workspace?.preview?.predicted;
  const region = curve ? curve.roi_name : workspace?.selected_roi;
  const heading = region ? 'ROI signal' : 'Image signal';
  const label = curve ? signalAxisLabel(curve.x_name, curve.x_unit) : '';
  return (
    <section
      className="curve-panel"
      aria-label={heading}
      tabIndex={0}
      onContextMenu={menu.onContextMenu}
      onKeyDown={menu.onKeyDown}
    >
      <div className="section-heading curve-heading">
        <h2 title={region ?? undefined}>{heading}</h2>
        <div className="curve-selectors">
          <label className="curve-axis">
            X axis
            <select
              aria-label="Signal x-axis"
              value={String(workspace?.display.plot_axis ?? 'auto')}
              disabled={!workspace?.loaded || busy}
              onChange={(event) => onAxisChange(event.target.value)}
            >
              <option value="auto">Automatic</option>
              {Object.keys(workspace?.dimensions ?? {}).map((dimension) => (
                <option key={dimension}>{dimension}</option>
              ))}
            </select>
          </label>
          <label className="curve-axis curve-roi">
            ROI
            <select
              aria-label="Signal ROI"
              value={region ?? ''}
              disabled={!workspace?.loaded || busy}
              onChange={(event) => onRoiChange(event.target.value)}
            >
              <option value="">Whole image mean</option>
              {workspace?.rois
                .filter((roi) => roi.enabled)
                .map((roi) => (
                  <option key={roi.name}>{roi.name}</option>
                ))}
            </select>
          </label>
        </div>
        <div className="curve-key">
          <span className="dot observed" />
          Observed
          {predicted && (
            <>
              <span className="dot fitted" />
              Fit
            </>
          )}
          <button className="secondary" disabled={!curve} onClick={() => setTable(!table)}>
            {table ? 'Chart' : 'Data table'}
          </button>
          <button
            className="secondary signal-open"
            disabled={!curve}
            onClick={onOpen}
            aria-label="Open signal curves"
            title="Open detailed signal viewer"
          >
            <WorkbenchIcon name="timecourse" />
          </button>
        </div>
      </div>
      {!curve ? (
        <div className="curve-empty">Open an experiment to view its mean signal.</div>
      ) : table ? (
        <div className="table-scroll" tabIndex={0} aria-label={heading + ' table scroll area'}>
          <table>
            <caption>{heading} values</caption>
            <thead>
              <tr>
                <th>{label}</th>
                <th>Observed</th>
                {predicted && <th>Predicted</th>}
                <th>Included</th>
              </tr>
            </thead>
            <tbody>
              {curve.x.map((x, i) => (
                <tr key={i}>
                  <td>{x ?? 'Missing'}</td>
                  <td>{curve.observed[i]?.toPrecision(7) ?? 'Missing'}</td>
                  {predicted && <td>{predicted[i]?.toPrecision(7) ?? 'Missing'}</td>}
                  <td>{curve.included?.[i] === false ? 'No' : 'Yes'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <CompactSignalPlot
          x={curve.x}
          included={curve.included}
          axisLabel={label}
          label={
            predicted
              ? 'Observed ROI signal and fitted decay curve'
              : region
                ? 'Observed ROI signal profile'
                : 'Whole image mean signal'
          }
          traces={[
            { name: 'Observed', values: curve.observed, colour: 'var(--accent)', points: true },
            ...(predicted
              ? [{ name: 'Fitted', values: predicted, colour: 'var(--fit)', fitted: true }]
              : []),
          ]}
        />
      )}
      {menu.menu}
    </section>
  );
});
