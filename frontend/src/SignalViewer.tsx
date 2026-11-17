import { useMemo, useState, type RefObject } from 'react';
import type { Options, Workspace } from './api';
import { InspectorSection } from './InspectorSection';
import { SignalPlot, signalColour } from './SignalPlot';
import {
  automaticSignalAxes,
  defaultSignalOptions,
  finiteSignal,
  signalBounds,
  signalFitSummary,
  signalNumber,
  signalSummary,
  type SignalData,
  type SignalOptions,
  type SignalTrace,
} from './signal';

export function signalDomains(data: SignalData, traces: SignalTrace[], options: SignalOptions) {
  const x = signalBounds(data.x, options.xScale, options.xMin, options.xMax);
  const values = traces.flatMap((t) =>
    t.values.filter(
      (_, i) =>
        finiteSignal(data.x[i]) &&
        data.x[i]! >= x.domain[0] &&
        data.x[i]! <= x.domain[1] &&
        (options.excluded || data.included[i] !== false || t.fitted),
    ),
  );
  const y = signalBounds(values, options.yScale, options.yMin, options.yMax);
  return { x, y };
}

export function SignalViewer({
  active,
  data,
  options,
  update,
  figureRef,
}: {
  active: boolean;
  data: SignalData | null;
  options: SignalOptions;
  update: (patch: Partial<SignalOptions>) => void;
  figureRef: RefObject<SVGSVGElement | null>;
}) {
  const [page, setPage] = useState<'plot' | 'samples' | 'statistics'>('plot');
  const [tablePage, setTablePage] = useState(0);
  const traces = useMemo(
    () => data?.traces.filter((t) => !options.hidden.includes(t.id)) ?? [],
    [data, options.hidden],
  );
  const domains = useMemo(
    () => (data ? signalDomains(data, traces, options) : null),
    [data, traces, options],
  );
  const summaries = useMemo(
    () =>
      data && domains
        ? traces.map((trace) => ({
            trace,
            ...signalSummary(trace.values, data.x, data.included, domains.x.domain),
          }))
        : [],
    [data, domains, traces],
  );
  const fit = useMemo(
    () => (data && domains ? signalFitSummary(data, domains.x.domain) : null),
    [data, domains],
  );
  if (!active) return <section hidden aria-label="Scientific signal viewer" />;
  const error = !traces.length
    ? 'Select at least one curve.'
    : domains?.x.error || domains?.y.error;
  const pages = data ? Math.ceil(data.x.length / 100) : 0;
  const currentPage = Math.min(tablePage, Math.max(0, pages - 1));
  return (
    <section className="signal-viewer" aria-label="Scientific signal viewer">
      <header className="signal-viewer-heading">
        <div>
          <h2>Signal curves</h2>
          <p className="control-note">
            {data?.description ??
              (options.source === 'volume'
                ? 'Choose ROI volumes and plot their curves using the controls.'
                : options.source === 'fit'
                  ? 'Run a preview fit in Fit.'
                  : 'Open an experiment to view its signal profile.')}
          </p>
        </div>
        <div className="signal-pages" role="tablist" aria-label="Signal data views">
          {(['plot', 'samples', 'statistics'] as const).map((value) => (
            <button
              key={value}
              role="tab"
              id={'signal-tab-' + value}
              aria-selected={page === value}
              aria-controls={'signal-page-' + value}
              tabIndex={page === value ? 0 : -1}
              onClick={() => setPage(value)}
              onKeyDown={(event) => {
                const list = ['plot', 'samples', 'statistics'] as const;
                const index = list.indexOf(value);
                const next =
                  event.key === 'ArrowRight'
                    ? (index + 1) % 3
                    : event.key === 'ArrowLeft'
                      ? (index + 2) % 3
                      : event.key === 'Home'
                        ? 0
                        : event.key === 'End'
                          ? 2
                          : null;
                if (next !== null) {
                  event.preventDefault();
                  setPage(list[next]);
                  document.getElementById('signal-tab-' + list[next])?.focus();
                }
              }}
            >
              {value === 'plot' ? 'Plot' : value === 'samples' ? 'Samples' : 'Statistics'}
            </button>
          ))}
        </div>
      </header>
      {!data ? (
        <div className="signal-empty">No signal data selected.</div>
      ) : (
        <>
          <div
            id="signal-page-plot"
            role="tabpanel"
            aria-labelledby="signal-tab-plot"
            hidden={page !== 'plot'}
          >
            {error ? (
              <p role="alert" className="signal-empty">
                {error}
              </p>
            ) : (
              domains && (
                <SignalPlot
                  data={data}
                  traces={traces}
                  options={options}
                  xDomain={domains.x.domain}
                  yDomain={domains.y.domain}
                  svgRef={figureRef}
                  update={update}
                />
              )
            )}
          </div>
          <div
            id="signal-page-samples"
            role="tabpanel"
            aria-labelledby="signal-tab-samples"
            hidden={page !== 'samples'}
          >
            <div className="signal-table-tools">
              <span>{data.x.length} samples · full precision is preserved in CSV exports</span>
              <button disabled={currentPage === 0} onClick={() => setTablePage(currentPage - 1)}>
                Previous samples
              </button>
              <span>
                Page {currentPage + 1} / {pages}
              </span>
              <button
                disabled={currentPage >= pages - 1}
                onClick={() => setTablePage(currentPage + 1)}
              >
                Next samples
              </button>
            </div>
            <div
              className="signal-table-scroll"
              tabIndex={0}
              aria-label="Signal samples table scroll area"
            >
              <table aria-label="Scientific signal samples">
                <thead>
                  <tr>
                    <th scope="col">Sample</th>
                    <th scope="col">{data.xLabel}</th>
                    <th scope="col">Included in fit</th>
                    {traces.map((trace) => (
                      <th scope="col" key={trace.id}>
                        {trace.name}
                        {trace.counts ? ' · Value / valid n' : ''}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.x.slice(currentPage * 100, (currentPage + 1) * 100).map((x, j) => {
                    const i = currentPage * 100 + j;
                    return (
                      <tr key={i}>
                        <th scope="row">{i + 1}</th>
                        <td>{signalNumber(x, 12)}</td>
                        <td>{data.included[i] === false ? 'No' : 'Yes'}</td>
                        {traces.map((trace) => (
                          <td key={trace.id}>
                            {signalNumber(trace.values[i], 12)}
                            {trace.counts ? ` / ${trace.counts[i]}` : ''}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <div
            id="signal-page-statistics"
            role="tabpanel"
            aria-labelledby="signal-tab-statistics"
            hidden={page !== 'statistics'}
          >
            <p className="control-note">
              Statistics use included, finite samples in the current X range. SD uses n − 1. Y
              display limits affect only the plot.
            </p>
            <div
              className="signal-table-scroll"
              tabIndex={0}
              aria-label="Signal statistics table scroll area"
            >
              <table aria-label="Signal range statistics">
                <thead>
                  <tr>
                    {['Curve', 'Valid n', 'Mean', 'SD', 'Median', 'Minimum', 'Maximum'].map((v) => (
                      <th scope="col" key={v}>
                        {v}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {summaries.map((s) => (
                    <tr key={s.trace.id}>
                      <th scope="row">{s.trace.name}</th>
                      <td>{s.n}</td>
                      <td>{signalNumber(s.mean)}</td>
                      <td>{signalNumber(s.sd)}</td>
                      <td>{signalNumber(s.median)}</td>
                      <td>{signalNumber(s.min)}</td>
                      <td>{signalNumber(s.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {fit && (
              <section className="signal-fit-summary" aria-label="Signal fit diagnostics">
                <h3>Fit diagnostics</h3>
                <p className="control-note">
                  Observed − fitted, using finite included pairs in the X range. R² is undefined for
                  a constant signal.
                </p>
                <dl>
                  <dt>Paired samples</dt>
                  <dd>{fit.n}</dd>
                  <dt>Mean residual</dt>
                  <dd>{signalNumber(fit.bias)}</dd>
                  <dt>RMSE</dt>
                  <dd>{signalNumber(fit.rmse)}</dd>
                  <dt>R²</dt>
                  <dd>{signalNumber(fit.r2)}</dd>
                </dl>
              </section>
            )}
          </div>
        </>
      )}
    </section>
  );
}

export function SignalSourceControls({
  workspace: w,
  options,
  update,
  busy,
  changeView,
}: {
  workspace: Workspace | null;
  options: SignalOptions;
  busy: boolean;
  update: (patch: Partial<SignalOptions>) => void;
  changeView: (command: Options) => Promise<void>;
}) {
  const rois = w?.rois.filter((roi) => roi.enabled && roi.pixels > 0) ?? [];
  return (
    <InspectorSection title="Signal data" icon="timecourse">
      <fieldset className="control-form" disabled={busy || !w?.loaded}>
        <label>
          Source
          <select
            aria-label="Signal source"
            value={options.source}
            onChange={(event) =>
              update({
                source: event.target.value as SignalOptions['source'],
                residuals: false,
                hidden: [],
                ...automaticSignalAxes,
              })
            }
          >
            <option value="profile">
              {w?.non_imaging ? 'Current signal' : 'Current slice profile'}
            </option>
            <option value="fit" disabled={!w?.preview}>
              Saved fit input
            </option>
            <option value="volume" disabled={!rois.length}>
              ROI volume curves
            </option>
          </select>
        </label>
        {options.source !== 'volume' && rois.length > 0 && (
          <label>
            ROI
            <select
              aria-label="Signal ROI"
              value={(options.source === 'fit' ? w?.preview : w?.timecourse)?.roi_name ?? ''}
              onChange={(event) => {
                const roi = event.target.value;
                update({ source: 'profile', ...automaticSignalAxes });
                void changeView(roi ? { selected_roi: roi } : { plot_whole_image: true });
              }}
            >
              <option value="">Whole image mean</option>
              {rois.map((roi) => (
                <option key={roi.name}>{roi.name}</option>
              ))}
            </select>
          </label>
        )}
        {options.source === 'profile' && (
          <label>
            Component
            <select
              aria-label="Signal component"
              value={String(w?.display.component ?? 'auto')}
              onChange={(event) => void changeView({ component: event.target.value })}
            >
              <option value="auto">Automatic</option>
              <option value="magnitude">Magnitude</option>
              <option value="real">Real</option>
              <option value="imag">Imaginary</option>
              <option value="phase">Phase</option>
            </select>
          </label>
        )}
        {options.source === 'profile' && (
          <label>
            Profile axis
            <select
              aria-label="Signal profile axis"
              value={String(w?.display.plot_axis ?? 'auto')}
              onChange={(event) => {
                update(automaticSignalAxes);
                void changeView({ plot_axis: event.target.value });
              }}
            >
              <option value="auto">Automatic</option>
              {Object.keys(w?.dimensions ?? {}).map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
        )}
        {options.source === 'fit' && (
          <label>
            Fit display
            <select
              aria-label="Signal fit display"
              value={options.residuals ? 'residual' : 'signal'}
              onChange={(event) =>
                update({
                  residuals: event.target.value === 'residual',
                  hidden: [],
                  ...automaticSignalAxes,
                })
              }
            >
              <option value="signal">Observed and fitted</option>
              <option value="residual">Residuals</option>
            </select>
          </label>
        )}
      </fieldset>
      <p className="control-note">
        {options.source === 'profile'
          ? (rois.length
              ? 'ROI mean in the current slice.'
              : 'Whole-image mean in the current slice.') +
            ' Other coordinates follow the workspace.'
          : options.source === 'fit'
            ? 'Recorded observations and predictions from the preview fit.'
            : 'Curves summarize whole ROI volumes. Other acquisition coordinates follow the workspace.'}
      </p>
    </InspectorSection>
  );
}

export function SignalAppearanceControls({
  data,
  options,
  update,
}: {
  data: SignalData | null;
  options: SignalOptions;
  update: (patch: Partial<SignalOptions>) => void;
}) {
  return (
    <div className="control-layout signal-appearance">
      <InspectorSection title="Axes" icon="settings" className="control-wide">
        <div className="field-grid">
          {(['x', 'y'] as const).map((axis) => (
            <div key={axis} className="signal-axis-controls">
              <label>
                {axis.toUpperCase()} scale
                <select
                  aria-label={`Signal ${axis.toUpperCase()} scale`}
                  value={axis === 'x' ? options.xScale : options.yScale}
                  onChange={(e) =>
                    update(
                      axis === 'x'
                        ? { xScale: e.target.value as SignalOptions['xScale'], xMin: '', xMax: '' }
                        : { yScale: e.target.value as SignalOptions['yScale'], yMin: '', yMax: '' },
                    )
                  }
                >
                  <option value="linear">Linear</option>
                  <option value="log">Logarithmic (base 10)</option>
                </select>
              </label>
              <div className="field-grid">
                {(['Min', 'Max'] as const).map((bound) => {
                  const key = `${axis}${bound}` as 'xMin' | 'xMax' | 'yMin' | 'yMax';
                  return (
                    <label key={bound}>
                      {bound}
                      <input
                        aria-label={`Signal ${axis.toUpperCase()} ${bound.toLowerCase()}`}
                        inputMode="decimal"
                        placeholder="Auto"
                        value={options[key]}
                        onChange={(event) => update({ [key]: event.target.value })}
                      />
                    </label>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <button className="secondary" onClick={() => update(automaticSignalAxes)}>
          Auto range
        </button>
      </InspectorSection>
      <InspectorSection title="Curves" icon="layers" className="control-wide">
        {!data ? (
          <p className="control-note">Choose a signal source first.</p>
        ) : (
          <div className="signal-trace-list">
            {data.traces.map((trace, i) => (
              <label className="check-label" key={trace.id}>
                <input
                  type="checkbox"
                  aria-label={'Plot ' + trace.name}
                  checked={!options.hidden.includes(trace.id)}
                  onChange={(e) =>
                    update({
                      hidden: e.target.checked
                        ? options.hidden.filter((id) => id !== trace.id)
                        : [...options.hidden, trace.id],
                    })
                  }
                />
                <i
                  className="signal-trace-swatch"
                  style={{ background: signalColour(i, options.theme) }}
                />
                <span>{trace.name}</span>
              </label>
            ))}
          </div>
        )}
        {(['lines', 'markers', 'excluded', 'grid', 'legend'] as const).map((key) => (
          <label className="check-label" key={key}>
            <input
              type="checkbox"
              aria-label={'Signal ' + key}
              checked={options[key]}
              onChange={(e) => update({ [key]: e.target.checked })}
            />
            {
              {
                lines: 'Connect observations',
                markers: 'Show sample markers',
                excluded: 'Show excluded samples',
                grid: 'Show grid',
                legend: 'Show legend',
              }[key]
            }
          </label>
        ))}
      </InspectorSection>
      <details className="control-advanced control-wide">
        <summary>Figure appearance</summary>
        <div className="field-grid">
          <label>
            Figure theme
            <select
              aria-label="Signal figure theme"
              value={options.theme}
              onChange={(e) => update({ theme: e.target.value as SignalOptions['theme'] })}
            >
              <option value="auto">Interface theme</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </label>
          <label>
            Text size
            <select
              aria-label="Signal text size"
              value={options.textSize}
              onChange={(e) => update({ textSize: Number(e.target.value) })}
            >
              {[14, 16, 18, 20, 24].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
          <label>
            Line width
            <input
              type="number"
              aria-label="Signal line width"
              min={1}
              max={5}
              step={0.5}
              value={options.lineWidth}
              onChange={(e) =>
                update({ lineWidth: Math.max(1, Math.min(5, Number(e.target.value) || 1)) })
              }
            />
          </label>
        </div>
        <label>
          Title
          <input
            aria-label="Signal figure title"
            maxLength={120}
            value={options.title}
            onChange={(e) => update({ title: e.target.value })}
          />
        </label>
        <label>
          X title
          <input
            aria-label="Signal X title"
            maxLength={100}
            placeholder={data?.xLabel ?? 'Automatic'}
            value={options.xLabel}
            onChange={(e) => update({ xLabel: e.target.value })}
          />
        </label>
        <label>
          Y title
          <input
            aria-label="Signal Y title"
            maxLength={100}
            placeholder={data?.yLabel ?? 'Automatic'}
            value={options.yLabel}
            onChange={(e) => update({ yLabel: e.target.value })}
          />
        </label>
        <p className="control-note">
          Titles and axes change presentation only. Save the figure or source values from Export
          data.
        </p>
        <button
          className="secondary"
          onClick={() =>
            update({
              ...defaultSignalOptions,
              source: options.source,
              residuals: options.residuals,
            })
          }
        >
          Reset plot style
        </button>
      </details>
    </div>
  );
}
