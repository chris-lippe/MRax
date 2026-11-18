import { useEffect, useRef, useState } from 'react';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { api, type TimecourseResult, type Workspace } from './api';
import { roiColor } from './scene';

type Draft = {
  input?: string;
  component?: string;
  statistic?: string;
  axis?: string;
  rois?: string[];
  respectExclusions?: boolean;
};

export function TimecourseControls({
  workspace: w,
  busy,
  onResult,
  viewer = false,
}: {
  workspace: Workspace | null;
  busy: boolean;
  onResult: (result: TimecourseResult | null) => void;
  viewer?: boolean;
}) {
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [results, setResults] = useState<Record<string, { key: string; result: TimecourseResult }>>(
    {},
  );
  const [pending, setPending] = useState('');
  const [error, setError] = useState<{ key: string; message: string } | null>(null);
  const request = useRef(0);
  const context = JSON.stringify([w?.id, w?.experiment]);
  const loaded = results[context];
  const draft = drafts[context] ?? {};
  const update = (patch: Partial<Draft>) =>
    setDrafts((previous) => ({ ...previous, [context]: { ...previous[context], ...patch } }));
  const axes = Object.keys(w?.dimensions ?? {}).filter(
    (name) => !['x', 'y', 'z'].includes(name) && w!.dimensions[name] > 1,
  );
  const axis = axes.includes(draft.axis ?? '') ? draft.axis! : (axes[0] ?? '');
  const rois = w?.rois.filter((roi) => roi.enabled && roi.pixels > 0) ?? [];
  const defaults = rois.find((roi) => roi.name === w?.selected_roi) ?? rois[0];
  const selected = (draft.rois ?? (defaults ? [defaults.name] : [])).filter((name) =>
    rois.some((roi) => roi.name === name),
  );
  const input =
    draft.input === 'reconstruction' && !w?.has_reconstruction ? 'image' : (draft.input ?? 'image');
  const options = {
    input,
    component:
      input === 'reconstruction' && ['real', 'imag'].includes(draft.component ?? '')
        ? 'auto'
        : (draft.component ?? 'auto'),
    statistic: draft.statistic ?? 'mean',
    axis,
    roi_names: selected,
    respect_exclusions: draft.respectExclusions ?? false,
  };
  const key = JSON.stringify([
    context,
    w?.revision,
    w?.indices,
    options,
    input === 'view' ? w?.visual_state : null,
    input === 'reconstruction'
      ? [w?.reconstruction_component, w?.reconstruction_coil, w?.unwrap_phase]
      : null,
  ]);
  const result = loaded?.key === key ? loaded.result : null;
  useEffect(() => onResult(result), [result, onResult]);

  return (
    <section className="timecourse-controls" aria-label="Timecourse analysis">
      <form
        className="control-layout"
        onSubmit={(event) => {
          event.preventDefault();
          if (!w || !axis || !selected.length) return;
          const attempt = ++request.current;
          setPending(key);
          setError(null);
          void api<TimecourseResult>(`/api/workspaces/${w.id}/analysis/timecourse`, 'POST', {
            ...options,
            revision: w.revision,
            view_revision: w.view_revision,
          })
            .then((data) => {
              if (attempt === request.current)
                setResults((previous) => ({ ...previous, [context]: { key, result: data } }));
            })
            .catch((failure) => {
              if (attempt === request.current) setError({ key, message: String(failure) });
            })
            .finally(() => {
              if (attempt === request.current) setPending('');
            });
        }}
      >
        <fieldset className="control-form" disabled={!w?.loaded || busy || pending === key}>
          <InspectorSection
            title="Signal and statistic"
            icon="timecourse"
            help="Summarize each full ROI volume along an acquisition axis. Other acquisition axes follow the current frame."
          >
            <div className="field-grid">
              <label>
                Input data
                <select
                  aria-label="Timecourse input"
                  value={input}
                  onChange={(event) => update({ input: event.target.value })}
                >
                  <option value="image">Current image data</option>
                  <option value="original">Original image data</option>
                  <option value="view">Image with active corrections</option>
                  <option value="reconstruction" disabled={!w?.has_reconstruction}>
                    {w?.reconstruction_label ?? 'Reconstruction'}
                  </option>
                </select>
              </label>
              <label>
                Component
                <select
                  aria-label="Timecourse component"
                  value={options.component}
                  onChange={(event) => update({ component: event.target.value })}
                >
                  <option value="auto">Automatic</option>
                  <option value="magnitude">Magnitude</option>
                  <option value="real" disabled={input === 'reconstruction'}>
                    Real
                  </option>
                  <option value="imag" disabled={input === 'reconstruction'}>
                    Imaginary
                  </option>
                  <option value="phase">Phase</option>
                </select>
              </label>
              <label>
                Statistic
                <select
                  aria-label="Timecourse statistic"
                  value={options.statistic}
                  onChange={(event) => update({ statistic: event.target.value })}
                >
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                  <option value="std">Standard deviation</option>
                  <option value="variance">Variance</option>
                  <option value="sum">Sum</option>
                </select>
              </label>
              {input === 'reconstruction' && (
                <p className="control-note">Coil selection and phase unwrapping follow View.</p>
              )}
              <label>
                Acquisition axis
                <select
                  aria-label="Timecourse axis"
                  value={axis}
                  onChange={(event) => update({ axis: event.target.value })}
                >
                  {!axes.length && <option value="">No acquisition series</option>}
                  {axes.map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
            </div>
          </InspectorSection>
          <InspectorSection title="ROIs and frames" icon="rois">
            <fieldset className="timecourse-rois">
              <legend>ROI selection</legend>
              {!rois.length && <p className="control-note">Create or enable ROIs in Segment.</p>}
              {rois.map((roi) => (
                <label className="check-label" key={roi.name}>
                  <input
                    type="checkbox"
                    aria-label={`Analyze ROI ${roi.name}`}
                    checked={selected.includes(roi.name)}
                    onChange={(event) =>
                      update({
                        rois: event.target.checked
                          ? [...selected, roi.name]
                          : selected.filter((name) => name !== roi.name),
                      })
                    }
                  />
                  <span
                    className="roi-swatch"
                    style={{ background: roiColor(roi.name, 1, w!.scene.roi_styles[roi.name]) }}
                  />
                  {roi.name}
                </label>
              ))}
              <div className="button-row">
                <button
                  type="button"
                  className="secondary"
                  disabled={!rois.length}
                  onClick={() => update({ rois: rois.map((roi) => roi.name) })}
                >
                  All ROIs
                </button>
                <button
                  type="button"
                  className="secondary"
                  disabled={!selected.length}
                  onClick={() => update({ rois: [] })}
                >
                  Clear selection
                </button>
              </div>
            </fieldset>
            <label className="check-label">
              <input
                type="checkbox"
                checked={options.respect_exclusions}
                onChange={(event) => update({ respectExclusions: event.target.checked })}
              />
              Respect fit frame exclusions
            </label>
          </InspectorSection>
          <div className="inspector-actionbar">
            <button className="primary full" disabled={!axis || !selected.length}>
              <WorkbenchIcon name="analyze" />
              {pending === key ? 'Analyzing…' : viewer ? 'Plot ROI curves' : 'Analyze timecourse'}
            </button>
          </div>
        </fieldset>
      </form>
      {error?.key === key && <p role="alert">{error.message}</p>}
      {result && (
        <p role="status">
          Timecourse ready: {result.series.length} {result.series.length === 1 ? 'ROI' : 'ROIs'},{' '}
          {result.x.length} {result.x.length === 1 ? 'sample' : 'samples'}.
        </p>
      )}
    </section>
  );
}
