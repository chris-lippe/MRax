import { useEffect, useState } from 'react';
import { api, type Options } from './api';
import type { AnalysisProps } from './analysisTypes';
import { JsonEditor, NumberField, Report } from './Controls';
import { fitRois, roiColor } from './scene';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';

type Spec = { initial: number; bounds: [number, number] | null; unit: string };
type FitChoices = {
  pools: string[];
  default_pools: string[];
  reference_scans: string[];
  prior_maps: Record<string, { ref: string; label: string }[]>;
};
export function FitControls({
  workspace: w,
  busy,
  action,
  changeView,
  run,
  openWorkflows,
}: AnalysisProps) {
  const [choices, setChoices] = useState<FitChoices | null>(null);
  const [regularizers, setRegularizers] = useState<Record<string, Options[]>>({}),
    [constants, setConstants] = useState<Record<string, string[]>>({});
  const [previewSteps, setPreviewSteps] = useState(600);
  const [specs, setSpecs] = useState<Record<string, Spec>>({}),
    [overrides, setOverrides] = useState<Record<string, Options>>({}),
    [excluded, setExcluded] = useState('');
  const [steps, setSteps] = useState(Number(w?.fit_settings.num_steps ?? 600)),
    [rate, setRate] = useState(Number(w?.fit_settings.step_size ?? 0.02));
  const [device, setDevice] = useState('cpu'),
    [devices, setDevices] = useState<{ id: string; label: string }[]>([]);
  const useRoi = w?.scene.fit_region !== 'all';
  const selectedRois = w ? fitRois(w) : [];
  const [error, setError] = useState('');
  useEffect(() => {
    void api<{ devices: { id: string; label: string }[] }>('/api/catalog')
      .then((c) => setDevices(c.devices))
      .catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    let cancelled = false;
    if (w?.loaded)
      void api<FitChoices>(`/api/workspaces/${w.id}/fit-options`)
        .then((value) => {
          if (!cancelled) setChoices(value);
        })
        .catch((e) => {
          if (!cancelled) setError(String(e));
        });
    if (w?.loaded)
      void api<Record<string, Spec>>(`/api/workspaces/${w.id}/parameters`)
        .then((value) => {
          if (!cancelled) setSpecs(value);
        })
        .catch((e) => {
          if (!cancelled) setError(String(e));
        });
    setSteps(Number(w?.fit_settings.num_steps ?? 600));
    setRate(Number(w?.fit_settings.step_size ?? 0.02));
    const values = { ...(w?.fit_settings.param_overrides ?? {}) } as Record<string, Options>;
    for (const [pool, parameters] of Object.entries(
      (w?.fit_settings.pool_overrides ?? {}) as Record<string, Record<string, Options>>,
    ))
      for (const [parameter, value] of Object.entries(parameters))
        values[`${parameter}_${pool}`] = value;
    setOverrides(values);
    setRegularizers((w?.fit_settings.regularizer_configs ?? {}) as Record<string, Options[]>);
    setConstants((w?.fit_settings.parameter_constant_dims ?? {}) as Record<string, string[]>);
    setExcluded(w?.frame_exclusions.join(', ') ?? '');
    return () => {
      cancelled = true;
    };
  }, [w?.id, w?.revision, w?.experiment]);
  const setParameter = (name: string, key: string, value: unknown) =>
    setOverrides((p) => ({
      ...p,
      [name]: {
        ...p[name],
        [key]: value,
        ...(key === 'bounds' ? { transform: 'bounded' } : {}),
        ...(key === 'init' && p[name]?.trainable === false ? { value } : {}),
      },
    }));
  const configuration = () => {
    const plain: Record<string, Options> = {},
      pools: Record<string, Record<string, Options>> = {};
    for (const [name, value] of Object.entries(overrides)) {
      const pool = choices?.pools.find((p) => name.endsWith(`_${p}`));
      if (pool) {
        pools[pool] ??= {};
        pools[pool][name.slice(0, -pool.length - 1)] = value;
      } else plain[name] = value;
    }
    return {
      num_steps: steps,
      step_size: rate,
      param_overrides: plain,
      pool_overrides: pools,
      regularizer_configs: regularizers,
      parameter_constant_dims: constants,
    };
  };
  const apply = () => action('fit.settings', configuration());
  return (
    <div className="control-layout fit-controls">
      <button
        className="secondary control-wide workflow-library-trigger"
        disabled={busy}
        onClick={() =>
          openWorkflows?.('fitting', {
            settings: configuration(),
            use_roi: useRoi,
            roi_names: useRoi ? (w?.scene.fit_rois ?? null) : null,
            excluded_frames: excluded
              .split(',')
              .filter((value) => value.trim())
              .map(Number),
          })
        }
      >
        <WorkbenchIcon name="pipeline" />
        Fitting workflows
      </button>
      {error && <p role="alert">{error}</p>}
      <InspectorSection title="Model and solver" icon="fit" className="fit-model">
        <div className="field-grid">
          <label className="field-wide">
            Model
            <select
              aria-label="Model"
              disabled={!w?.loaded || busy}
              value={w?.model ?? 'identity'}
              onChange={(e) =>
                void action('fit.settings', {
                  model_name: e.target.value,
                  param_overrides: {},
                  pool_overrides: {},
                  selected_pools: [],
                  regularizer_configs: {},
                  parameter_constant_dims: {},
                })
              }
            >
              {w?.models.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <div className="bounds-grid">
            <NumberField
              label="Fit steps"
              value={steps}
              min={1}
              step="1"
              disabled={!w?.loaded || busy}
              onChange={setSteps}
            />
            <NumberField
              label="Step size"
              value={rate}
              min={1e-8}
              disabled={!w?.loaded || busy}
              onChange={setRate}
            />
          </div>
          <label>
            Optimizer
            <select
              aria-label="Optimizer"
              disabled={!w || busy}
              value={String(w?.fit_settings.optimizer_name ?? 'auto')}
              onChange={(e) => void action('fit.settings', { optimizer_name: e.target.value })}
            >
              {['auto', 'adam', 'adamw', 'sgd', 'rmsprop', 'fista', 'split_bregman'].map((v) => (
                <option key={v}>{v}</option>
              ))}
            </select>
          </label>
          <label>
            Compute device
            <select
              aria-label="Compute device"
              value={device}
              onChange={(e) => setDevice(e.target.value)}
            >
              <option value="cpu">CPU</option>
              {devices
                .filter((d) => !d.id.startsWith('cpu'))
                .map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.label}
                  </option>
                ))}
            </select>
          </label>
        </div>
      </InspectorSection>
      <InspectorSection
        title="Preview"
        icon="timecourse"
        className="fit-preview"
        help="Preview fits use the selected ROI mean signal. Full map fits use the separate fitting region."
      >
        <label>
          ROI for preview
          <select
            aria-label="Fit ROI"
            value={w?.selected_roi ?? ''}
            disabled={!w || busy}
            onChange={(event) => void changeView({ selected_roi: event.target.value })}
          >
            <option value="" disabled>
              Select an ROI
            </option>
            {w?.rois.map((roi) => (
              <option key={roi.name} value={roi.name} disabled={!roi.enabled || !roi.pixels}>
                {roi.name} · {roi.pixels} voxels
              </option>
            ))}
          </select>
        </label>
        <button
          className="secondary full"
          disabled={
            !w?.rois.some((r) => r.name === w.selected_roi && r.enabled && r.pixels > 0) || busy
          }
          onClick={() =>
            void run('preview', {
              roi_name: w?.selected_roi,
              steps: previewSteps,
              device,
              settings: configuration(),
            })
          }
        >
          <WorkbenchIcon name="fit" />
          Run preview fit
        </button>
        {w?.preview && (
          <div className="fit-result" aria-label="Preview fit results">
            {Object.entries(w.preview.parameters ?? {}).map(([name, value]) => (
              <div key={name}>
                <span>{name === 't2' ? 'T2' : name}</span>
                <strong aria-label={name === 't2' ? 'Fitted T2' : `Fitted ${name}`}>
                  {value.toFixed(2)}
                </strong>
                <small>{specs[name]?.unit ?? ''}</small>
              </div>
            ))}
            <div>
              <span>Relative RMSE</span>
              <span aria-label="Relative RMSE">{w.preview.relative_rmse?.toExponential(3)}</span>
            </div>
          </div>
        )}
      </InspectorSection>
      <InspectorSection
        title="Map fitting region"
        icon="rois"
        className="fit-region-section"
        help="Fits use the current experiment state. Selected ROIs are combined; display visibility does not affect fitting."
      >
        <fieldset className="fit-region" disabled={!w?.loaded || busy}>
          <legend>Map fitting region</legend>
          <label>
            Fit region
            <select
              aria-label="Fit region"
              value={useRoi ? 'rois' : 'all'}
              onChange={(event) => void changeView({ scene: { fit_region: event.target.value } })}
            >
              <option value="rois">Selected ROIs</option>
              <option value="all">Whole image</option>
            </select>
          </label>
          {useRoi && (
            <>
              {w?.rois
                .filter((r) => r.enabled)
                .map((roi) => (
                  <label className="check-label" key={roi.name}>
                    <input
                      type="checkbox"
                      aria-label={`Fit mask ${roi.name}`}
                      checked={selectedRois.includes(roi.name)}
                      disabled={!roi.pixels}
                      onChange={(event) =>
                        void changeView({
                          scene: {
                            fit_rois: event.target.checked
                              ? [...selectedRois, roi.name]
                              : selectedRois.filter((name) => name !== roi.name),
                          },
                        })
                      }
                    />
                    <span
                      className="roi-swatch"
                      style={{ background: roiColor(roi.name, 1, w.scene.roi_styles[roi.name]) }}
                    />
                    {roi.name}
                  </label>
                ))}
              <div className="button-row">
                <button
                  className="secondary"
                  onClick={() => void changeView({ scene: { fit_rois: null } })}
                >
                  All ROIs
                </button>
                <button
                  className="secondary"
                  onClick={() => void changeView({ scene: { fit_rois: [] } })}
                >
                  Clear selection
                </button>
              </div>
              {!selectedRois.length && (
                <p className="control-note">
                  Select at least one nonempty ROI, or choose Whole image.
                </p>
              )}
            </>
          )}
        </fieldset>
        <button
          className="primary full"
          disabled={!w?.loaded || busy || (useRoi && !selectedRois.length)}
          data-tutorial="fit-apply"
          onClick={() =>
            void run('full_fit', {
              roi_name: w?.selected_roi,
              use_roi: useRoi,
              roi_names: selectedRois,
              device,
              settings: configuration(),
            })
          }
        >
          <WorkbenchIcon name="fit" />
          Run full map fit
        </button>
      </InspectorSection>
      <details className="control-advanced control-wide">
        <summary>Parameters and fit options</summary>
        {!!choices?.pools.length && (
          <label>
            Model pools
            <select
              aria-label="Model pools"
              multiple
              value={
                (w?.fit_settings.selected_pools as string[] | undefined)?.length
                  ? (w!.fit_settings.selected_pools as string[])
                  : choices.default_pools
              }
              onChange={(e) =>
                void action('fit.settings', {
                  selected_pools: [...e.target.selectedOptions].map((o) => o.value),
                  param_overrides: {},
                  pool_overrides: {},
                  regularizer_configs: {},
                  parameter_constant_dims: {},
                })
              }
            >
              {choices.pools.map((pool) => (
                <option key={pool}>{pool}</option>
              ))}
            </select>
          </label>
        )}
        <NumberField
          label="Preview steps"
          value={previewSteps}
          min={1}
          step="1"
          onChange={setPreviewSteps}
        />
        <div className="parameter-table">
          {Object.entries(specs).map(([name, spec]) => (
            <fieldset key={name}>
              <legend>
                {name} <small>{spec.unit}</small>
              </legend>
              <NumberField
                label={`${name} initial`}
                value={Number(overrides[name]?.init ?? spec.initial)}
                onChange={(v) => setParameter(name, 'init', v)}
              />
              {spec.bounds && (
                <div className="bounds-grid">
                  <NumberField
                    label={`${name} lower bound`}
                    value={Number(
                      (overrides[name]?.bounds as number[] | undefined)?.[0] ?? spec.bounds[0],
                    )}
                    onChange={(v) =>
                      setParameter(name, 'bounds', [
                        v,
                        (overrides[name]?.bounds as number[] | undefined)?.[1] ?? spec.bounds![1],
                      ])
                    }
                  />
                  <NumberField
                    label={`${name} upper bound`}
                    value={Number(
                      (overrides[name]?.bounds as number[] | undefined)?.[1] ?? spec.bounds[1],
                    )}
                    onChange={(v) =>
                      setParameter(name, 'bounds', [
                        (overrides[name]?.bounds as number[] | undefined)?.[0] ?? spec.bounds![0],
                        v,
                      ])
                    }
                  />
                </div>
              )}
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={overrides[name]?.trainable === false}
                  onChange={(e) =>
                    setOverrides((p) => ({
                      ...p,
                      [name]: e.target.checked
                        ? {
                            ...p[name],
                            trainable: false,
                            value: Number(p[name]?.init ?? spec.initial),
                            transform: 'identity',
                          }
                        : { init: Number(p[name]?.value ?? spec.initial) },
                    }))
                  }
                />
                Fix {name}
              </label>
              <label>
                Regularizer
                <select
                  aria-label={`${name} regularizer`}
                  value={String(regularizers[name]?.[0]?.type ?? 'none')}
                  onChange={(e) =>
                    setRegularizers((p) => ({
                      ...p,
                      [name]:
                        e.target.value === 'none'
                          ? []
                          : [
                              {
                                ...p[name]?.[0],
                                type: e.target.value,
                                weight: p[name]?.[0]?.weight ?? 0.01,
                              },
                            ],
                    }))
                  }
                >
                  {['none', 'smoothness', 'tv', 'tv2'].map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
              </label>
              {!!regularizers[name]?.length && (
                <>
                  <NumberField
                    label={`${name} regularization weight`}
                    value={Number(regularizers[name][0].weight ?? 0)}
                    min={0}
                    onChange={(v) =>
                      setRegularizers((p) => ({ ...p, [name]: [{ ...p[name][0], weight: v }] }))
                    }
                  />
                  <label>
                    Regularization axes
                    <input
                      aria-label={`${name} regularization axes`}
                      value={((regularizers[name][0].axes ?? []) as string[]).join(',')}
                      onChange={(e) =>
                        setRegularizers((p) => ({
                          ...p,
                          [name]: [
                            {
                              ...p[name][0],
                              axes: e.target.value
                                .split(',')
                                .map((v) => v.trim())
                                .filter(Boolean),
                            },
                          ],
                        }))
                      }
                    />
                  </label>
                </>
              )}
              <label>
                Constant dimensions
                <input
                  aria-label={`${name} constant dimensions`}
                  value={(constants[name] ?? []).join(',')}
                  onChange={(e) =>
                    setConstants((p) => ({
                      ...p,
                      [name]: e.target.value
                        .split(',')
                        .map((v) => v.trim())
                        .filter(Boolean),
                    }))
                  }
                />
              </label>
            </fieldset>
          ))}
        </div>
        <button className="secondary full" disabled={!w || busy} onClick={() => void apply()}>
          Apply fit configuration
        </button>
        <label>
          Excluded frames (zero based)
          <input
            aria-label="Excluded frames"
            value={excluded}
            onChange={(e) => setExcluded(e.target.value)}
          />
        </label>
        <button
          className="secondary full"
          disabled={!w || busy}
          onClick={() =>
            void action('frames.exclude', {
              indices: excluded.trim() ? excluded.split(',').map(Number) : [],
            })
          }
        >
          Apply frame exclusions
        </button>

        <label className="check-label">
          <input
            type="checkbox"
            checked={Boolean(w?.fit_settings.auto_mask_nan_observables)}
            disabled={!w || busy}
            onChange={(e) =>
              void action('fit.settings', { auto_mask_nan_observables: e.target.checked })
            }
          />
          Automatically exclude nonfinite observables
        </label>

        <label>
          Derived maps
          <select
            aria-label="Derived maps"
            value={
              w?.fit_settings.compute_derived_maps === null
                ? 'default'
                : String(w?.fit_settings.compute_derived_maps ?? 'default')
            }
            onChange={(e) =>
              void action('fit.settings', {
                compute_derived_maps:
                  e.target.value === 'default' ? null : e.target.value === 'true',
              })
            }
          >
            <option value="default">Model default</option>
            <option value="true">Compute</option>
            <option value="false">Skip</option>
          </select>
        </label>
        <label>
          Reference scan
          <select
            aria-label="Reference scan"
            value={String(w?.fit_settings.sat_trans_reference_scan ?? '')}
            onChange={(e) =>
              void action('fit.settings', { sat_trans_reference_scan: e.target.value || null })
            }
          >
            <option value="">None</option>
            {choices?.reference_scans.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
        {['t1', 'b1', 'b0'].map((kind) => (
          <label key={kind}>
            {kind.toUpperCase()} prior map
            <select
              aria-label={`${kind.toUpperCase()} prior map`}
              value={String(w?.fit_settings[`ab_mt_${kind}_map_ref`] ?? '')}
              onChange={(e) =>
                void action('fit.settings', { [`ab_mt_${kind}_map_ref`]: e.target.value || null })
              }
            >
              <option value="">None</option>
              {choices?.prior_maps[kind].map((map) => (
                <option key={map.ref} value={map.ref}>
                  {map.label}
                </option>
              ))}
            </select>
          </label>
        ))}
        <JsonEditor
          label="Advanced fit settings"
          value={w?.fit_settings ?? {}}
          disabled={!w || busy}
          onApply={(value) => action('fit.settings', value)}
        />
        <p className="muted">
          Advanced settings include selected_pools, pool_overrides, parameter_constant_dims,
          regularizer_configs (smoothness / tv / tv2), derived maps, reference scans and prior-map
          references.
        </p>
      </details>
      {w?.fit_info && (
        <details>
          <summary>Latest full-fit history and diagnostics</summary>
          <Report value={w.fit_info} label="Fit diagnostics" />
        </details>
      )}
    </div>
  );
}
