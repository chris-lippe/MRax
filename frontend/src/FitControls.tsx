import { useEffect, useRef, useState } from 'react';
import { api, type Options } from './api';
import type { AnalysisProps } from './analysisTypes';
import { JsonEditor, NumberField, Report } from './Controls';
import { fitRois, roiColor } from './scene';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { PoolEditor } from './PoolEditor';
import { FitReadiness } from './FitReadiness';
import { FitParameterEditor, type ParameterSpec } from './FitParameterEditor';
import type { FitDimension } from './FitDimensionSelector';
import { FitFrameSelector } from './FitFrameSelector';
import { WorkflowButton } from './WorkflowButton';
import { usePipelineOptions } from './usePipelineOptions';

type FitChoices = {
  workspace_id: string;
  experiment: string;
  model: string;
  inputs: { reference_scan: boolean; prior_maps: string[]; derived_maps: boolean };
  pools: string[];
  default_pools: string[];
  field_dimensions: FitDimension[];
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
  editingStep,
  stagePipelineOptions,
}: AnalysisProps) {
  const editing = editingStep?.kind === 'fit';
  const edited = useRef(false);
  const [choices, setChoices] = useState<FitChoices | null>(null);
  const [loadedSetup, setLoadedSetup] = useState('');
  const setupContext = JSON.stringify([w?.id, w?.revision, w?.experiment, w?.model]);
  const setupLoading = Boolean(w?.loaded && loadedSetup !== setupContext);
  const [regularizers, setRegularizers] = useState<Record<string, Options[]>>({}),
    [constants, setConstants] = useState<Record<string, (string | number)[]>>({});
  const [settingsDraft, setSettingsDraft] = useState<Options>({});
  const [framesExpanded, setFramesExpanded] = useState(false);
  const [previewSteps, setPreviewSteps] = useState(600);
  const [specs, setSpecs] = useState<Record<string, ParameterSpec>>({}),
    [overrides, setOverrides] = useState<Record<string, Options>>({}),
    [excluded, setExcluded] = useState<number[]>([]);
  const [steps, setSteps] = useState(Number(w?.fit_settings.num_steps ?? 600)),
    [rate, setRate] = useState(Number(w?.fit_settings.step_size ?? 0.02));
  const [device, setDevice] = useState(String(editingStep?.options.device ?? 'cpu')),
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
    setError('');
    if (w?.loaded) {
      const request = editing
        ? api<{ choices: FitChoices; parameters: Record<string, ParameterSpec> }>(
            '/api/workspaces/' + w.id + '/fit-options',
            'POST',
            { revision: w.revision, settings: w.fit_settings },
          ).then((result) => [result.choices, result.parameters] as const)
        : Promise.all([
            api<FitChoices>('/api/workspaces/' + w.id + '/fit-options'),
            api<Record<string, ParameterSpec>>('/api/workspaces/' + w.id + '/parameters'),
          ]);
      void request
        .then(([options, parameters]) => {
          if (cancelled) return;
          setChoices(options);
          setSpecs(parameters);
          setLoadedSetup(setupContext);
        })
        .catch((error) => {
          if (!cancelled) setError(String(error));
        });
    } else {
      setChoices(null);
      setSpecs({});
    }
    setSteps(Number(w?.fit_settings.num_steps ?? 600));
    setRate(Number(w?.fit_settings.step_size ?? 0.02));
    const values = { ...(w?.fit_settings.param_overrides ?? {}) } as Record<string, Options>;
    for (const [pool, parameters] of Object.entries(
      (w?.fit_settings.pool_overrides ?? {}) as Record<string, Record<string, Options>>,
    ))
      for (const [parameter, value] of Object.entries(parameters)) {
        if (value && typeof value === 'object' && !Array.isArray(value))
          values[parameter + '_' + pool] = value;
      }
    setOverrides(values);
    setRegularizers((w?.fit_settings.regularizer_configs ?? {}) as Record<string, Options[]>);
    setConstants(
      (w?.fit_settings.parameter_constant_dims ?? {}) as Record<string, (string | number)[]>,
    );
    setExcluded(w?.frame_exclusions ?? []);
    setSettingsDraft({});
    return () => {
      cancelled = true;
    };
  }, [w?.id, w?.revision, w?.experiment, w?.model, setupContext, editing]);
  const fitInputs =
    choices?.workspace_id === w?.id &&
    choices?.experiment === w?.experiment &&
    choices?.model === w?.model
      ? choices?.inputs
      : null;
  const activePools =
    fitInputs && choices && choices.default_pools.length > 0
      ? (((w?.fit_settings.selected_pools as string[] | undefined)?.length
          ? w!.fit_settings.selected_pools
          : choices.default_pools) as string[])
      : [];
  const poolsByLength = [...activePools].sort((left, right) => right.length - left.length);
  const parameterPool = (name: string) => poolsByLength.find((pool) => name.endsWith('_' + pool));
  const advancedSettings = Object.fromEntries(
    Object.entries(w?.fit_settings ?? {}).filter(([key]) =>
      ['mask_fill_value', 'use_view_state_data'].includes(key),
    ),
  );
  const settingValue = (key: string) =>
    Object.hasOwn(settingsDraft, key) ? settingsDraft[key] : w?.fit_settings[key];
  const updateSetting = (key: string, value: unknown) =>
    setSettingsDraft((old) => ({ ...old, [key]: value }));
  const setParameter = (name: string, key: string, value: unknown) =>
    setOverrides((p) => ({
      ...p,
      [name]: {
        ...p[name],
        [key]: value,
        ...(key === 'bounds'
          ? {
              transform:
                (p[name]?.trainable ?? specs[name]?.trainable) === false ? 'identity' : 'bounded',
            }
          : {}),
        ...(key === 'init' && (p[name]?.trainable ?? specs[name]?.trainable) === false
          ? { value, trainable: false, transform: 'identity' }
          : {}),
      },
    }));
  const setFixedParameter = (name: string, fixed: boolean) =>
    setOverrides((previous) => {
      const old = previous[name] ?? {};
      const initial = Number(old.init ?? old.value ?? specs[name].initial);
      const bounds = old.bounds ?? specs[name].bounds;
      const next: Options = {
        ...old,
        init: initial,
        trainable: !fixed,
        transform: fixed || !bounds ? 'identity' : 'bounded',
      };
      if (fixed) next.value = initial;
      else delete next.value;
      return { ...previous, [name]: next };
    });
  const configuration = () => {
    const plain: Record<string, Options> = {};
    const pools = structuredClone(
      (w?.fit_settings.pool_overrides ?? {}) as Record<string, Options>,
    );
    const poolNames = [
      ...new Set([
        ...(fitInputs ? (choices?.pools ?? []) : []),
        ...Object.keys(pools),
        ...((w?.fit_settings.selected_pools ?? []) as string[]),
      ]),
    ].sort((left, right) => right.length - left.length);
    for (const [name, value] of Object.entries(overrides)) {
      const pool = poolNames.find((pool) => name.endsWith('_' + pool));
      if (pool) {
        pools[pool] ??= {};
        pools[pool][name.slice(0, -pool.length - 1)] = value;
      } else plain[name] = value;
    }
    return {
      ...w?.fit_settings,
      ...settingsDraft,
      num_steps: steps,
      step_size: rate,
      param_overrides: plain,
      pool_overrides: pools,
      regularizer_configs: regularizers,
      parameter_constant_dims: constants,
    };
  };
  const apply = () => action('fit.settings', { ...configuration(), excluded_frames: excluded });
  usePipelineOptions(
    Boolean(editing && !setupLoading),
    edited.current,
    {
      settings: configuration(),
      use_roi: useRoi,
      roi_names: selectedRois,
      device,
      excluded_frames: excluded,
    },
    stagePipelineOptions,
    useRoi && !selectedRois.length ? 'Choose a fitting ROI or Whole image.' : undefined,
  );
  const renderParameter = (name: string, spec: ParameterSpec, caption = name) => (
    <FitParameterEditor
      key={name}
      name={name}
      caption={caption}
      spec={spec}
      override={overrides[name]}
      regularizers={regularizers[name]}
      constants={constants[name]}
      dimensions={choices?.field_dimensions ?? []}
      disabled={!w?.loaded || busy || setupLoading}
      onChange={(key, value) => setParameter(name, key, value)}
      onFixedChange={(fixed) => setFixedParameter(name, fixed)}
      onRegularizersChange={(values) => setRegularizers((old) => ({ ...old, [name]: values }))}
      onConstantsChange={(values) => setConstants((old) => ({ ...old, [name]: values }))}
    />
  );
  return (
    <div
      className="control-layout fit-controls"
      onChangeCapture={() => {
        edited.current = true;
      }}
    >
      <WorkflowButton
        disabled={busy}
        onClick={() =>
          openWorkflows?.('fitting', {
            settings: configuration(),
            use_roi: useRoi,
            roi_names: useRoi ? (w?.scene.fit_rois ?? null) : null,
            excluded_frames: excluded,
          })
        }
      />
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
                  ...configuration(),
                  excluded_frames: excluded,
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
          {(fitInputs?.reference_scan || Boolean(fitInputs?.prior_maps.length)) && (
            <div className="field-grid field-wide" role="group" aria-label="Model inputs">
              {fitInputs?.reference_scan && (
                <label>
                  Reference scan
                  <select
                    disabled={!w?.loaded || busy}
                    aria-label="Reference scan"
                    value={String(settingValue('sat_trans_reference_scan') ?? '')}
                    onChange={(e) =>
                      updateSetting('sat_trans_reference_scan', e.target.value || null)
                    }
                  >
                    <option value="">Automatic</option>
                    {choices?.reference_scans.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                  <span className="control-note">
                    Use Register to align the reference image to the scan before fitting.
                  </span>
                </label>
              )}
              {fitInputs?.prior_maps.map((kind) => (
                <label key={kind}>
                  {kind.toUpperCase()} prior map
                  <select
                    disabled={!w?.loaded || busy}
                    aria-label={`${kind.toUpperCase()} prior map`}
                    value={String(settingValue(`ab_mt_${kind}_map_ref`) ?? '')}
                    onChange={(e) => updateSetting(`ab_mt_${kind}_map_ref`, e.target.value || null)}
                  >
                    <option value="">Automatic</option>
                    {choices?.prior_maps[kind].map((map) => (
                      <option key={map.ref} value={map.ref}>
                        {map.label}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          )}
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
              value={String(settingValue('optimizer_name') ?? 'auto')}
              onChange={(e) => updateSetting('optimizer_name', e.target.value)}
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
      {Object.keys(specs).some((name) => !parameterPool(name)) && (
        <InspectorSection title="General parameters" icon="fit" className="control-wide">
          <div className="parameter-table">
            {Object.entries(specs)
              .filter(([name]) => !parameterPool(name))
              .map(([name, spec]) => renderParameter(name, spec))}
          </div>
        </InspectorSection>
      )}
      {fitInputs && choices && choices.default_pools.length > 0 && (
        <PoolEditor
          key={(w?.id ?? '') + ':' + w?.experiment + ':' + w?.model}
          pools={activePools}
          templates={choices.pools}
          busy={busy || setupLoading}
          onEdit={(edit) =>
            action('fit.pool', { ...edit, settings: configuration(), excluded_frames: excluded })
          }
          renderParameters={(pool) =>
            Object.entries(specs)
              .filter(([name]) => parameterPool(name) === pool)
              .map(([name, spec]) => renderParameter(name, spec, name.slice(0, -pool.length - 1)))
          }
        />
      )}
      {w?.loaded && (
        <details
          className="control-advanced control-wide"
          aria-label="Frames for fitting"
          open={framesExpanded}
          onToggle={(event) => setFramesExpanded(event.currentTarget.open)}
        >
          <summary>
            Frames for fitting{' '}
            <small>
              ({w.frame_count - excluded.length} of {w.frame_count} included)
            </small>
          </summary>
          {framesExpanded && (
            <FitFrameSelector
              workspace={w}
              excluded={excluded}
              disabled={busy || setupLoading}
              onChange={setExcluded}
              onView={(frame) => void changeView({ view: 'Image', frame })}
            />
          )}
          <label className="check-label">
            <input
              type="checkbox"
              checked={Boolean(settingValue('auto_mask_nan_observables'))}
              disabled={busy}
              onChange={(event) => updateSetting('auto_mask_nan_observables', event.target.checked)}
            />
            Automatically exclude nonfinite observables
          </label>
        </details>
      )}
      <button
        className="secondary control-wide"
        disabled={!w?.loaded || busy || setupLoading}
        onClick={() => void apply()}
      >
        Apply fit configuration
      </button>
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
        <NumberField
          label="Preview steps"
          value={previewSteps}
          min={1}
          step="1"
          onChange={setPreviewSteps}
        />
        <button
          className="secondary full"
          disabled={
            !w?.rois.some((r) => r.name === w.selected_roi && r.enabled && r.pixels > 0) ||
            busy ||
            setupLoading
          }
          onClick={() =>
            void run('preview', {
              roi_name: w?.selected_roi,
              steps: previewSteps,
              device,
              settings: configuration(),
              excluded_frames: excluded,
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
                  {Math.abs(value) > 0 && Math.abs(value) < 0.01
                    ? value.toExponential(3)
                    : value.toFixed(2)}
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
        help="Fitting uses the current experiment state and combines the selected ROIs. ROI visibility is a separate display setting."
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
        {fitInputs?.derived_maps && (
          <label>
            Derived maps
            <select
              aria-label="Derived maps"
              value={
                settingValue('compute_derived_maps') === null
                  ? 'default'
                  : String(settingValue('compute_derived_maps') ?? 'default')
              }
              onChange={(e) =>
                updateSetting(
                  'compute_derived_maps',
                  e.target.value === 'default' ? null : e.target.value === 'true',
                )
              }
            >
              <option value="default">Model default</option>
              <option value="true">Compute</option>
              <option value="false">Skip</option>
            </select>
          </label>
        )}
        <FitReadiness
          workspace={w}
          disabled={!w?.loaded || busy || setupLoading}
          options={{
            settings: configuration(),
            use_roi: useRoi,
            roi_names: selectedRois,
            device,
            excluded_frames: excluded,
          }}
        />
        <button
          className="primary full"
          disabled={!w?.loaded || busy || setupLoading || (useRoi && !selectedRois.length)}
          data-tutorial="fit-apply"
          onClick={() =>
            void run('full_fit', {
              step_id: editing ? editingStep.id : undefined,
              roi_name: w?.selected_roi,
              use_roi: useRoi,
              roi_names: selectedRois,
              device,
              settings: configuration(),
              excluded_frames: excluded,
            })
          }
        >
          <WorkbenchIcon name="fit" />
          Run full map fit
        </button>
      </InspectorSection>
      <details className="control-advanced control-wide" aria-label="Additional fit settings">
        <summary>Additional fit settings</summary>
        <JsonEditor
          label="Additional fit settings"
          value={advancedSettings}
          disabled={!w || busy}
          onApply={(value) =>
            action('fit.settings', { ...configuration(), ...value, excluded_frames: excluded })
          }
        />
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
