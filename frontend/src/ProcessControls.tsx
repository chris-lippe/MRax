import { useEffect, useId, useState } from 'react';
import type { components } from './api.generated';
import { api, type Options } from './api';
import type { AnalysisProps } from './analysisTypes';
import { roiColor } from './scene';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { NoiseEstimationControls } from './NoiseEstimationControls';
import { processingGroups, type ProcessingSection } from './WorkflowNav';

type Catalog = components['schemas']['ProcessingCatalog'];
type Draft = {
  method?: string;
  source?: string;
  component?: string;
  axis?: string;
  axes?: Record<string, string[]>;
  parameters?: Record<string, Options>;
  reference?: string;
  referenceFrame?: number;
  limited?: boolean;
  rois?: string[];
  noiseRois?: string[] | null;
  scope?: 'image' | 'package' | 'dataset';
  indices?: Record<string, number>;
  planeDims?: string[];
};

const choiceLabels: Record<string, string> = {
  real_imag: 'Real and imaginary',
  magnitude: 'Magnitude',
  reflect: 'Reflect',
  symmetric: 'Symmetric',
  edge: 'Nearest edge',
  wrap: 'Wrap',
  subtract: 'Subtract baseline',
  divide: 'Divide by baseline',
  tv: 'Total variation',
  tv2: 'Second order total variation',
  smoothness: 'Spatial smoothness',
  full: 'Full',
  basic: 'Hard thresholding',
  adam: 'Adam',
  adamw: 'AdamW',
  sgd: 'SGD',
  rmsprop: 'RMSprop',
};

const actionLabels: Record<string, string> = {
  normalize: 'Normalize signal',
  dilution: 'Apply dilution correction',
  baseline: 'Apply baseline correction',
  temporal_gaussian: 'Apply temporal smoothing',
  despike: 'Remove spikes',
  dti: 'Compute derived maps',
};

const methodLabels: Record<ProcessingSection, string> = {
  denoise: 'Denoiser',
  normalize: 'Normalization method',
  series: 'Time-series correction',
  analyze: 'Map calculation',
};

export function ProcessControls({
  workspace: w,
  busy,
  run,
  editingStep,
  section,
  active,
}: Pick<AnalysisProps, 'workspace' | 'busy' | 'run' | 'editingStep'> & {
  section: ProcessingSection;
  active: boolean;
}) {
  const [loaded, setLoaded] = useState<{ key: string; value: Catalog } | null>(null);
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [error, setError] = useState('');
  const axesHelpId = useId();
  const context = JSON.stringify([w?.id, w?.experiment, section]);
  const editing =
    editingStep?.kind === 'process' &&
    (editingStep.family === section ||
      (section === 'series' &&
        ['baseline', 'temporal_gaussian', 'despike'].includes(editingStep.method)));
  const editOptions = editing ? JSON.stringify(editingStep.options) : '';
  const editKey = editing ? editingStep.id + editingStep.version : '';
  useEffect(() => {
    if (!editOptions || !editKey) return;
    const options = JSON.parse(editOptions) as Options;
    const method = String(options.method);
    setDrafts((previous) => ({
      ...previous,
      [context]: {
        method,
        source: String(options.input ?? 'image'),
        component: String(options.component ?? 'auto'),
        axis: String(options.axis ?? ''),
        axes: { [method]: (options.axes as string[]) ?? [] },
        parameters: { [method]: (options.parameters as Options) ?? {} },
        reference: String(options.reference_experiment ?? ''),
        referenceFrame: Number(options.reference_frame ?? 0),
        limited: options.roi_names !== null && options.roi_names !== undefined,
        rois: (options.roi_names as string[]) ?? [],
        noiseRois: (options.noise_roi_names as string[] | null) ?? null,
        scope: (options.scope as Draft['scope']) ?? 'dataset',
        indices: (options.indices as Record<string, number>) ?? {},
        planeDims: options.plane_dims as string[] | undefined,
      },
    }));
  }, [context, editOptions, editKey]);
  const draft = drafts[context] ?? {};
  const targetIndices = editing && draft.indices ? draft.indices : (w?.indices ?? {});
  const targetPackage = targetIndices.slice_package ?? w?.indices.slice_package;
  const requestedSource = draft.source === 'view' ? 'image' : (draft.source ?? 'image');
  const requestKey = JSON.stringify([
    w?.id,
    w?.experiment,
    w?.revision,
    requestedSource,
    targetPackage,
    w?.plane,
    w?.plane_dims,
  ]);
  const methodLabel = methodLabels[section];
  const catalog = loaded?.key === requestKey ? loaded.value : null;
  useEffect(() => {
    let stopped = false;
    setError('');
    if (w?.loaded && active)
      void api<Catalog>(
        `/api/workspaces/${w.id}/processing?input_source=${encodeURIComponent(requestedSource)}${targetPackage === undefined ? '' : '&package=' + targetPackage}`,
      )
        .then((value) => {
          if (!stopped) setLoaded({ key: requestKey, value });
        })
        .catch((failure) => {
          if (!stopped) setError(String(failure));
        });
    return () => {
      stopped = true;
    };
  }, [
    w?.id,
    w?.experiment,
    w?.revision,
    w?.loaded,
    requestKey,
    requestedSource,
    targetPackage,
    active,
  ]);
  const update = (patch: Partial<Draft>) =>
    setDrafts((previous) => ({ ...previous, [context]: { ...previous[context], ...patch } }));
  const groups: string[] = [processingGroups[section]];
  const methods = catalog?.methods.filter((m) => groups.includes(m.group)) ?? [];
  const method = methods.find((m) => m.id === draft.method) ?? methods[0];
  const axisCount = method?.axis_count;
  const source =
    draft.source !== 'view' && catalog?.sources.some((s) => s.id === draft.source)
      ? draft.source!
      : 'image';
  const inputAxes = catalog?.axis_info.length
    ? catalog.axis_info.map((axis) => axis.name)
    : source.startsWith('map:')
      ? (catalog?.spatial_axes ?? [])
      : (catalog?.axes ?? []);
  const seriesAxes = source.startsWith('map:') ? [] : (catalog?.series_axes ?? []);
  const axis = draft.axis && seriesAxes.includes(draft.axis) ? draft.axis : (seriesAxes[0] ?? '');
  const batchScope = w?.dimensions.slice_package ? 'package' : 'dataset';
  const scope = draft.scope ?? batchScope;
  const planeDims = (editing ? draft.planeDims : undefined) ?? w?.plane_dims ?? [];
  const availableAxes =
    scope === 'image' && method?.axes === 'filter'
      ? inputAxes.filter((name) => planeDims.includes(name))
      : inputAxes;
  const axisInfo = Object.fromEntries(catalog?.axis_info.map((axis) => [axis.name, axis]) ?? []);
  const axes = method
    ? (
        draft.axes?.[method.id] ??
        (method.id === 'bm3d' ? w?.plane_dims : catalog?.spatial_axes) ??
        []
      ).filter((name) => availableAxes.includes(name))
    : [];
  const independentAxes = inputAxes.filter((name) => !axes.includes(name));
  const maxAxes = axisCount ?? 3;
  const fixedNames =
    scope === 'image'
      ? [
          ...inputAxes.filter((name) => !planeDims.includes(name)),
          ...(w?.dimensions.slice_package ? ['slice_package'] : []),
        ]
      : scope === 'package'
        ? ['slice_package']
        : [];
  const selectedIndices = Object.fromEntries(
    fixedNames.map((name) => [name, targetIndices[name] ?? 0]),
  );
  const targetLabel = fixedNames
    .map((name) =>
      name === 'slice_package'
        ? `Package ${selectedIndices[name] + 1}`
        : `${axisInfo[name]?.label ?? name}${axisInfo[name]?.spatial ? ' slice' : ''} ${selectedIndices[name] + 1}`,
    )
    .join(' · ');
  const scopeOptions =
    method?.axes === 'filter'
      ? { scope, indices: selectedIndices, plane_dims: scope === 'image' ? planeDims : undefined }
      : {};
  const reference = draft.reference ?? w?.experiment ?? '';
  const referenceCount = w?.experiment_catalog.find((e) => e.name === reference)?.frames ?? 1;
  const rois =
    draft.rois ?? w?.rois.filter((r) => r.enabled && r.pixels > 0).map((r) => r.name) ?? [];
  const scopedRois = rois.filter((name) =>
    w?.rois.some((r) => r.name === name && r.enabled && r.pixels > 0),
  );
  const limited = Boolean(draft.limited);
  const noiseRois = draft.noiseRois ?? null;
  const unavailable =
    method?.unavailable ??
    (axisCount && availableAxes.length < axisCount
      ? `${method?.label} requires ${axisCount} filter dimensions. ${scope === 'image' ? 'Choose the active slice package or whole dataset to use a 3D block.' : 'This input has too few dimensions.'}`
      : (method?.axes === 'series' || method?.reference === 'series') && !seriesAxes.length
        ? 'Choose image data with an acquisition series.'
        : null);
  const parameters = Object.fromEntries(
    method?.fields.map((field) => [
      field.key,
      Object.hasOwn(draft.parameters?.[method.id] ?? {}, field.key)
        ? draft.parameters![method.id][field.key]
        : field.key === 'baseline_frames' && axis
          ? Math.min(Number(field.default), w?.dimensions[axis] ?? 1)
          : field.default,
    ]) ?? [],
  );
  const parameter = (key: string, value: unknown) => {
    if (method)
      update({
        parameters: {
          ...draft.parameters,
          [method.id]: { ...draft.parameters?.[method.id], [key]: value },
        },
      });
  };
  const invalid =
    !w?.loaded ||
    !method ||
    !catalog ||
    Boolean(unavailable) ||
    (method.axes === 'filter' &&
      (!axes.length || axes.length > 3 || (axisCount != null && axes.length !== axisCount))) ||
    (limited && !scopedRois.length) ||
    (['bm3d', 'bm4d'].includes(method?.id ?? '') &&
      parameters.sigma == null &&
      noiseRois !== null &&
      !noiseRois.length);

  const fieldControls = method?.fields.map((field) => {
    const value = parameters[field.key];
    return field.kind === 'boolean' ? (
      <label className="check-label" key={field.key}>
        <input
          type="checkbox"
          aria-label={field.label}
          checked={Boolean(value)}
          onChange={(event) => parameter(field.key, event.target.checked)}
        />
        {field.label}
      </label>
    ) : (
      <label key={`${method.id}-${field.key}`}>
        {field.label}
        {field.kind === 'choice' ? (
          <select
            aria-label={field.label}
            value={String(value)}
            onChange={(event) => parameter(field.key, event.target.value)}
          >
            {field.choices.map((choice) => (
              <option key={choice} value={choice}>
                {choiceLabels[choice] ?? choice}
              </option>
            ))}
          </select>
        ) : (
          <input
            aria-label={field.label}
            type="number"
            value={value == null ? '' : Number(value)}
            required={!field.nullable}
            min={field.minimum ?? undefined}
            max={
              field.key === 'baseline_frames' && axis
                ? w?.dimensions[axis]
                : (field.maximum ?? undefined)
            }
            step={field.kind === 'integer' ? 1 : 'any'}
            onChange={(event) =>
              parameter(field.key, event.target.value === '' ? null : Number(event.target.value))
            }
          />
        )}
      </label>
    );
  });

  const primary = (index: number) =>
    method?.id === 'dilution'
      ? ['baseline_frames', 'smooth_sigma'].includes(method.fields[index].key)
      : index < 2;
  const primaryFields = fieldControls?.filter((_field, index) => primary(index));
  const advancedFields = fieldControls?.filter(
    (_field, index) =>
      !primary(index) && !['noise_model', 'noise_coils'].includes(method!.fields[index].key),
  );
  const noiseOptions = {
    ...scopeOptions,
    method: method?.id,
    input: source,
    component: draft.component ?? 'auto',
    axes,
    parameters,
    step_id: editing ? editingStep?.id : undefined,
    noise_roi_names: noiseRois,
  };

  return (
    <div className="processing-controls">
      {error && <p role="alert">{error}</p>}
      <form
        className="control-layout"
        onSubmit={(event) => {
          event.preventDefault();
          if (!method || invalid) return;
          void run('process', {
            ...scopeOptions,
            method: method.id,
            input: source,
            component: draft.component ?? 'auto',
            axes,
            axis,
            parameters,
            step_id: editing ? editingStep?.id : undefined,
            roi_names: limited ? scopedRois : null,
            noise_roi_names:
              ['bm3d', 'bm4d'].includes(method.id) && parameters.sigma == null ? noiseRois : null,
            reference_experiment: method.reference ? reference : undefined,
            reference_frame: method.reference === 'frame' ? (draft.referenceFrame ?? 0) : undefined,
          });
        }}
      >
        <fieldset className="control-form" disabled={busy || !w?.loaded || !catalog}>
          <InspectorSection
            title={methodLabel + ' and input'}
            icon={
              section === 'denoise'
                ? 'denoise'
                : section === 'normalize'
                  ? 'normalize'
                  : section === 'analyze'
                    ? 'maps'
                    : 'series'
            }
          >
            <div className="field-grid">
              <label className="field-wide">
                {methodLabel}
                <select
                  aria-label={methodLabel}
                  value={method?.id ?? ''}
                  onChange={(event) => update({ method: event.target.value })}
                >
                  {!catalog && <option>Load an experiment</option>}
                  {groups.map((group) => (
                    <optgroup label={group} key={group}>
                      {methods
                        .filter((m) => m.group === group)
                        .map((m) => (
                          <option key={m.id} value={m.id} disabled={Boolean(m.unavailable)}>
                            {m.label}
                            {m.unavailable ? ' (unavailable)' : ''}
                          </option>
                        ))}
                    </optgroup>
                  ))}
                </select>
              </label>
              {method?.id !== 'dti' && (
                <>
                  <label>
                    Input data
                    <select
                      aria-label="Processing input"
                      value={source}
                      onChange={(event) => update({ source: event.target.value })}
                    >
                      {catalog?.sources
                        .filter((s) => s.id !== 'view')
                        .map((s) => (
                          <option value={s.id} key={s.id}>
                            {s.id === 'view' || s.id === 'image'
                              ? 'Current pipeline state'
                              : s.label}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label>
                    Component
                    <select
                      aria-label="Processing component"
                      value={draft.component ?? 'auto'}
                      onChange={(event) => update({ component: event.target.value })}
                    >
                      <option value="auto">Preserve complex data</option>
                      <option value="magnitude">Magnitude</option>
                      <option value="real">Real</option>
                      <option value="imag">Imaginary</option>
                      <option value="phase">Phase</option>
                    </select>
                  </label>
                </>
              )}
            </div>
          </InspectorSection>
          {(method?.axes !== 'none' || Boolean(fieldControls?.length) || unavailable) && (
            <InspectorSection
              title="Settings"
              icon="settings"
              help={
                method?.axes === 'filter'
                  ? 'Selected data dimensions are filtered jointly. Each combination of the remaining dimensions is processed independently.'
                  : undefined
              }
            >
              {method?.axes === 'filter' && (
                <fieldset className="processing-axes">
                  <legend>Filter dimensions{axisCount ? ` · select ${axisCount}` : ''}</legend>
                  <label>
                    Apply to
                    <select
                      aria-label="Denoising scope"
                      value={scope}
                      onChange={(event) => {
                        const next = event.target.value as Draft['scope'];
                        update({
                          scope: next,
                          indices: undefined,
                          planeDims: undefined,
                          axes: {
                            ...draft.axes,
                            [method.id]:
                              (next === 'image' || method.id === 'bm3d'
                                ? w?.plane_dims
                                : catalog?.spatial_axes) ?? [],
                          },
                        });
                      }}
                    >
                      <option value="image">Viewed image (current frame and slice)</option>
                      <option value={batchScope}>
                        {batchScope === 'package' ? 'Active slice package' : 'Whole dataset'}
                      </option>
                      {scope === 'dataset' && batchScope === 'package' && (
                        <option value="dataset">Whole dataset (all packages)</option>
                      )}
                    </select>
                  </label>
                  {targetLabel && (
                    <p className="control-note" aria-label="Denoising target">
                      {targetLabel}
                    </p>
                  )}
                  {editing && scope !== 'dataset' && (
                    <button
                      type="button"
                      className="quiet"
                      onClick={() => update({ indices: undefined, planeDims: undefined })}
                    >
                      Use current selection
                    </button>
                  )}
                  <div className="axis-options">
                    {inputAxes.map((name) => (
                      <label className="check-label" key={name}>
                        <input
                          type="checkbox"
                          aria-label={`Filter axis ${axisInfo[name]?.label ?? name}`}
                          aria-describedby={axesHelpId}
                          title={`Stored data dimension: ${name}`}
                          checked={axes.includes(name)}
                          disabled={
                            !axes.includes(name) &&
                            (axes.length >= maxAxes || !availableAxes.includes(name))
                          }
                          onChange={(event) =>
                            update({
                              axes: {
                                ...draft.axes,
                                [method.id]: event.target.checked
                                  ? [...axes, name]
                                  : axes.filter((n) => n !== name),
                              },
                            })
                          }
                        />
                        {axisInfo[name]?.label ?? name} (
                        {axisInfo[name]?.size ?? w?.dimensions[name] ?? 1})
                      </label>
                    ))}
                  </div>
                  <p className="control-note" id={axesHelpId}>
                    {axisCount
                      ? `${method.label.split(' ')[0]} denoises one ${axisCount === 2 ? '2D plane' : '3D block'} at a time. `
                      : 'Choose one to three data dimensions to filter jointly. '}
                    {scope === 'image'
                      ? 'Applies to the current image. Other frames and slices are retained.'
                      : independentAxes.length
                        ? `Processed sequentially: ${independentAxes.map((name) => `${axisInfo[name]?.label ?? name} (${axisInfo[name]?.size ?? w?.dimensions[name] ?? 1})`).join(', ')}.`
                        : 'The selected dimensions cover the entire input.'}
                  </p>
                </fieldset>
              )}
              {method?.axes === 'series' && (
                <label>
                  Acquisition axis
                  <select
                    aria-label="Processing acquisition axis"
                    value={axis}
                    onChange={(event) => update({ axis: event.target.value })}
                  >
                    {seriesAxes.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
              )}
              {unavailable && (
                <p className="control-note" role="status">
                  {unavailable}
                </p>
              )}
              <div className="field-grid">{primaryFields}</div>
              {(method?.id === 'bm3d' || method?.id === 'bm4d') && (
                <p className="control-note">
                  Leave sigma empty for automatic estimation. A manual value overrides estimation.
                </p>
              )}
              {w && ['bm3d', 'bm4d'].includes(method?.id ?? '') && parameters.sigma == null && (
                <NoiseEstimationControls
                  workspace={w}
                  busy={busy}
                  options={noiseOptions}
                  roiNames={noiseRois}
                  setRois={(noiseRois) => update({ noiseRois })}
                  parameter={parameter}
                />
              )}
              {Boolean(advancedFields?.length) && (
                <details className="control-advanced">
                  <summary>More settings</summary>
                  <div className="field-grid">{advancedFields}</div>
                </details>
              )}
            </InspectorSection>
          )}
          {method?.reference && (
            <InspectorSection
              title="Reference"
              icon="image"
              help="Reference grids must match. Register mismatched experiments first."
            >
              <div className="field-grid">
                <label>
                  Reference experiment
                  <select
                    aria-label="Processing reference experiment"
                    value={reference}
                    onChange={(event) =>
                      update({ reference: event.target.value, referenceFrame: 0 })
                    }
                  >
                    {w?.experiments.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                {method.reference === 'frame' && (
                  <label>
                    Reference frame (zero-based)
                    <input
                      aria-label="Processing reference frame"
                      type="number"
                      value={draft.referenceFrame ?? 0}
                      min={0}
                      max={referenceCount - 1}
                      step={1}
                      required
                      onChange={(event) => update({ referenceFrame: Number(event.target.value) })}
                    />
                  </label>
                )}
              </div>
            </InspectorSection>
          )}
          <InspectorSection title="Region" icon="rois">
            <label className="check-label">
              <input
                type="checkbox"
                checked={limited}
                onChange={(event) => update({ limited: event.target.checked })}
              />
              Limit processing to ROIs
            </label>
            {limited && (
              <div className="processing-rois">
                {w?.rois
                  .filter((r) => r.enabled)
                  .map((roi) => (
                    <label className="check-label" key={roi.name}>
                      <input
                        type="checkbox"
                        aria-label={`Process mask ${roi.name}`}
                        checked={scopedRois.includes(roi.name)}
                        disabled={!roi.pixels}
                        onChange={(event) =>
                          update({
                            rois: event.target.checked
                              ? [...scopedRois, roi.name]
                              : scopedRois.filter((name) => name !== roi.name),
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
              </div>
            )}
            {limited && !scopedRois.length && (
              <p className="control-note">Select a nonempty mask, or process the entire image.</p>
            )}
            {limited &&
              (method?.id === 'normalize' ||
                (method?.id === 'baseline' && parameters.mode === 'divide')) && (
                <p className="control-note">
                  Normalized results have missing values outside the selected masks.
                </p>
              )}
          </InspectorSection>
          <div className="inspector-actionbar">
            <button
              className="primary full"
              disabled={invalid}
              data-tutorial={section === 'denoise' ? 'denoise-apply' : undefined}
            >
              <WorkbenchIcon name="apply" />
              {editing
                ? 'Stage ' +
                  (section === 'denoise'
                    ? 'denoising'
                    : section === 'normalize'
                      ? 'normalization'
                      : 'time-series') +
                  ' settings'
                : section === 'denoise'
                  ? 'Apply denoising'
                  : (actionLabels[method?.id ?? ''] ?? 'Apply processing')}
            </button>
          </div>
        </fieldset>
      </form>
      {methods.some((m) => m.id.startsWith('bm') && m.unavailable) && (
        <details>
          <summary>Optional denoisers</summary>
          {methods
            .filter((m) => m.id.startsWith('bm') && m.unavailable)
            .map((m) => (
              <p className="control-note" key={m.id}>
                {m.label}: {m.unavailable}
              </p>
            ))}
        </details>
      )}
    </div>
  );
}
