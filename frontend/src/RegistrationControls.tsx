import { useEffect, useState } from 'react';
import { api, type Options } from './api';
import type { AnalysisProps } from './analysisTypes';
import { JsonEditor, NumberField, Report } from './Controls';
import type { WorkflowTask } from './WorkflowNav';
import { InspectorSection, InspectorChoice } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { WorkflowButton } from './WorkflowButton';

type RegistrationPreset = {
  key: string;
  label: string;
  description: string;
  settings: Options;
  contexts: string[];
  average_contrasts?: boolean;
};

export function RegistrationControls({
  workspace: w,
  busy,
  action,
  changeView,
  run,
  task,
  editingStep,
  openWorkflows,
}: AnalysisProps & { task: WorkflowTask }) {
  const [reference, setReference] = useState(''),
    [moving, setMoving] = useState(''),
    [referenceFrame, setReferenceFrame] = useState(0),
    [movingFrame, setMovingFrame] = useState(0);
  const [referenceAggregation, setReferenceAggregation] = useState('frame'),
    [movingAggregation, setMovingAggregation] = useState('frame');
  const [referenceSource, setReferenceSource] = useState('image'),
    [movingSource, setMovingSource] = useState('image'),
    [referenceMap, setReferenceMap] = useState(''),
    [movingMap, setMovingMap] = useState('');
  const [referenceRoi, setReferenceRoi] = useState(''),
    [movingRoi, setMovingRoi] = useState(''),
    [volume, setVolume] = useState(false),
    [manual, setManual] = useState(false);
  const [shiftX, setShiftX] = useState(0),
    [shiftY, setShiftY] = useState(0),
    [rotation, setRotation] = useState(0),
    [scaleX, setScaleX] = useState(1),
    [scaleY, setScaleY] = useState(1);
  const [settings, setSettings] = useState<Options>({}),
    [drift, setDrift] = useState<Options>({}),
    [transform, setTransform] = useState(''),
    [direction, setDirection] = useState('moving_to_reference');
  const [error, setError] = useState('');
  const [driftMask, setDriftMask] = useState('');
  const [seriesMethod, setSeriesMethod] = useState('framewise');
  const [averagedDimensions, setAveragedDimensions] = useState<string[] | null>(null);
  const [presets, setPresets] = useState<RegistrationPreset[]>([]);
  const editingAlignment =
    editingStep?.kind === 'register' || editingStep?.kind === 'stored_registration';
  const editingDrift = editingStep?.kind === 'drift';
  const editOptions = editingStep ? JSON.stringify(editingStep.options) : '';
  const editKey = editingStep ? editingStep.id + editingStep.version : '';
  const editKind = editingStep?.kind;
  useEffect(() => {
    if (!editOptions || !editKey) return;
    const options = JSON.parse(editOptions) as Options;
    if (editKind === 'drift') {
      const saved = (options.settings as Options) ?? {};
      setDrift(saved);
      setSeriesMethod(String(saved.method ?? 'linear'));
      setAveragedDimensions((saved.estimation_dimensions as string[] | undefined) ?? []);
      if (saved.registration)
        setSettings((previous) => ({ ...previous, ...(saved.registration as Options) }));
      setDriftMask(String(options.roi_name ?? ''));
      return;
    }
    if (editKind === 'stored_registration') {
      setTransform(String(options.key ?? ''));
      setDirection(String(options.direction ?? 'moving_to_reference'));
      return;
    }
    if (editKind !== 'register') return;
    setReference(String(options.reference_experiment ?? ''));
    setMoving(String(options.moving_experiment ?? ''));
    setReferenceFrame(Number(options.reference_frame ?? 0));
    setMovingFrame(Number(options.moving_frame ?? 0));
    setReferenceAggregation(String(options.reference_aggregation ?? 'frame'));
    setMovingAggregation(String(options.moving_aggregation ?? 'frame'));
    setReferenceSource(String(options.reference_source ?? 'image'));
    setMovingSource(String(options.moving_source ?? 'image'));
    setReferenceMap(String(options.reference_map ?? ''));
    setMovingMap(String(options.moving_map ?? ''));
    setReferenceRoi(String(options.reference_roi ?? ''));
    setMovingRoi(String(options.moving_roi ?? ''));
    setVolume(options.scope === 'volume');
    setManual(Boolean(options.manual));
    const affine = (options.manual_settings as Options) ?? {};
    setShiftX(Number(affine.shift_x ?? 0));
    setShiftY(Number(affine.shift_y ?? 0));
    setRotation(Number(affine.rotation_degrees ?? 0));
    setScaleX(Number(affine.scale_x ?? 1));
    setScaleY(Number(affine.scale_y ?? 1));
    setSettings((options.settings as Options) ?? {});
  }, [editOptions, editKey, editKind]);
  useEffect(() => {
    setReference('');
    setMoving('');
    setReferenceFrame(0);
    setMovingFrame(0);
    setReferenceAggregation('frame');
    setMovingAggregation('frame');
    setReferenceSource('image');
    setMovingSource('image');
    setReferenceMap('');
    setMovingMap('');
    setReferenceRoi('');
    setMovingRoi('');
  }, [w?.id]);
  useEffect(() => {
    setTransform(String(w?.registrations.at(-1)?.key ?? ''));
  }, [w?.id, w?.registrations.length]);
  useEffect(() => {
    void api<{ registration: Options; drift: Options; registration_presets: RegistrationPreset[] }>(
      '/api/catalog',
    )
      .then((c) => {
        setSettings((previous) => ({ ...c.registration, ...previous }));
        setDrift((previous) => ({ ...c.drift, ...previous }));
        setPresets(c.registration_presets);
      })
      .catch((e) => setError(String(e)));
  }, []);
  const row = w?.registrations.find((r) => r.key === transform);
  const driftDimensions = Object.entries(w?.dimensions ?? {}).filter(
    ([name, size]) => !['x', 'y', 'z', 'slice_package'].includes(name) && size > 1,
  );
  const saturationSeries =
    w?.experiment_catalog.find((experiment) => experiment.name === w.experiment)?.exp_type ===
    'sat_trans';
  const selectedDriftDimensions = Array.isArray(drift.dimensions)
    ? (drift.dimensions as string[]).filter((name) => driftDimensions.some(([key]) => key === name))
    : driftDimensions.some(([name]) => name === 't')
      ? ['t']
      : saturationSeries
        ? []
        : driftDimensions.map(([name]) => name);
  const contrastDimensions = driftDimensions.filter(
    ([name]) => !selectedDriftDimensions.includes(name),
  );
  const estimationDimensions = contrastDimensions
    .filter(([name]) => averagedDimensions === null || averagedDimensions.includes(name))
    .map(([name]) => name);
  const volumeDrift = ['x', 'y', 'z'].filter((name) => name in (w?.dimensions ?? {})).length > 2;
  const selectedDriftMask = w?.rois.some(
    (roi) => roi.name === driftMask && roi.enabled && roi.pixels > 0,
  )
    ? driftMask
    : '';
  const [transferRoi, setTransferRoi] = useState('');
  const transformMetadata = row?.metadata as Options | undefined;
  const transformSource = String(
    transformMetadata?.[
      direction === 'moving_to_reference' ? 'moving_experiment' : 'reference_experiment'
    ] ?? '',
  );
  const sourceRois = w?.experiment_catalog.find((e) => e.name === transformSource)?.rois ?? [];
  const transferredName = sourceRois.includes(transferRoi)
    ? transferRoi
    : sourceRois.includes(w?.selected_roi ?? '')
      ? w!.selected_roi!
      : (sourceRois[0] ?? '');
  const activateTransformSource = async () => {
    if (transformMetadata)
      await changeView({
        experiment: transformSource,
        view: 'Image',
        plane: transformMetadata.plane,
        frame:
          transformMetadata[
            direction === 'moving_to_reference' ? 'moving_frame' : 'reference_frame'
          ],
      });
  };
  const applyPreset = (key: string, series: boolean) => {
    const preset = presets.find((entry) => entry.key === key);
    if (!preset) return;
    setSettings({ ...preset.settings });
    if (series) {
      setSeriesMethod('framewise');
      setAveragedDimensions(
        preset.average_contrasts ? contrastDimensions.map(([name]) => name) : [],
      );
      setDrift((previous) => ({
        ...previous,
        reference: preset.average_contrasts ? 'median' : 'first',
        ...(preset.average_contrasts ? { dimensions: ['t'] } : {}),
      }));
    } else {
      setMovingAggregation(key === 'different_contrasts' ? 'mean' : 'frame');
    }
  };
  const presetControls = (series: boolean) => {
    const available = presets.filter((preset) =>
      preset.contexts.includes(series ? 'series' : 'pair'),
    );
    const selected = available.find(
      (preset) =>
        Object.entries(preset.settings).every(
          ([key, value]) => JSON.stringify(settings[key]) === JSON.stringify(value),
        ) &&
        (!series ||
          (seriesMethod === 'framewise' &&
            (preset.average_contrasts
              ? drift.reference === 'median' &&
                selectedDriftDimensions.length === 1 &&
                selectedDriftDimensions[0] === 't' &&
                estimationDimensions.length === contrastDimensions.length &&
                estimationDimensions.length > 0
              : estimationDimensions.length === 0))),
    );
    return (
      <>
        <label>
          Preset
          <select
            aria-label={series ? 'Series alignment preset' : 'Registration preset'}
            value={selected?.key ?? 'custom'}
            onChange={(event) => applyPreset(event.target.value, series)}
          >
            <option value="custom">Custom settings</option>
            {available.map((preset) => (
              <option
                key={preset.key}
                value={preset.key}
                disabled={Boolean(
                  preset.average_contrasts &&
                  (!driftDimensions.some(([name]) => name === 't') || !contrastDimensions.length),
                )}
              >
                {preset.label}
              </option>
            ))}
          </select>
        </label>
        {selected && <p className="control-note">{selected.description}</p>}
      </>
    );
  };
  const stabilityControls = (series: boolean) => (
    <div className="field-grid">
      {settings.transform !== 'translation' && (
        <label className="check-label">
          <input
            type="checkbox"
            aria-label={
              series
                ? 'Series translation initialization'
                : 'Registration translation initialization'
            }
            checked={Boolean(settings.prealign_translation)}
            onChange={(event) =>
              setSettings((previous) => ({
                ...previous,
                prealign_translation: event.target.checked,
              }))
            }
          />
          Initialize with translation
        </label>
      )}
      <label className="check-label">
        <input
          type="checkbox"
          aria-label={series ? 'Limit series motion' : 'Limit registration motion'}
          checked={settings.max_displacement_voxels != null}
          onChange={(event) =>
            setSettings((previous) => ({
              ...previous,
              max_displacement_voxels: event.target.checked ? 6 : null,
            }))
          }
        />
        Limit estimated motion
      </label>
      {settings.max_displacement_voxels != null && (
        <NumberField
          label={series ? 'Maximum series motion' : 'Maximum registration motion'}
          caption="Maximum motion (voxels)"
          min={0.1}
          value={Number(settings.max_displacement_voxels)}
          onChange={(value) =>
            setSettings((previous) => ({ ...previous, max_displacement_voxels: value }))
          }
        />
      )}
    </div>
  );

  const alignmentOptions = {
    step_id: editingAlignment ? editingStep?.id : undefined,
    reference_experiment: reference || w?.experiment,
    moving_experiment: moving || w?.experiment,
    reference_source: referenceSource,
    moving_source: movingSource,
    reference_aggregation: referenceSource === 'map' ? 'frame' : referenceAggregation,
    moving_aggregation: movingSource === 'map' ? 'frame' : movingAggregation,
    reference_frame: referenceFrame,
    moving_frame: movingFrame,
    reference_map: referenceMap || null,
    moving_map: movingMap || null,
    reference_roi: referenceRoi || null,
    moving_roi: movingRoi || null,
    scope: volume ? 'volume' : 'plane',
    manual,
    manual_settings: {
      shift_x: shiftX,
      shift_y: shiftY,
      rotation_degrees: rotation,
      scale_x: scaleX,
      scale_y: scaleY,
    },
    settings,
  };
  const seriesOptions = {
    settings:
      seriesMethod === 'framewise'
        ? {
            method: 'framewise',
            dimensions: selectedDriftDimensions,
            reference: drift.reference ?? 'first',
            estimation_dimensions: estimationDimensions,
            registration: settings,
          }
        : { ...drift, method: 'linear', dimensions: selectedDriftDimensions },
    roi_name: selectedDriftMask || null,
    step_id: editingDrift ? editingStep?.id : undefined,
  };
  return (
    <div className="registration-controls">
      <WorkflowButton
        data-tutorial="processing-workflows"
        disabled={busy}
        onClick={() =>
          openWorkflows?.('registration', {
            operation: task === 'drift' ? 'drift' : 'register',
            options: task === 'drift' ? seriesOptions : alignmentOptions,
          })
        }
      />
      {error && <p role="alert">{error}</p>}
      <div className="control-layout" hidden={task !== 'alignment'}>
        {[
          [
            'Reference',
            reference,
            setReference,
            referenceSource,
            setReferenceSource,
            referenceMap,
            setReferenceMap,
            referenceRoi,
            setReferenceRoi,
          ] as const,
          [
            'Moving',
            moving,
            setMoving,
            movingSource,
            setMovingSource,
            movingMap,
            setMovingMap,
            movingRoi,
            setMovingRoi,
          ] as const,
        ].map(([label, name, setName, source, setSource, map, setMap, roi, setRoi]) => {
          const experiment = w?.experiment_catalog.find((e) => e.name === (name || w.experiment));
          return (
            <InspectorSection key={label} title={label + ' image'} icon="image">
              <fieldset className="control-form" disabled={!w?.loaded || busy}>
                <div className="field-grid registration-image-fields">
                  <label className="field-wide">
                    {label} experiment
                    <select
                      aria-label={`${label} experiment`}
                      value={name}
                      onChange={(e) => {
                        setName(e.target.value);
                        setSource('image');
                        setMap('');
                        setRoi('');
                        (label === 'Reference' ? setReferenceFrame : setMovingFrame)(0);
                      }}
                    >
                      <option value="">Current experiment</option>
                      {w?.experiments.map((v) => (
                        <option key={v}>{v}</option>
                      ))}
                    </select>
                  </label>
                  <label className="registration-source">
                    Data
                    <select
                      aria-label={`${label} source`}
                      value={source}
                      onChange={(e) => setSource(e.target.value)}
                    >
                      <option value="image">Image</option>
                      <option value="reconstruction" disabled={!experiment?.has_reconstruction}>
                        {experiment?.reconstruction_label ?? 'Reconstruction'}
                      </option>
                      <option value="map" disabled={!experiment?.maps.length}>
                        Fitted map
                      </option>
                    </select>
                  </label>
                  {source === 'map' && (
                    <label>
                      {label} map
                      <select
                        aria-label={`${label} map`}
                        value={map}
                        onChange={(e) => setMap(e.target.value)}
                      >
                        <option value="">Choose map</option>
                        {experiment?.maps.map((name) => (
                          <option key={name}>{name}</option>
                        ))}
                      </select>
                    </label>
                  )}
                  {source !== 'map' && (
                    <InspectorChoice
                      label={`${label} image selection`}
                      caption="Image selection"
                      value={label === 'Reference' ? referenceAggregation : movingAggregation}
                      choices={[
                        ['frame', 'Single frame'],
                        ['mean', 'Mean magnitude'],
                        ['median', 'Median magnitude'],
                      ]}
                      onChange={
                        label === 'Reference' ? setReferenceAggregation : setMovingAggregation
                      }
                    />
                  )}
                  <NumberField
                    caption={
                      (label === 'Reference' ? referenceAggregation : movingAggregation) === 'frame'
                        ? 'Frame'
                        : 'Slice/package frame'
                    }
                    className="registration-frame"
                    label={`${label} frame`}
                    value={label === 'Reference' ? referenceFrame : movingFrame}
                    min={0}
                    step="1"
                    onChange={label === 'Reference' ? setReferenceFrame : setMovingFrame}
                  />
                  <details className="control-advanced field-wide">
                    <summary>Metric mask{roi ? ' · ' + roi : ''}</summary>
                    <label>
                      {label} metric mask
                      <select
                        aria-label={`${label} metric mask`}
                        value={roi}
                        onChange={(e) => setRoi(e.target.value)}
                      >
                        <option value="">None</option>
                        {experiment?.rois.map((name) => (
                          <option key={name}>{name}</option>
                        ))}
                      </select>
                    </label>
                  </details>
                </div>
              </fieldset>
            </InspectorSection>
          );
        })}
        <InspectorSection title="Alignment" icon="registration" className="control-wide">
          <fieldset className="control-form" disabled={!w?.loaded || busy}>
            {!manual && presetControls(false)}
            <div className="field-grid">
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={volume}
                  onChange={(e) => setVolume(e.target.checked)}
                />
                Register full spatial volume
              </label>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={manual}
                  onChange={(e) => setManual(e.target.checked)}
                />
                Manual affine transform
              </label>
            </div>
            {!manual && (
              <div className="field-grid">
                <InspectorChoice
                  label="Registration transform"
                  caption="Transform"
                  value={String(settings.transform ?? '')}
                  choices={[
                    ['translation', 'Translation'],
                    ['rigid', 'Rigid'],
                    ['similarity', 'Similarity'],
                    ['affine', 'Affine'],
                  ]}
                  onChange={(value) =>
                    setSettings((previous) => ({ ...previous, transform: value }))
                  }
                />
                <InspectorChoice
                  label="Registration metric"
                  caption="Similarity metric"
                  value={String(settings.metric ?? '')}
                  choices={[
                    ['mattes_mi', 'Mutual information (Mattes)'],
                    ['joint_histogram_mi', 'Mutual information (histogram)'],
                    ['mean_squares', 'Mean squared difference'],
                    ['correlation', 'Correlation'],
                    ['ants_correlation', 'Local correlation'],
                  ]}
                  onChange={(value) => setSettings((previous) => ({ ...previous, metric: value }))}
                />
                <InspectorChoice
                  label="Registration interpolation"
                  caption="Interpolation"
                  value={String(settings.interpolator ?? '')}
                  choices={[
                    ['linear', 'Linear'],
                    ['nearest', 'Nearest neighbour'],
                    ['bspline', 'B-spline'],
                  ]}
                  onChange={(value) =>
                    setSettings((previous) => ({ ...previous, interpolator: value }))
                  }
                />
                <NumberField
                  label="Registration iterations"
                  caption="Iterations"
                  value={Number(settings.iterations ?? 200)}
                  min={1}
                  step="1"
                  onChange={(value) =>
                    setSettings((previous) => ({ ...previous, iterations: value }))
                  }
                />
                <NumberField
                  label="Registration step size"
                  caption="Step size (voxels)"
                  value={Number(settings.learning_rate ?? 0.25)}
                  min={0.000001}
                  onChange={(value) =>
                    setSettings((previous) => ({ ...previous, learning_rate: value }))
                  }
                />
              </div>
            )}
            {!manual && stabilityControls(false)}
            {!manual &&
              (reference || w?.experiment) !== (moving || w?.experiment) &&
              !['mattes_mi', 'joint_histogram_mi'].includes(String(settings.metric)) && (
                <p className="control-note">
                  Correlation and squared difference work best with similar contrast. Use mutual
                  information for different contrasts.
                </p>
              )}
            {manual && (
              <div className="field-grid">
                <NumberField label="Horizontal shift" value={shiftX} onChange={setShiftX} />
                <NumberField label="Vertical shift" value={shiftY} onChange={setShiftY} />
                <NumberField
                  label="Registration rotation"
                  value={rotation}
                  onChange={setRotation}
                />
                <NumberField
                  label="Horizontal scale"
                  value={scaleX}
                  min={0.01}
                  onChange={setScaleX}
                />
                <NumberField
                  label="Vertical scale"
                  value={scaleY}
                  min={0.01}
                  onChange={setScaleY}
                />
              </div>
            )}
            {!manual && (
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={Boolean(settings.normalize_intensity)}
                  onChange={(event) =>
                    setSettings((previous) => ({
                      ...previous,
                      normalize_intensity: event.target.checked,
                    }))
                  }
                />
                Scale intensities for estimation
              </label>
            )}
            <details className="control-advanced">
              <summary>Registration optimizer and metric</summary>
              <JsonEditor
                label="Registration settings"
                value={Object.fromEntries(
                  Object.entries(settings).filter(
                    ([key]) =>
                      ![
                        'transform',
                        'metric',
                        'interpolator',
                        'normalize_intensity',
                        'iterations',
                        'learning_rate',
                        'prealign_translation',
                        'max_displacement_voxels',
                      ].includes(key),
                  ),
                )}
                onApply={async (value) => {
                  setSettings((previous) => ({ ...previous, ...value }));
                }}
              />
            </details>
          </fieldset>
        </InspectorSection>
        <div className="inspector-actionbar">
          <button
            className="primary full"
            disabled={!w?.loaded || busy}
            onClick={() => void run('register', alignmentOptions)}
          >
            <WorkbenchIcon name="registration" />
            {editingAlignment ? 'Stage registration settings' : 'Run registration'}
          </button>
        </div>
        <details className="control-advanced control-wide" open={Boolean(row)}>
          <summary>Stored transforms</summary>
          <InspectorSection title="Transform tools" icon="layers">
            <div className="field-grid">
              <label>
                Stored transform
                <select
                  aria-label="Stored transform"
                  value={transform}
                  onChange={(e) => setTransform(e.target.value)}
                >
                  <option value="">Choose registration</option>
                  {w?.registrations.map((r) => (
                    <option key={String(r.key)} value={String(r.key)}>
                      {String((r.metadata as Options | undefined)?.moving_experiment ?? 'Moving')} →{' '}
                      {String(
                        (r.metadata as Options | undefined)?.reference_experiment ?? 'Reference',
                      )}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Transform direction
                <select
                  aria-label="Transform direction"
                  value={direction}
                  onChange={(e) => setDirection(e.target.value)}
                >
                  <option value="moving_to_reference">Moving to reference</option>
                  <option value="reference_to_moving">Reference to moving</option>
                </select>
              </label>
            </div>
            <div className="button-row">
              <button
                className="secondary"
                aria-label="View registered overlay"
                disabled={!row || busy}
                onClick={() => void action('registration.show', { key: transform })}
              >
                View overlay
              </button>
              <button
                className="secondary"
                aria-label="Apply stored transform"
                disabled={!transform || busy}
                onClick={async () => {
                  await activateTransformSource();
                  await run('pipeline', {
                    action: 'registration',
                    key: transform,
                    direction,
                    step_id: editingAlignment ? editingStep?.id : undefined,
                  });
                }}
              >
                Apply transform
              </button>
              <button
                className="secondary"
                aria-label="Transfer labelled ROI"
                disabled={!transform || !transferredName || busy}
                onClick={async () => {
                  await activateTransformSource();
                  await action('roi.transform', { name: transferredName, transform, direction });
                }}
              >
                Transfer ROI
              </button>
            </div>
            {row && (
              <label>
                ROI to transfer
                <select
                  aria-label="ROI to transfer"
                  value={transferredName}
                  disabled={busy || !sourceRois.length}
                  onChange={(event) => setTransferRoi(event.target.value)}
                >
                  {!sourceRois.length && (
                    <option value="">No enabled masks in {transformSource}</option>
                  )}
                  {sourceRois.map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
            )}
            {row && (
              <details>
                <summary>Registration metrics</summary>
                <Report value={row} label="Registration metrics" />
              </details>
            )}
          </InspectorSection>
        </details>
      </div>
      <div className="control-layout" hidden={task !== 'drift'}>
        <fieldset className="control-form" disabled={!w?.loaded || busy}>
          <InspectorSection title="Motion model" icon="drift">
            {presetControls(true)}
            <div className="field-grid">
              <InspectorChoice
                label="Series alignment method"
                caption="Method"
                value={seriesMethod}
                choices={[
                  ['framewise', 'Frame-wise registration'],
                  ['linear', 'Linear drift'],
                ]}
                onChange={setSeriesMethod}
              />
              <InspectorChoice
                label="Drift transform"
                caption="Transform"
                value={String(
                  seriesMethod === 'framewise'
                    ? (settings.transform ?? 'rigid')
                    : (drift.transform ?? 'translation'),
                )}
                choices={
                  seriesMethod === 'linear' && volumeDrift
                    ? [['translation', 'Translation']]
                    : [
                        ['translation', 'Translation'],
                        ['rigid', 'Rigid'],
                        ['similarity', 'Similarity'],
                        ['affine', 'Affine'],
                      ]
                }
                onChange={(value) =>
                  seriesMethod === 'framewise'
                    ? setSettings((previous) => ({ ...previous, transform: value }))
                    : setDrift((previous) => ({ ...previous, transform: value }))
                }
              />
              <InspectorChoice
                label="Drift metric"
                caption="Similarity metric"
                value={String(
                  seriesMethod === 'framewise'
                    ? (settings.metric ?? 'mattes_mi')
                    : (drift.metric ?? 'ncc'),
                )}
                choices={
                  seriesMethod === 'framewise'
                    ? [
                        ['mattes_mi', 'Mutual information (Mattes)'],
                        ['joint_histogram_mi', 'Mutual information (histogram)'],
                        ['correlation', 'Correlation'],
                        ['ants_correlation', 'Local correlation'],
                        ['mean_squares', 'Mean squared difference'],
                      ]
                    : [
                        ['ncc', 'Normalized correlation'],
                        ['negative_mse', 'Mean squared difference'],
                        ['mutual_information', 'Mutual information'],
                      ]
                }
                onChange={(value) =>
                  seriesMethod === 'framewise'
                    ? setSettings((previous) => ({ ...previous, metric: value }))
                    : setDrift((previous) => ({ ...previous, metric: value }))
                }
              />
            </div>
          </InspectorSection>
          <InspectorSection title="Dimensions and mask" icon="layers">
            <fieldset className="fit-dimensions" aria-label="Drift dimensions">
              <legend>Align along</legend>
              <div className="dimension-choices">
                {driftDimensions.map(([name, size]) => (
                  <label className="check-label" key={name}>
                    <input
                      type="checkbox"
                      aria-label={'Drift dimension ' + name}
                      checked={selectedDriftDimensions.includes(name)}
                      onChange={(event) =>
                        setDrift((previous) => ({
                          ...previous,
                          dimensions: event.target.checked
                            ? [...selectedDriftDimensions, name]
                            : selectedDriftDimensions.filter((value) => value !== name),
                        }))
                      }
                    />
                    {name} <small>({size})</small>
                  </label>
                ))}
              </div>
            </fieldset>
            {seriesMethod === 'framewise' &&
              contrastDimensions.map(([name]) => (
                <label className="check-label" key={name}>
                  <input
                    type="checkbox"
                    checked={estimationDimensions.includes(name)}
                    onChange={(event) =>
                      setAveragedDimensions(
                        event.target.checked
                          ? [...estimationDimensions, name]
                          : estimationDimensions.filter((dimension) => dimension !== name),
                      )
                    }
                  />
                  Average {name} for motion estimation
                </label>
              ))}
            <p className="control-note">
              {seriesMethod === 'framewise' && estimationDimensions.length
                ? 'Each selected frame shares one transform across its averaged contrasts.'
                : 'Other acquisition dimensions form separate groups.'}{' '}
              Slice packages stay separate.
            </p>
            {saturationSeries && (
              <p className="control-note">
                Use time with contrast averaging for dynamic saturation series. Low-signal images
                need visual review.
              </p>
            )}
            <label>
              Registration mask
              <select
                aria-label="Drift mask"
                value={selectedDriftMask}
                onChange={(event) => setDriftMask(event.target.value)}
              >
                <option value="">Whole image</option>
                {w?.rois
                  .filter((roi) => roi.enabled && roi.pixels > 0)
                  .map((roi) => (
                    <option key={roi.name}>{roi.name}</option>
                  ))}
              </select>
            </label>
          </InspectorSection>
          <InspectorSection title="Reference and solver" icon="settings">
            {seriesMethod === 'framewise' && stabilityControls(true)}
            <div className="field-grid">
              <InspectorChoice
                label="Drift reference"
                caption="Reference"
                value={String(drift.reference ?? 'first')}
                choices={[
                  ['first', 'First frame'],
                  ['current', 'Current frame'],
                  ['mean', 'Mean image'],
                  ['median', 'Median image'],
                ]}
                onChange={(value) => setDrift((previous) => ({ ...previous, reference: value }))}
              />
              <NumberField
                label={seriesMethod === 'framewise' ? 'Series iterations' : 'Drift steps'}
                value={Number(
                  seriesMethod === 'framewise'
                    ? (settings.iterations ?? 200)
                    : (drift.num_steps ?? 200),
                )}
                min={1}
                step="1"
                onChange={(value) =>
                  seriesMethod === 'framewise'
                    ? setSettings((previous) => ({ ...previous, iterations: value }))
                    : setDrift((previous) => ({ ...previous, num_steps: value }))
                }
              />
              <NumberField
                label={seriesMethod === 'framewise' ? 'Series learning rate' : 'Drift step size'}
                value={Number(
                  seriesMethod === 'framewise'
                    ? (settings.learning_rate ?? 0.25)
                    : (drift.step_size ?? 0.05),
                )}
                min={0.000001}
                onChange={(value) =>
                  seriesMethod === 'framewise'
                    ? setSettings((previous) => ({ ...previous, learning_rate: value }))
                    : setDrift((previous) => ({ ...previous, step_size: value }))
                }
              />
              <InspectorChoice
                label="Drift optimizer"
                caption="Optimizer"
                value={String(
                  seriesMethod === 'framewise'
                    ? (settings.optimizer ?? 'regular_step_gradient_descent')
                    : (drift.optimizer ?? 'adam'),
                )}
                choices={(seriesMethod === 'framewise'
                  ? [
                      'gradient_descent',
                      'regular_step_gradient_descent',
                      'lbfgsb',
                      'powell',
                      'amoeba',
                    ]
                  : ['adam', 'adamw', 'sgd', 'lion']
                ).map((name) => [name, name] as const)}
                onChange={(value) =>
                  seriesMethod === 'framewise'
                    ? setSettings((previous) => ({ ...previous, optimizer: value }))
                    : setDrift((previous) => ({ ...previous, optimizer: value }))
                }
              />
              <InspectorChoice
                label="Drift interpolation"
                caption="Output interpolation"
                value={String(
                  seriesMethod === 'framewise'
                    ? (settings.interpolator ?? 'linear')
                    : (drift.interpolation ?? 'linear'),
                )}
                choices={(seriesMethod === 'framewise'
                  ? ['linear', 'nearest', 'bspline']
                  : ['linear', 'nearest']
                ).map((name) => [name, name] as const)}
                onChange={(value) =>
                  seriesMethod === 'framewise'
                    ? setSettings((previous) => ({ ...previous, interpolator: value }))
                    : setDrift((previous) => ({ ...previous, interpolation: value }))
                }
              />
              {seriesMethod === 'framewise' ? (
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={Boolean(settings.normalize_intensity)}
                    onChange={(event) =>
                      setSettings((previous) => ({
                        ...previous,
                        normalize_intensity: event.target.checked,
                      }))
                    }
                  />
                  Scale intensities for estimation
                </label>
              ) : (
                <>
                  <label className="check-label">
                    <input
                      type="checkbox"
                      checked={Boolean(drift.normalize_intensity ?? true)}
                      onChange={(event) =>
                        setDrift((previous) => ({
                          ...previous,
                          normalize_intensity: event.target.checked,
                        }))
                      }
                    />
                    Scale intensities for estimation
                  </label>
                  <label className="check-label">
                    <input
                      type="checkbox"
                      checked={Boolean(drift.fit_intercept)}
                      onChange={(event) =>
                        setDrift((previous) => ({
                          ...previous,
                          fit_intercept: event.target.checked,
                        }))
                      }
                    />
                    Fit a free intercept
                  </label>
                </>
              )}
            </div>
          </InspectorSection>
          <div className="inspector-actionbar">
            {seriesMethod === 'framewise' && (
              <details className="control-advanced">
                <summary>Additional registration settings</summary>
                <JsonEditor
                  label="Series registration settings"
                  value={Object.fromEntries(
                    Object.entries(settings).filter(
                      ([key]) =>
                        ![
                          'transform',
                          'metric',
                          'interpolator',
                          'normalize_intensity',
                          'iterations',
                          'learning_rate',
                          'optimizer',
                          'prealign_translation',
                          'max_displacement_voxels',
                        ].includes(key),
                    ),
                  )}
                  onApply={async (value) => setSettings((previous) => ({ ...previous, ...value }))}
                />
              </details>
            )}
            <button
              className="primary full"
              disabled={
                !w?.loaded ||
                busy ||
                !selectedDriftDimensions.length ||
                (seriesMethod === 'linear' &&
                  volumeDrift &&
                  String(drift.transform ?? 'translation') !== 'translation')
              }
              onClick={() => void run('drift', seriesOptions)}
            >
              <WorkbenchIcon name="drift" />
              {editingDrift ? 'Stage series settings' : 'Run series alignment'}
            </button>
          </div>
        </fieldset>
      </div>
    </div>
  );
}
