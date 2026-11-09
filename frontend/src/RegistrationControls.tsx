import { useEffect, useState } from 'react';
import { api, type Options } from './api';
import type { AnalysisProps } from './analysisTypes';
import { JsonEditor, NumberField, Report } from './Controls';
import type { WorkflowTask } from './WorkflowNav';
import { InspectorSection, InspectorChoice } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';

export function RegistrationControls({
  workspace: w,
  busy,
  action,
  changeView,
  run,
  task,
  editingStep,
}: AnalysisProps & { task: WorkflowTask }) {
  const [reference, setReference] = useState(''),
    [moving, setMoving] = useState(''),
    [referenceFrame, setReferenceFrame] = useState(0),
    [movingFrame, setMovingFrame] = useState(0);
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
      setDrift((options.settings as Options) ?? {});
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
    void api<{ registration: Options; drift: Options }>('/api/catalog')
      .then((c) => {
        setSettings((previous) => ({ ...c.registration, ...previous }));
        setDrift((previous) => ({ ...c.drift, ...previous }));
      })
      .catch((e) => setError(String(e)));
  }, []);
  const row = w?.registrations.find((r) => r.key === transform);
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
  return (
    <div className="registration-controls">
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
                        MRax reconstruction
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
                  <NumberField
                    caption="Frame"
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
              </div>
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
            <details className="control-advanced">
              <summary>Registration optimizer and metric</summary>
              <JsonEditor
                label="Registration settings"
                value={settings}
                onApply={async (value) => {
                  setSettings(value);
                }}
              />
            </details>
          </fieldset>
        </InspectorSection>
        <div className="inspector-actionbar">
          <button
            className="primary full"
            disabled={!w?.loaded || busy}
            onClick={() =>
              void run('register', {
                step_id: editingAlignment ? editingStep?.id : undefined,
                reference_experiment: reference || w?.experiment,
                moving_experiment: moving || w?.experiment,
                reference_source: referenceSource,
                moving_source: movingSource,
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
              })
            }
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
            <div className="field-grid">
              <InspectorChoice
                label="Drift transform"
                caption="Transform"
                value={String(drift.transform ?? '')}
                choices={[
                  ['translation', 'Translation'],
                  ['rigid', 'Rigid'],
                  ['similarity', 'Similarity'],
                  ['affine', 'Affine'],
                ]}
                onChange={(value) => setDrift((previous) => ({ ...previous, transform: value }))}
              />
              <InspectorChoice
                label="Drift metric"
                caption="Similarity metric"
                value={String(drift.metric ?? '')}
                choices={[
                  ['ncc', 'Normalized correlation'],
                  ['negative_mse', 'Mean squared difference'],
                  ['mutual_information', 'Mutual information'],
                ]}
                onChange={(value) => setDrift((previous) => ({ ...previous, metric: value }))}
              />
            </div>
          </InspectorSection>
          <InspectorSection title="Reference and solver" icon="settings">
            <div className="field-grid">
              <InspectorChoice
                label="Drift reference"
                caption="Reference"
                value={String(drift.reference ?? '')}
                choices={[
                  ['first', 'First frame'],
                  ['current', 'Current frame'],
                  ['mean', 'Mean image'],
                  ['median', 'Median image'],
                ]}
                onChange={(value) => setDrift((previous) => ({ ...previous, reference: value }))}
              />
              <NumberField
                label="Drift steps"
                value={Number(drift.num_steps ?? 200)}
                min={1}
                step="1"
                onChange={(value) => setDrift((previous) => ({ ...previous, num_steps: value }))}
              />
              <NumberField
                label="Drift step size"
                value={Number(drift.step_size ?? 0.05)}
                min={0.000001}
                onChange={(value) => setDrift((previous) => ({ ...previous, step_size: value }))}
              />
            </div>
          </InspectorSection>
          <details className="control-advanced control-wide">
            <summary>Drift model and reference</summary>
            <JsonEditor
              label="Drift settings"
              value={drift}
              onApply={async (value) => {
                setDrift(value);
              }}
            />
          </details>
          <div className="inspector-actionbar">
            <button
              className="primary full"
              disabled={!w?.loaded || busy}
              onClick={() =>
                void run('drift', {
                  settings: drift,
                  roi_name: w?.selected_roi,
                  step_id: editingDrift ? editingStep?.id : undefined,
                })
              }
            >
              <WorkbenchIcon name="drift" />
              {editingDrift ? 'Stage drift settings' : 'Run drift correction'}
            </button>
          </div>
        </fieldset>
      </div>
    </div>
  );
}
