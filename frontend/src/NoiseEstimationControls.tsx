import { useEffect, useRef, useState } from 'react';
import { api, type Options, type Workspace } from './api';

type Estimate = {
  sigma: number;
  range: [number, number];
  blocks: number;
  estimated_blocks: number;
  scope: string;
  warnings: string[];
  components?: Record<string, number>;
};

const number = (value: number) =>
  new Intl.NumberFormat('en', { maximumSignificantDigits: 4 }).format(value);

export function NoiseEstimationControls({
  workspace,
  options,
  busy,
  roiNames,
  setRois,
  parameter,
}: {
  workspace: Workspace;
  options: Options;
  busy: boolean;
  roiNames: string[] | null;
  setRois: (names: string[] | null) => void;
  parameter: (key: string, value: unknown) => void;
}) {
  const [estimate, setEstimate] = useState<{ key: string; value: Estimate } | null>(null);
  const [pending, setPending] = useState('');
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const readout = useRef<HTMLDivElement>(null);
  const parameters = options.parameters as Options;
  const model = String(parameters.noise_model ?? 'auto');
  const key = JSON.stringify([workspace.id, workspace.revision, options]);
  useEffect(() => {
    if (estimate?.key === key) readout.current?.scrollIntoView({ block: 'nearest' });
  }, [estimate, key]);
  const stored = workspace.pipeline.steps.find((step) =>
    options.step_id ? step.id === options.step_id : step.family === 'denoise',
  );
  const matches =
    stored &&
    stored.method === options.method &&
    JSON.stringify(stored.options.noise_roi_names ?? null) === JSON.stringify(roiNames) &&
    ['input', 'component', 'axes'].every(
      (field) => JSON.stringify(stored.options[field]) === JSON.stringify(options[field]),
    ) &&
    ['noise_model', 'noise_coils', 'complex_mode'].every(
      (field) =>
        ((stored.options.parameters as Options)?.[field] ?? parameters[field]) ===
        parameters[field],
    );
  const value =
    estimate?.key === key
      ? estimate.value
      : matches
        ? (stored?.noise_estimate as Estimate | null)
        : null;
  const eligible = workspace.rois.filter((roi) => roi.pixels > 0);
  return (
    <div className="noise-estimation">
      <div className="field-grid">
        <label>
          Noise estimation model
          <select
            aria-label="Noise estimation model"
            value={model}
            onChange={(event) => {
              parameter('noise_model', event.target.value);
              if (event.target.value === 'background' && roiNames === null) setRois([]);
            }}
          >
            <option value="auto">MRI signal patches</option>
            <option value="gaussian">Uniform signal / Gaussian</option>
            <option value="background">Noise-only magnitude background</option>
          </select>
        </label>
        {model === 'background' && (
          <label>
            Effective noise coils
            <input
              aria-label="Effective noise coils"
              type="number"
              min={1}
              max={64}
              step={1}
              value={Number(parameters.noise_coils ?? 1)}
              required
              onChange={(event) => parameter('noise_coils', Number(event.target.value))}
            />
          </label>
        )}
      </div>
      <label className="check-label">
        <input
          type="checkbox"
          checked={roiNames !== null}
          disabled={model === 'background'}
          onChange={(event) => setRois(event.target.checked ? [] : null)}
        />
        Estimate noise from ROIs
      </label>
      {roiNames !== null && (
        <div className="processing-rois" role="group" aria-label="Noise estimation ROIs">
          {eligible.map((roi) => (
            <label className="check-label" key={roi.name}>
              <input
                type="checkbox"
                aria-label={`Estimate noise in ${roi.name}`}
                checked={roiNames.includes(roi.name)}
                onChange={(event) =>
                  setRois(
                    event.target.checked
                      ? [...roiNames, roi.name]
                      : roiNames.filter((name) => name !== roi.name),
                  )
                }
              />
              {roi.name}
            </label>
          ))}
          {!eligible.length && (
            <p className="control-note">Create a uniform ROI in Segment first.</p>
          )}
        </div>
      )}
      <p className="control-note">
        Sigma uses input intensity units. Noise ROIs do not limit processing.
        {model === 'background'
          ? ' Select signal-free magnitude background. One effective coil means Rician noise. Receiver count may differ.'
          : ' Use a uniform region and avoid tissue boundaries.'}
      </p>
      <button
        type="button"
        disabled={busy || pending === key || (roiNames !== null && !roiNames.length)}
        onClick={() => {
          setPending(key);
          setFailure(null);
          void api<Estimate>(`/api/workspaces/${workspace.id}/noise-estimate`, 'POST', {
            revision: workspace.revision,
            options,
          })
            .then((value) => setEstimate({ key, value }))
            .catch((error) => setFailure({ key, message: String(error) }))
            .finally(() => setPending((current) => (current === key ? '' : current)));
        }}
      >
        {pending === key ? 'Estimating noise…' : 'Estimate noise'}
      </button>
      {failure?.key === key && <p role="alert">{failure.message}</p>}
      {value && (
        <div className="noise-estimate-result" ref={readout} role="status" aria-live="polite">
          <span>
            Estimated σ <output aria-label="Estimated noise sigma">{number(value.sigma)}</output>
          </span>
          <span className="control-note">
            Range {number(value.range[0])}–{number(value.range[1])} · {value.estimated_blocks} of{' '}
            {value.blocks} blocks sampled.
          </span>
          {value.components && (
            <span className="control-note">
              Real σ {number(value.components.real)} · Imaginary σ{' '}
              {number(value.components.imaginary)}
            </span>
          )}
          {value.scope === 'roi' && (
            <span className="control-note">The ROI estimate is shared across filter blocks.</span>
          )}
          {value.warnings.map((warning) => (
            <span className="control-note" key={warning}>
              {warning}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
