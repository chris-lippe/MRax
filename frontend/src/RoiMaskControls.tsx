import { useState } from 'react';
import type { AnalysisProps, DrawingProps } from './analysisTypes';
import { NumberField } from './Controls';
import { InspectorChoice, InspectorSection } from './InspectorSection';

export function RoiMaskControls({
  workspace: w,
  busy,
  action,
  mode,
  label,
}: Pick<AnalysisProps, 'workspace' | 'busy' | 'action'> &
  Pick<DrawingProps, 'label'> & { mode: 'labels' | 'volume' | 'combine' }) {
  const [targets, setTargets] = useState('');
  const [interpolation, setInterpolation] = useState('linear');
  const [extrapolation, setExtrapolation] = useState('none');
  const [radius, setRadius] = useState(1);
  const [replacement, setReplacement] = useState(1);
  const [sources, setSources] = useState<string[]>([]);
  const [name, setName] = useState('Combined ROI');
  const [output, setOutput] = useState('binary');
  const [overlap, setOverlap] = useState('first');
  const [target, setTarget] = useState('');
  const selected = w?.selected_roi ?? w?.rois.find((r) => r.enabled)?.name;
  const roi = w?.rois.find((r) => r.name === selected);
  const blocked = busy || !w?.editable;
  const axis = w?.slice_dim ?? 'z';
  const sliceCount = w?.dimensions[axis] ?? 1;
  const targetIndices = targets.trim() ? targets.split(',').map((v) => Number(v.trim()) - 1) : null;
  const validTargets =
    !targetIndices || targetIndices.every((v) => Number.isInteger(v) && v >= 0 && v < sliceCount);
  const selectedSources = sources.filter((key) => w?.rois.some((r) => r.name === key));
  const replacing = output === 'merge' || output === 'subtract';
  const targetName = w?.rois.some((r) => r.name === target) ? target : (selected ?? '');
  const operands = selectedSources.filter((name) => !replacing || name !== targetName);
  const sourceLabel =
    output === 'subtract'
      ? 'ROIs to subtract'
      : output === 'merge'
        ? 'ROIs to merge'
        : 'Source ROIs';
  const labelNames = roi?.metadata.label_names as Record<string, string> | undefined;

  if (mode === 'labels')
    return (
      <InspectorSection
        title="Mask labels"
        icon="layers"
        help="Select a label in ROI appearance to paint it. New label values can be entered in the drawing controls."
      >
        <p className="control-note">
          Selected label: {label}
          {labelNames?.[String(label)] ? ' · ' + labelNames[String(label)] : ''}
        </p>
        {!roi?.labels.length && (
          <p className="control-note">Paint a mask or select a labelled ROI.</p>
        )}
        <NumberField
          label="Replacement label"
          value={replacement}
          min={1}
          step="1"
          onChange={setReplacement}
        />
        <div className="button-row">
          <button
            className="secondary"
            disabled={
              blocked ||
              !roi?.labels.includes(label) ||
              !Number.isInteger(replacement) ||
              replacement < 1 ||
              replacement > 65535
            }
            onClick={() =>
              void action('roi.relabel', { name: roi?.name, old: label, label: replacement })
            }
          >
            Relabel selected label
          </button>
          <button
            className="secondary danger"
            disabled={blocked || !roi?.labels.includes(label)}
            onClick={() => void action('roi.delete_label', { name: roi?.name, label })}
          >
            Delete selected label
          </button>
        </div>
      </InspectorSection>
    );

  if (mode === 'volume')
    return (
      <>
        <InspectorSection
          title="Expand across slices"
          icon="layers"
          help="Expand along the current viewing plane's slice axis. Copy replaces target slices. Interpolation preserves existing labelled voxels. Undo restores the previous mask."
        >
          <p className="control-note">
            Axis {axis} · current slice {(w?.indices[axis] ?? w?.z ?? 0) + 1} of {sliceCount}
          </p>
          <label>
            Target slices (1 based. Leave blank for all)
            <input
              aria-label="Target slices"
              placeholder="1, 2, 3"
              value={targets}
              onChange={(event) => setTargets(event.target.value)}
            />
          </label>
          {!validTargets && (
            <p className="control-note">Use slice numbers from 1 to {sliceCount}.</p>
          )}
          <div className="button-row">
            <button
              className="secondary"
              disabled={blocked || !roi?.pixels || sliceCount < 2 || !validTargets}
              onClick={() =>
                void action('roi.copy_slice', {
                  name: roi?.name,
                  axis,
                  source: w?.indices[axis] ?? w?.z ?? 0,
                  ...(targetIndices ? { targets: targetIndices } : {}),
                })
              }
            >
              Copy current slice
            </button>
            <button
              className="secondary"
              disabled={blocked || !roi}
              onClick={() => void action('roi.delete_slice', { name: roi?.name })}
            >
              Delete current slice
            </button>
          </div>
          <div className="field-grid">
            <InspectorChoice
              label="Slice interpolation"
              value={interpolation}
              choices={[
                ['linear', 'Linear'],
                ['nearest', 'Nearest'],
                ['spline', 'Cubic spline'],
              ]}
              onChange={setInterpolation}
            />
            <InspectorChoice
              label="Outside labelled slices"
              value={extrapolation}
              choices={[
                ['none', 'Leave empty'],
                ['nearest', 'Copy nearest label'],
              ]}
              onChange={setExtrapolation}
            />
          </div>
          <button
            className="secondary full"
            disabled={blocked || !roi?.pixels || sliceCount < 2}
            onClick={() =>
              void action('roi.interpolate', {
                name: roi?.name,
                axis,
                method: interpolation,
                extrapolation,
              })
            }
          >
            Interpolate labelled slices
          </button>
        </InspectorSection>
        <InspectorSection
          title="Grow in 3D"
          icon="rois"
          help="Grow into background voxels by Euclidean distance in voxel units. New voxels take the nearest label. Existing labels are retained."
        >
          <NumberField
            label="Growth (voxels)"
            value={radius}
            min={1}
            step="1"
            onChange={setRadius}
          />
          <button
            className="primary full"
            disabled={
              blocked || !roi?.pixels || !Number.isFinite(radius) || radius <= 0 || radius > 64
            }
            onClick={() => void action('roi.grow', { name: roi?.name, radius })}
          >
            Grow ROI in 3D
          </button>
        </InspectorSection>
      </>
    );

  return (
    <InspectorSection
      title="Combine ROIs"
      icon="layers"
      help="Create a new mask, merge ROIs or subtract from a target ROI. Undo restores the previous masks. Indexed labels follow selection order."
    >
      <fieldset className="control-form" disabled={blocked}>
        <InspectorChoice
          label="Combine method"
          value={output}
          choices={[
            ['binary', 'Joint binary ROI'],
            ['indexed', 'Indexed label mask'],
            ['merge', 'Merge into an ROI'],
            ['subtract', 'Subtract from an ROI'],
          ]}
          onChange={setOutput}
        />
        {replacing && (
          <label>
            {output === 'subtract' ? 'Subtract from' : 'Merge into'}
            <select
              aria-label={output === 'subtract' ? 'Subtract from' : 'Merge into'}
              value={targetName}
              onChange={(event) => setTarget(event.target.value)}
            >
              <option value="" disabled>
                Choose target ROI
              </option>
              {w?.rois.map((item) => (
                <option key={item.name}>{item.name}</option>
              ))}
            </select>
          </label>
        )}
        <p className="roi-source-heading">{sourceLabel}</p>
        <div className="roi-source-list" role="group" aria-label={sourceLabel}>
          {w?.rois
            .filter((source) => !replacing || source.name !== targetName)
            .map((source) => (
              <label className="check-label" key={source.name}>
                <input
                  type="checkbox"
                  aria-label={'Include ' + source.name}
                  checked={selectedSources.includes(source.name)}
                  onChange={(event) =>
                    setSources(
                      event.target.checked
                        ? [...selectedSources, source.name]
                        : selectedSources.filter((key) => key !== source.name),
                    )
                  }
                />
                <span>{source.name}</span>
                {output === 'indexed' && selectedSources.includes(source.name) && (
                  <span className="label-badge">
                    Label {selectedSources.indexOf(source.name) + 1}
                  </span>
                )}
              </label>
            ))}
        </div>
        <div className="field-grid">
          {!replacing && (
            <label>
              New ROI name
              <input
                aria-label="Combined ROI name"
                value={name}
                maxLength={80}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
          )}
          {output === 'indexed' && (
            <InspectorChoice
              label="Overlapping voxels"
              value={overlap}
              choices={[
                ['first', 'First selected ROI wins'],
                ['last', 'Last selected ROI wins'],
                ['error', 'Reject overlaps'],
              ]}
              onChange={setOverlap}
            />
          )}
        </div>
        <button
          className="primary full roi-combine-action"
          disabled={
            replacing
              ? !targetName || operands.length === 0
              : operands.length < 2 || !name.trim() || w?.rois.some((r) => r.name === name.trim())
          }
          onClick={() =>
            replacing
              ? void action(output === 'subtract' ? 'roi.subtract' : 'roi.combine', {
                  name: targetName,
                  others: operands,
                })
              : void action('roi.compose', {
                  name: name.trim(),
                  sources: operands,
                  output,
                  overlap,
                })
          }
        >
          {output === 'subtract'
            ? `Subtract ${operands.length ? operands.join(', ') : 'ROIs'} from ${targetName || 'target ROI'}`
            : output === 'merge'
              ? `Merge ${operands.length ? operands.join(', ') : 'ROIs'} into ${targetName || 'target ROI'}`
              : output === 'binary'
                ? 'Create joint binary ROI'
                : 'Create indexed label mask'}
        </button>
      </fieldset>
    </InspectorSection>
  );
}
