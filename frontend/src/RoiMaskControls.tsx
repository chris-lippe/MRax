import { useState } from 'react';
import type { AnalysisProps, DrawingProps } from './analysisTypes';
import { NumberField } from './Controls';
import { InspectorChoice, InspectorSection } from './InspectorSection';
import { roiColor } from './scene';

export function RoiMaskControls({
  workspace: w,
  busy,
  action,
  mode,
  label,
  setLabel,
}: Pick<AnalysisProps, 'workspace' | 'busy' | 'action'> &
  Pick<DrawingProps, 'label' | 'setLabel'> & { mode: 'labels' | 'volume' | 'combine' }) {
  const [targets, setTargets] = useState('');
  const [interpolation, setInterpolation] = useState('linear');
  const [extrapolation, setExtrapolation] = useState('none');
  const [radius, setRadius] = useState(1);
  const [replacement, setReplacement] = useState(1);
  const [sources, setSources] = useState<string[]>([]);
  const [name, setName] = useState('Combined ROI');
  const [output, setOutput] = useState('binary');
  const [overlap, setOverlap] = useState('first');
  const selected = w?.selected_roi ?? w?.rois.find((r) => r.enabled)?.name;
  const roi = w?.rois.find((r) => r.name === selected);
  const blocked = busy || !w?.editable;
  const axis = w?.slice_dim ?? 'z';
  const sliceCount = w?.dimensions[axis] ?? 1;
  const targetIndices = targets.trim() ? targets.split(',').map((v) => Number(v.trim()) - 1) : null;
  const validTargets =
    !targetIndices || targetIndices.every((v) => Number.isInteger(v) && v >= 0 && v < sliceCount);
  const selectedSources = sources.filter((key) => w?.rois.some((r) => r.name === key));
  const labelNames = roi?.metadata.label_names as Record<string, string> | undefined;

  if (mode === 'labels')
    return (
      <InspectorSection
        title="Mask labels"
        icon="layers"
        help="Choose an existing label to paint or erase it. New label values can be entered in the drawing controls."
      >
        <div className="roi-label-choices" role="group" aria-label="Mask labels">
          {roi?.labels.map((value) => (
            <button
              key={value}
              className="secondary roi-label-choice"
              aria-label={'Use label ' + value}
              aria-pressed={label === value}
              disabled={blocked}
              onClick={() => setLabel(value)}
            >
              <span
                className="roi-label-swatch"
                style={{ background: roiColor(roi.name, value, w!.scene.roi_styles[roi.name]) }}
              />
              {value}
              {labelNames?.[String(value)] ? ' · ' + labelNames[String(value)] : ''}
            </button>
          ))}
        </div>
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
      title="Combined mask"
      icon="layers"
      className="control-wide"
      help="Create a new mask from source ROIs on the same grid. Original source masks are retained. Indexed labels follow selection order."
    >
      <fieldset className="control-form" disabled={blocked}>
        <div className="roi-source-list" role="group" aria-label="Source ROIs">
          {w?.rois.map((source) => (
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
          <label>
            New ROI name
            <input
              aria-label="Combined ROI name"
              value={name}
              maxLength={80}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <InspectorChoice
            label="Combined mask type"
            value={output}
            choices={[
              ['binary', 'Joint binary ROI'],
              ['indexed', 'Indexed label mask'],
            ]}
            onChange={setOutput}
          />
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
          className="primary full"
          disabled={
            selectedSources.length < 2 ||
            !name.trim() ||
            w?.rois.some((r) => r.name === name.trim())
          }
          onClick={() =>
            void action('roi.compose', {
              name: name.trim(),
              sources: selectedSources,
              output,
              overlap,
            })
          }
        >
          {output === 'binary' ? 'Create joint binary ROI' : 'Create indexed label mask'}
        </button>
        <div className="button-row">
          <button
            className="secondary"
            disabled={!roi || !selectedSources.some((name) => name !== roi.name)}
            onClick={() =>
              void action('roi.combine', {
                name: roi?.name,
                others: selectedSources.filter((name) => name !== roi?.name),
              })
            }
          >
            Merge into selected ROI
          </button>
          <button
            className="secondary"
            disabled={!roi || !selectedSources.some((name) => name !== roi.name)}
            onClick={() =>
              void action('roi.subtract', {
                name: roi?.name,
                others: selectedSources.filter((name) => name !== roi?.name),
              })
            }
          >
            Subtract
          </button>
        </div>
      </fieldset>
    </InspectorSection>
  );
}
