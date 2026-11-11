import { useState } from 'react';
import type { Bounds, Tool, Options } from './api';
import type { AnalysisProps, DrawingProps } from './analysisTypes';
import { NumberField, JsonEditor } from './Controls';
import { InspectorGroup, InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { RoiMaskControls } from './RoiMaskControls';
import { RoiAppearanceControls } from './RoiAppearanceControls';
import { MaskPreviewDialog } from './MaskPreviewDialog';
import { roiColor } from './scene';

export function RoiControls({
  workspace: w,
  busy,
  action,
  changeView,
  previewRoiOpacity,
  drawing,
  tool,
  brushSize,
  label,
  setDrawing,
  setTool,
  setBrushSize,
  setLabel,
  drawOptions,
  setDrawOptions,
}: AnalysisProps & DrawingProps) {
  const [bounds, setBounds] = useState<Bounds>([5, 5, 12, 10]),
    [name, setName] = useState('');
  const [maskPreview, setMaskPreview] = useState(false);
  const angle = Number(drawOptions.angle ?? 0),
    sides = Number(drawOptions.sides ?? 6);
  const setAngle = (value: number) => setDrawOptions({ ...drawOptions, angle: value });
  const setSides = (value: number) => setDrawOptions({ ...drawOptions, sides: value });
  const [shift, setShift] = useState('0,0,0'),
    [newName, setNewName] = useState('');
  const selected = w?.selected_roi ?? w?.rois.find((r) => r.enabled)?.name;
  const roi = w?.rois.find((r) => r.name === selected);
  const geometric = !['brush', 'erase', 'pan'].includes(tool);
  const mode = String(drawOptions.mode ?? 'new');
  const coordinateTool = ['rect', 'ellipse', 'regular_polygon'].includes(tool);
  const coordinateShape =
    tool === 'rect' ? 'rectangle' : tool === 'ellipse' ? 'ellipse' : 'polygon';
  const coordinateAction = mode === 'subtract' ? 'Subtract' : 'Add';
  const options = () => ({
    ...drawOptions,
    name: name || `ROI ${(w?.rois.length ?? 0) + 1}`,
    tool: ['brush', 'erase', 'pan'].includes(tool) ? 'rect' : tool,
    bounds,
    angle,
    sides,
    label,
  });
  const command = (op: string, extra = {}) => action(`roi.${op}`, { name: selected, ...extra });
  const editMask = async () => {
    if (!roi) return;
    if (!roi.metadata.consolidated) {
      await command('consolidate');
      if (roi.labels.length === 1) setLabel(roi.labels[0]);
    }
    setMaskPreview(false);
    setTool('brush');
    setDrawing(true);
  };
  const create = (value: Options = options()) =>
    action(mode === 'new' ? 'roi.create' : 'roi.draw', {
      ...value,
      ...(mode === 'new' ? {} : { name: selected, mode }),
    });
  return (
    <div className="control-layout">
      <InspectorSection
        title="Draw and paint"
        icon="rois"
        help="Choose a tool, then activate drawing on the image. Labels are positive integers. Label 0 is background."
      >
        {geometric && (
          <label>
            Drawing mode
            <select
              aria-label="Drawing mode"
              value={mode}
              onChange={(event) => setDrawOptions({ ...drawOptions, mode: event.target.value })}
            >
              <option value="new">Create a new ROI</option>
              <option value="add">Add round to selected ROI</option>
              <option value="subtract">Subtract round from selected ROI</option>
            </select>
          </label>
        )}
        <div className="roi-tool-palette" role="group" aria-label="ROI tool">
          {(
            [
              ['rect', 'Rectangle', 'rectangle'],
              ['ellipse', 'Ellipse', 'ellipse'],
              ['regular_polygon', 'Regular polygon', 'polygon'],
              ['polygon', 'Polygon', 'polygon'],
              ['lasso', 'Lasso', 'lasso'],
              ['brush', 'Brush', 'brush'],
              ['erase', 'Eraser', 'eraser'],
              ['wand', 'Magic wand', 'wand'],
              ['pan', 'Pan image', 'pan'],
            ] as const
          ).map(([value, title, icon]) => (
            <button
              key={value}
              type="button"
              className="secondary"
              aria-label={title + ' tool'}
              title={title}
              aria-pressed={tool === value}
              disabled={!w?.editable || busy}
              onClick={() => setTool(value as Tool)}
            >
              <WorkbenchIcon name={icon} />
              <span>{title}</span>
            </button>
          ))}
        </div>
        <button
          className={`${drawing ? 'primary' : 'secondary'} full`}
          aria-pressed={drawing}
          data-tutorial="roi-start"
          disabled={!w?.editable || busy || (geometric && mode !== 'new' && !roi)}
          onClick={() => setDrawing(!drawing)}
        >
          {drawing
            ? `Drawing ${tool === 'rect' ? 'rectangle' : tool}…`
            : tool === 'brush'
              ? 'Paint labels'
              : tool === 'erase'
                ? 'Erase labels'
                : tool === 'rect'
                  ? 'Draw rectangle'
                  : `Draw ${tool.replace('_', ' ')}`}
        </button>
        <div className="bounds-grid">
          <NumberField label="Label value" value={label} min={1} step="1" onChange={setLabel} />
          <NumberField
            label="Brush size"
            value={brushSize}
            min={1}
            step="1"
            onChange={setBrushSize}
          />
        </div>
        <details className="control-advanced" hidden={!coordinateTool}>
          <summary>Coordinates</summary>
          <form
            aria-label="ROI coordinate editor"
            onSubmit={(e) => {
              e.preventDefault();
              void create();
            }}
          >
            {mode === 'new' ? (
              <label>
                New ROI name
                <input
                  aria-label="New ROI name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={`ROI ${(w?.rois.length ?? 0) + 1}`}
                  maxLength={80}
                />
              </label>
            ) : (
              <p className="control-note">Selected ROI: {selected ?? 'Select an enabled ROI'}</p>
            )}
            <div className="bounds-grid">
              {['Column', 'Row', 'Width', 'Height'].map((l, index) => (
                <label key={l}>
                  {l}
                  <input
                    aria-label={`ROI ${l.toLowerCase()}`}
                    type="number"
                    min={index < 2 ? 0 : 0.5}
                    step=".5"
                    value={bounds[index]}
                    onChange={(e) =>
                      setBounds(
                        bounds.map((v, i) => (index === i ? Number(e.target.value) : v)) as Bounds,
                      )
                    }
                    required
                  />
                </label>
              ))}
            </div>
            <button
              className="secondary full"
              disabled={!w?.editable || busy || (mode !== 'new' && !roi?.enabled)}
            >
              {coordinateAction} {coordinateShape} {mode === 'new' ? 'by coordinates' : 'round'}
            </button>
          </form>
        </details>
        {tool === 'lasso' && (
          <label className="check-label">
            <input
              type="checkbox"
              checked={Boolean(drawOptions.snap)}
              onChange={(e) => setDrawOptions({ ...drawOptions, snap: e.target.checked })}
            />
            Snap lasso to image edges
          </label>
        )}
        <label className="check-label">
          <input
            type="checkbox"
            checked={Boolean(drawOptions.auto_subtract)}
            disabled={
              busy || !w?.editable || tool === 'erase' || (geometric && mode === 'subtract')
            }
            onChange={(e) => setDrawOptions({ ...drawOptions, auto_subtract: e.target.checked })}
          />
          <span title="Remove newly drawn voxels from other enabled ROIs. Undo restores the whole edit.">
            Auto subtract
          </span>
        </label>
        {(tool === 'brush' || tool === 'erase' || (geometric && mode === 'subtract')) && (
          <label className="check-label">
            <input
              type="checkbox"
              checked={Boolean(drawOptions.label_only)}
              onChange={(event) =>
                setDrawOptions({ ...drawOptions, label_only: event.target.checked })
              }
            />
            Erase only label {label}
          </label>
        )}
        {tool === 'wand' && (
          <NumberField
            label="Wand tolerance"
            value={Number(drawOptions.tolerance ?? 0.08)}
            min={0}
            onChange={(v) => setDrawOptions({ ...drawOptions, tolerance: v })}
          />
        )}
      </InspectorSection>
      <InspectorSection title="ROIs" icon="layers">
        {!w?.rois.length && (
          <p className="control-note">Draw on the image or add a shape by coordinates.</p>
        )}
        <div className="roi-cards">
          {w?.rois.map((r) => (
            <div key={r.name} className={`roi-card ${selected === r.name ? 'selected' : ''}`}>
              <input
                aria-label={`Enable ${r.name}`}
                type="checkbox"
                checked={r.enabled}
                disabled={busy}
                onChange={() =>
                  void action(`roi.${r.enabled ? 'disable' : 'enable'}`, { name: r.name })
                }
              />
              <button
                className="secondary"
                disabled={!r.enabled || busy}
                onClick={() => void changeView({ selected_roi: r.name })}
              >
                <i
                  className="roi-swatch"
                  style={{ background: roiColor(r.name, 1, w.scene.roi_styles[r.name]) }}
                />
                <strong>{r.name}</strong>
                <span>{r.pixels} voxels</span>
              </button>
              <span className="label-badge">{r.labels.join(', ') || 'Empty'}</span>
            </div>
          ))}
        </div>
        {w && roi && (
          <>
            <div className="roi-selected-controls">
              <p className="control-note">
                <strong>{roi.name}</strong> ·{' '}
                {roi.metadata.consolidated || roi.metadata.kind === 'mask'
                  ? 'Pixel mask'
                  : 'Shape ROI'}
              </p>
              <RoiAppearanceControls
                workspace={w}
                name={roi.name}
                busy={busy}
                changeView={changeView}
                previewOpacity={previewRoiOpacity}
              />
              <div className="button-row">
                <button
                  className="secondary"
                  disabled={busy || !w.editable || !roi.enabled}
                  title="Keep the current labelled voxels and edit them with a brush. Undo restores the shape."
                  data-tutorial="roi-pixel-mask"
                  onClick={() => void editMask()}
                >
                  {roi.metadata.consolidated || roi.metadata.kind === 'mask'
                    ? 'Edit with brush'
                    : 'Convert to pixel mask'}
                </button>
                <button className="secondary" disabled={busy} onClick={() => setMaskPreview(true)}>
                  Preview export mask
                </button>
              </div>
            </div>
            {maskPreview && (
              <MaskPreviewDialog
                key={w.id + roi.name}
                workspace={w}
                name={roi.name}
                busy={busy}
                onClose={() => setMaskPreview(false)}
                onEdit={() => void editMask()}
              />
            )}
          </>
        )}
        <div className="button-row">
          <button
            className="secondary"
            disabled={!w?.roi_undo || busy}
            onClick={() => void action('roi.undo')}
          >
            Undo ROI edit
          </button>
          <button
            className="secondary"
            disabled={!w?.rois.length || busy}
            onClick={() => void action('roi.apply_all')}
          >
            Apply to all experiments
          </button>
        </div>
      </InspectorSection>
      {roi && (roi.metadata.consolidated || roi.metadata.kind === 'mask' || !geometric) && (
        <RoiMaskControls
          workspace={w}
          busy={busy}
          action={action}
          mode="labels"
          label={label}
          setLabel={(value) => {
            setLabel(value);
            setTool('brush');
          }}
        />
      )}
      <InspectorGroup title="3D expansion" icon="layers">
        <RoiMaskControls
          workspace={w}
          busy={busy}
          action={action}
          mode="volume"
          label={label}
          setLabel={setLabel}
        />
      </InspectorGroup>
      <InspectorGroup title="Combine ROIs" icon="layers">
        <RoiMaskControls
          workspace={w}
          busy={busy}
          action={action}
          mode="combine"
          label={label}
          setLabel={setLabel}
        />
      </InspectorGroup>
      <details className="control-advanced control-wide">
        <summary>Geometry and ROI management</summary>
        <div className="bounds-grid">
          <NumberField label="Rotation (degrees)" value={angle} onChange={setAngle} />
          <NumberField label="Polygon sides" value={sides} min={3} step="1" onChange={setSides} />
        </div>
        <div className="button-row">
          <button
            className="secondary"
            disabled={!roi || Boolean(roi.metadata.consolidated) || busy}
            onClick={() => void command('geometry', { ...options(), name: selected })}
          >
            Update selected geometry
          </button>
        </div>
        <button
          className="secondary full"
          disabled={!w?.editable || busy}
          onClick={() => void create({ ...options(), tool: 'otsu' })}
        >
          Create Otsu selection
        </button>
        <label>
          Rename selected ROI
          <input
            aria-label="Rename selected ROI"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
        </label>
        <div className="button-row">
          <button
            className="secondary"
            disabled={!roi || !newName || busy}
            onClick={() => void command('rename', { new_name: newName })}
          >
            Rename ROI
          </button>
          <button
            className="secondary danger"
            disabled={!roi || busy}
            onClick={() => void command('remove')}
          >
            Delete ROI
          </button>
        </div>
        <label>
          Shift in{' '}
          {Object.keys(w?.dimensions ?? {})
            .filter((v) => ['x', 'y', 'z'].includes(v))
            .join(', ')}{' '}
          voxels
          <input aria-label="ROI shift" value={shift} onChange={(e) => setShift(e.target.value)} />
        </label>
        <button
          className="secondary full"
          disabled={!roi || busy}
          onClick={() =>
            void command('shift', {
              shifts: Object.fromEntries(
                ['x', 'y', 'z']
                  .filter((v) => v in (w?.dimensions ?? {}))
                  .map((v, i) => [v, Number(shift.split(',')[i] ?? 0)]),
              ),
            })
          }
        >
          Shift ROI
        </button>
        <JsonEditor
          label="ROI geometry options"
          value={{
            tool: 'polygon',
            points: [
              [4, 4],
              [15, 4],
              [10, 15],
            ],
            snap: false,
            tolerance: 0.08,
            auto_subtract: false,
          }}
          disabled={!w?.editable || busy}
          onApply={(value) => create({ ...options(), ...value })}
        />
        <div className="button-row">
          <button
            className="secondary"
            disabled={!w || busy}
            onClick={() => void action('roi.enable_all')}
          >
            Enable all
          </button>
          <button
            className="secondary"
            disabled={!w || busy}
            onClick={() => void action('roi.disable_all')}
          >
            Disable all
          </button>
        </div>
      </details>
    </div>
  );
}
