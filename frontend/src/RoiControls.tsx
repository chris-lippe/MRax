import { useState } from 'react';
import type { Bounds, Tool, Options } from './api';
import type { AnalysisProps, DrawingProps } from './analysisTypes';
import { NumberField, JsonEditor } from './Controls';
import { InspectorGroup, InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { RoiMaskControls } from './RoiMaskControls';
import { RoiAppearanceControls } from './RoiAppearanceControls';
import { MaskPreviewDialog } from './MaskPreviewDialog';
import { nextRoiName, roiColor } from './scene';

export function RoiControls({
  workspace: w,
  busy,
  action,
  changeView,
  previewRoiOpacity,
  tool,
  brushSize,
  label,
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
  const geometric = !['brush', 'erase'].includes(tool);
  const mode = String(drawOptions.mode ?? 'new');
  const coordinateTool = ['pan', 'rect', 'ellipse', 'regular_polygon'].includes(tool);
  const coordinateShape =
    tool === 'rect' || tool === 'pan' ? 'rectangle' : tool === 'ellipse' ? 'ellipse' : 'polygon';
  const coordinateAction = mode === 'subtract' ? 'Subtract' : 'Add';
  const defaultName = nextRoiName(w?.rois ?? []);
  const options = () => ({
    ...drawOptions,
    name: name || defaultName,
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
        help="Click a tool to draw. Click it again to return to panning and shape handles. Alt-drag or the middle mouse button pans with any tool. Label 0 is background."
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
              <option value="add" disabled={!roi}>
                Add drawn region to {selected ?? 'selected ROI'}
              </option>
              <option value="subtract" disabled={!roi}>
                Subtract drawn region from {selected ?? 'selected ROI'}
              </option>
            </select>
          </label>
        )}
        {geometric && mode !== 'new' && !roi && (
          <p className="control-note">Select an ROI or choose Create a new ROI.</p>
        )}
        {geometric && mode === 'subtract' && roi && (
          <p className="control-note">
            Draw the region to remove from <strong>{roi.name}</strong>.
          </p>
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
            ] as const
          ).map(([value, title, icon]) => (
            <button
              key={value}
              type="button"
              className="secondary"
              aria-label={title + ' tool'}
              title={title}
              aria-pressed={tool === value}
              data-tutorial={value === 'rect' ? 'roi-start' : undefined}
              disabled={!w?.editable || busy}
              onClick={() => setTool(tool === value ? 'pan' : (value as Tool))}
            >
              <WorkbenchIcon name={icon} />
              <span>{title}</span>
            </button>
          ))}
          <button
            type="button"
            className="secondary"
            aria-label="Otsu threshold"
            title={
              mode === 'subtract'
                ? `Subtract Otsu selection from ${selected ?? 'selected ROI'}`
                : 'Select a region using Otsu thresholding'
            }
            disabled={!w?.editable || busy || (mode !== 'new' && !roi)}
            onClick={() => {
              setTool('pan');
              void create({ ...options(), tool: 'otsu', name: defaultName });
            }}
          >
            <WorkbenchIcon name="threshold" />
            <span>Otsu</span>
          </button>
        </div>
        <p className="control-note">
          {tool === 'pan'
            ? 'Drag to pan. Select a tool to draw.'
            : 'Click the selected tool again to pan.'}
        </p>
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
            aria-label="Auto subtract from other ROIs"
            disabled={
              busy || !w?.editable || tool === 'erase' || (geometric && mode === 'subtract')
            }
            onChange={(e) => setDrawOptions({ ...drawOptions, auto_subtract: e.target.checked })}
          />
          <span title="Remove newly drawn voxels from other enabled ROIs. Undo restores the whole edit.">
            Auto subtract from other ROIs
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
            {tool === 'brush'
              ? 'Keep other labels when painting ' + label
              : 'Erase only label ' + label}
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
                activeLabel={label}
                editLabel={async (value) => {
                  const style = w.scene.roi_styles[roi.name];
                  if (!roi.visible || style?.hidden_labels?.includes(value)) {
                    await changeView({
                      scene: {
                        hidden_rois: w.rois
                          .filter((item) => !item.visible && item.name !== roi.name)
                          .map((item) => item.name),
                        roi_styles: {
                          ...w.scene.roi_styles,
                          [roi.name]: {
                            ...style,
                            hidden_labels: (style?.hidden_labels ?? []).filter(
                              (label) => label !== value,
                            ),
                          },
                        },
                      },
                    });
                  }
                  setLabel(value);
                  setDrawOptions({ ...drawOptions, label_only: true });
                  await editMask();
                }}
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
        <label>
          Rename selected ROI
          <input
            aria-label="Rename selected ROI"
            disabled={busy || !roi}
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
      </InspectorSection>
      {roi && (roi.metadata.consolidated || roi.metadata.kind === 'mask' || !geometric) && (
        <RoiMaskControls workspace={w} busy={busy} action={action} mode="labels" label={label} />
      )}
      <RoiMaskControls workspace={w} busy={busy} action={action} mode="volume" label={label} />
      <RoiMaskControls workspace={w} busy={busy} action={action} mode="combine" label={label} />
      <InspectorGroup title="Geometry edits" icon="rois" wide={false}>
        <form
          aria-label="ROI coordinate editor"
          hidden={!coordinateTool}
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
                placeholder={defaultName}
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
      </InspectorGroup>
    </div>
  );
}
