import type { Workspace, Options } from './api';
import { OpacityControl } from './OpacityControl';
import { roiColor, type RoiAppearance } from './scene';

export function RoiAppearanceControls({
  workspace: w,
  name,
  busy,
  changeView,
  previewOpacity,
}: {
  workspace: Workspace;
  name: string;
  busy: boolean;
  changeView: (command: Options) => Promise<void>;
  previewOpacity?: (name: string, value: number) => void;
}) {
  const roi = w.rois.find((r) => r.name === name);
  if (!roi) return null;
  const style = w.scene.roi_styles[name] ?? {};
  const change = (patch: Partial<RoiAppearance>) =>
    changeView({
      scene: { roi_styles: { ...w.scene.roi_styles, [name]: { ...style, ...patch } } },
    });
  return (
    <div className="roi-appearance" role="group" aria-label="ROI appearance">
      <label className="roi-color-field">
        Color
        <input
          type="color"
          aria-label={'Color for ' + name}
          value={style.color ?? roiColor(name)}
          disabled={busy}
          onChange={(e) => void change({ color: e.target.value })}
        />
        <span>{style.color ?? roiColor(name)}</span>
      </label>
      <OpacityControl
        label={'Opacity for ' + name}
        value={style.opacity ?? w.scene.roi_opacity}
        disabled={busy}
        preview={(value) => previewOpacity?.(name, value)}
        commit={(opacity) => change({ opacity })}
      />
      {roi.labels.length > 1 && (
        <details className="control-advanced">
          <summary>Label colors</summary>
          {roi.labels.map((value) => (
            <label className="roi-color-field" key={value}>
              Label {value}
              <input
                type="color"
                aria-label={'Color for label ' + value + ' in ' + name}
                value={roiColor(name, value, style)}
                disabled={busy}
                onChange={(event) =>
                  void change({
                    label_colors: { ...style.label_colors, [value]: event.target.value },
                  })
                }
              />
            </label>
          ))}
        </details>
      )}
      <button
        className="secondary"
        disabled={busy || !(name in w.scene.roi_styles)}
        onClick={() => {
          const styles = { ...w.scene.roi_styles };
          delete styles[name];
          void changeView({ scene: { roi_styles: styles } });
        }}
      >
        Reset ROI appearance
      </button>
    </div>
  );
}
