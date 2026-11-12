import type { Workspace, Options } from './api';
import { OpacityControl } from './OpacityControl';
import { roiColor, type RoiAppearance } from './scene';
import { WorkbenchIcon } from './WorkbenchIcon';

export function RoiAppearanceControls({
  workspace: w,
  name,
  busy,
  changeView,
  previewOpacity,
  activeLabel,
  editLabel,
}: {
  workspace: Workspace;
  name: string;
  busy: boolean;
  changeView: (command: Options) => Promise<void>;
  previewOpacity?: (name: string, value: number, label?: number) => void;
  activeLabel?: number;
  editLabel?: (label: number) => void;
}) {
  const roi = w.rois.find((r) => r.name === name);
  if (!roi) return null;
  const style = w.scene.roi_styles[name] ?? {};
  const labelNames = roi.metadata.label_names as Record<string, string> | undefined;
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
      {(roi.labels.length > 1 || roi.metadata.consolidated || roi.metadata.kind === 'mask') && (
        <details className="control-advanced" open={Boolean(editLabel)}>
          <summary>Labels</summary>
          <p className="control-note">Exports include all label values.</p>
          <div className="roi-label-styles">
            {roi.labels.map((value) => {
              const title =
                'Label ' +
                value +
                (labelNames?.[String(value)] ? ' · ' + labelNames[String(value)] : '');
              const hidden = style.hidden_labels?.includes(value) ?? false;
              return (
                <div
                  className="roi-label-style"
                  role="group"
                  aria-label={title + ' in ' + name}
                  key={value}
                >
                  <div className="roi-label-heading">
                    {editLabel ? (
                      <button
                        className="secondary roi-label-choice"
                        aria-label={'Use label ' + value}
                        aria-pressed={activeLabel === value}
                        title={'Paint ' + title}
                        disabled={busy || !w.editable || !roi.enabled}
                        onClick={() => editLabel(value)}
                      >
                        <span
                          className="roi-label-swatch"
                          style={{ background: roiColor(name, value, style) }}
                        />
                        <span>{title}</span>
                        <WorkbenchIcon name="brush" />
                      </button>
                    ) : (
                      <strong>{title}</strong>
                    )}
                    <label className="check-label">
                      <input
                        type="checkbox"
                        aria-label={'Show label ' + value + ' in ' + name}
                        checked={!hidden}
                        disabled={busy}
                        onChange={(event) =>
                          void change({
                            hidden_labels: event.target.checked
                              ? (style.hidden_labels ?? []).filter((label) => label !== value)
                              : [...(style.hidden_labels ?? []), value],
                          })
                        }
                      />
                      Visible
                    </label>
                  </div>
                  <div className="roi-label-fields">
                    <label className="roi-color-field">
                      Color
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
                    <OpacityControl
                      label={'Opacity for label ' + value + ' in ' + name}
                      caption="Opacity"
                      value={
                        style.label_opacities?.[String(value)] ??
                        style.opacity ??
                        w.scene.roi_opacity
                      }
                      disabled={busy}
                      preview={(alpha) => previewOpacity?.(name, alpha, value)}
                      commit={(alpha) =>
                        change({ label_opacities: { ...style.label_opacities, [value]: alpha } })
                      }
                    />
                    <button
                      className="secondary"
                      aria-label={'Reset appearance for label ' + value + ' in ' + name}
                      disabled={
                        busy ||
                        !(
                          String(value) in (style.label_colors ?? {}) ||
                          String(value) in (style.label_opacities ?? {}) ||
                          hidden
                        )
                      }
                      onClick={() => {
                        const colors = { ...style.label_colors },
                          alphas = { ...style.label_opacities };
                        delete colors[String(value)];
                        delete alphas[String(value)];
                        void change({
                          label_colors: colors,
                          label_opacities: alphas,
                          hidden_labels: (style.hidden_labels ?? []).filter(
                            (label) => label !== value,
                          ),
                        });
                      }}
                    >
                      Reset
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
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
