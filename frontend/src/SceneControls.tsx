import { InspectorSection } from './InspectorSection';
import { useState } from 'react';
import type { Options, Workspace } from './api';
import { CheckField } from './Controls';
import { layerInput, roiColor, type Layer } from './scene';
import { OpacityControl } from './OpacityControl';
import { RoiAppearanceControls } from './RoiAppearanceControls';

export function SceneControls({
  workspace: w,
  busy,
  changeView,
  previewOpacity,
  previewRoiOpacity,
}: {
  workspace: Workspace;
  busy: boolean;
  changeView: (command: Options) => Promise<void>;
  previewOpacity: (id: string | null, value: number) => void;
  previewRoiOpacity?: (name: string, value: number) => void;
}) {
  const [source, setSource] = useState('');
  const layers = w.scene.layers;
  const change = (scene: Options) => changeView({ scene });
  const update = (id: string, patch: Partial<Layer>) =>
    change({
      layers: layers.map((layer) => ({ ...layerInput(layer), ...(layer.id === id ? patch : {}) })),
    });
  const spatial =
    ['Image', 'Map', 'MRax reconstruction'].includes(w.view) &&
    (w.editable || w.view === 'MRax reconstruction');
  return (
    <div className="control-layout">
      <InspectorSection
        title="Image layers"
        icon="layers"
        className="control-wide scene-controls"
        help="Layers blend over the base image. Unregistered images must share its grid."
      >
        <div className="add-layer">
          <label>
            Overlay source
            <select
              aria-label="Overlay source"
              value={source}
              disabled={busy || !w.loaded}
              onChange={(event) => setSource(event.target.value)}
            >
              <option value="">Choose image, map or registration</option>
              {w.scene.sources.map((choice, index) => (
                <option key={index} value={index}>
                  {choice.label}
                </option>
              ))}
            </select>
          </label>
          <button
            className="secondary full"
            disabled={busy || source === '' || layers.length >= 8 || !spatial}
            onClick={() => {
              const { experiment, kind, map_name, registration_key } =
                w.scene.sources[Number(source)];
              void change({
                layers: [
                  ...layers.map(layerInput),
                  {
                    experiment,
                    kind,
                    map_name,
                    registration_key,
                    id: crypto.randomUUID(),
                    frame: 0,
                    component: 'auto',
                    colormap: 'viridis',
                    opacity: 0.5,
                    visible: true,
                    window: null,
                  },
                ],
              });
            }}
          >
            Add overlay
          </button>
        </div>
        {!spatial && (
          <p className="control-note">
            Image layers are available on native images, reconstructions and maps.
          </p>
        )}
        <div className="scene-layer-list">
          {layers.map((layer, index) => {
            const choice = w.scene.sources.find(
              (s) =>
                s.kind === layer.kind &&
                s.experiment === layer.experiment &&
                s.map_name === layer.map_name &&
                s.registration_key === layer.registration_key,
            );
            return (
              <section className="scene-layer" key={layer.id} aria-label={`Overlay ${index + 1}`}>
                <div className="layer-heading">
                  <CheckField
                    label={layer.label}
                    checked={layer.visible}
                    disabled={busy}
                    onChange={(visible) => update(layer.id, { visible })}
                  />
                  <button
                    className="icon-button"
                    aria-label={`Remove overlay ${index + 1}`}
                    title="Remove layer"
                    disabled={busy}
                    onClick={() =>
                      void change({
                        layers: layers.filter((l) => l.id !== layer.id).map(layerInput),
                      })
                    }
                  >
                    ×
                  </button>
                </div>
                {layer.unavailable && (
                  <p className="control-note" role="status">
                    {layer.unavailable}
                  </p>
                )}
                <OpacityControl
                  label={`Overlay ${index + 1} opacity`}
                  value={layer.opacity}
                  disabled={busy}
                  preview={(value) => previewOpacity(layer.id, value)}
                  commit={(opacity) => update(layer.id, { opacity })}
                />
                <details>
                  <summary>Layer settings</summary>
                  <label>
                    Colormap
                    <select
                      aria-label={`Overlay ${index + 1} colormap`}
                      value={layer.colormap}
                      disabled={busy}
                      onChange={(event) =>
                        void update(layer.id, { colormap: event.target.value as Layer['colormap'] })
                      }
                    >
                      {['gray', 'viridis', 'diverging', 'hot'].map((name) => (
                        <option key={name}>{name}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Component
                    <select
                      aria-label={`Overlay ${index + 1} component`}
                      value={layer.component}
                      disabled={busy}
                      onChange={(event) =>
                        void update(layer.id, {
                          component: event.target.value as Layer['component'],
                        })
                      }
                    >
                      {['auto', 'magnitude', 'phase', 'real', 'imag'].map((name) => (
                        <option key={name}>{name}</option>
                      ))}
                    </select>
                  </label>
                  {choice && choice.frames > 1 && (
                    <label>
                      Frame
                      <select
                        aria-label={`Overlay ${index + 1} frame`}
                        value={layer.frame ?? ''}
                        disabled={busy}
                        onChange={(event) =>
                          void update(layer.id, {
                            frame: event.target.value === '' ? null : Number(event.target.value),
                          })
                        }
                      >
                        <option value="">Follow base frame</option>
                        {Array.from({ length: choice.frames }, (_, frame) => (
                          <option key={frame} value={frame}>
                            Frame {frame}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  <CheckField
                    label={`Automatic window for overlay ${index + 1}`}
                    checked={!layer.window}
                    disabled={busy}
                    onChange={(auto) =>
                      update(layer.id, { window: auto ? null : { center: 0.5, width: 1 } })
                    }
                  />
                  {layer.window && (
                    <div className="bounds-grid">
                      {(['center', 'width'] as const).map((key) => (
                        <label key={key}>
                          Window {key}
                          <input
                            aria-label={`Overlay ${index + 1} window ${key}`}
                            type="number"
                            step="any"
                            min={key === 'width' ? '0.000001' : undefined}
                            defaultValue={layer.window![key]}
                            key={`${layer.id}-${key}-${layer.window![key]}`}
                            disabled={busy}
                            onBlur={(event) => {
                              const value = Number(event.target.value);
                              if (
                                Number.isFinite(value) &&
                                (key !== 'width' || value > 0) &&
                                value !== layer.window![key]
                              )
                                void update(layer.id, {
                                  window: { ...layer.window!, [key]: value },
                                });
                            }}
                          />
                        </label>
                      ))}
                    </div>
                  )}
                  <div className="button-row">
                    <button
                      className="secondary"
                      disabled={busy || index === 0}
                      onClick={() => {
                        const reordered = layers.map(layerInput);
                        [reordered[index - 1], reordered[index]] = [
                          reordered[index],
                          reordered[index - 1],
                        ];
                        void change({ layers: reordered });
                      }}
                    >
                      Move down
                    </button>
                    <button
                      className="secondary"
                      disabled={busy || index === layers.length - 1}
                      onClick={() => {
                        const reordered = layers.map(layerInput);
                        [reordered[index + 1], reordered[index]] = [
                          reordered[index],
                          reordered[index + 1],
                        ];
                        void change({ layers: reordered });
                      }}
                    >
                      Move up
                    </button>
                  </div>
                </details>
              </section>
            );
          })}
        </div>
      </InspectorSection>
      <InspectorSection
        title="ROI overlays"
        icon="rois"
        className="control-wide scene-controls"
        help="Visibility affects this view. Choose fitting masks separately in Fit."
      >
        <label>
          Active ROI
          <select
            aria-label="Active ROI"
            value={w.selected_roi ?? ''}
            disabled={busy || !w.rois.some((r) => r.enabled)}
            onChange={(event) => void changeView({ selected_roi: event.target.value })}
          >
            <option value="" disabled>
              Select an ROI
            </option>
            {w.rois
              .filter((r) => r.enabled)
              .map((roi) => (
                <option key={roi.name}>{roi.name}</option>
              ))}
          </select>
        </label>
        {!w.rois.some((r) => r.enabled) && (
          <p className="control-note">Create or enable ROIs in Segment.</p>
        )}
        {w.rois
          .filter((r) => r.enabled)
          .map((roi) => (
            <div className="roi-visibility" key={roi.name}>
              <span
                className="roi-swatch"
                style={{ background: roiColor(roi.name, 1, w.scene.roi_styles[roi.name]) }}
              />
              <CheckField
                label={`Show ${roi.name}`}
                checked={roi.visible}
                disabled={busy}
                onChange={(visible) =>
                  change({
                    hidden_rois: visible
                      ? w.scene.hidden_rois.filter((n) => n !== roi.name)
                      : [...w.scene.hidden_rois, roi.name],
                  })
                }
              />
            </div>
          ))}
        <OpacityControl
          label="ROI opacity"
          value={w.scene.roi_opacity}
          disabled={busy || !w.loaded}
          preview={(value) => previewOpacity(null, value)}
          commit={(roi_opacity) => change({ roi_opacity })}
        />
        <p className="control-note">Default opacity for ROIs without an individual setting.</p>
        {w.selected_roi && (
          <RoiAppearanceControls
            workspace={w}
            name={w.selected_roi}
            busy={busy}
            changeView={changeView}
            previewOpacity={previewRoiOpacity}
          />
        )}
      </InspectorSection>
    </div>
  );
}
