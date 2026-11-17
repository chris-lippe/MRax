import type { Options } from './api';
import { NumberField } from './Controls';
import { FitDimensionSelector, type FitDimension } from './FitDimensionSelector';

export type ParameterSpec = {
  initial: number;
  bounds: [number, number] | null;
  unit: string;
  trainable: boolean;
};

export function FitParameterEditor({
  name,
  caption = name,
  spec,
  override = {},
  regularizers = [],
  constants = [],
  dimensions,
  disabled,
  onChange,
  onFixedChange,
  onRegularizersChange,
  onConstantsChange,
}: {
  name: string;
  caption?: string;
  spec: ParameterSpec;
  override?: Options;
  regularizers?: Options[];
  constants?: (string | number)[];
  dimensions: FitDimension[];
  disabled: boolean;
  onChange: (key: string, value: unknown) => void;
  onFixedChange: (fixed: boolean) => void;
  onRegularizersChange: (values: Options[]) => void;
  onConstantsChange: (values: (string | number)[]) => void;
}) {
  const fixed = override.trainable === false || (override.trainable == null && !spec.trainable);
  const bounds = (override.bounds ?? spec.bounds) as [number, number] | null;
  const initial = Number((fixed ? override.value : undefined) ?? override.init ?? spec.initial);
  const updateRegularizer = (index: number, value: Options) =>
    onRegularizersChange(
      regularizers.map((entry, i) => (i === index ? { ...entry, ...value } : entry)),
    );
  const spatial = dimensions.filter(
    (dimension) => ['x', 'y', 'z'].includes(dimension.name) && dimension.regularizable,
  );
  const temporal = dimensions.filter(
    (dimension) => ['t', 'frame'].includes(dimension.name) && dimension.regularizable,
  );
  const addRegularizer = (axes: FitDimension[], type: string) =>
    onRegularizersChange([
      ...regularizers,
      { type, weight: 0.01, axes: axes.map((dimension) => dimension.axis) },
    ]);
  return (
    <fieldset disabled={disabled} className="fit-parameter">
      <legend>
        {caption} <small>{spec.unit}</small>
      </legend>
      <div className="field-grid parameter-value">
        <NumberField
          label={name + ' initial'}
          caption={fixed ? 'Fixed value' : 'Initial value'}
          value={initial}
          onChange={(value) => onChange('init', value)}
        />
        <label className="check-label">
          <input
            type="checkbox"
            aria-label={'Fix ' + name}
            checked={fixed}
            onChange={(event) => onFixedChange(event.target.checked)}
          />
          Fixed
        </label>
      </div>
      {bounds && (
        <div className="bounds-grid">
          <NumberField
            label={name + ' lower bound'}
            caption="Lower bound"
            value={bounds[0]}
            onChange={(value) => onChange('bounds', [value, bounds[1]])}
          />
          <NumberField
            label={name + ' upper bound'}
            caption="Upper bound"
            value={bounds[1]}
            onChange={(value) => onChange('bounds', [bounds[0], value])}
          />
        </div>
      )}
      <details className="parameter-spatial">
        <summary>Dimension settings</summary>
        {regularizers.map((entry, index) => {
          const suffix = index ? ' ' + (index + 1) : '';
          return (
            <fieldset className="fit-regularizer" key={index}>
              <legend>Penalty {index + 1}</legend>
              <div className="bounds-grid">
                <label>
                  Regularizer
                  <select
                    aria-label={name + ' regularizer' + suffix}
                    value={String(entry.type)}
                    onChange={(event) => updateRegularizer(index, { type: event.target.value })}
                  >
                    <option value="smoothness">Smoothness</option>
                    <option value="tv">Total variation</option>
                    <option value="tv2">Second-order TV</option>
                  </select>
                </label>
                <NumberField
                  label={name + ' regularization weight' + suffix}
                  caption="Weight"
                  value={Number(entry.weight ?? 0)}
                  min={0}
                  onChange={(value) => updateRegularizer(index, { weight: value })}
                />
              </div>
              <FitDimensionSelector
                label={name + ' regularization dimension' + suffix}
                caption="Regularization dimensions"
                dimensions={dimensions}
                selected={
                  (entry.axes ??
                    dimensions
                      .filter((dimension) => dimension.regularizable)
                      .map((dimension) => dimension.axis)) as (string | number)[]
                }
                numeric
                disabled={disabled}
                onChange={(axes) => updateRegularizer(index, { axes })}
              />
              <button
                type="button"
                className="secondary small"
                aria-label={'Remove ' + name + ' regularizer ' + (index + 1)}
                onClick={() => onRegularizersChange(regularizers.filter((_, i) => i !== index))}
              >
                Remove penalty
              </button>
            </fieldset>
          );
        })}
        <div className="button-row">
          {!!spatial.length && (
            <button
              type="button"
              className="secondary small"
              aria-label={'Add ' + name + ' spatial regularizer'}
              onClick={() => addRegularizer(spatial, 'tv')}
            >
              Add spatial penalty
            </button>
          )}
          {!!temporal.length && (
            <button
              type="button"
              className="secondary small"
              aria-label={'Add ' + name + ' temporal regularizer'}
              onClick={() => addRegularizer(temporal, 'smoothness')}
            >
              Add temporal penalty
            </button>
          )}
          {!spatial.length &&
            !temporal.length &&
            dimensions.some((dimension) => dimension.regularizable) && (
              <button
                type="button"
                className="secondary small"
                aria-label={'Add ' + name + ' regularizer'}
                onClick={() =>
                  addRegularizer(
                    dimensions.filter((dimension) => dimension.regularizable),
                    'smoothness',
                  )
                }
              >
                Add penalty
              </button>
            )}
        </div>
        <FitDimensionSelector
          label={name + ' constant dimension'}
          caption="Constant across dimensions"
          dimensions={dimensions}
          selected={constants}
          disabled={disabled}
          onChange={onConstantsChange}
        />
      </details>
    </fieldset>
  );
}
