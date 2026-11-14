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
  const updateRegularizer = (value: Options) =>
    onRegularizersChange([{ ...regularizers[0], ...value }, ...regularizers.slice(1)]);
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
        <summary>Spatial settings</summary>
        <label>
          Regularizer
          <select
            aria-label={name + ' regularizer'}
            value={String(regularizers[0]?.type ?? 'none')}
            onChange={(event) =>
              event.target.value === 'none'
                ? onRegularizersChange([])
                : updateRegularizer({
                    type: event.target.value,
                    weight: regularizers[0]?.weight ?? 0.01,
                    axes:
                      regularizers[0]?.axes ??
                      dimensions
                        .filter((dimension) => dimension.regularizable)
                        .map((dimension) => dimension.axis),
                  })
            }
          >
            {['none', 'smoothness', 'tv', 'tv2'].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        {!!regularizers.length && (
          <>
            <NumberField
              label={name + ' regularization weight'}
              caption="Weight"
              value={Number(regularizers[0].weight ?? 0)}
              min={0}
              onChange={(value) => updateRegularizer({ weight: value })}
            />
            <FitDimensionSelector
              label={name + ' regularization dimension'}
              caption="Regularization dimensions"
              dimensions={dimensions}
              selected={
                (regularizers[0].axes ??
                  dimensions
                    .filter((dimension) => dimension.regularizable)
                    .map((dimension) => dimension.axis)) as (string | number)[]
              }
              numeric
              disabled={disabled}
              onChange={(axes) => updateRegularizer({ axes })}
            />
          </>
        )}
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
