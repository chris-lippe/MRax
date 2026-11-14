export type FitDimension = {
  name: string;
  label: string;
  size: number;
  axis: number;
  regularizable: boolean;
};

export function FitDimensionSelector({
  label,
  caption,
  dimensions,
  selected,
  numeric = false,
  disabled,
  onChange,
}: {
  label: string;
  caption: string;
  dimensions: FitDimension[];
  selected: (string | number)[];
  numeric?: boolean;
  disabled: boolean;
  onChange: (values: (string | number)[]) => void;
}) {
  const dimensionFor = (value: string | number) =>
    dimensions.find(
      (dimension) =>
        value === dimension.name ||
        value === dimension.axis ||
        (numeric &&
          typeof value === 'number' &&
          value < 0 &&
          value + dimensions.length === dimension.axis),
    );
  const normalized = selected.map((value) => {
    const dimension = dimensionFor(value);
    return dimension ? (numeric ? dimension.axis : dimension.name) : value;
  });
  const unavailable = selected.filter((value) => !dimensionFor(value));
  return (
    <fieldset className="fit-dimensions" aria-label={label} disabled={disabled}>
      <legend>{caption}</legend>
      <div className="dimension-choices">
        {dimensions.map((dimension) => {
          const value = numeric ? dimension.axis : dimension.name;
          const checked = normalized.includes(value);
          return (
            <label className="check-label" key={dimension.name}>
              <input
                type="checkbox"
                aria-label={label + ' ' + dimension.name}
                checked={checked}
                disabled={
                  !checked && (dimension.size <= 1 || (numeric && !dimension.regularizable))
                }
                onChange={(event) =>
                  onChange(
                    event.target.checked
                      ? [...normalized, value]
                      : normalized.filter((entry) => entry !== value),
                  )
                }
              />
              {dimension.label} <small>({dimension.size})</small>
            </label>
          );
        })}
      </div>
      {!dimensions.length && <p className="control-note">No parameter dimensions available.</p>}
      {!!unavailable.length && (
        <p role="alert">
          Unavailable dimensions: {unavailable.join(', ')}.
          <button
            type="button"
            className="secondary small"
            onClick={() => onChange(normalized.filter((value) => !unavailable.includes(value)))}
          >
            Clear unavailable
          </button>
        </p>
      )}
    </fieldset>
  );
}
