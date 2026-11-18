export function ImageAxes({
  dims,
  labels = dims,
  directions = [1, 1],
}: {
  dims: string[];
  labels?: string[];
  directions?: number[];
}) {
  return (
    <>
      {[1, 0].map((index) => {
        const horizontal = index === 1;
        const reversed = directions[index] < 0;
        const label = labels[index] ?? dims[index];
        const direction = horizontal ? (reversed ? 'left' : 'right') : reversed ? 'up' : 'down';
        const description = `${label} increases ${direction}; data ${horizontal ? 'column' : 'row'} axis ${dims[index]}`;
        return (
          <span
            className={`orientation ${horizontal ? 'top' : 'left'}`}
            key={index}
            role="img"
            aria-label={description}
            title={description}
          >
            <span>{label}</span>
            <svg
              viewBox="0 0 32 10"
              aria-hidden="true"
              style={{ transform: `rotate(${(horizontal ? 0 : 90) + (reversed ? 180 : 0)}deg)` }}
            >
              <path d="M2 5h27m-5-4 5 4-5 4" />
            </svg>
          </span>
        );
      })}
    </>
  );
}
