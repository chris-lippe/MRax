import { Component, useId, useState, type ReactNode } from 'react';
import { numberError, type DataNode, type DataPath } from './structuredData';

export class TableBoundary extends Component<
  {
    children: ReactNode;
    onError: () => void;
  },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export function DataFormat({
  label,
  raw,
  onChange,
}: {
  label: string;
  raw: boolean;
  onChange: (raw: boolean) => void;
}) {
  return (
    <div className="data-format" role="group" aria-label={`${label} format`}>
      {['Table', 'Raw'].map((name, index) => (
        <button
          key={name}
          type="button"
          aria-pressed={raw === Boolean(index)}
          onClick={() => onChange(Boolean(index))}
        >
          {name}
        </button>
      ))}
    </div>
  );
}

type DataProps = {
  node: DataNode;
  path: DataPath;
  label: string;
  depth: number;
  onChange?: (path: DataPath, node: DataNode) => void;
  disabled?: boolean;
};

const fieldName = (label: string, path: DataPath) => [label, ...path].join(' / ');

function Scalar({ node, path, label, onChange, disabled }: DataProps) {
  const errorId = useId();
  const name = fieldName(label, path);
  if (node.kind === 'array' || node.kind === 'object') return null;
  if (!onChange || node.kind === 'null')
    return <span className="data-value">{node.kind === 'null' ? 'null' : String(node.value)}</span>;
  if (node.kind === 'boolean')
    return (
      <input
        aria-label={name}
        type="checkbox"
        checked={node.value}
        disabled={disabled}
        onChange={(e) => onChange(path, { kind: 'boolean', value: e.target.checked })}
      />
    );
  const invalid = node.kind === 'number' && numberError(node.value);
  const attributes = {
    'aria-label': name,
    value: node.value,
    disabled,
    onChange: (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
      onChange(path, { ...node, value: event.target.value }),
  };
  return (
    <div className={node.kind === 'number' ? 'data-number' : undefined}>
      {node.kind === 'string' ? (
        <textarea
          {...attributes}
          rows={node.value.includes('\n') || node.value.length > 100 ? 3 : 1}
          spellCheck={false}
        />
      ) : (
        <input
          {...attributes}
          type="text"
          inputMode={node.kind === 'number' ? 'decimal' : undefined}
          aria-invalid={invalid || undefined}
          aria-describedby={invalid ? errorId : undefined}
        />
      )}
      {invalid && (
        <span id={errorId} className="data-error">
          Enter a finite JSON number.
        </span>
      )}
    </div>
  );
}

function summary(node: DataNode): string {
  if (node.kind === 'object')
    return `${node.entries.length} ${node.entries.length === 1 ? 'field' : 'fields'}`;
  if (node.kind !== 'array') return '';
  const preview = node.items
    .slice(0, 4)
    .map((item) =>
      item.kind === 'object'
        ? '{…}'
        : item.kind === 'array'
          ? '[…]'
          : item.kind === 'null'
            ? 'null'
            : String(item.value),
    )
    .join(', ');
  return `${node.items.length} ${node.items.length === 1 ? 'value' : 'values'}${preview ? ` · ${preview.slice(0, 80)}${node.items.length > 4 || preview.length > 80 ? '…' : ''}` : ''}`;
}

function Cell(props: DataProps) {
  if (props.node.kind !== 'object' && props.node.kind !== 'array') return <Scalar {...props} />;
  return (
    <details className="data-nested" open={props.depth < 2 && props.node.kind === 'object'}>
      <summary aria-label={`Expand ${fieldName(props.label, props.path)}`}>
        {summary(props.node)}
      </summary>
      <DataTable {...props} />
    </details>
  );
}

function quantity(node: DataNode) {
  if (node.kind !== 'object' || node.entries.length !== 2) return null;
  const values = node.entries.find(([key]) => key === 'values');
  const unit = node.entries.find(([key]) => key === 'unit');
  return values && unit?.[1].kind === 'string' ? { values: values[1], unit: unit[1] } : null;
}

export function DataTable(props: DataProps) {
  const [page, setPage] = useState(0);
  const { node, path, label, depth } = props;
  if (node.kind !== 'array' && node.kind !== 'object') return <Scalar {...props} />;
  const rows: [string | number, DataNode][] =
    node.kind === 'array' ? node.items.map((item, i) => [i, item]) : node.entries;
  const columns =
    node.kind === 'array' && rows.length && rows.every(([, item]) => item.kind === 'object')
      ? Array.from(
          new Set(
            rows.flatMap(([, item]) =>
              item.kind === 'object' ? item.entries.map(([key]) => key) : [],
            ),
          ),
        )
      : [];
  const records = columns.length > 0 && columns.length <= 12;
  const units = !records && rows.some(([, item]) => quantity(item));
  const last = Math.max(0, Math.ceil(rows.length / 50) - 1);
  const current = Math.min(page, last);
  const shown = rows.slice(current * 50, (current + 1) * 50);
  const child = (value: DataNode, childPath: DataPath) => (
    <Cell {...props} node={value} path={childPath} depth={depth + 1} />
  );
  return (
    <div className="structured-table">
      {rows.length === 0 ? (
        <p className="control-note">{node.kind === 'array' ? 'Empty list' : 'No fields'}</p>
      ) : (
        <div className="structured-scroll">
          <table aria-label={`${fieldName(label, path)} table`}>
            <thead>
              <tr>
                <th scope="col">{node.kind === 'array' ? 'Index' : 'Field'}</th>
                {records ? (
                  columns.map((key) => (
                    <th key={key} scope="col">
                      {key}
                    </th>
                  ))
                ) : (
                  <th scope="col">Value</th>
                )}
                {units && <th scope="col">Unit</th>}
              </tr>
            </thead>
            <tbody>
              {shown.map(([key, value]) => {
                const qty = units ? quantity(value) : null;
                return (
                  <tr key={key}>
                    <th scope="row">{key}</th>
                    {records && value.kind === 'object' ? (
                      columns.map((column) => {
                        const entry = value.entries.find(([name]) => name === column);
                        return (
                          <td key={column}>
                            {entry ? (
                              child(entry[1], [...path, key, column])
                            ) : (
                              <span className="muted">—</span>
                            )}
                          </td>
                        );
                      })
                    ) : (
                      <td>
                        {child(qty ? qty.values : value, [
                          ...path,
                          key,
                          ...(qty ? ['values'] : []),
                        ])}
                      </td>
                    )}
                    {units && (
                      <td className="data-unit">
                        {qty ? child(qty.unit, [...path, key, 'unit']) : '—'}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {last > 0 && (
        <div className="data-pages">
          <button
            className="secondary"
            type="button"
            aria-label={`Previous rows for ${fieldName(label, path)}`}
            disabled={current === 0}
            onClick={() => setPage(current - 1)}
          >
            Previous
          </button>
          <span>
            {current * 50 + 1}–{Math.min((current + 1) * 50, rows.length)} of {rows.length}
          </span>
          <button
            className="secondary"
            type="button"
            aria-label={`Next rows for ${fieldName(label, path)}`}
            disabled={current === last}
            onClick={() => setPage(current + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
