import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import type { Options } from './api';
import { DataFormat, DataTable, TableBoundary } from './StructuredDataControls';
import {
  compileTable,
  parseData,
  printData,
  readData,
  replaceData,
  serializeData,
  type DataNode,
  type DataPath,
} from './structuredData';

export type DataDraft = { raw: boolean; text: string; node: DataNode | null; error: string };
export function dataDraft(text: string): DataDraft {
  try {
    return { raw: false, text, node: compileTable(text), error: '' };
  } catch (failure) {
    return { raw: true, text, node: null, error: tableError(failure) };
  }
}
const tableError = (failure: unknown) =>
  `Table view unavailable. ${failure instanceof Error ? failure.message : 'Use Raw to inspect or edit this value.'}`;

function useDataDraft(value: unknown) {
  const serialized = serializeData(value);
  const [draft, setDraft] = useState(() => dataDraft(serialized));
  useEffect(() => {
    setDraft((previous) => ({ ...dataDraft(serialized), ...(previous.raw ? { raw: true } : {}) }));
  }, [serialized]);
  return dataController(draft, setDraft);
}

export function dataController(draft: DataDraft, setDraft: Dispatch<SetStateAction<DataDraft>>) {
  const text = !draft.raw && draft.node ? printData(draft.node) : draft.text;
  const chooseFormat = (raw: boolean) => {
    if (raw) setDraft({ raw: true, text, node: null, error: '' });
    else setDraft(dataDraft(text));
  };
  const change = (path: DataPath, node: DataNode) =>
    setDraft((previous) => ({
      ...previous,
      node: previous.node ? replaceData(previous.node, path, node) : null,
      error: '',
    }));
  const fallback = () =>
    setDraft({
      raw: true,
      text,
      node: null,
      error: 'Table view unavailable. The data is still available in Raw.',
    });
  return { draft, setDraft, text, chooseFormat, change, fallback };
}

export function JsonEditor({
  label,
  value,
  onApply,
  disabled = false,
}: {
  label: string;
  value: unknown;
  onApply: (value: Options) => Promise<void>;
  disabled?: boolean;
}) {
  const controller = useDataDraft(value);
  return <DataEditor label={label} controller={controller} onApply={onApply} disabled={disabled} />;
}

export function DataEditor({
  label,
  controller,
  onApply,
  disabled = false,
  path = [],
  applyLabel,
}: {
  label: string;
  controller: ReturnType<typeof dataController>;
  onApply: (value: Options) => Promise<void>;
  disabled?: boolean;
  path?: DataPath;
  applyLabel?: string;
}) {
  const { draft, setDraft, text, chooseFormat, change, fallback } = controller;
  const selected = draft.node ? readData(draft.node, path) : undefined;
  const [pending, setPending] = useState(false);
  return (
    <form
      className="json-editor structured-data"
      aria-label={`${label} editor`}
      aria-busy={pending}
      onSubmit={async (e) => {
        e.preventDefault();
        if (disabled || pending) return;
        try {
          const data = parseData(text);
          if (!data || typeof data !== 'object' || Array.isArray(data))
            throw Error('Enter a JSON object');
          setDraft((previous) => ({ ...previous, error: '' }));
          setPending(true);
          await onApply(data as Options);
        } catch (failure) {
          setDraft({
            raw: true,
            text,
            node: null,
            error: failure instanceof Error ? failure.message : 'Invalid JSON',
          });
        } finally {
          setPending(false);
        }
      }}
    >
      <div className="data-heading">
        <strong>{label}</strong>
        <DataFormat label={label} raw={draft.raw} onChange={chooseFormat} />
      </div>
      {draft.raw ? (
        <>
          {path.length > 0 && <p className="control-note">Raw shows the entire document.</p>}
          <textarea
            aria-label={label}
            value={text}
            spellCheck={false}
            disabled={disabled || pending}
            onChange={(e) => setDraft({ raw: true, text: e.target.value, node: null, error: '' })}
            rows={8}
          />
        </>
      ) : (
        draft.node && (
          <TableBoundary onError={fallback}>
            <DataTable
              node={selected ?? draft.node}
              label={label}
              path={selected ? path : []}
              depth={0}
              onChange={change}
              disabled={disabled || pending}
            />
          </TableBoundary>
        )
      )}
      {draft.error && <p role="alert">{draft.error}</p>}
      <button className="secondary full" disabled={disabled || pending}>
        {applyLabel ?? `Apply ${label.toLowerCase()}`}
      </button>
    </form>
  );
}

export function NumberField({
  label,
  caption = label,
  className,
  value,
  onChange,
  disabled = false,
  min,
  step = 'any',
}: {
  label: string;
  caption?: string;
  className?: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  min?: number;
  step?: string;
}) {
  return (
    <label className={className}>
      {caption}
      <input
        aria-label={label}
        type="number"
        required
        disabled={disabled}
        value={value}
        min={min}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function Report({ value, label = 'Report' }: { value: unknown; label?: string }) {
  const { draft, text, chooseFormat, fallback } = useDataDraft(value);
  return (
    <section className="structured-data data-report" aria-label={`${label} report`}>
      <div className="data-heading">
        <strong>{label}</strong>
        <DataFormat label={label} raw={draft.raw} onChange={chooseFormat} />
      </div>
      {draft.error && <p role="alert">{draft.error}</p>}
      {draft.raw ? (
        <pre className="report" aria-label={`${label} raw`}>
          {text}
        </pre>
      ) : (
        draft.node && (
          <TableBoundary onError={fallback}>
            <DataTable node={draft.node} label={label} path={[]} depth={0} />
          </TableBoundary>
        )
      )}
    </section>
  );
}

export function CheckField({
  label,
  checked,
  disabled = false,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (value: boolean) => Promise<void>;
}) {
  const [pending, setPending] = useState<boolean | null>(null);
  return (
    <label className="check-label">
      <input
        aria-label={label}
        type="checkbox"
        checked={pending ?? checked}
        disabled={disabled || pending !== null}
        onChange={(event) => {
          const value = event.target.checked;
          setPending(value);
          void onChange(value).finally(() => setPending(null));
        }}
      />
      {label}
    </label>
  );
}
