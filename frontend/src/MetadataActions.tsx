import { useMemo, useRef, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { WorkbenchIcon } from './WorkbenchIcon';
import { api } from './api';
import { useFileSystem } from './FileSystemDialog';
import { Report } from './Controls';
import { importMetadata, metadataPointer, parseMetadataImport, type ImportMode } from './metadata';
import { compileTable, parseData, printData, type DataNode, type DataPath } from './structuredData';

export function MetadataActions({
  node,
  path,
  label,
  disabled,
  quantityDefault,
  onChange,
}: {
  node: DataNode | undefined;
  path: DataPath;
  label: string;
  disabled: boolean;
  quantityDefault: boolean;
  onChange: (value: DataNode, selected?: DataPath) => void;
}) {
  const choose = useFileSystem();
  const [reading, setReading] = useState(false);
  const [filename, setFilename] = useState('');
  const [dialog, setDialog] = useState<'add' | 'import' | null>(null);
  const [name, setName] = useState('');
  const [kind, setKind] = useState('string');
  const [value, setValue] = useState('');
  const [unit, setUnit] = useState('');
  const [error, setError] = useState('');
  const [source, setSource] = useState('');
  const [format, setFormat] = useState<'json' | 'csv' | 'tsv'>('json');
  const [mode, setMode] = useState<ImportMode>('merge');
  const origin = useRef<HTMLButtonElement | null>(null);
  const container = node?.kind === 'array' || node?.kind === 'object';
  const preview = useMemo(() => {
    if (dialog !== 'import' || !source || !node) return null;
    try {
      const updates = parseMetadataImport(source, format);
      return { value: importMetadata(node, updates, mode), entries: updates.length, error: '' };
    } catch (failure) {
      return {
        value: null,
        entries: 0,
        error: failure instanceof Error ? failure.message : 'Cannot read metadata.',
      };
    }
  }, [dialog, source, format, mode, node]);
  const open = (action: 'add' | 'import') => {
    setError('');
    setName('');
    setValue('');
    setUnit('');
    setKind(quantityDefault ? 'quantity' : 'string');
    setSource('');
    setFilename('');
    setMode('merge');
    setDialog(action);
  };
  return (
    <>
      <button
        className="secondary"
        disabled={disabled || !container}
        onClick={(event) => {
          origin.current = event.currentTarget;
          open('add');
        }}
      >
        {node?.kind === 'array' ? 'Add item' : 'Add field'}
      </button>
      <button
        className="secondary"
        disabled={disabled || !container}
        onClick={(event) => {
          origin.current = event.currentTarget;
          open('import');
        }}
      >
        Import from file
      </button>
      <Dialog.Root
        open={dialog !== null}
        onOpenChange={(open) => {
          if (!open) setDialog(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="drawer-overlay" />
          <Dialog.Content
            className="metadata-action-dialog"
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              if (origin.current?.isConnected && !origin.current.disabled) origin.current.focus();
              else {
                const field = document.querySelector<HTMLElement>(
                  '.metadata-browser:not([hidden]) .metadata-content .json-editor input:not(:disabled), .metadata-browser:not([hidden]) .metadata-content .json-editor textarea:not(:disabled)',
                );
                (
                  field ??
                  document.querySelector<HTMLElement>(
                    '.metadata-breadcrumbs [aria-current="location"]',
                  )
                )?.focus();
              }
            }}
          >
            <Dialog.Title>
              {dialog === 'add'
                ? node?.kind === 'array'
                  ? 'Add metadata item'
                  : 'Add metadata field'
                : 'Import metadata from file'}
            </Dialog.Title>
            <Dialog.Description>
              Destination: {label} {metadataPointer(path) || '/'}. Changes stay in the draft until
              Apply.
            </Dialog.Description>
            <Dialog.Close className="drawer-close" aria-label="Close metadata form">
              <WorkbenchIcon name="close" />
            </Dialog.Close>
            {dialog === 'add' ? (
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  if (!node || disabled) return;
                  try {
                    let parsed: unknown;
                    if (kind === 'string') parsed = value;
                    else if (kind === 'null') parsed = null;
                    else if (kind === 'boolean') parsed = value === 'true';
                    else {
                      parsed = parseData(value);
                      if (
                        (kind === 'number' && typeof parsed !== 'number') ||
                        (kind === 'array' && !Array.isArray(parsed)) ||
                        (kind === 'object' &&
                          (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)))
                      )
                        throw Error('The value does not match its selected type.');
                      if (kind === 'quantity') {
                        if (!unit.trim()) throw Error('Enter a unit for this quantity.');
                        const numeric = (v: unknown): boolean =>
                          typeof v === 'number' || (Array.isArray(v) && v.every(numeric));
                        if (!numeric(parsed))
                          throw Error('A quantity needs a number or a numeric array.');
                        parsed = { values: parsed, unit: unit.trim() };
                      }
                    }
                    const added = compileTable(JSON.stringify(parsed));
                    if (node.kind === 'array')
                      onChange({ kind: 'array', items: [...node.items, added] }, [
                        ...path,
                        node.items.length,
                      ]);
                    else if (node.kind === 'object') {
                      if (!name.trim()) throw Error('Enter a field name.');
                      if (node.entries.some(([key]) => key === name.trim()))
                        throw Error('A field with that name already exists.');
                      onChange(
                        { kind: 'object', entries: [...node.entries, [name.trim(), added]] },
                        [...path, name.trim()],
                      );
                    }
                    setDialog(null);
                  } catch (failure) {
                    setError(failure instanceof Error ? failure.message : 'Cannot add this field.');
                  }
                }}
              >
                {node?.kind !== 'array' && (
                  <label>
                    Field name
                    <input
                      aria-label="Metadata field name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                    />
                  </label>
                )}
                <label>
                  Type
                  <select
                    aria-label="Metadata field type"
                    value={kind}
                    onChange={(e) => {
                      setKind(e.target.value);
                      setValue(
                        e.target.value === 'boolean'
                          ? 'false'
                          : e.target.value === 'array'
                            ? '[]'
                            : e.target.value === 'object'
                              ? '{}'
                              : '',
                      );
                    }}
                  >
                    <option value="string">Text</option>
                    <option value="number">Number</option>
                    <option value="boolean">Boolean</option>
                    <option value="null">Null</option>
                    <option value="object">Object</option>
                    <option value="array">Array</option>
                    <option value="quantity">Quantity with unit</option>
                  </select>
                </label>
                {kind !== 'null' &&
                  (kind === 'boolean' ? (
                    <label>
                      Value
                      <select
                        aria-label="New metadata value"
                        value={value || 'false'}
                        onChange={(e) => setValue(e.target.value)}
                      >
                        <option value="false">False</option>
                        <option value="true">True</option>
                      </select>
                    </label>
                  ) : (
                    <label>
                      {kind === 'string' ? 'Value' : 'Value (JSON)'}
                      <textarea
                        aria-label="New metadata value"
                        rows={3}
                        value={value}
                        onChange={(e) => setValue(e.target.value)}
                        required={kind !== 'string'}
                        spellCheck={false}
                      />
                    </label>
                  ))}
                {kind === 'quantity' && (
                  <label>
                    Unit
                    <input
                      aria-label="New metadata unit"
                      value={unit}
                      onChange={(e) => setUnit(e.target.value)}
                      required
                    />
                  </label>
                )}
                {error && <p role="alert">{error}</p>}
                <button className="secondary full" disabled={disabled}>
                  Add to draft
                </button>
              </form>
            ) : (
              <div>
                <button
                  className="secondary full"
                  disabled={disabled || reading}
                  onClick={() => {
                    setError('');
                    setReading(true);
                    void (async () => {
                      try {
                        const file = await choose({
                          title: 'Open metadata file',
                          mode: 'file',
                          extensions: ['.json', '.csv', '.tsv'],
                        });
                        if (!file) return;
                        const content = await api<{ content: string }>(
                          '/api/files/text?' +
                            new URLSearchParams({ root: file.root, path: file.path }),
                        );
                        setFormat(file.name.split('.').at(-1)!.toLowerCase() as typeof format);
                        setFilename(file.name);
                        setSource(content.content);
                      } catch (failure) {
                        setError(
                          failure instanceof Error ? failure.message : 'The file could not be read',
                        );
                      } finally {
                        setReading(false);
                      }
                    })();
                  }}
                >
                  Open metadata file…
                </button>
                {filename && <p className="file-source">{filename}</p>}
                <label>
                  Import mode
                  <select
                    aria-label="Metadata import mode"
                    value={mode}
                    onChange={(e) => setMode(e.target.value as ImportMode)}
                  >
                    <option value="merge">Merge and update matching fields</option>
                    <option value="add">Add new fields only</option>
                  </select>
                </label>
                <p className="control-note">
                  JSON applies to the selected group. CSV and TSV use name (or path), value, and
                  optional type and unit columns.
                </p>
                {source && (
                  <details className="metadata-import-source" open={Boolean(preview?.error)}>
                    <summary>Raw import source</summary>
                    <textarea
                      aria-label="Metadata import source"
                      rows={8}
                      spellCheck={false}
                      value={source}
                      onChange={(e) => setSource(e.target.value)}
                    />
                  </details>
                )}
                {(error || preview?.error) && <p role="alert">{error || preview?.error}</p>}
                {preview?.value && (
                  <>
                    <p role="status">Preview of the selected group after import</p>
                    <Report
                      label="Metadata import preview"
                      value={parseData(printData(preview.value))}
                    />
                  </>
                )}
                <button
                  className="secondary full"
                  disabled={disabled || !preview?.value || Boolean(error)}
                  onClick={() => {
                    if (preview?.value) {
                      onChange(preview.value);
                      setDialog(null);
                    }
                  }}
                >
                  Import to draft
                </button>
              </div>
            )}
            <Dialog.Close asChild>
              <button className="secondary full">Cancel</button>
            </Dialog.Close>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
}
