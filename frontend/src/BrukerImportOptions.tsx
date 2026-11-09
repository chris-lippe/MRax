import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type Options } from './api';
import type { FileSelection } from './FileSystemDialog';

type Experiment = {
  scan: string;
  method: unknown;
  protocol: unknown;
  frame_count: number | null;
  detected_type: string;
  detected_model: string;
  field_strength_mhz: number | null;
  pdata_index: string;
  reconstructions: string[];
  importable: boolean;
  has_raw: boolean;
};
type Choice = { selected: boolean; pdata: string; type: string; model: string };
export type ImportReview = { key: string; kind: string; ready: boolean; options: Options };
export const importReviewKey = (source: FileSelection, kind: string) =>
  JSON.stringify([source.root, source.path, kind]);

export function BrukerImportOptions({
  source,
  kind,
  settings,
  onReview,
}: {
  source: FileSelection;
  kind: string;
  settings: string;
  onReview: (review: ImportReview) => void;
}) {
  const [rows, setRows] = useState<Experiment[] | null>(null);
  const [detected, setDetected] = useState('');
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const [page, setPage] = useState(0);
  const [frequency, setFrequency] = useState('');
  const [raw, setRaw] = useState(false);
  const [error, setError] = useState('');
  const [models, setModels] = useState<{ key: string; label: string }[]>([]);
  const supplied = useRef(settings);
  supplied.current = settings;
  const key = importReviewKey(source, kind);
  useEffect(() => {
    const controller = new AbortController();
    setRows(null);
    setError('');
    setDetected('');
    setPage(0);
    onReview({ key, kind: '', ready: false, options: {} });
    void (async () => {
      try {
        const extra = JSON.parse(supplied.current) as Options;
        const preview = await api<{ kind: string; experiments?: Experiment[] }>(
          '/api/files/preview',
          'POST',
          {
            root: source.root,
            path: source.path,
            options: { ...extra, kind },
          },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setDetected(preview.kind);
        const experiments = preview.experiments ?? [];
        setRows(experiments);
        setChoices(
          Object.fromEntries(
            experiments.map((row) => [
              row.scan,
              {
                selected: row.importable,
                pdata: row.pdata_index,
                type: String(
                  (extra.type_overrides as Record<string, string> | undefined)?.[row.scan] ??
                    row.detected_type,
                ),
                model: String(
                  (extra.model_overrides_by_name as Record<string, string> | undefined)?.[
                    row.scan
                  ] ?? '',
                ),
              },
            ]),
          ),
        );
        const frequencies = [
          ...new Set(
            experiments.map((row) => row.field_strength_mhz).filter((value) => value !== null),
          ),
        ];
        setFrequency(
          extra.field_strength_mhz !== undefined
            ? String(extra.field_strength_mhz)
            : frequencies.length === 1
              ? String(frequencies[0])
              : '',
        );
        setRaw(Boolean(extra.load_raw_kspace));
        if (preview.kind === 'bruker')
          void api<{ models: typeof models }>('/api/catalog')
            .then((catalog) => {
              if (!controller.signal.aborted) setModels(catalog.models);
            })
            .catch(() => {});
      } catch (failure) {
        if (!controller.signal.aborted)
          setError(
            failure instanceof Error ? failure.message : 'Could not read acquisition headers',
          );
      }
    })();
    return () => controller.abort();
  }, [source.root, source.path, kind, key, onReview]);
  useEffect(() => {
    if (!rows) return;
    try {
      const extra = JSON.parse(settings) as Options;
      if (extra.field_strength_mhz !== undefined) setFrequency(String(extra.field_strength_mhz));
      if (extra.load_raw_kspace !== undefined) setRaw(Boolean(extra.load_raw_kspace));
      setChoices((previous) =>
        Object.fromEntries(
          rows.map((row) => {
            const old = previous[row.scan];
            if (!old)
              return [
                row.scan,
                {
                  selected: row.importable,
                  pdata: row.pdata_index,
                  type: row.detected_type,
                  model: '',
                },
              ];
            const indices = extra.pdata_index;
            return [
              row.scan,
              {
                ...old,
                ...(Array.isArray(extra.selected_experiments)
                  ? { selected: extra.selected_experiments.includes(row.scan) }
                  : {}),
                pdata:
                  typeof indices === 'string' || typeof indices === 'number'
                    ? String(indices)
                    : String(
                        (indices as Record<string, string> | undefined)?.[row.scan] ?? old.pdata,
                      ),
                type: String(
                  (extra.type_overrides as Record<string, string> | undefined)?.[row.scan] ??
                    old.type,
                ),
                model: String(
                  (extra.model_overrides_by_name as Record<string, string> | undefined)?.[
                    row.scan
                  ] ?? old.model,
                ),
              },
            ];
          }),
        ),
      );
    } catch {
      /* Keep the last valid selection while the raw JSON is being edited. */
    }
  }, [settings, rows]);
  const review = useMemo<ImportReview>(() => {
    const selected = rows?.filter((row) => choices[row.scan]?.selected) ?? [];
    return {
      key,
      kind: detected,
      ready:
        rows !== null &&
        !error &&
        (detected !== 'bruker' || selected.length > 0) &&
        (!frequency || (Number.isFinite(Number(frequency)) && Number(frequency) > 0)),
      options:
        detected === 'bruker'
          ? {
              selected_experiments: selected.map((row) => row.scan),
              pdata_index: Object.fromEntries(
                selected.map((row) => [row.scan, choices[row.scan].pdata]),
              ),
              type_overrides: Object.fromEntries(
                selected
                  .filter((row) => choices[row.scan].type !== row.detected_type)
                  .map((row) => [row.scan, choices[row.scan].type]),
              ),
              model_overrides_by_name: Object.fromEntries(
                selected
                  .filter((row) => choices[row.scan].model)
                  .map((row) => [row.scan, choices[row.scan].model]),
              ),
              load_raw_kspace: raw,
              ...(frequency ? { field_strength_mhz: Number(frequency) } : {}),
            }
          : {},
    };
  }, [key, detected, rows, choices, raw, frequency, error]);
  useEffect(() => {
    onReview(review);
  }, [review, onReview]);
  if (error) return <p role="alert">{error}</p>;
  if (rows === null) return <p role="status">Reading acquisition headers…</p>;
  if (detected !== 'bruker') return null;
  const selected = rows.filter((row) => choices[row.scan]?.selected).length;
  const first = Math.min(page * 50, Math.max(0, Math.floor((rows.length - 1) / 50) * 50));
  const update = (scan: string, patch: Partial<Choice>) =>
    setChoices((previous) => ({
      ...previous,
      [scan]: { ...previous[scan], ...patch },
    }));
  return (
    <section className="import-selection" aria-label="Bruker import selection">
      <h3>Select experiments</h3>
      <div className="import-selection-toolbar">
        <p role="status">
          {selected} of {rows.length} experiments selected
        </p>
        <button
          className="secondary"
          onClick={() =>
            setChoices((previous) =>
              Object.fromEntries(
                Object.entries(previous).map(([scan, choice]) => [
                  scan,
                  {
                    ...choice,
                    selected: rows
                      .find((row) => row.scan === scan)!
                      .reconstructions.includes(choice.pdata),
                  },
                ]),
              ),
            )
          }
        >
          Select all
        </button>
        <button
          className="secondary"
          onClick={() =>
            setChoices((previous) =>
              Object.fromEntries(
                Object.entries(previous).map(([scan, choice]) => [
                  scan,
                  { ...choice, selected: false },
                ]),
              ),
            )
          }
        >
          Select none
        </button>
      </div>
      <div className="import-experiment-table">
        <table>
          <thead>
            <tr>
              <th scope="col">Import</th>
              <th scope="col">Experiment / protocol</th>
              <th scope="col">Frames</th>
              <th scope="col">Reconstruction</th>
              <th scope="col">Type</th>
              <th scope="col">Model</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(first, first + 50).map((row) => {
              const choice = choices[row.scan];
              if (!choice) return null;
              return (
                <tr key={row.scan}>
                  <td>
                    <input
                      type="checkbox"
                      aria-label={'Import experiment ' + row.scan}
                      checked={choice.selected}
                      disabled={!row.reconstructions.includes(choice.pdata)}
                      onChange={(event) => update(row.scan, { selected: event.target.checked })}
                    />
                  </td>
                  <th scope="row">
                    <span>{row.scan}</span>
                    <small>{String(row.protocol || row.method || '')}</small>
                  </th>
                  <td>{choice.pdata === row.pdata_index ? (row.frame_count ?? '—') : '—'}</td>
                  <td>
                    <select
                      aria-label={'Reconstruction for ' + row.scan}
                      value={choice.pdata}
                      onChange={(event) => update(row.scan, { pdata: event.target.value })}
                    >
                      {!row.reconstructions.includes(choice.pdata) && (
                        <option value={choice.pdata}>{choice.pdata} (unavailable)</option>
                      )}
                      {row.reconstructions.map((index) => (
                        <option key={index}>{index}</option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <input
                      aria-label={'Type for ' + row.scan}
                      value={choice.type}
                      onChange={(event) => update(row.scan, { type: event.target.value })}
                    />
                  </td>
                  <td>
                    <select
                      aria-label={'Model for ' + row.scan}
                      value={choice.model}
                      onChange={(event) => update(row.scan, { model: event.target.value })}
                    >
                      <option value="">
                        Automatic
                        {choice.type === row.detected_type ? ` (${row.detected_model})` : ''}
                      </option>
                      {models.map((model) => (
                        <option key={model.key} value={model.key}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length > 50 && (
        <div className="import-selection-toolbar">
          <p>
            Experiments {first + 1}–{Math.min(first + 50, rows.length)} of {rows.length}
          </p>
          <button
            className="secondary"
            disabled={first === 0}
            onClick={() => setPage(Math.max(0, page - 1))}
          >
            Previous experiments
          </button>
          <button
            className="secondary"
            disabled={first + 50 >= rows.length}
            onClick={() => setPage(page + 1)}
          >
            Next experiments
          </button>
        </div>
      )}
      {!rows.length && <p role="alert">No Bruker experiments were found in this folder.</p>}
      <div className="import-selection-settings">
        <label>
          Frequency (MHz)
          <input
            type="number"
            aria-label="Import frequency (MHz)"
            value={frequency}
            min="0"
            step="any"
            placeholder="From study constants"
            onChange={(event) => setFrequency(event.target.value)}
          />
        </label>
        <label className="check-label">
          <input type="checkbox" checked={raw} onChange={(event) => setRaw(event.target.checked)} />
          Load raw k-space
        </label>
      </div>
      <p className="control-note">
        Only selected experiments and reconstructions are loaded. Raw k-space is optional.
      </p>
    </section>
  );
}
