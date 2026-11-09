import { useEffect, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { api, type Job, type Options, type Workspace } from './api';
import { NumberField } from './Controls';
import { useFileSystem, type FileSelection } from './FileSystemDialog';
import { BrukerImportOptions, importReviewKey, type ImportReview } from './BrukerImportOptions';

export function FileDialog({
  open,
  onClose,
  onAfterClose,
  workspace,
  currentStudy,
  onDestinationChange,
  onJob,
  onError,
}: {
  open: boolean;
  onClose: () => void;
  onAfterClose: () => void;
  workspace: Workspace | null;
  currentStudy: Workspace | null;
  onDestinationChange: (intoCurrent: boolean) => void;
  onJob: (job: Job) => void;
  onError: (message: string) => void;
}) {
  const choose = useFileSystem();
  const [source, setSource] = useState<FileSelection | null>(null);
  const [busy, setBusy] = useState(false);
  const [kind, setKind] = useState('auto'),
    [name, setName] = useState(''),
    [shape, setShape] = useState('8,24,32,3'),
    [dims, setDims] = useState('t,x,y,z'),
    [dtype, setDtype] = useState('float32'),
    [offset, setOffset] = useState(0),
    [endian, setEndian] = useState('native'),
    [order, setOrder] = useState('C');
  const [options, setOptions] = useState('{}');
  const [preview, setPreview] = useState<unknown>(null);
  const [review, setReview] = useState<ImportReview | null>(null);
  const [importError, setImportError] = useState('');
  const needsReview = Boolean(source?.directory && ['auto', 'bruker'].includes(kind));
  const currentReview = source && review?.key === importReviewKey(source, kind) ? review : null;
  const directorySource = Boolean(source?.directory);
  const newStudyRequired =
    Boolean(currentStudy?.loaded) &&
    (['bruker', 'pulseq', 'dicom-study', 'study'].includes(kind) ||
      (['auto', 'dicom'].includes(kind) && directorySource) ||
      (kind === 'auto' &&
        /(?:\.mrax|(?:study_manifest|mrax_package)\.json)$/i.test(source?.path ?? '')));
  useEffect(() => {
    if (open && workspace && newStudyRequired) onDestinationChange(false);
  }, [open, workspace, newStudyRequired, onDestinationChange]);
  async function task(fn: () => Promise<void>) {
    setBusy(true);
    setImportError('');
    try {
      await fn();
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not open this file';
      setImportError(message);
      onError(message);
    } finally {
      setBusy(false);
    }
  }
  const selection = () => (source ? { root: source.root, path: source.path } : {});
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content
          className="file-dialog"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            onAfterClose();
          }}
        >
          <Dialog.Title>Import a study or experiment</Dialog.Title>
          <Dialog.Description>
            Open a data file or a study folder. Select the Bruker study folder directly.
          </Dialog.Description>
          <Dialog.Close className="drawer-close" aria-label="Close importer">
            ×
          </Dialog.Close>
          <label className="import-destination">
            Import destination
            <select
              aria-label="Import destination"
              value={workspace ? 'current' : 'new'}
              disabled={busy}
              onChange={(event) => onDestinationChange(event.target.value === 'current')}
            >
              <option value="new">New study</option>
              <option value="current" disabled={!currentStudy || newStudyRequired}>
                Add experiment to {currentStudy?.label ?? 'selected study'}
              </option>
            </select>
          </label>
          {newStudyRequired && <p className="control-note">This format opens as a new study.</p>}
          <div className="file-grid">
            <section>
              <h3>Source</h3>
              <button
                className="secondary full"
                disabled={busy}
                onClick={() => {
                  void choose({ title: 'Open data file or study folder', mode: 'open' }).then(
                    (value) => {
                      if (value) {
                        setSource(value);
                        setPreview(null);
                      }
                    },
                  );
                }}
              >
                Browse files and folders…
              </button>
              <p className="file-source" role="status">
                {source?.path ?? 'No source selected'}
              </p>
            </section>
            <section>
              <h3>Import settings</h3>
              <label>
                Import format
                <select
                  aria-label="Import format"
                  value={kind}
                  onChange={(e) => setKind(e.target.value)}
                >
                  {[
                    ['auto', 'Detect automatically'],
                    ['study', 'MRax study / state / package'],
                    ['bruker', 'Bruker study'],
                    ['pulseq', 'Pulseq study'],
                    ['dicom', 'DICOM study / image'],
                    ['nifti', 'NIfTI image'],
                    ['numpy', 'NumPy array'],
                    ['raw', 'Raw binary array'],
                    ['blank', 'Blank experiment'],
                  ].map(([v, l]) => (
                    <option key={v} value={v}>
                      {l}
                    </option>
                  ))}
                </select>
              </label>
              {kind !== 'bruker' && currentReview?.kind !== 'bruker' && (
                <label>
                  Experiment name
                  <input
                    aria-label="Experiment name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Automatic"
                  />
                </label>
              )}
              {['raw', 'blank', 'numpy', 'nifti'].includes(kind) && (
                <>
                  <label>
                    Dimension order
                    <input
                      aria-label="Dimension order"
                      value={dims}
                      onChange={(e) => setDims(e.target.value)}
                      placeholder="t,x,y,z"
                    />
                  </label>
                  {['raw', 'blank'].includes(kind) && (
                    <>
                      <label>
                        Array shape
                        <input
                          aria-label="Array shape"
                          value={shape}
                          onChange={(e) => setShape(e.target.value)}
                        />
                      </label>
                      <label>
                        Data type
                        <select
                          aria-label="Data type"
                          value={dtype}
                          onChange={(e) => setDtype(e.target.value)}
                        >
                          {[
                            'uint8',
                            'uint16',
                            'int16',
                            'int32',
                            'float32',
                            'float64',
                            'complex64',
                            'complex128',
                          ].map((v) => (
                            <option key={v}>{v}</option>
                          ))}
                        </select>
                      </label>
                      <NumberField
                        label="Byte offset"
                        value={offset}
                        min={0}
                        step="1"
                        onChange={setOffset}
                      />
                      <label>
                        Byte order
                        <select
                          aria-label="Byte order"
                          value={endian}
                          onChange={(e) => setEndian(e.target.value)}
                        >
                          <option>native</option>
                          <option>little</option>
                          <option>big</option>
                        </select>
                      </label>
                      <label>
                        Memory order
                        <select
                          aria-label="Memory order"
                          value={order}
                          onChange={(e) => setOrder(e.target.value)}
                        >
                          <option>C</option>
                          <option>F</option>
                        </select>
                      </label>
                    </>
                  )}
                </>
              )}
              <details>
                <summary>Import options and observables</summary>
                <p className="muted">
                  Set model_name, exp_type, observables with values and unit, field_strength_mhz,
                  pdata_index, load_raw_kspace, npz_key, type_overrides, or vendor_options.
                </p>
                <label>
                  Import options
                  <textarea
                    aria-label="Import options"
                    rows={8}
                    value={options}
                    onChange={(e) => setOptions(e.target.value)}
                    spellCheck={false}
                  />
                </label>
              </details>
            </section>
          </div>
          {needsReview && source && (
            <BrukerImportOptions
              source={source}
              kind={kind}
              settings={options}
              onReview={setReview}
            />
          )}
          {importError && <p role="alert">{importError}</p>}
          <div className="button-row">
            <button
              className="secondary"
              disabled={busy || (!source && kind !== 'blank')}
              onClick={() =>
                void task(async () =>
                  setPreview(
                    await api('/api/files/preview', 'POST', {
                      ...selection(),
                      options: { ...JSON.parse(options), kind },
                    }),
                  ),
                )
              }
            >
              Inspect source
            </button>
            <button
              className="primary"
              disabled={
                busy || (!source && kind !== 'blank') || (needsReview && !currentReview?.ready)
              }
              onClick={() =>
                void task(async () => {
                  const extra: Options = JSON.parse(options);
                  const job = await api<Job>('/api/import', 'POST', {
                    ...selection(),
                    workspace_id: workspace?.id ?? null,
                    revision: workspace?.revision ?? null,
                    options: {
                      ...extra,
                      kind,
                      ...(currentReview?.kind === 'bruker'
                        ? {
                            ...currentReview.options,
                            type_overrides: {
                              ...((extra.type_overrides as Options) ?? {}),
                              ...((currentReview.options.type_overrides as Options) ?? {}),
                            },
                            model_overrides_by_name: {
                              ...((extra.model_overrides_by_name as Options) ?? {}),
                              ...((currentReview.options.model_overrides_by_name as Options) ?? {}),
                            },
                          }
                        : {}),
                      ...(name ? { name } : {}),
                      ...(['raw', 'blank', 'numpy', 'nifti'].includes(kind)
                        ? {
                            dim_order: dims
                              .split(',')
                              .map((v) => v.trim())
                              .filter(Boolean),
                          }
                        : {}),
                      ...(['raw', 'blank'].includes(kind)
                        ? {
                            shape: shape.split(',').map(Number),
                            dtype,
                            byte_offset: offset,
                            endian,
                            memory_order: order,
                          }
                        : {}),
                    },
                  });
                  onJob(job);
                  onClose();
                })
              }
            >
              Import {workspace ? 'into study' : 'study'}
            </button>
          </div>

          {preview !== null && (
            <pre className="report" aria-label="Import discovery">
              {JSON.stringify(preview, null, 2)}
            </pre>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
