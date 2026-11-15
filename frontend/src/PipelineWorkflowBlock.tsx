import { useEffect, useState } from 'react';
import { api, type Options, type Workspace } from './api';
import { JsonEditor } from './Controls';
import { useFileSystem } from './FileSystemDialog';
import { WorkbenchIcon } from './WorkbenchIcon';
import {
  type WorkflowDocument,
  type WorkflowEntry,
  type WorkflowTarget,
} from './reusableWorkflows';

export type BlockOptions = Options & {
  workflow: WorkflowDocument;
  bindings: Record<string, string>;
};
type Catalog = {
  defaults: Record<string, BlockOptions>;
  presets: { id: string; name: string; document: WorkflowDocument; compatible: boolean }[];
  resources: WorkflowTarget['experiments'];
};

export function PipelineWorkflowBlock({
  workspace: w,
  kind,
  value,
  onChange,
  disabled,
}: {
  workspace: Workspace;
  kind: 'reconstruct' | 'fit';
  value: Options | null;
  onChange: (options: BlockOptions) => void;
  disabled: boolean;
}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [library, setLibrary] = useState<WorkflowEntry[]>([]);
  const [error, setError] = useState('');
  const [fileBusy, setFileBusy] = useState(false);
  const choose = useFileSystem();
  const expected = kind === 'reconstruct' ? 'reconstruction' : 'fitting';
  const extension = kind === 'reconstruct' ? '.mflow' : '.mflow';
  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      api<Catalog>('/api/workspaces/' + w.id + '/pipeline-blocks'),
      api<WorkflowEntry[]>('/api/workflows'),
    ])
      .then(([catalog, library]) => {
        if (!cancelled) {
          setCatalog(catalog);
          setLibrary(library);
        }
      })
      .catch((failure) => {
        if (!cancelled) setError(String(failure));
      });
    return () => {
      cancelled = true;
    };
  }, [w.id, w.experiment, w.revision]);
  const fallback = catalog?.defaults[kind];
  useEffect(() => {
    if (!value?.workflow && fallback) {
      const next = structuredClone(fallback);
      if (kind === 'reconstruct' && value) {
        for (const key of ['pipeline', 'component'])
          if (value[key] !== undefined) next.workflow.steps[0].settings[key] = value[key];
      }
      onChange(next);
    }
  }, [value, fallback, kind, onChange]);
  const options = (value?.workflow ? value : fallback) as BlockOptions | undefined;
  const document = options?.workflow;
  const candidates = [
    ...(fallback
      ? [{ id: 'current', name: 'Current configuration', options: fallback, compatible: true }]
      : []),
    ...(kind === 'reconstruct'
      ? (catalog?.presets.map((preset) => ({
          id: 'preset:' + preset.id,
          name: preset.name,
          options: { workflow: preset.document, bindings: {} } as BlockOptions,
          compatible: preset.compatible,
        })) ?? [])
      : []),
    ...library
      .filter((item) => item.document.kind === expected)
      .map((item) => ({
        id: item.id,
        name: item.document.name,
        compatible: true,
        options: {
          workflow: item.document,
          bindings: Object.fromEntries(
            item.document.bindings.map((binding) => [
              binding.id,
              item.origin?.workspace_id === w.id && item.origin.experiment === w.experiment
                ? (item.suggested_bindings[binding.id] ?? '')
                : binding.default === 'target'
                  ? '$target'
                  : '',
            ]),
          ),
        } as BlockOptions,
      })),
  ];
  const selection =
    candidates.find((item) => JSON.stringify(item.options.workflow) === JSON.stringify(document))
      ?.id ?? '';
  async function fileOperation(save: boolean) {
    setFileBusy(true);
    setError('');
    try {
      const file = await choose({
        title: (save ? 'Save ' : 'Open ') + expected + ' workflow',
        mode: save ? 'save-file' : 'file',
        extensions: [extension],
      });
      if (!file) return;
      if (save && document) {
        const entry = await api<WorkflowEntry>('/api/workflows', 'POST', { document });
        await api('/api/workflows/' + entry.id + '/save-file', 'POST', {
          revision: entry.revision,
          root: file.root,
          path: file.path,
          overwrite: file.overwrite ?? false,
        });
        setLibrary((old) => [...old, entry]);
      } else {
        const entry = await api<WorkflowEntry>('/api/workflows/open-file', 'POST', {
          root: file.root,
          path: file.path,
        });
        if (entry.document.kind !== expected)
          throw Error('Choose a ' + extension + ' workflow for this block.');
        onChange({
          workflow: entry.document,
          bindings: Object.fromEntries(
            entry.document.bindings.map((binding) => [
              binding.id,
              binding.default === 'target' ? '$target' : '',
            ]),
          ),
        });
        setLibrary((old) => [...old, entry]);
      }
    } catch (failure) {
      setError(String(failure));
    } finally {
      setFileBusy(false);
    }
  }
  return (
    <fieldset className="pipeline-workflow-block" disabled={disabled || fileBusy || !catalog}>
      {error && <p role="alert">{error}</p>}
      <label>
        {kind === 'reconstruct' ? 'Reconstruction workflow' : 'Fitting workflow'}
        <select
          aria-label={expected + ' workflow'}
          value={selection}
          onChange={(event) => {
            const selected = candidates.find((item) => item.id === event.target.value);
            if (selected) onChange(structuredClone(selected.options));
          }}
        >
          <option value="" disabled>
            Custom workflow
          </option>
          {candidates.map((item) => (
            <option key={item.id} value={item.id} disabled={!item.compatible}>
              {item.name}
            </option>
          ))}
        </select>
      </label>
      {document && options && (
        <>
          <label>
            Block name
            <input
              aria-label="Workflow block name"
              value={document.name}
              required
              maxLength={160}
              onChange={(event) =>
                onChange({ ...options, workflow: { ...document, name: event.target.value } })
              }
            />
          </label>
          <div className="button-row">
            <button type="button" onClick={() => void fileOperation(false)}>
              <WorkbenchIcon name="open" />
              Open {extension}
            </button>
            <button type="button" onClick={() => void fileOperation(true)}>
              <WorkbenchIcon name="save" />
              Save {extension}
            </button>
          </div>
          {kind === 'fit' && (
            <p className="control-note">
              Model: {String((document.steps[0].settings.settings as Options)?.model_name ?? '')}.
              Maps are stored separately. Image data pass through this block.
            </p>
          )}
          {document.bindings.map((binding) => {
            const reference = binding.experiment
              ? options.bindings[binding.experiment]
              : w.experiment;
            const resources =
              catalog?.resources.find(
                (row) => row.experiment === (reference === '$target' ? w.experiment : reference),
              )?.resources[binding.kind] ?? [];
            return (
              <label key={binding.id}>
                {binding.label}
                <select
                  aria-label={binding.label + ' for workflow block'}
                  value={options.bindings[binding.id] ?? ''}
                  onChange={(event) =>
                    onChange({
                      ...options,
                      bindings: { ...options.bindings, [binding.id]: event.target.value },
                    })
                  }
                  required
                >
                  <option value="">Choose input</option>
                  {binding.kind === 'experiment' && (
                    <option value="$target">Current experiment</option>
                  )}
                  {resources.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
            );
          })}
          <details>
            <summary>Workflow definition</summary>
            <JsonEditor
              label="Embedded workflow definition"
              value={document}
              disabled={disabled || fileBusy}
              onApply={async (definition) => {
                const entry = await api<WorkflowEntry>('/api/workflows', 'POST', {
                  document: definition,
                });
                if (entry.document.kind !== expected)
                  throw Error('The workflow kind must match this block.');
                onChange({
                  ...options,
                  workflow: entry.document,
                  bindings: Object.fromEntries(
                    entry.document.bindings.map((binding) => [
                      binding.id,
                      options.bindings[binding.id] ??
                        (binding.default === 'target' ? '$target' : ''),
                    ]),
                  ),
                });
              }}
            />
          </details>
        </>
      )}
    </fieldset>
  );
}
