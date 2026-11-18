import { useEffect, useRef, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { api, type Job, type Options, type Workspace } from './api';
import { useFileSystem } from './FileSystemDialog';
import { WorkbenchIcon } from './WorkbenchIcon';
import {
  printWorkflow,
  targetKey,
  waitWorkflow,
  workflowExtension,
  type WorkflowDocument,
  type WorkflowEntry,
  type WorkflowKind,
  type WorkflowPreset,
  type WorkflowTarget,
} from './reusableWorkflows';

export function WorkflowDialog({
  request,
  workspace,
  onClose,
  onUpdated,
}: {
  request: { kind: WorkflowKind; draft?: Options };
  workspace: Workspace | null;
  onClose: () => void;
  onUpdated: (operation?: string) => Promise<void>;
}) {
  const choose = useFileSystem();
  const [kind, setKind] = useState<WorkflowKind | 'all'>(request.kind);
  const [source, setSource] = useState<'presets' | 'session'>('presets');
  const [library, setLibrary] = useState<WorkflowEntry[]>([]);
  const [presets, setPresets] = useState<WorkflowPreset[]>([]);
  const [preset, setPreset] = useState<WorkflowPreset | null>(null);
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<WorkflowEntry | null>(null);
  const [text, setText] = useState('');
  const [definition, setDefinition] = useState(false);
  const [targets, setTargets] = useState<WorkflowTarget[]>([]);
  const [selection, setSelection] = useState<string[]>(
    workspace?.loaded ? [targetKey(workspace.id, workspace.experiment)] : [],
  );
  const [bindings, setBindings] = useState<Record<string, Record<string, string>>>({});
  const [mode, setMode] = useState('replace');
  const [execute, setExecute] = useState(false);
  const [device, setDevice] = useState('cpu');
  const [devices, setDevices] = useState<{ id: string; label: string }[]>([]);
  const [busy, setBusy] = useState(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [report, setReport] = useState<string[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const cancel = useRef(false);
  const dirty = Boolean(selected && text !== printWorkflow(selected.document));
  const entry = selected?.document;
  const matches = (item: { document: WorkflowDocument }) =>
    (kind === 'all' || item.document.kind === kind) &&
    (item.document.name + ' ' + item.document.description)
      .toLocaleLowerCase()
      .includes(search.trim().toLocaleLowerCase());
  const visible = library.filter(matches);
  const visiblePresets = presets.filter(matches);
  let fields: Partial<WorkflowDocument> = {};
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) fields = parsed;
  } catch {
    /* Invalid definitions remain editable until the user validates them. */
  }

  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    void Promise.all([
      api<WorkflowEntry[]>('/api/workflows', 'GET', undefined, controller.signal),
      api<WorkflowPreset[]>('/api/workflow-presets', 'GET', undefined, controller.signal),
      api<WorkflowTarget[]>('/api/workflow-targets', 'GET', undefined, controller.signal),
      api<{ devices: { id: string; label: string }[] }>(
        '/api/catalog',
        'GET',
        undefined,
        controller.signal,
      ),
    ])
      .then(([library, presets, targets, catalog]) => {
        if (!stopped) {
          setLibrary(library);
          setPresets(presets);
          const saved = library.find((item) => item.document.kind === request.kind);
          if (saved) {
            setSource('session');
            select(saved);
          } else {
            setPreset(
              presets.find((item) => item.document.kind === request.kind) ?? presets[0] ?? null,
            );
          }
          setTargets(targets);
          setDevices(catalog.devices);
        }
      })
      .catch((failure) => {
        if (!stopped) setError(String(failure));
      })
      .finally(() => {
        if (!stopped) setBusy(false);
      });
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [request.kind]);

  function select(item: WorkflowEntry) {
    setPreset(null);
    setSelected(item);
    setText(printWorkflow(item.document));
    setExecute(false);
    setMode(
      ['denoising', 'registration', 'fitting'].includes(item.document.kind) ? 'append' : 'replace',
    );
    setBindings({});
    setError('');
    setMessage('');
    setReport([]);
    setDefinition(false);
  }
  async function operation(action: () => Promise<void>) {
    setBusy(true);
    setWorking(true);
    setError('');
    setMessage('');
    try {
      await action();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
      setWorking(false);
      setJob(null);
    }
  }
  function remember(item: WorkflowEntry) {
    setSource('session');
    setKind('all');
    setSearch('');
    setLibrary((old) =>
      old.some((row) => row.id === item.id)
        ? old.map((row) => (row.id === item.id ? item : row))
        : [...old, item],
    );
    select(item);
  }
  async function capture(captureKind: WorkflowKind = request.kind) {
    if (!workspace) return;
    await operation(async () =>
      remember(
        await api<WorkflowEntry>('/api/workspaces/' + workspace.id + '/workflows', 'POST', {
          revision: workspace.revision,
          kind: captureKind,
          draft: captureKind === request.kind ? (request.draft ?? {}) : {},
        }),
      ),
    );
  }
  async function openFile() {
    const file = await choose({
      title: 'Open workflow',
      mode: 'file',
      extensions: [workflowExtension],
    });
    if (file)
      await operation(async () =>
        remember(
          await api<WorkflowEntry>('/api/workflows/open-file', 'POST', {
            root: file.root,
            path: file.path,
          }),
        ),
      );
  }
  async function saveChanges() {
    if (!selected) return null;
    const updated = await api<WorkflowEntry>('/api/workflows/' + selected.id, 'PUT', {
      revision: selected.revision,
      content: text,
    });
    setLibrary((old) => old.map((item) => (item.id === updated.id ? updated : item)));
    setSelected(updated);
    setText(printWorkflow(updated.document));
    setDefinition(false);
    setReport([]);
    return updated;
  }
  async function saveFile() {
    if (!selected) return;
    await operation(async () => {
      const item = dirty ? await saveChanges() : selected;
      if (!item) return;
      const file = await choose({
        title: 'Save workflow',
        mode: 'save-file',
        suggestedName:
          item.document.name
            .split('')
            .map((character) => (character.charCodeAt(0) < 32 ? '-' : character))
            .join('')
            .replace(/[<>:"/\\|?*]/g, '-')
            .slice(0, 100) + workflowExtension,
        extensions: [workflowExtension],
      });
      if (!file) return;
      await api('/api/workflows/' + item.id + '/save-file', 'POST', {
        revision: item.revision,
        root: file.root,
        path: file.path,
        overwrite: file.overwrite ?? false,
      });
      setMessage('Workflow saved.');
    });
  }
  function mapping(study: WorkflowTarget, experiment: string) {
    const key = targetKey(study.workspace_id, experiment);
    return Object.fromEntries(
      (entry?.bindings ?? []).map((binding) => [
        binding.id,
        bindings[key]?.[binding.id] ??
          (selected?.origin?.workspace_id === study.workspace_id &&
          selected.origin.experiment === experiment
            ? (selected.suggested_bindings[binding.id] ?? '')
            : binding.default === 'target'
              ? '$target'
              : ''),
      ]),
    );
  }
  function choices(
    study: WorkflowTarget,
    experiment: string,
    binding: WorkflowDocument['bindings'][number],
  ) {
    const values = mapping(study, experiment);
    const reference = binding.experiment ? values[binding.experiment] : experiment;
    return (
      study.experiments.find(
        (row) => row.experiment === (reference === '$target' ? experiment : reference),
      )?.resources[binding.kind] ?? []
    );
  }
  function updateDocument(update: (document: WorkflowDocument) => WorkflowDocument) {
    try {
      setText(printWorkflow(update(JSON.parse(text) as WorkflowDocument)));
      setError('');
    } catch {
      setError('Fix the JSON definition before editing these fields.');
    }
  }
  async function checkedApplications() {
    if (!selected || dirty) throw new Error('Keep workflow changes before checking inputs.');
    const groups = targets
      .map((study) => ({
        study,
        experiments: study.experiments
          .filter((row) => selection.includes(targetKey(study.workspace_id, row.experiment)))
          .map((row) => row.experiment),
      }))
      .filter((row) => row.experiments.length);
    if (!groups.length) throw new Error('Select at least one target experiment.');
    const commands = groups.map(({ study, experiments }) => ({
      revision: study.revision,
      workflow_revision: selected.revision,
      experiments,
      bindings: Object.fromEntries(
        experiments.map((experiment) => [experiment, mapping(study, experiment)]),
      ),
      mode,
      run: ['fitting', 'reconstruction'].includes(entry?.kind ?? '') ? execute : true,
      device,
    }));
    const problems: string[] = [];
    for (const [index, { study }] of groups.entries()) {
      const checks = await api<{
        targets: { experiment: string; ready: boolean; errors: string[] }[];
      }>(
        '/api/workspaces/' + study.workspace_id + '/workflows/' + selected.id + '/validate',
        'POST',
        commands[index],
      );
      for (const row of checks.targets)
        if (!row.ready)
          problems.push(study.label + ' / ' + row.experiment + ': ' + row.errors.join('. '));
    }
    if (problems.length) {
      setReport(problems);
      throw new Error('Resolve target compatibility and input assignments before applying.');
    }
    return { groups, commands };
  }
  async function checkInputs() {
    setReport([]);
    await operation(async () => {
      await checkedApplications();
      setMessage('Inputs are ready for the selected experiments.');
    });
  }
  async function apply() {
    if (!selected || dirty) return;
    cancel.current = false;
    setReport([]);
    await operation(async () => {
      const { groups, commands } = await checkedApplications();
      let changed = false;
      try {
        for (const [index, { study, experiments }] of groups.entries()) {
          if (cancel.current) break;
          const running = await api<Job>(
            '/api/workspaces/' + study.workspace_id + '/workflows/' + selected.id + '/apply',
            'POST',
            commands[index],
          );
          await waitWorkflow(running, setJob);
          changed = true;
          setReport((old) => [
            ...old,
            study.label + ': applied to ' + experiments.length + ' experiment(s).',
          ]);
        }
        setMessage(
          cancel.current
            ? 'Remaining studies were skipped.'
            : 'Workflow applied to all selected targets.',
        );
      } finally {
        if (changed)
          await onUpdated(
            entry?.kind === 'fitting' && execute
              ? 'full_fit'
              : entry?.kind === 'reconstruction' && execute
                ? 'reconstruct'
                : undefined,
          );
        setTargets(await api<WorkflowTarget[]>('/api/workflow-targets'));
      }
    });
  }
  return (
    <Dialog.Root
      open
      onOpenChange={(open) => {
        if (!open && !working) {
          if (dirty) setError('Keep or discard your workflow edits before closing.');
          else onClose();
        }
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content
          className="workflow-dialog"
          onChangeCapture={() => {
            setError('');
            setMessage('');
            setReport([]);
          }}
          onEscapeKeyDown={(event) => {
            if (working) event.preventDefault();
          }}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            document
              .querySelector<HTMLButtonElement>('.workflow-nav [role="tab"][aria-selected="true"]')
              ?.focus();
          }}
        >
          <div className="settings-heading">
            <Dialog.Title>
              <WorkbenchIcon name="pipeline" />
              Reusable workflows
            </Dialog.Title>
            <Dialog.Close className="icon-button" aria-label="Close workflows" disabled={working}>
              <WorkbenchIcon name="close" />
            </Dialog.Close>
          </div>
          <Dialog.Description>
            Choose a preset, capture settings or open a workflow file.
          </Dialog.Description>
          <div className="workflow-library-layout">
            <nav className="workflow-library" aria-label="Workflow library">
              <div
                className="button-row workflow-library-source"
                role="group"
                aria-label="Library source"
              >
                <button
                  className="secondary"
                  aria-pressed={source === 'presets'}
                  disabled={busy || dirty}
                  onClick={() => setSource('presets')}
                >
                  Presets
                </button>
                <button
                  className="secondary"
                  aria-pressed={source === 'session'}
                  disabled={busy || dirty}
                  onClick={() => setSource('session')}
                >
                  Session
                </button>
              </div>
              <label>
                Show workflows
                <select
                  aria-label="Workflow category"
                  value={kind}
                  disabled={busy || dirty}
                  onChange={(event) => {
                    setKind(event.target.value as WorkflowKind | 'all');
                    setSelected(null);
                    setPreset(null);
                  }}
                >
                  <option value="all">All workflows</option>
                  <option value="processing">Processing and analysis</option>
                  <option value="reconstruction">Reconstruction</option>
                  <option value="denoising">Denoising</option>
                  <option value="registration">Registration</option>
                  <option value="fitting">Fitting</option>
                </select>
              </label>
              <label>
                Search workflows
                <input
                  type="search"
                  value={search}
                  disabled={busy || dirty}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </label>
              <ul
                className="workflow-library-list"
                aria-label={source === 'presets' ? 'Built-in presets' : 'Session workflows'}
              >
                {source === 'presets'
                  ? visiblePresets.map((item) => (
                      <li key={item.id}>
                        <button
                          className="workflow-library-item"
                          aria-current={preset?.id === item.id ? 'true' : undefined}
                          disabled={busy || dirty}
                          onClick={() => {
                            setSelected(null);
                            setPreset(item);
                            setError('');
                            setMessage('');
                            setReport([]);
                          }}
                        >
                          {item.document.name}
                          <small>{item.document.kind}</small>
                        </button>
                      </li>
                    ))
                  : visible.map((item) => (
                      <li key={item.id}>
                        <button
                          className="workflow-library-item"
                          aria-current={selected?.id === item.id ? 'true' : undefined}
                          disabled={busy || dirty}
                          onClick={() => select(item)}
                        >
                          {item.document.name}
                          <small>
                            {item.document.steps.length}{' '}
                            {item.document.steps.length === 1 ? 'step' : 'steps'}
                          </small>
                        </button>
                      </li>
                    ))}
              </ul>
              {!(source === 'presets' ? visiblePresets : visible).length && (
                <p className="hint">
                  No matching {source === 'presets' ? 'presets' : 'session workflows'}.
                </p>
              )}
              <div className="workflow-library-actions">
                <button
                  className="secondary"
                  disabled={busy || dirty || !workspace?.loaded}
                  onClick={() => void capture()}
                >
                  <WorkbenchIcon name="add" />
                  Capture current {request.kind === 'processing' ? 'pipeline' : 'configuration'}
                </button>
                {request.kind !== 'processing' && (
                  <button
                    className="secondary"
                    disabled={busy || dirty || !workspace?.loaded}
                    onClick={() => void capture('processing')}
                  >
                    <WorkbenchIcon name="pipeline" />
                    Capture current pipeline
                  </button>
                )}
                <button
                  className="secondary"
                  disabled={busy || dirty}
                  onClick={() => void openFile()}
                >
                  <WorkbenchIcon name="open" />
                  Open workflow file
                </button>
              </div>
            </nav>
            <div className="workflow-library-content">
              {entry ? (
                <>
                  <div className="workflow-document-toolbar">
                    <h3>{entry.name}</h3>
                    <button
                      className="secondary"
                      disabled={busy}
                      aria-pressed={definition}
                      onClick={() => setDefinition(!definition)}
                    >
                      <WorkbenchIcon name="metadata" />
                      {definition ? 'Show targets' : 'Edit definition'}
                    </button>
                    <button className="secondary" disabled={busy} onClick={() => void saveFile()}>
                      <WorkbenchIcon name="save" />
                      Save workflow file
                    </button>
                  </div>
                  <p className="control-note">
                    {workflowExtension} · MRax {entry.mrax_version}
                  </p>
                  {entry.description && <p>{entry.description}</p>}
                  <WorkflowRequirements document={entry} />
                  {definition ? (
                    <section aria-label="Workflow editor">
                      <label>
                        JSON definition
                        <textarea
                          aria-label="Workflow JSON definition"
                          rows={20}
                          value={text}
                          disabled={busy}
                          onChange={(event) => setText(event.target.value)}
                          spellCheck={false}
                        />
                      </label>
                      <p className="hint">
                        UTF-8 JSON · MRax {entry.mrax_version}. Resource bindings are assigned on
                        the target, outside the file.
                      </p>
                    </section>
                  ) : (
                    <>
                      <label>
                        Workflow name
                        <input
                          aria-label="Workflow name"
                          value={typeof fields.name === 'string' ? fields.name : entry.name}
                          disabled={busy}
                          onChange={(event) =>
                            updateDocument((doc) => ({ ...doc, name: event.target.value }))
                          }
                        />
                      </label>
                      <details className="control-advanced">
                        <summary>Steps and description</summary>
                        <label>
                          Description
                          <textarea
                            aria-label="Workflow description"
                            rows={2}
                            value={
                              typeof fields.description === 'string'
                                ? fields.description
                                : entry.description
                            }
                            disabled={busy}
                            onChange={(event) =>
                              updateDocument((doc) => ({ ...doc, description: event.target.value }))
                            }
                          />
                        </label>
                        <ol className="workflow-step-list">
                          {entry.steps.map((step, index) => (
                            <li key={step.id}>
                              <span>
                                {index + 1}.{' '}
                                {step.operation === 'process'
                                  ? String(step.settings.method)
                                  : step.operation}
                              </span>
                              <small>{step.enabled ? 'Enabled' : 'Disabled'}</small>
                            </li>
                          ))}
                        </ol>
                      </details>
                      <section className="workflow-targets" aria-label="Workflow targets">
                        <h3>Apply to experiments</h3>
                        <p className="hint">
                          Choose target experiments and assign any required inputs. Source images
                          remain protected.
                        </p>
                        {targets.map((study) => (
                          <fieldset key={study.workspace_id} disabled={busy}>
                            <legend>{study.label}</legend>
                            <button
                              className="secondary small"
                              onClick={() =>
                                setSelection((old) => [
                                  ...new Set([
                                    ...old,
                                    ...study.experiments.map((row) =>
                                      targetKey(study.workspace_id, row.experiment),
                                    ),
                                  ]),
                                ])
                              }
                            >
                              Select all in study
                            </button>
                            {study.experiments.map((target) => {
                              const key = targetKey(study.workspace_id, target.experiment);
                              const checked = selection.includes(key);
                              return (
                                <div className="workflow-target" key={key}>
                                  <label className="check-label">
                                    <input
                                      type="checkbox"
                                      aria-label={
                                        'Target ' + target.experiment + ' in ' + study.label
                                      }
                                      checked={checked}
                                      onChange={(event) =>
                                        setSelection((old) =>
                                          event.target.checked
                                            ? [...old, key]
                                            : old.filter((item) => item !== key),
                                        )
                                      }
                                    />
                                    {target.experiment}
                                    <small>
                                      {Object.entries(target.dims)
                                        .map(([name, size]) => name + ' ' + size)
                                        .join(' · ')}
                                    </small>
                                  </label>
                                  {checked && entry.bindings.length > 0 && (
                                    <div className="workflow-binding-grid">
                                      {entry.bindings.map((binding) => (
                                        <label key={binding.id}>
                                          {binding.label}
                                          <select
                                            aria-label={binding.label + ' for ' + target.experiment}
                                            value={
                                              mapping(study, target.experiment)[binding.id] ?? ''
                                            }
                                            onChange={(event) =>
                                              setBindings((old) => ({
                                                ...old,
                                                [key]: {
                                                  ...mapping(study, target.experiment),
                                                  [binding.id]: event.target.value,
                                                },
                                              }))
                                            }
                                          >
                                            <option value="">Choose an input</option>
                                            {binding.kind === 'experiment' && (
                                              <option value="$target">Target experiment</option>
                                            )}
                                            {choices(study, target.experiment, binding).map(
                                              (choice) => (
                                                <option key={choice.value} value={choice.value}>
                                                  {choice.label}
                                                </option>
                                              ),
                                            )}
                                          </select>
                                        </label>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </fieldset>
                        ))}
                      </section>
                      <div className="workflow-apply-options">
                        {['processing', 'registration', 'denoising'].includes(entry.kind) && (
                          <label>
                            Existing processing
                            <select
                              aria-label="Existing processing"
                              value={mode}
                              disabled={busy}
                              onChange={(event) => setMode(event.target.value)}
                            >
                              <option value="replace">Replace processing steps</option>
                              <option value="append">Append workflow steps</option>
                            </select>
                          </label>
                        )}
                        {['fitting', 'reconstruction'].includes(entry.kind) && (
                          <label className="check-label">
                            <input
                              type="checkbox"
                              aria-label={
                                entry.kind === 'fitting'
                                  ? 'Run fits after loading configuration'
                                  : 'Run reconstruction after loading configuration'
                              }
                              checked={execute}
                              disabled={busy}
                              onChange={(event) => setExecute(event.target.checked)}
                            />
                            {entry.kind === 'fitting'
                              ? 'Run fits after loading configuration'
                              : 'Run reconstruction after loading configuration'}
                          </label>
                        )}
                        {entry.steps.some(
                          (step) =>
                            step.enabled &&
                            ((step.operation === 'reconstruct' &&
                              (entry.kind !== 'reconstruction' || execute)) ||
                              (step.operation === 'fit' && (entry.kind !== 'fitting' || execute))),
                        ) && (
                          <label>
                            Compute device
                            <select
                              aria-label="Workflow compute device"
                              value={device}
                              disabled={busy}
                              onChange={(event) => setDevice(event.target.value)}
                            >
                              <option value="cpu">CPU</option>
                              {devices
                                .filter((item) => !item.id.startsWith('cpu'))
                                .map((item) => (
                                  <option key={item.id} value={item.id}>
                                    {item.label}
                                  </option>
                                ))}
                            </select>
                          </label>
                        )}
                      </div>
                      <p className="hint">
                        {entry.kind === 'reconstruction'
                          ? 'Reconstruction is replaced. Later processing is reapplied. Fit maps are retained.'
                          : entry.kind === 'fitting'
                            ? 'Fitting uses the current images. Results are stored as named maps.'
                            : 'Replace starts from imported images. Append continues the current pipeline. Existing fit maps are retained.'}
                      </p>
                    </>
                  )}
                  {dirty && (
                    <div className="button-row">
                      <button
                        className="primary"
                        disabled={busy}
                        onClick={() =>
                          void operation(async () => {
                            await saveChanges();
                            setMessage('Workflow updated in this session.');
                          })
                        }
                      >
                        Keep workflow changes
                      </button>
                      <button
                        className="secondary"
                        disabled={busy}
                        onClick={() => {
                          setText(printWorkflow(entry));
                          setError('');
                          setMessage('');
                          setReport([]);
                        }}
                      >
                        Discard workflow edits
                      </button>
                    </div>
                  )}
                </>
              ) : preset ? (
                <section aria-label="Preset preview">
                  <h3>{preset.document.name}</h3>
                  <p>{preset.document.description}</p>
                  <h4>Input requirements</h4>
                  <ul>
                    {preset.requirements.map((requirement) => (
                      <li key={requirement}>{requirement}</li>
                    ))}
                  </ul>
                  <ol className="workflow-step-list">
                    {preset.document.steps.map((step, index) => (
                      <li key={step.id}>
                        {index + 1}.{' '}
                        {step.operation === 'process'
                          ? String(step.settings.method)
                          : step.operation}
                      </li>
                    ))}
                  </ol>
                  <button
                    className="primary"
                    disabled={busy}
                    onClick={() =>
                      void operation(async () => {
                        remember(
                          await api<WorkflowEntry>(
                            '/api/workflow-presets/' + preset.id + '/open',
                            'POST',
                            {},
                          ),
                        );
                        setMessage(
                          'Preset copied to this session. Review its settings and inputs.',
                        );
                      })
                    }
                  >
                    Use preset
                  </button>
                </section>
              ) : (
                <p className="workflow-library-empty">
                  Choose a built-in preset or a session workflow from the library.
                </p>
              )}
            </div>
          </div>
          {job && ['queued', 'running', 'cancelling'].includes(job.status) && (
            <div className="workflow-run-progress" role="status">
              <progress aria-label="Workflow progress" max={1} value={job.progress ?? undefined} />
              {job.detail}
              <button
                className="secondary"
                disabled={job.status === 'cancelling'}
                onClick={() => {
                  cancel.current = true;
                  void api('/api/jobs/' + job.id + '/cancel', 'POST').catch((failure) =>
                    setError(String(failure)),
                  );
                }}
              >
                Cancel remaining work
              </button>
            </div>
          )}
          {report.length > 0 && (
            <ul className="workflow-run-report">
              {report.map((line, index) => (
                <li key={index}>{line}</li>
              ))}
            </ul>
          )}
          {error && <p role="alert">{error}</p>}
          {message && <p role="status">{message}</p>}
          <div className="settings-footer">
            {selected && (
              <button
                className="secondary"
                disabled={busy || dirty}
                onClick={() =>
                  void operation(async () => {
                    await api(
                      '/api/workflows/' + selected.id + '?revision=' + selected.revision,
                      'DELETE',
                    );
                    setLibrary((old) => old.filter((row) => row.id !== selected.id));
                    setSelected(null);
                  })
                }
              >
                Remove from session
              </button>
            )}
            {entry && !definition && (
              <>
                <button
                  className="secondary"
                  disabled={busy || dirty || !selection.length}
                  onClick={() => void checkInputs()}
                >
                  Check inputs
                </button>
                <button
                  className="primary"
                  disabled={busy || dirty || !selection.length}
                  onClick={() => void apply()}
                >
                  {['fitting', 'reconstruction'].includes(entry.kind) && !execute
                    ? entry.kind === 'fitting'
                      ? 'Load fitting settings'
                      : 'Load reconstruction settings'
                    : 'Apply workflow'}
                </button>
              </>
            )}
            <Dialog.Close className="secondary" disabled={working || dirty}>
              Done
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function WorkflowRequirements({ document }: { document: WorkflowDocument }) {
  const metadata = document.extensions['org.mrax.preset'];
  const requirements =
    metadata && typeof metadata === 'object' && !Array.isArray(metadata)
      ? (metadata as Record<string, unknown>).requirements
      : undefined;
  if (!Array.isArray(requirements) || !requirements.every((value) => typeof value === 'string'))
    return null;
  return (
    <details className="control-advanced">
      <summary>Input requirements</summary>
      <ul>
        {requirements.map((value) => (
          <li key={value}>{value}</li>
        ))}
      </ul>
    </details>
  );
}
