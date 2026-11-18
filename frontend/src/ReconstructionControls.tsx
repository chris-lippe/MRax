import { useEffect, useRef, useState } from 'react';
import { api } from './api';
import { ComputeDeviceSelect } from './ComputeDeviceSelect';
import type { AnalysisProps } from './analysisTypes';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';
import { ReconstructionDisplayOptions } from './ReconstructionDisplayOptions';
import { ReconstructionPipelineEditor } from './ReconstructionPipelineEditor';
import { useFileSystem } from './FileSystemDialog';
import type { WorkflowEntry } from './reusableWorkflows';
import { recoProblem, type RecoCatalog, type RecoPlan } from './reconstructionPlan';
import { WorkflowButton } from './WorkflowButton';
import { usePipelineOptions } from './usePipelineOptions';

type Draft = {
  plan: RecoPlan;
  component: string;
  device: string;
  generation: number;
  submitted?: boolean;
};

export function ReconstructionControls({
  workspace: w,
  busy,
  run,
  changeView,
  editingStep,
  openWorkflows,
  stagePipelineOptions,
  active,
}: Pick<
  AnalysisProps,
  | 'workspace'
  | 'busy'
  | 'run'
  | 'changeView'
  | 'editingStep'
  | 'openWorkflows'
  | 'stagePipelineOptions'
> & { active: boolean }) {
  const context = JSON.stringify([w?.id, w?.experiment, w?.pipeline.id]);
  const requestKey = JSON.stringify([context, w?.revision]);
  const [loaded, setLoaded] = useState<{ context: string; key: string; value: RecoCatalog } | null>(
    null,
  );
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [error, setError] = useState('');
  const edited = useRef('');
  const [fileBusy, setFileBusy] = useState(false);
  const choose = useFileSystem();
  const editing = editingStep?.kind === 'reconstruct';
  const editKey = editing ? editingStep.id + editingStep.version : '';
  const editOptions = editing ? JSON.stringify(editingStep.options) : '';
  const savedStep = editing
    ? editingStep
    : w?.pipeline.steps.find((step) => step.kind === 'reconstruct');
  const savedDevice = String(savedStep?.options.device ?? 'cpu');
  const catalog = loaded?.context === context ? loaded.value : null;
  useEffect(() => {
    let stopped = false;
    if (w?.loaded && active)
      void api<RecoCatalog>('/api/workspaces/' + w.id + '/reconstruction')
        .then((value) => {
          if (!stopped) {
            setLoaded({ context, key: requestKey, value });
            setError('');
          }
        })
        .catch((failure) => {
          if (!stopped) setError(String(failure));
        });
    return () => {
      stopped = true;
    };
  }, [w?.id, w?.loaded, w?.revision, active, context, requestKey]);
  useEffect(() => {
    if (!editKey || !editOptions) return;
    const source = JSON.parse(editOptions) as {
      pipeline?: RecoPlan;
      component?: string;
      workflow?: { steps: { settings: { pipeline?: RecoPlan; component?: string } }[] };
    };
    const options = source.workflow?.steps[0].settings ?? source;
    if (options.pipeline)
      setDrafts((previous) => ({
        ...previous,
        [context]: {
          plan: structuredClone(options.pipeline!),
          component: options.component ?? 'magnitude',
          device: savedDevice,
          generation: -1,
        },
      }));
  }, [context, editKey, editOptions, savedDevice]);
  useEffect(() => {
    const draft = drafts[context];
    if (
      !draft?.submitted ||
      !catalog ||
      loaded?.key !== requestKey ||
      !w ||
      w.pipeline.generation === draft.generation
    )
      return;
    if (
      JSON.stringify(draft.plan) === JSON.stringify(catalog.plan) &&
      draft.component === catalog.component &&
      draft.device === savedDevice
    )
      setDrafts((previous) => {
        const next = { ...previous };
        delete next[context];
        return next;
      });
  }, [drafts, context, catalog, loaded?.key, requestKey, savedDevice, w]);
  const draft = drafts[context];
  const plan = draft?.plan ?? catalog?.plan;
  const component = draft?.component ?? catalog?.component ?? 'magnitude';
  const device = draft?.device ?? savedDevice;
  const change = (plan: RecoPlan, output = component, target = device) => {
    if (!w) return;
    setDrafts((previous) => ({
      ...previous,
      [context]: { plan, component: output, device: target, generation: w.pipeline.generation },
    }));
  };
  const loadPreset = (plan: RecoPlan, output = component) => {
    change(plan, output);
    if (!editing)
      void run('pipeline', {
        action: 'configure',
        kind: 'reconstruct',
        options: { pipeline: plan, component: output, device },
      });
  };
  const issue = plan && catalog ? recoProblem(plan, catalog.stages, catalog.encoding) : '';
  const rawAvailable = Boolean(w?.can_reconstruct || w?.can_load_raw);
  usePipelineOptions(
    Boolean(editing && active && plan && catalog),
    Boolean(editKey && edited.current === editKey),
    { view: 'mrax_reconstruction', component, pipeline: plan, device },
    stagePipelineOptions,
    issue || undefined,
  );
  return (
    <div
      className="reconstruction-controls"
      onChangeCapture={() => {
        edited.current = editKey;
      }}
      onClickCapture={() => {
        edited.current = editKey;
      }}
    >
      <WorkflowButton
        disabled={busy}
        onClick={() => openWorkflows?.('reconstruction', { component, pipeline: plan })}
      />
      {error && <p role="alert">{error}</p>}
      <details className="reconstruction-raw">
        <summary>
          <WorkbenchIcon name="reconstruct" />
          Raw k-space
          <span>
            {w?.has_kspace ? 'Loaded' : w?.can_load_raw ? 'Available on disk' : 'Unavailable'}
          </span>
        </summary>
        <button
          type="button"
          className="secondary full"
          disabled={busy || !w?.loaded || !w.can_load_raw || w.has_kspace}
          onClick={() => void run('reconstruct', { view: 'k_space', device })}
        >
          <WorkbenchIcon name="open" />
          Load raw k-space
        </button>
      </details>
      <form
        className="control-layout"
        onSubmit={(event) => {
          event.preventDefault();
          if (!plan || issue || !rawAvailable) return;
          const settings = { ...plan, name: plan.name.trim() };
          change(settings);
          setDrafts((previous) => ({
            ...previous,
            [context]: { ...previous[context], submitted: !editing },
          }));
          void run('reconstruct', {
            view: 'mrax_reconstruction',
            component,
            device,
            pipeline: settings,
            step_id: editing ? editingStep.id : undefined,
          });
        }}
      >
        <fieldset className="control-form" disabled={busy || fileBusy || !w?.loaded || !catalog}>
          <InspectorSection title="Reconstruction pipeline" icon="pipeline">
            <label>
              Reconstruction preset
              <select
                aria-label="Reconstruction preset"
                value={
                  catalog?.presets.find(
                    (preset) => JSON.stringify(preset.plan) === JSON.stringify(plan),
                  )?.id ?? ''
                }
                onChange={(event) => {
                  const preset = catalog?.presets.find((item) => item.id === event.target.value);
                  if (preset) loadPreset(structuredClone(preset.plan));
                }}
              >
                <option value="" disabled>
                  Custom pipeline
                </option>
                {catalog?.presets.map((preset) => (
                  <option value={preset.id} disabled={!preset.compatible} key={preset.id}>
                    {preset.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Pipeline name
              <input
                aria-label="Reconstruction pipeline name"
                value={plan?.name ?? ''}
                maxLength={160}
                onChange={(event) => {
                  if (plan) change({ ...plan, name: event.target.value });
                }}
              />
            </label>
            <div className="reconstruction-file-actions">
              <button
                type="button"
                onClick={() => {
                  setFileBusy(true);
                  setError('');
                  void (async () => {
                    try {
                      const file = await choose({
                        title: 'Open reconstruction pipeline',
                        mode: 'file',
                        extensions: ['.mflow'],
                      });
                      if (!file) return;
                      const entry = await api<WorkflowEntry>('/api/workflows/open-file', 'POST', {
                        root: file.root,
                        path: file.path,
                      });
                      if (entry.document.kind !== 'reconstruction')
                        throw Error('Choose a reconstruction (.mflow) workflow.');
                      if (entry.document.bindings.length)
                        throw Error('Use Workflows to assign this file’s experiment inputs.');
                      const settings = entry.document.steps[0].settings;
                      const imported = (settings.pipeline as RecoPlan | undefined) ?? catalog?.plan;
                      if (imported)
                        loadPreset(
                          structuredClone(imported),
                          String(settings.component ?? 'magnitude'),
                        );
                    } catch (failure) {
                      setError(String(failure));
                    } finally {
                      setFileBusy(false);
                    }
                  })();
                }}
              >
                <WorkbenchIcon name="open" />
                Open .mflow
              </button>
              {draft && !editing && (
                <button
                  type="button"
                  onClick={() =>
                    setDrafts((previous) => {
                      const next = { ...previous };
                      delete next[context];
                      return next;
                    })
                  }
                >
                  Discard changes
                </button>
              )}
            </div>
          </InspectorSection>
          {plan && catalog && (
            <ReconstructionPipelineEditor
              plan={plan}
              catalog={catalog}
              change={change}
              disabled={busy}
            />
          )}
          <InspectorSection title="Output" icon="image">
            <label>
              Reconstruction data
              <select
                aria-label="Reconstruction data"
                value={component}
                onChange={(event) => {
                  if (plan) change(plan, event.target.value);
                }}
              >
                <option value="magnitude">Magnitude</option>
                <option value="phase">Phase</option>
              </select>
            </label>
            <ComputeDeviceSelect
              label="Reconstruction compute device"
              value={device}
              onChange={(target) => {
                if (plan) change(plan, component, target);
              }}
              onError={setError}
            />
            <p className="control-note">
              Reconstruction runs from acquired samples. Later image processing is reapplied to its
              output.
            </p>
            {!rawAvailable && w?.has_reconstruction && (
              <p className="control-note">
                Raw k-space is required to edit reconstruction. Cached images remain available in
                View.
              </p>
            )}
          </InspectorSection>
          {issue && <p role="alert">{issue}</p>}
          <div className="inspector-actionbar">
            <button
              className="primary full"
              data-tutorial="reconstruct-apply"
              disabled={busy || !w?.loaded || !plan || Boolean(issue) || !rawAvailable}
            >
              <WorkbenchIcon name="reconstruct" />
              Apply reconstruction
            </button>
          </div>
        </fieldset>
      </form>
      <ReconstructionDisplayOptions workspace={w} busy={busy} changeView={changeView} />
    </div>
  );
}
