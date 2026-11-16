import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
  type CSSProperties,
  type ReactNode,
} from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { createPortal } from 'react-dom';
import { WorkbenchIcon } from './WorkbenchIcon';
import {
  api,
  image,
  rememberImages,
  retainImagePlanes,
  clearImageCache,
  originalComparison,
  type Bounds,
  type Job,
  type Workspace,
  type Options,
  type Pixels,
  type Tool,
  type TimecourseResult,
  type PipelineStep,
} from './api';
import { Curve } from './Curve';
import {
  fetchNavigation,
  navigationCache,
  navigationContext,
  navigationIndices,
  navigationKey,
  nearbyIndices,
  rememberNavigation,
  type NavigationFrame,
} from './navigation';
import { TimecoursePlot } from './TimecoursePlot';
import { SignalPreview } from './SignalPreview';
import { ResizablePreview } from './ResizablePreview';
import { serializeSignalFigure } from './signalExport';
import {
  automaticSignalAxes,
  defaultSignalOptions,
  signalData,
  signalCsv,
  type SignalOptions,
} from './signal';
import { Viewer } from './Viewer';
import { ViewerControls } from './ViewerControls';
import {
  captureView,
  contextView,
  viewContextKey,
  type SavedView,
  type ViewingContext,
} from './viewing';
import { ComparisonView } from './ComparisonView';
import { AnalysisPanel } from './AnalysisPanel';
import { FileDialog } from './FileDialog';
import { useFileSystem } from './FileSystemDialog';
import { StudyBrowser } from './StudyBrowser';
import { StudyHeader } from './StudyPanel';
import { ExportDialog } from './ExportDialog';
import { MetadataBrowser } from './MetadataBrowser';
import type { MetadataCategory } from './metadata';
import { WorkflowNav, initialWorkflowTasks, type Workflow, type WorkflowTask } from './WorkflowNav';
import { ImageControls } from './ImageControls';
import { Pipeline } from './Pipeline';
import {
  draftChanged,
  draftRequest,
  insertedStep,
  newPipelineDraft,
  type PipelineDraft,
} from './pipelineDraft';
import { InspectorDivider, StudyDivider, useWorkspacePanels } from './InspectorDivider';
import { SettingsDialog } from './SettingsDialog';
import { usePreferences } from './preferences';
import { useBindings, useShortcuts, type ShortcutAction } from './shortcuts';
import { Tutorial, type TutorialSession } from './Tutorial';
import { tutorialSteps } from './tutorialSteps';
import type { WorkflowKind } from './reusableWorkflows';
import { NewStudyDialog, type NewStudyKind } from './NewStudyDialog';
import { useImageDisplay } from './useImageDisplay';
import { nextRoiName, roiDrawingMode, type ScenePreview } from './scene';

type Panel = 'tools' | null;
const WorkflowDialog = lazy(() =>
  import('./WorkflowDialog').then((module) => ({ default: module.WorkflowDialog })),
);
const SignalViewer = lazy(() =>
  import('./SignalViewer').then((m) => ({ default: m.SignalViewer })),
);
const SignalSourceControls = lazy(() =>
  import('./SignalViewer').then((m) => ({ default: m.SignalSourceControls })),
);
const SignalAppearanceControls = lazy(() =>
  import('./SignalViewer').then((m) => ({ default: m.SignalAppearanceControls })),
);
const active = (job: Job | null) =>
  !!job && ['queued', 'running', 'cancelling'].includes(job.status);

function focusDataControl(button: HTMLButtonElement | null) {
  if (button?.getClientRects().length) button.focus();
  else document.querySelector<HTMLButtonElement>('#study-controls button')?.focus();
}

function useMedia(query: string) {
  const [matches, setMatches] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const media = matchMedia(query);
    const update = () => setMatches(media.matches);
    media.addEventListener('change', update);
    update();
    return () => media.removeEventListener('change', update);
  }, [query]);
  return matches;
}

export function App() {
  useEffect(() => {
    const root = document.documentElement;
    const setMode = (mode: 'pointer' | 'keyboard') => {
      if (root.dataset.inputModality !== mode) root.dataset.inputModality = mode;
    };
    const pointer = () => setMode('pointer');
    const keyboard = (event: KeyboardEvent) => {
      if (!['Shift', 'Control', 'Alt', 'Meta'].includes(event.key)) setMode('keyboard');
    };
    document.addEventListener('pointerdown', pointer, true);
    document.addEventListener('keydown', keyboard, true);
    return () => {
      document.removeEventListener('pointerdown', pointer, true);
      document.removeEventListener('keydown', keyboard, true);
      delete root.dataset.inputModality;
    };
  }, []);
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [values, setValues] = useState<Pixels | null>(null);
  const [layerValues, setLayerValues] = useState<Record<string, Pixels>>({});
  const [scenePreview, setScenePreview] = useState<ScenePreview>({ layers: {} });
  const previewRoiOpacity = (name: string, value: number, label?: number) =>
    setScenePreview((previous) =>
      label === undefined
        ? { ...previous, rois: { ...previous.rois, [name]: value } }
        : {
            ...previous,
            labels: {
              ...previous.labels,
              [name]: { ...previous.labels?.[name], [label]: value },
            },
          },
    );
  const [connected, setConnected] = useState(false);
  const [taskBusy, setTaskBusy] = useState(false);
  const [navigating, setNavigating] = useState(false);
  const [pendingIndices, setPendingIndices] = useState<Record<string, number>>({});
  const navigationVersion = useRef(0);
  const navigationRun = useRef<{
    version: number;
    context: string;
    id: string;
    desired: Record<string, number>;
    queued: boolean;
    preferred: { key: string; direction: number };
  } | null>(null);
  const prefetchTimer = useRef<number | null>(null);
  const prefetchAbort = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      if (prefetchTimer.current !== null) window.clearTimeout(prefetchTimer.current);
      prefetchAbort.current?.abort();
    },
    [],
  );
  const busy = taskBusy || navigating;
  const [error, setError] = useState('');
  const [message, setMessage] = useState('No study open');
  const [tool, setTool] = useState<Tool>('pan'),
    [brushSize, setBrushSize] = useState(5),
    [label, setLabel] = useState(1);
  const [importOpen, setImportOpen] = useState(false),
    [importIntoCurrent, setImportIntoCurrent] = useState(false);
  const current = useRef<Workspace | null>(null);
  const [drawOptions, setDrawOptions] = useState<Options>({
    angle: 0,
    sides: 6,
    snap: false,
    tolerance: 0.08,
    auto_subtract: false,
  });
  const [job, setJob] = useState<Job | null>(null);
  const [panel, setPanel] = useState<Panel>(null);
  const [workflow, setWorkflow] = useState<Workflow>('image');
  const [workflowTasks, setWorkflowTasks] = useState(initialWorkflowTasks);
  const [signalOpened, setSignalOpened] = useState(false);
  const workflowTask = workflowTasks[workflow];
  const activeRoi = workspace?.selected_roi ?? workspace?.rois.find((roi) => roi.enabled)?.name;
  const drawingMode = roiDrawingMode(tool, drawOptions, Boolean(activeRoi));
  const requiresRoi =
    tool === 'erase' || (tool !== 'pan' && ['add', 'subtract'].includes(drawingMode));
  const drawing =
    workflowTask === 'roi-edit' &&
    !['pan', 'otsu'].includes(tool) &&
    Boolean(workspace?.editable) &&
    (!requiresRoi || Boolean(activeRoi));
  const previousDrawingTool = useRef<Tool>('rect');
  useEffect(() => {
    if (tool !== 'pan') previousDrawingTool.current = tool;
  }, [tool]);
  const viewingContext = useRef<ViewingContext>({ section: 'image', task: 'display' });
  const displayedContext = useRef<string | null>(null);
  const savedViews = useRef(new Map<string, SavedView>());
  const viewPalettes = useRef(new Map<string, string>());
  const [comparisonControlsHost, setComparisonControlsHost] = useState<HTMLDivElement | null>(null);
  const [editingPipeline, setEditingPipeline] = useState<{
    context: string;
    id: string;
    version: string;
    options: Options;
    session: string;
  } | null>(null);
  const pipelineContext = JSON.stringify([
    workspace?.id,
    workspace?.experiment,
    workspace?.pipeline.id,
  ]);
  const [pipelineDrafts, setPipelineDrafts] = useState<Record<string, PipelineDraft>>({});
  const [pipelineRuns, setPipelineRuns] = useState<Record<string, Job>>({});
  const pipelineDraft = pipelineDrafts[pipelineContext] ?? null;
  const pipelineDraftRef = useRef(pipelineDrafts);
  pipelineDraftRef.current = pipelineDrafts;
  const [editorOptions, setEditorOptions] = useState<{
    key: string;
    options: Options;
    form: number;
  } | null>(null);
  const editorKey =
    editingPipeline?.context === pipelineContext
      ? JSON.stringify([
          editingPipeline.context,
          editingPipeline.id,
          editingPipeline.version,
          editingPipeline.session,
        ])
      : '';
  const editingStep =
    editingPipeline?.context === pipelineContext
      ? ((pipelineDraft
          ? pipelineDraft.steps.find((step) => step.id === editingPipeline.id)
          : workspace?.pipeline.steps.find(
              (step) => step.id === editingPipeline.id && step.version === editingPipeline.version,
            )) ?? null)
      : null;
  function editPipelineStep(step: PipelineStep) {
    if (workspace)
      setPipelineDrafts((previous) => ({
        ...previous,
        [pipelineContext]: previous[pipelineContext] ?? newPipelineDraft(workspace),
      }));
    setEditingPipeline({
      context: pipelineContext,
      id: step.id,
      version: step.version,
      options: structuredClone(step.options),
      session: crypto.randomUUID(),
    });
    if (step.kind === 'reconstruct') selectWorkflow('reconstruct', 'reconstruction');
    else if (step.kind === 'fit') selectWorkflow('fit', 'model-fit');
    else if (step.kind === 'register' || step.kind === 'stored_registration')
      selectWorkflow('registration', 'alignment');
    else if (step.kind === 'drift') selectWorkflow('registration', 'drift');
    else if (step.family === 'denoise') selectWorkflow('denoise', 'denoising');
    else selectWorkflow('analyze', step.family === 'analyze' ? 'statistics' : 'timecourse');
  }
  useEffect(() => {
    if (!editingPipeline || editingPipeline.context !== pipelineContext || !workspace) return;
    const snapshot = editingPipeline;
    let stopped = false;
    setEditorOptions(null);
    const selectedStep = (
      pipelineDraftRef.current[pipelineContext]?.steps ?? workspace.pipeline.steps
    ).find((step) => step.id === snapshot.id);
    const kind = selectedStep?.kind;
    const options = selectedStep?.options ?? snapshot.options;
    const request =
      kind && ['reconstruct', 'fit'].includes(kind)
        ? api<{ options: Options }>(
            '/api/workspaces/' + workspace.id + '/pipeline-editor',
            'POST',
            {
              revision: workspace.revision,
              kind,
              options,
            },
          ).then((result) => ({ ...result.options, device: options.device ?? 'cpu' }))
        : Promise.resolve(options);
    void request
      .then((options) => {
        if (!stopped) setEditorOptions({ key: editorKey, options, form: 0 });
      })
      .catch((failure) => {
        if (!stopped) setError(String(failure));
      });
    return () => {
      stopped = true;
    };
    // Settings are captured when an edit starts. Local draft updates keep the form mounted.
  }, [editorKey]);
  const controlStep =
    editingStep && editorOptions?.key === editorKey
      ? { ...editingStep, options: editorOptions.options }
      : null;
  const stagePipelineOptions = useCallback(
    (options: Options, problem?: string) => {
      if (!workspace || !editingPipeline || editingPipeline.context !== pipelineContext) return;
      const clean = Object.fromEntries(
        Object.entries(options).filter(
          ([key, value]) =>
            key !== 'step_id' && key !== 'name' && !key.startsWith('_') && value !== undefined,
        ),
      );
      const draft = pipelineDraftRef.current[pipelineContext];
      if (!draft || draft.submission) return;
      const next = {
        ...draft,
        submission: undefined,
        steps: draft.steps.map((step) =>
          step.id === editingPipeline.id ? { ...step, options: clean, problem } : step,
        ),
      };
      pipelineDraftRef.current = { ...pipelineDraftRef.current, [pipelineContext]: next };
      setPipelineDrafts(pipelineDraftRef.current);
    },
    [workspace, editingPipeline, pipelineContext],
  );
  const editFitSettings = (patch: Options) => {
    if (!controlStep || controlStep.kind !== 'fit') return;
    const staged = pipelineDraftRef.current[pipelineContext]?.steps.find(
      (step) => step.id === controlStep.id,
    )?.options;
    const previous = staged && !staged.workflow ? staged : controlStep.options;
    const { excluded_frames, ...settings } = patch;
    const next = {
      ...previous,
      settings: { ...(previous.settings as Options), ...settings },
      ...(excluded_frames === undefined ? {} : { excluded_frames }),
    };
    stagePipelineOptions(next);
    setEditorOptions((current) =>
      current ? { ...current, options: next, form: current.form + 1 } : current,
    );
  };
  const editorChangeView = async (command: Options) => {
    const scene = command.scene as Options | undefined;
    if (controlStep?.kind === 'fit' && scene && ('fit_region' in scene || 'fit_rois' in scene)) {
      const staged = pipelineDraftRef.current[pipelineContext]?.steps.find(
        (step) => step.id === controlStep.id,
      )?.options;
      const previous = staged && !staged.workflow ? staged : controlStep.options;
      const next = {
        ...previous,
        ...('fit_region' in scene ? { use_roi: scene.fit_region !== 'all' } : {}),
        ...('fit_rois' in scene ? { roi_names: scene.fit_rois } : {}),
      };
      stagePipelineOptions(next);
      setEditorOptions((current) =>
        current ? { ...current, options: next, form: current.form + 1 } : current,
      );
      return;
    }
    await changeView(command);
  };
  useEffect(() => {
    if (job?.pipeline) {
      const key = JSON.stringify([
        job.workspace_id,
        job.pipeline.experiment,
        job.pipeline.pipeline_id,
      ]);
      setPipelineRuns((previous) => ({ ...previous, [key]: job }));
    }
    if (job?.status === 'completed') {
      setPipelineDrafts((previous) =>
        Object.fromEntries(
          Object.entries(previous).filter(([, draft]) => draft.submission !== job.id),
        ),
      );
      if (job.pipeline && job.operation === 'pipeline') {
        const key = JSON.stringify([
          job.workspace_id,
          job.pipeline.experiment,
          job.pipeline.pipeline_id,
        ]);
        setEditingPipeline((previous) => (previous?.context === key ? null : previous));
      }
    }
  }, [job]);
  function setPipelineDraft(next: PipelineDraft | null) {
    const drafts = { ...pipelineDraftRef.current };
    if (next) drafts[pipelineContext] = next;
    else delete drafts[pipelineContext];
    pipelineDraftRef.current = drafts;
    setPipelineDrafts((previous) => {
      const result = { ...previous };
      if (next) result[pipelineContext] = next;
      else delete result[pipelineContext];
      return result;
    });
    if (!next) setEditingPipeline(null);
  }
  async function applyPipelineDraft(draft: PipelineDraft) {
    if (!workspace) throw Error('Select an experiment.');
    const targetContext = pipelineContext;
    const next = await api<Job>(`/api/workspaces/${workspace.id}/tasks`, 'POST', {
      revision: workspace.revision,
      operation: 'pipeline',
      options: draftRequest(draft),
    });
    pipelineDraftRef.current = {
      ...pipelineDraftRef.current,
      [targetContext]: { ...draft, submission: next.id },
    };
    setPipelineDrafts(pipelineDraftRef.current);
    setJob(next);
  }
  const [timecourse, setTimecourse] = useState<TimecourseResult | null>(null);
  const [signalDrafts, setSignalDrafts] = useState<Record<string, SignalOptions>>({});
  const signalContext = JSON.stringify([workspace?.id, workspace?.experiment]);
  const signalOptions = signalDrafts[signalContext] ?? defaultSignalOptions;
  const signalFigure = useRef<SVGSVGElement>(null);
  function updateSignal(patch: Partial<SignalOptions>) {
    setSignalDrafts((previous) => ({
      ...previous,
      [signalContext]: { ...(previous[signalContext] ?? defaultSignalOptions), ...patch },
    }));
  }
  const display = useImageDisplay(
    workspace
      ? JSON.stringify([
          workspace.id,
          workspace.experiment,
          workspace.view,
          workspace.plane,
          workspace.view === 'Map' ? workspace.active_map : null,
          workspace.display.component ?? 'auto',
          workspace.reconstruction_component,
          workspace.reconstruction_coil,
          workspace.kspace_mode,
        ])
      : undefined,
    values,
    Boolean(workspace && workspace.channels > 1),
  );
  const { theme, setTheme, accessibility, setAccessibility } = usePreferences();
  const { bindings, setBindings } = useBindings();
  const [tutorial, setTutorial] = useState<TutorialSession | null>(null);
  const [viewerControlsOpen, setViewerControlsOpen] = useState(
    () => localStorage.getItem('mrax.viewerControlsOpen') !== 'false',
  );
  function viewControls(children: ReactNode) {
    return (
      workspace && (
        <ViewerControls
          workspace={workspace}
          context={viewingContext.current}
          busy={busy}
          change={changeView}
          expanded={viewerControlsOpen}
          onExpandedChange={(open) => {
            setViewerControlsOpen(open);
            localStorage.setItem('mrax.viewerControlsOpen', String(open));
          }}
        >
          {children}
        </ViewerControls>
      )
    );
  }
  const selected = useRef<string | null>(null);
  const imageRequest = useRef(0);
  const imageAbort = useRef<AbortController | null>(null);
  const pending = useRef(0),
    viewRequest = useRef(0);
  const choose = useFileSystem();
  const closeStudyTrigger = useRef<HTMLButtonElement>(null);
  const restoreStudyFocus = useRef(false);
  const [closeTarget, setCloseTarget] = useState<Workspace | null>(null);
  const [metadataCategory, setMetadataCategory] = useState<MetadataCategory>('experiment');
  const [exportOpen, setExportOpen] = useState(false);
  const importButton = useRef<HTMLButtonElement>(null);
  const exportButton = useRef<HTMLButtonElement>(null);
  const settingsButton = useRef<HTMLButtonElement>(null);
  const newStudyButton = useRef<HTMLButtonElement>(null);
  const [newStudyOpen, setNewStudyOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [workflowRequest, setWorkflowRequest] = useState<{
    kind: WorkflowKind;
    draft?: Options;
  } | null>(null);
  const large = useMedia('(min-width: 1120px)');
  const { browser, inspector } = useWorkspacePanels(large && workflowTask !== 'metadata');

  useEffect(() => {
    if (busy || !restoreStudyFocus.current) return;
    restoreStudyFocus.current = false;
    const trigger = closeStudyTrigger.current;
    const next = document.querySelector<HTMLButtonElement>('.study-item[aria-current="true"]');
    focusDataControl(trigger?.isConnected ? trigger : (next ?? newStudyButton.current));
  }, [busy, workspaces]);

  useEffect(() => {
    if (large && panel === 'tools') setPanel(null);
  }, [large, panel]);

  useShortcuts(bindings, (name: ShortcutAction) => {
    if (name === 'settings') {
      setSettingsOpen(true);
      return true;
    }
    const sections: Partial<Record<ShortcutAction, Workflow>> = {
      view: 'image',
      reconstruct: 'reconstruct',
      denoise: 'denoise',
      register: 'registration',
      segment: 'rois',
      fit: 'fit',
      analyze: 'analyze',
    };
    const section = sections[name];
    if (section) {
      selectWorkflow(section);
      return true;
    }
    if (name === 'drawing') {
      if (busy || active(job) || !current.current?.editable || workflowTask !== 'roi-edit')
        return false;
      setTool(tool === 'pan' ? previousDrawingTool.current : 'pan');
      return true;
    }
    if (!current.current?.loaded) return false;
    if (name === 'resetImage') display.resetPosition();
    else if (name === 'zoomIn') display.zoomIn();
    else if (name === 'zoomOut') display.zoomOut();
    else return false;
    return true;
  });

  async function startTutorial() {
    setSettingsOpen(false);
    if (tutorial && workspaces.some((w) => w.id === tutorial.studyId)) {
      await resumeTutorial();
      return;
    }
    const previous = {
      previousId: current.current?.id ?? null,
      previousContext: { ...viewingContext.current },
      previousWorkflow: workflow,
      previousTasks: { ...workflowTasks },
      previousTool: tool,
      previousBrushSize: brushSize,
      previousLabel: label,
      previousDrawOptions: { ...drawOptions },
      previousControlsOpen: viewerControlsOpen,
    };
    await task(async () => {
      const next = await api<Workspace>('/api/workspaces/synthetic?kind=tutorial', 'POST');
      viewingContext.current = { section: 'image', task: 'display' };
      setWorkflow('image');
      setWorkflowTasks({ ...initialWorkflowTasks });
      setViewerControlsOpen(true);
      setTool('pan');
      setBrushSize(5);
      setLabel(1);
      setDrawOptions({ angle: 0, sides: 6, snap: false, tolerance: 0.08, auto_subtract: false });
      setPanel(null);
      await show(next);
      setTutorial({
        ...previous,
        studyId: next.id,
        initialFrame: next.frame,
        initialSlice: next.z,
        step: 0,
        completed: [],
      });
    });
  }

  async function resumeTutorial() {
    if (!tutorial) return;
    await task(async () => {
      await show(await api<Workspace>(`/api/workspaces/${tutorial.studyId}`));
      setPanel(null);
    });
  }

  function openTutorialTask() {
    if (!tutorial || current.current?.id !== tutorial.studyId) return;
    setViewerControlsOpen(true);
    const step = tutorialSteps[tutorial.step];
    if (step.context) selectWorkflow(step.context.section, step.context.task);
    if (step.id === 'roi') {
      setTool('pan');
    }
    if (step.id === 'pipeline')
      document.querySelector('.pipeline-panel')?.scrollIntoView({ block: 'start' });
  }

  async function endTutorial() {
    if (!tutorial) return;
    const finished = tutorial;
    setTutorial(null);
    setSettingsOpen(false);
    setTool(finished.previousTool);
    setBrushSize(finished.previousBrushSize);
    setLabel(finished.previousLabel);
    setDrawOptions(finished.previousDrawOptions);
    setViewerControlsOpen(finished.previousControlsOpen);
    if (finished.previousId && workspaces.some((w) => w.id === finished.previousId)) {
      viewingContext.current = finished.previousContext;
      setWorkflow(finished.previousWorkflow);
      setWorkflowTasks(finished.previousTasks);
      await task(async () => show(await api<Workspace>('/api/workspaces/' + finished.previousId)));
    }
  }

  async function show(
    next: Workspace,
    preserveImage = false,
    completedOperation?: string,
    keepRequestedView = false,
    unchangedPixels?: 'display' | 'roi',
  ) {
    const previousImage = current.current;
    navigationVersion.current += 1;
    navigationRun.current = null;
    stopPrefetch();
    setNavigating(false);
    setPendingIndices({});
    const request = ++imageRequest.current;
    imageAbort.current?.abort();
    const controller = new AbortController();
    imageAbort.current = controller;
    const context = viewContextKey(next, viewingContext.current);
    const showResult =
      (completedOperation === 'full_fit' && viewingContext.current.section === 'fit') ||
      (completedOperation === 'reconstruct' &&
        next.has_reconstruction &&
        viewingContext.current.section === 'reconstruct');
    if (next.loaded && (displayedContext.current !== context || showResult)) {
      const previous = current.current;
      if (previous?.loaded && displayedContext.current)
        savedViews.current.set(displayedContext.current, captureView(previous));
      const view = keepRequestedView
        ? captureView(next)
        : contextView(
            next,
            viewingContext.current,
            showResult ? undefined : savedViews.current.get(context),
          );
      if (JSON.stringify(view) !== JSON.stringify(captureView(next)))
        next = await api<Workspace>(`/api/workspaces/${next.id}/view`, 'PATCH', {
          revision: next.revision,
          ...view,
        });
      if (request !== imageRequest.current) return;
      displayedContext.current = context;
    }
    if (previousImage && unchangedPixels)
      retainImagePlanes(previousImage, next, unchangedPixels === 'roi');
    current.current = next;
    if (!preserveImage || selected.current !== next.id) {
      current.current = next;
      setWorkspace(next);
      setValues(null);
      setLayerValues({});
    }
    selected.current = next.id;
    let pixels: Pixels | null;
    let layers: (readonly [string, Pixels])[];
    try {
      [pixels, layers] = await Promise.all([
        next.loaded && !next.scene.comparison.enabled
          ? image(next, undefined, controller.signal)
          : null,
        Promise.all(
          next.scene.layers
            .filter(
              (layer) => !next.scene.comparison.enabled && layer.visible && !layer.unavailable,
            )
            .map(
              async (layer) => [layer.id, await image(next, layer.id, controller.signal)] as const,
            ),
        ),
      ]);
    } catch (failure) {
      if (controller.signal.aborted) return;
      throw failure;
    }
    if (request === imageRequest.current) {
      current.current = next;
      setWorkspace(next);
      setValues(pixels);
      setLayerValues((previous) => {
        const nextLayers = Object.fromEntries(layers);
        return Object.keys(previous).length === layers.length &&
          layers.every(([id, data]) => previous[id] === data)
          ? previous
          : nextLayers;
      });
      setScenePreview({ layers: {} });
      if (next.loaded) savedViews.current.set(context, captureView(next));
      setWorkspaces((previous) =>
        previous.some((item) => item.id === next.id)
          ? previous.map((item) => (item.id === next.id ? next : item))
          : [...previous, next],
      );
      if (next.loaded) {
        if (!next.scene.comparison.enabled)
          rememberNavigation({
            workspace: next,
            values: pixels,
            layers: Object.fromEntries(layers),
          });
        startPrefetch(next);
      }
    }
  }

  async function task(action: () => Promise<void>, propagate = false) {
    stopPrefetch();
    pending.current += 1;
    setTaskBusy(true);
    setError('');
    try {
      await action();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The operation failed.');
      if (propagate) throw failure;
    } finally {
      pending.current -= 1;
      setTaskBusy(pending.current > 0);
    }
  }

  useEffect(() => {
    let stopped = false;
    void task(async () => {
      await api('/api/health');
      const opened = await api<Workspace[]>('/api/workspaces');
      if (!stopped) {
        setConnected(true);
        setWorkspaces(opened);
        if (opened.length) {
          await show(opened[opened.length - 1]);
          setMessage('Workspace reconnected');
        }
      }
    });
    return () => {
      stopped = true;
    };
  }, []);

  useEffect(() => {
    if (!job) return;
    const id = job.id;
    let stopped = false;
    async function poll() {
      try {
        while (!stopped) {
          const next = await api<Job>(`/api/jobs/${id}`);
          if (stopped) return;
          if (!active(next)) {
            setMessage(next.detail);
            if (
              next.status === 'completed' &&
              (selected.current === next.workspace_id || next.operation === 'import')
            )
              await show(
                await api<Workspace>(`/api/workspaces/${next.workspace_id}`),
                false,
                next.operation,
              );
            if (next.status === 'failed' || next.status === 'stale') setError(next.detail);
            setJob(next);
            return;
          }
          setJob(next);
          await new Promise((resolve) => setTimeout(resolve, 250));
        }
      } catch (failure) {
        if (!stopped) {
          setError(failure instanceof Error ? failure.message : 'The fit connection was lost.');
          setJob(null);
        }
      }
    }
    void poll();
    return () => {
      stopped = true;
    };
  }, [job?.id]);

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (workspace?.dirty || active(job) || Object.values(pipelineDrafts).some(draftChanged)) {
        event.preventDefault();
        event.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [workspace?.dirty, job?.status, pipelineDrafts]);

  async function addRectangle(rectangle: Bounds) {
    if (
      Number(drawOptions.angle ?? 0) !== 0 ||
      Boolean(drawOptions.auto_subtract) ||
      ['add', 'subtract'].includes(String(drawOptions.mode))
    ) {
      await shape({ tool: 'rect', bounds: rectangle });
      return;
    }
    if (!workspace) return;
    await task(async () => {
      const next = await api<Workspace>(`/api/workspaces/${workspace.id}/rois`, 'POST', {
        revision: workspace.revision,
        view_revision: workspace.view_revision,
        name: nextRoiName(workspace.rois),
        bounds: rectangle,
      });
      await show(next, true, undefined, true, 'roi');
      setMessage('ROI added on the native image grid');
    });
  }

  async function changeView(command: Options) {
    const currentWorkspace = current.current;
    if (!currentWorkspace) return;
    if (typeof command.view === 'string' && command.view !== currentWorkspace.view) {
      const paletteKey = (view: string) =>
        JSON.stringify([currentWorkspace.id, currentWorkspace.experiment, view]);
      viewPalettes.current.set(
        paletteKey(currentWorkspace.view),
        String(currentWorkspace.display.colormap ?? 'gray'),
      );
      if (command.colormap === undefined)
        command = {
          ...command,
          colormap:
            viewPalettes.current.get(paletteKey(command.view)) ??
            (command.view === 'Map'
              ? 'viridis'
              : command.view === 'Vendor MRax Diff'
                ? 'diverging'
                : 'gray'),
        };
    }
    const request = ++viewRequest.current;
    if (command.scene || command.selected_roi) {
      const patch = (command.scene ?? {}) as Partial<Workspace['scene']>;
      const layers = patch.layers
        ? patch.layers.flatMap((layer) => {
            const previous = currentWorkspace.scene.layers.find((item) => item.id === layer.id);
            return previous ? [{ ...previous, ...layer }] : [];
          })
        : currentWorkspace.scene.layers;
      const scene = { ...currentWorkspace.scene, ...patch, layers };
      setWorkspace({
        ...currentWorkspace,
        scene,
        rois: currentWorkspace.rois.map((roi) => ({
          ...roi,
          visible: roi.enabled && !scene.hidden_rois.includes(roi.name),
        })),
        ...(command.selected_roi
          ? { selected_roi: String(command.selected_roi), timecourse: null, preview: null }
          : {}),
      });
    }
    await task(async () => {
      try {
        const next = await api<Workspace>(`/api/workspaces/${currentWorkspace.id}/view`, 'PATCH', {
          revision: currentWorkspace.revision,
          ...command,
        });
        if (request === viewRequest.current) await show(next, true, undefined, false, 'display');
      } catch (failure) {
        if (request === viewRequest.current) {
          setWorkspace(current.current);
          setScenePreview({ layers: {} });
        }
        throw failure;
      }
    });
  }

  function stopPrefetch() {
    if (prefetchTimer.current !== null) window.clearTimeout(prefetchTimer.current);
    prefetchTimer.current = null;
    prefetchAbort.current?.abort();
    prefetchAbort.current = null;
  }

  function startPrefetch(w: Workspace, preferred?: { key: string; direction: number }) {
    stopPrefetch();
    const controller = new AbortController();
    prefetchAbort.current = controller;
    const version = navigationVersion.current;
    prefetchTimer.current = window.setTimeout(() => {
      prefetchTimer.current = null;
      void (async () => {
        for (const indices of nearbyIndices(w, preferred)) {
          if (controller.signal.aborted || version !== navigationVersion.current) return;
          if (navigationCache.get(navigationKey(w, indices))) continue;
          try {
            await fetchNavigation(w, indices, false, controller.signal);
          } catch {
            return;
          } // Background previews must never interrupt interaction.
        }
      })();
    }, 64);
  }

  function displayNavigation(frame: NavigationFrame, preview = false) {
    const next = preview
      ? { ...frame.workspace, view_revision: current.current!.view_revision, dirty: true }
      : frame.workspace;
    rememberImages(next, frame.values, frame.layers, frame.comparison);
    if (!preview) {
      current.current = next;
      savedViews.current.set(viewContextKey(next, viewingContext.current), captureView(next));
      setWorkspaces((previous) => previous.map((item) => (item.id === next.id ? next : item)));
    }
    setWorkspace(next);
    setValues(frame.values);
    setLayerValues(frame.layers);
    setScenePreview({ layers: {} });
  }

  async function navigateIndex(key: string, value: number) {
    const target = current.current;
    if (!target) return;
    stopPrefetch();
    const context = navigationContext(target);
    let run = navigationRun.current;
    const running = run?.context === context && run.version === navigationVersion.current;
    if (!running) {
      run = {
        version: navigationVersion.current,
        context,
        id: target.id,
        desired: {},
        queued: false,
        preferred: { key, direction: 1 },
      };
      navigationRun.current = run;
      imageRequest.current += 1;
      imageAbort.current?.abort();
    }
    const active = run!;
    const previous = active.desired[key] ?? navigationIndices(target)[key] ?? 0;
    active.preferred = { key, direction: Math.sign(value - previous) || 1 };
    active.desired = { ...active.desired, [key]: value };
    active.queued = true;
    setPendingIndices(active.desired);
    const cached = navigationCache.get(navigationKey(target, active.desired));
    if (cached) displayNavigation(cached, true);
    if (running) return;
    setNavigating(true);
    setError('');
    try {
      while (active.queued) {
        active.queued = false;
        const latest = current.current;
        if (!latest || navigationRun.current !== active || navigationContext(latest) !== context)
          break;
        const frame = await fetchNavigation(latest, active.desired, true);
        if (
          navigationRun.current !== active ||
          active.version !== navigationVersion.current ||
          selected.current !== active.id
        )
          break;
        current.current = frame.workspace;
        if (navigationKey(frame.workspace) === navigationKey(frame.workspace, active.desired))
          active.queued = false;
        if (!active.queued) displayNavigation(frame);
      }
    } catch (failure) {
      if (navigationRun.current === active) {
        setError(failure instanceof Error ? failure.message : 'Could not update the image.');
        try {
          const restored = await api<Workspace>(`/api/workspaces/${active.id}`);
          if (navigationRun.current === active && selected.current === active.id)
            await show(restored, true);
        } catch {
          // Keep the last confirmed plane if the connection is unavailable.
          if (navigationRun.current === active && current.current) {
            const confirmed = navigationCache.get(navigationKey(current.current));
            if (confirmed) displayNavigation(confirmed, true);
          }
        }
      }
    } finally {
      if (navigationRun.current === active) {
        navigationRun.current = null;
        setPendingIndices({});
        setNavigating(false);
        if (current.current) startPrefetch(current.current, active.preferred);
      }
    }
  }

  async function action(name: string, options: Options = {}) {
    if (name === 'fit.pool' && controlStep?.kind === 'fit' && workspace) {
      await task(async () => {
        const result = await api<{ settings: Options }>(
          '/api/workspaces/' + workspace.id + '/fit-options',
          'POST',
          {
            revision: workspace.revision,
            settings: options.settings,
            pool_edit: options,
          },
        );
        editFitSettings({ ...result.settings, excluded_frames: options.excluded_frames });
      });
      return;
    }
    if (name === 'fit.settings' && controlStep?.kind === 'fit') {
      editFitSettings(options);
      return;
    }
    await task(async () => {
      const w = current.current;
      if (!w) return;
      await show(
        await api<Workspace>(`/api/workspaces/${w.id}/actions`, 'POST', {
          revision: w.revision,
          view_revision: w.view_revision,
          action: name,
          options,
        }),
        name.startsWith('roi.'),
        undefined,
        true,
        name.startsWith('roi.') ? 'roi' : undefined,
      );
      setMessage('Workspace updated');
    });
  }
  async function run(operation: string, options: Options = {}) {
    const kind =
      operation === 'full_fit'
        ? 'fit'
        : operation === 'reconstruct' && options.view !== 'k_space'
          ? 'reconstruct'
          : ['process', 'register', 'drift'].includes(operation) && options.method !== 'dti'
            ? operation
            : operation === 'pipeline' && options.action === 'registration'
              ? 'stored_registration'
              : null;
    const staged = pipelineDraftRef.current[pipelineContext];
    const requested = typeof options.step_id === 'string' ? options.step_id : null;
    if (workspace && kind && (requested || (staged && draftChanged(staged)))) {
      await task(async () => {
        const draft = pipelineDraftRef.current[pipelineContext] ?? newPipelineDraft(workspace);
        const clean = Object.fromEntries(
          Object.entries(options).filter(
            ([key, value]) =>
              !['step_id', 'name', 'action'].includes(key) &&
              !key.startsWith('_') &&
              value !== undefined,
          ),
        );
        if (!['fit', 'reconstruct'].includes(kind)) clean.input ??= 'image';
        const family =
          kind === 'stored_registration'
            ? 'register'
            : kind === 'process'
              ? workflow === 'denoise'
                ? 'denoise'
                : String(clean.method)
              : kind;
        const previous = requested
          ? draft.steps.find((step) => step.id === requested)
          : draft.steps.find((step) =>
              ['fit', 'reconstruct'].includes(kind) ? step.kind === kind : step.family === family,
            );
        if (requested && !previous) throw Error('The selected pipeline block no longer exists.');
        const next = [...draft.steps];
        if (previous) {
          next[next.indexOf(previous)] =
            previous.kind === kind
              ? { ...previous, options: clean, problem: undefined }
              : insertedStep(kind, clean, previous.label, family);
        } else {
          const step = insertedStep(
            kind,
            clean,
            kind === 'process'
              ? String(clean.method)
              : kind === 'fit'
                ? 'Fitting'
                : kind === 'reconstruct'
                  ? 'Reconstruction'
                  : 'Registration',
            family,
          );
          next.splice(kind === 'reconstruct' ? 0 : next.length, 0, step);
        }
        const proposed = { ...draft, steps: next, submission: undefined };
        pipelineDraftRef.current = { ...pipelineDraftRef.current, [pipelineContext]: proposed };
        setPipelineDrafts(pipelineDraftRef.current);
        await applyPipelineDraft(proposed);
      });
      return;
    }
    await task(async () => {
      const w = current.current;
      if (!w) return;
      setJob(null);
      setJob(
        await api<Job>(`/api/workspaces/${w.id}/tasks`, 'POST', {
          revision: w.revision,
          operation,
          options,
        }),
      );
    });
  }
  async function openWorkspace() {
    const source = await choose({
      title: 'Open workspace',
      mode: 'open',
      extensions: ['.mrax', '.json'],
    });
    if (!source) return;
    await task(async () => {
      await show(
        await api<Workspace>('/api/workspaces/open', 'POST', {
          root: source.root,
          path: source.path,
        }),
      );
      setPanel(null);
      setMessage('Workspace opened');
    });
  }
  async function save() {
    const w = current.current;
    if (!w) return;
    const location = w.file_location
      ? await api<{ root: string; parent: string; name: string }>(
          '/api/files/stat?' + new URLSearchParams(w.file_location),
        ).catch(() => null)
      : null;
    const destination = await choose({
      title: 'Save workspace',
      mode: 'save-file',
      suggestedName: location?.name ?? 'mrax-workspace.mrax',
      extensions: ['.mrax'],
      initial: location ? { root: location.root, path: location.parent } : undefined,
    });
    if (!destination) return;
    await task(async () => {
      const saved = await api<Workspace>('/api/workspaces/' + w.id + '/save-file', 'POST', {
        revision: w.revision,
        root: destination.root,
        path: destination.path,
        overwrite: destination.overwrite ?? false,
      });
      if (current.current?.id === w.id) await show(saved);
      setMessage('Workspace saved');
    });
  }
  async function shape(options: Options) {
    options = { ...drawOptions, ...options };
    const w = current.current;
    if (!w) return;
    const selectedRoi = w.selected_roi ?? w.rois.find((roi) => roi.enabled)?.name;
    if (options.edit) {
      const edited = { ...options };
      delete edited.edit;
      await action('roi.geometry', edited);
    } else if (options.tool === 'brush') {
      options.mode = roiDrawingMode(
        options.erase ? 'erase' : 'brush',
        options,
        Boolean(selectedRoi),
      );
      delete options.brush_mode;
      if (options.mode === 'new') {
        await action('roi.create', { ...options, name: nextRoiName(w.rois) });
      } else if (selectedRoi) {
        await action('roi.brush', {
          ...options,
          name: selectedRoi,
          erase: Boolean(options.erase) || options.mode === 'subtract',
        });
      } else setError('Select an ROI to edit.');
    } else if (['add', 'subtract'].includes(String(options.mode))) {
      await action('roi.draw', { ...options, name: selectedRoi });
    } else await action('roi.create', { name: nextRoiName(w.rois), ...options });
  }

  function openImport() {
    setImportIntoCurrent(Boolean(workspace && !workspace.loaded));
    setImportOpen(true);
    setPanel(null);
  }
  async function closeStudy(w: Workspace) {
    await task(async () => {
      await api(`/api/workspaces/${w.id}`, 'DELETE', { revision: w.revision, discard: true });
      setPipelineDrafts((previous) =>
        Object.fromEntries(Object.entries(previous).filter(([key]) => JSON.parse(key)[0] !== w.id)),
      );
      setPipelineRuns((previous) =>
        Object.fromEntries(Object.entries(previous).filter(([key]) => JSON.parse(key)[0] !== w.id)),
      );
      if (tutorial?.studyId === w.id) setTutorial(null);
      clearImageCache(w.id);
      navigationCache.clear(w.id);
      if (selected.current === w.id) {
        stopPrefetch();
        navigationVersion.current += 1;
        navigationRun.current = null;
        setNavigating(false);
        setPendingIndices({});
      }
      const remaining = await api<Workspace[]>('/api/workspaces');
      setWorkspaces(remaining);
      if (selected.current === w.id) {
        if (remaining.length) await show(remaining[remaining.length - 1]);
        else {
          selected.current = null;
          current.current = null;
          displayedContext.current = null;
          setWorkspace(null);
          setValues(null);
        }
      }
      setMessage('Study closed');
      restoreStudyFocus.current = true;
    });
  }
  async function createStudy(kind: NewStudyKind) {
    await task(async () => {
      viewingContext.current = { section: 'image', task: 'display' };
      if (kind === 'empty') {
        await show(await api<Workspace>('/api/workspaces/new', 'POST'));
        selectWorkflow('image', 'metadata');
        setMetadataCategory('overview');
      } else {
        const query = kind === 'multimodal' ? '?kind=multimodal' : '';
        await show(await api<Workspace>(`/api/workspaces/synthetic${query}`, 'POST'));
        setWorkflow('image');
        setWorkflowTasks((previous) => ({ ...previous, image: 'display' }));
        setPanel(null);
        setMessage(
          kind === 'multimodal'
            ? 'Synthetic multimodal study loaded'
            : 'Synthetic T2 study loaded. Reference T2 = 80 ms',
        );
      }
    }, true);
  }
  function selectWorkflow(value: Workflow, selectedTask: WorkflowTask = workflowTasks[value]) {
    if (selectedTask === 'signal') setSignalOpened(true);
    setWorkflow(value);
    setWorkflowTasks((previous) => ({ ...previous, [value]: selectedTask }));
    if (value !== 'image') viewingContext.current = { section: value, task: selectedTask };
    const w = current.current;
    if (selectedTask === 'overlays' && w?.scene.comparison.enabled)
      void changeView({ scene: { comparison: { ...w.scene.comparison, enabled: false } } });
    if (value === 'rois' && w?.loaded && (w.scene.comparison.enabled || w.view !== 'Image'))
      displayedContext.current = null;
    if (
      value !== 'image' &&
      w?.loaded &&
      viewContextKey(w, viewingContext.current) !== displayedContext.current
    )
      void task(async () => {
        await show(w, true);
      });
    if (selectedTask === 'metadata') {
      setPanel(null);
      setError('');
    } else if (!large) setPanel('tools');
  }

  const primarySignal = Boolean(
    workspace?.non_imaging &&
    !['metadata', 'signal'].includes(workflowTask) &&
    workspace.view === 'Image' &&
    !workspace.scene.comparison.enabled &&
    workflow !== 'reconstruct',
  );
  const signalActive = workflowTask === 'signal' || primarySignal;
  const followProcessing = primarySignal && workflow !== 'image';
  const displayedSignalOptions = followProcessing
    ? { ...signalOptions, source: 'profile' as const, residuals: false }
    : signalOptions;
  const jobStudyLabel = workspaces.find((item) => item.id === job?.workspace_id)?.label;
  const currentTimecourse =
    timecourse &&
    timecourse.workspace_id === workspace?.id &&
    timecourse.experiment === workspace.experiment &&
    timecourse.revision === workspace.revision
      ? timecourse
      : null;
  const scientificSignal = useMemo(
    () =>
      signalData(workspace, currentTimecourse, {
        source: followProcessing ? 'profile' : signalOptions.source,
        residuals: followProcessing ? false : signalOptions.residuals,
      }),
    [workspace, currentTimecourse, signalOptions.source, signalOptions.residuals, followProcessing],
  );
  function openSignalViewer(source: SignalOptions['source']) {
    updateSignal({ source, residuals: false, hidden: [], xMin: '', xMax: '', yMin: '', yMax: '' });
    selectWorkflow('image', 'signal');
  }
  const studies = (
    <StudyBrowser
      workspace={workspace}
      workspaces={workspaces}
      busy={busy}
      connected={connected}
      running={Boolean(active(job))}
      importButton={importButton}
      exportButton={exportButton}
      settingsButton={settingsButton}
      newStudyButton={newStudyButton}
      onSettings={() => setSettingsOpen(true)}
      onNew={() => {
        setPanel(null);
        setError('');
        setNewStudyOpen(true);
      }}
      onOpen={() => void openWorkspace()}
      onImport={openImport}
      onExport={() => {
        setPanel(null);
        setError('');
        setExportOpen(true);
      }}
      onSave={() => void save()}
      onStudy={(id) =>
        void task(async () => {
          await show(await api<Workspace>(`/api/workspaces/${id}`));
          setPanel(null);
        })
      }
      onExperiment={(id, name) => {
        setPanel(null);
        if (current.current?.id === id) void changeView({ experiment: name });
        else
          void task(async () => {
            const target = await api<Workspace>(`/api/workspaces/${id}`);
            await show(
              await api<Workspace>(`/api/workspaces/${id}/view`, 'PATCH', {
                revision: target.revision,
                experiment: name,
              }),
            );
          });
      }}
      onClose={(study, trigger) => {
        closeStudyTrigger.current = trigger;
        setPanel(null);
        if (
          study.dirty ||
          Object.entries(pipelineDrafts).some(
            ([key, draft]) => JSON.parse(key)[0] === study.id && draftChanged(draft),
          )
        )
          setCloseTarget(study);
        else void closeStudy(study);
      }}
    />
  );

  const dimensionControls = workspace && (
    <div className="dimension-controls" hidden={workspace.scene.comparison.enabled}>
      {workspace.dimension_controls
        .filter((control) => !primarySignal || control.key !== workspace.timecourse?.dimension)
        .map((control) => {
          const key = String(control.key),
            size = Number(control.size),
            value = Number(pendingIndices[key] ?? workspace.indices[key] ?? 0);
          const name =
            (key === 'frame' || key === 't') && workspace.model === 't2_se'
              ? 'Echo'
              : key === workspace.slice_dim
                ? 'Slice'
                : String(control.label ?? key);
          if (key === 'slice_package') {
            const choices = (control.choices ?? []) as {
              value: number;
              label: string;
            }[];
            return (
              <label key={key}>
                Slice package
                <select
                  aria-label="Slice package"
                  value={value}
                  disabled={taskBusy}
                  onChange={(event) => void navigateIndex(key, Number(event.target.value))}
                >
                  {choices.map((choice) => (
                    <option key={choice.value} value={choice.value}>
                      {choice.label}
                    </option>
                  ))}
                </select>
              </label>
            );
          }
          return (
            <label key={key}>
              {name}{' '}
              <strong>
                {value + 1} / {size}
              </strong>
              <input
                aria-label={name}
                type="range"
                min={0}
                max={Math.max(0, size - 1)}
                value={value}
                disabled={taskBusy}
                onChange={(event) => void navigateIndex(key, Number(event.target.value))}
              />
            </label>
          );
        })}
    </div>
  );

  const tools = (
    <AnalysisPanel
      section={workflow}
      openWorkflows={(kind, draft) => {
        setPanel(null);
        setWorkflowRequest({ kind, draft });
      }}
      task={workflowTask}
      onJobChange={setJob}
      onTimecourseResult={setTimecourse}
      signalPrimary={primarySignal}
      signalComparison={
        workflow === 'image' && signalActive && displayedSignalOptions.source === 'volume'
      }
      signalSourceControls={
        (signalOpened || primarySignal) && (
          <Suspense fallback={<p>Loading signal controls…</p>}>
            <SignalSourceControls
              workspace={workspace}
              options={displayedSignalOptions}
              update={updateSignal}
              busy={busy || Boolean(active(job))}
              changeView={changeView}
            />
          </Suspense>
        )
      }
      signalDisplayControls={
        (signalOpened || primarySignal) && (
          <Suspense fallback={null}>
            <SignalAppearanceControls
              data={scientificSignal}
              options={displayedSignalOptions}
              update={updateSignal}
            />
          </Suspense>
        )
      }
      jobStudyLabel={jobStudyLabel}
      editingStep={controlStep}
      stagePipelineOptions={stagePipelineOptions}
      cancelPipelineEdit={() => setEditingPipeline(null)}
      editorFormKey={editorKey + ':' + (editorOptions?.form ?? 0)}
      previewRoiOpacity={previewRoiOpacity}
      displayControls={
        <ImageControls
          workspace={workspace}
          task={workflowTasks.image}
          busy={busy}
          changeView={changeView}
          comparisonControlsRef={setComparisonControlsHost}
          action={action}
          previewRoiOpacity={previewRoiOpacity}
          previewOpacity={(id, value) =>
            setScenePreview((previous) =>
              id === null
                ? { ...previous, roiOpacity: value }
                : { ...previous, layers: { ...previous.layers, [id]: value } },
            )
          }
        />
      }
      workspace={workspace}
      busy={busy || Boolean(editingStep && !controlStep)}
      job={job}
      action={action}
      changeView={editorChangeView}
      run={run}
      save={save}
      tool={tool}
      brushSize={brushSize}
      label={label}
      drawOptions={drawOptions}
      setDrawOptions={setDrawOptions}
      setTool={(value) => {
        setTool(value);
        setPanel(null);
      }}
      setBrushSize={setBrushSize}
      setLabel={setLabel}
    />
  );

  const experimentLabel = (name: string) => {
    const type = workspace?.experiment_catalog.find((entry) => entry.name === name)?.exp_type;
    return type ? `${name} · ${type}` : name;
  };

  return (
    <div
      className={'app' + (inspector.dragging || browser.dragging ? ' resizing-controls' : '')}
      style={
        {
          '--tools-width': inspector.width + 'px',
          '--browser-width': browser.width + 'px',
          '--study-column-width': browser.width + 'px',
          '--study-divider-width': '8px',
        } as CSSProperties
      }
    >
      <div className={`app-layout${workflowTask === 'metadata' ? ' metadata-open' : ''}`}>
        <header className="app-header" aria-label="MRax">
          <StudyHeader />
          <div className="header-workflows">
            <WorkflowNav value={workflow} tasks={workflowTasks} onChange={selectWorkflow} />
            <span className="header-drag-space" aria-hidden="true" />
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button aria-label="Dismiss error" onClick={() => setError('')}>
              ×
            </button>
          </div>
        )}
        {studies}
        <StudyDivider layout={browser} />
        <main
          className={
            signalActive
              ? primarySignal
                ? 'signal-active spectrum-active'
                : 'signal-active'
              : workflowTask === 'metadata'
                ? 'metadata-active'
                : workspace?.loaded
                  ? 'pipeline-active'
                  : undefined
          }
        >
          <div className="workspace-heading">
            <div>
              <h1>{workspace?.label ?? 'No study open'}</h1>
              <p className="muted">
                {workspace
                  ? workspace.scene.comparison.enabled &&
                    !['metadata', 'signal'].includes(workflowTask)
                    ? `${workspace.scene.comparison.mode === 'overlay' ? 'Overlay' : 'Split view'} · ${experimentLabel(workspace.scene.comparison.left.experiment)} / ${experimentLabel(workspace.scene.comparison.right.experiment)}`
                    : `${experimentLabel(workspace.experiment || 'Add an experiment')} · ${workspace.dimensions.slice_package ? `${workspace.dimensions.slice_package} packages · ` : ''}${workspace.frame_count / (workspace.dimensions.slice_package || 1)} frames · ${workspace.slice_count} ${workspace.slice_count === 1 ? 'slice' : 'slices'}`
                  : 'Open a workspace or import image data.'}
              </p>
            </div>
            <div className="compact-actions">
              {!large && workflowTask !== 'metadata' && (
                <button className="secondary" onClick={() => setPanel('tools')}>
                  Controls
                </button>
              )}
            </div>
          </div>
          {workspace?.loaded && !['metadata', 'signal'].includes(workflowTask) && (
            <Pipeline
              workspace={workspace}
              busy={busy}
              job={job}
              draft={pipelineDraft}
              onDraft={setPipelineDraft}
              apply={applyPipelineDraft}
              cancel={async (id) => {
                setJob(await api<Job>(`/api/jobs/${id}/cancel`, 'POST'));
              }}
              pipelineJob={pipelineRuns[pipelineContext] ?? null}
              run={run}
              edit={editPipelineStep}
              editing={editingStep?.id ?? null}
              cancelEdit={() => setEditingPipeline(null)}
              compare={(comparison) => {
                selectWorkflow('image', 'display');
                void changeView({
                  scene: { comparison: comparison ?? originalComparison(workspace) },
                });
              }}
            />
          )}
          <MetadataBrowser
            workspace={workspace}
            active={workflowTask === 'metadata'}
            busy={busy || Boolean(active(job))}
            category={metadataCategory}
            onCategoryChange={setMetadataCategory}
            apply={async (name, options) => {
              const currentWorkspace = current.current;
              if (!currentWorkspace) throw Error('No study is open.');
              let updated = currentWorkspace;
              await task(async () => {
                updated = await api<Workspace>(
                  `/api/workspaces/${currentWorkspace.id}/actions`,
                  'POST',
                  {
                    revision: currentWorkspace.revision,
                    view_revision: currentWorkspace.view_revision,
                    action: name,
                    options,
                  },
                );
                await show(updated);
              }, true);
              return updated;
            }}
          />
          {primarySignal && <div className="signal-navigation">{dimensionControls}</div>}
          {(signalOpened || primarySignal) && (
            <Suspense fallback={<p>Loading signal viewer…</p>}>
              <SignalViewer
                active={signalActive}
                data={scientificSignal}
                options={displayedSignalOptions}
                update={updateSignal}
                figureRef={signalFigure}
              />
            </Suspense>
          )}
          <ResizablePreview
            className="image-preview"
            label="Resize image viewer"
            bodyId="image-viewer-body"
            storageKey="mrax.imageHeight"
            minimum={360}
            maximum={2400}
            growWithContent
            hidden={signalActive || workflowTask === 'metadata'}
          >
            <section
              id="image-viewer-body"
              className="viewer-panel"
              aria-label="Image viewer"
              aria-busy={navigating}
            >
              <div className="viewer-toolbar">
                <span>
                  {viewingContext.current.section === 'reconstruct'
                    ? 'Reconstruction'
                    : viewingContext.current.section === 'fit'
                      ? 'Fitting'
                      : viewingContext.current.section === 'rois'
                        ? 'Segmentation'
                        : 'Image viewer'}
                </span>
                <span className="native-badge">
                  {workspace?.scene.comparison.enabled
                    ? 'Read-only'
                    : workspace?.editable
                      ? 'Native grid'
                      : 'Display view'}
                </span>
              </div>
              {workspace && (values || workspace.scene.comparison.enabled) ? (
                <>
                  {workspace.scene.comparison.enabled ? (
                    <ComparisonView
                      key={`${workspace.id}-${workspace.experiment}`}
                      workspace={workspace}
                      busy={busy}
                      processing={taskBusy}
                      pendingIndices={pendingIndices}
                      navigateIndex={navigateIndex}
                      display={display}
                      controls={viewControls}
                      advancedControls={(children) =>
                        comparisonControlsHost
                          ? createPortal(children, comparisonControlsHost)
                          : null
                      }
                      change={changeView}
                    />
                  ) : (
                    <Viewer
                      workspace={workspace}
                      controls={viewControls}
                      values={values!}
                      layerValues={layerValues}
                      scenePreview={scenePreview}
                      display={display}
                      editingEnabled={workflowTask === 'roi-edit'}
                      drawing={drawing}
                      tool={tool}
                      erasing={drawingMode === 'subtract'}
                      brushSize={brushSize}
                      label={label}
                      onShape={shape}
                      busy={busy}
                      onRectangle={addRectangle}
                    />
                  )}
                  {!primarySignal && dimensionControls}
                </>
              ) : (
                <div className="empty-view">
                  <h2>{busy ? 'Loading study…' : 'No image selected'}</h2>
                  <p>Open a workspace or import an acquisition from Data &amp; Studies.</p>
                </div>
              )}
            </section>
          </ResizablePreview>
          <SignalPreview
            hidden={
              signalActive ||
              (workflowTask !== 'timecourse' &&
                (['metadata', 'signal'].includes(workflowTask) ||
                  viewingContext.current.section === 'reconstruct' ||
                  Boolean(workspace?.scene.comparison.enabled)))
            }
          >
            <TimecoursePlot
              styles={workspace?.scene.roi_styles}
              result={currentTimecourse}
              hidden={workflowTask !== 'timecourse' || !currentTimecourse}
              onOpen={() => openSignalViewer('volume')}
            />
            <Curve
              workspace={workspace}
              busy={busy}
              onAxisChange={(axis) => {
                updateSignal(automaticSignalAxes);
                void changeView({ plot_axis: axis });
              }}
              onRoiChange={(roi) => {
                updateSignal({ source: 'profile', ...automaticSignalAxes });
                void changeView(roi ? { selected_roi: roi } : { plot_whole_image: true });
              }}
              hidden={
                (workflowTask === 'timecourse' && Boolean(currentTimecourse)) ||
                workflowTask === 'metadata' ||
                signalActive ||
                viewingContext.current.section === 'reconstruct' ||
                workspace?.scene.comparison.enabled
              }
              onOpen={() => openSignalViewer(workspace?.preview ? 'fit' : 'profile')}
            />
          </SignalPreview>
          <span className="visually-hidden" role="status" aria-label="Workspace status">
            {navigating ? 'Updating image…' : busy ? 'Working…' : message}
          </span>
        </main>
        {large && workflowTask !== 'metadata' && <InspectorDivider layout={inspector} />}
        {large && tools}
      </div>
      <FileDialog
        open={importOpen}
        onClose={() => setImportOpen(false)}
        onAfterClose={() => {
          focusDataControl(importButton.current);
        }}
        workspace={importIntoCurrent ? workspace : null}
        currentStudy={workspace}
        onDestinationChange={setImportIntoCurrent}
        onError={setError}
        onJob={(next) => {
          setJob(next);
          void task(async () => show(await api<Workspace>(`/api/workspaces/${next.workspace_id}`)));
        }}
      />
      <ExportDialog
        jobStudyLabel={jobStudyLabel}
        onJobChange={setJob}
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        onAfterClose={() => {
          focusDataControl(exportButton.current);
        }}
        workspace={workspace}
        busy={busy || Boolean(active(job))}
        job={job}
        action={action}
        changeView={changeView}
        run={run}
        save={save}
        timecourse={currentTimecourse}
        signalExport={
          signalActive && scientificSignal
            ? {
                csv: () =>
                  signalCsv(
                    scientificSignal,
                    scientificSignal.traces.filter(
                      (trace) => !signalOptions.hidden.includes(trace.id),
                    ),
                  ),
                svg: () => serializeSignalFigure(signalFigure.current),
              }
            : null
        }
        error={error}
      />
      <NewStudyDialog
        open={newStudyOpen}
        onOpenChange={setNewStudyOpen}
        onAfterClose={() => focusDataControl(newStudyButton.current)}
        create={createStudy}
        disabled={busy || !connected}
      />
      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        trigger={settingsButton}
        theme={theme}
        setTheme={setTheme}
        resetPanels={() => {
          browser.reset();
          inspector.reset();
        }}
        accessibility={accessibility}
        setAccessibility={setAccessibility}
        bindings={bindings}
        setBindings={setBindings}
        tutorialActive={Boolean(tutorial)}
        startTutorial={() => void startTutorial()}
        endTutorial={() => void endTutorial()}
        disabled={busy || Boolean(active(job)) || !connected}
      />
      {workflowRequest && (
        <Suspense fallback={null}>
          <WorkflowDialog
            request={workflowRequest}
            workspace={workspace}
            onClose={() => setWorkflowRequest(null)}
            onUpdated={async (operation) => {
              setWorkspaces(await api<Workspace[]>('/api/workspaces'));
              if (current.current)
                await show(
                  await api<Workspace>('/api/workspaces/' + current.current.id),
                  false,
                  operation,
                );
            }}
          />
        </Suspense>
      )}
      {tutorial && (
        <Tutorial
          session={tutorial}
          workspace={workspace}
          context={{ section: workflow, task: workflowTask }}
          drawing={drawing}
          pipelineChanged={Boolean(pipelineDraft && draftChanged(pipelineDraft))}
          blocked={busy || Boolean(active(job))}
          canResume={workspaces.some((w) => w.id === tutorial.studyId)}
          onStep={(step, completed = false) =>
            setTutorial((previous) =>
              previous
                ? {
                    ...previous,
                    step,
                    completed: completed
                      ? [...new Set([...previous.completed, tutorialSteps[previous.step].id])]
                      : previous.completed,
                  }
                : null,
            )
          }
          onOpen={openTutorialTask}
          onResume={() => void resumeTutorial()}
          onEnd={() => void endTutorial()}
        />
      )}
      <Dialog.Root
        open={Boolean(closeTarget)}
        onOpenChange={(open) => {
          if (!open) setCloseTarget(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="drawer-overlay" />
          <Dialog.Content
            className="confirm-dialog"
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              focusDataControl(closeStudyTrigger.current);
            }}
          >
            <Dialog.Title>Discard unsaved changes?</Dialog.Title>
            <Dialog.Description>
              Save a workspace bundle before closing {closeTarget?.label} to keep your changes.
              {closeTarget &&
                Object.entries(pipelineDrafts).some(
                  ([key, draft]) => JSON.parse(key)[0] === closeTarget.id && draftChanged(draft),
                ) &&
                ' Unapplied pipeline edits will be discarded.'}
            </Dialog.Description>
            <div className="button-row">
              <Dialog.Close asChild>
                <button className="secondary">Cancel</button>
              </Dialog.Close>
              <button
                disabled={busy || active(job)}
                onClick={() => {
                  setCloseTarget(null);
                  if (closeTarget) void closeStudy(closeTarget);
                }}
              >
                Discard and close
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root
        open={panel === 'tools' && !large}
        onOpenChange={(open) => {
          if (!open) setPanel(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="drawer-overlay" />
          <Dialog.Content className="drawer">
            <Dialog.Title className="visually-hidden">Analysis controls</Dialog.Title>
            <Dialog.Description className="visually-hidden">
              Controls for the selected action.
            </Dialog.Description>
            <Dialog.Close className="drawer-close" aria-label="Close panel">
              <WorkbenchIcon name="close" />
            </Dialog.Close>
            {tools}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
