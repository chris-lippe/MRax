import type { Job, Options, PipelineStep, Tool, Workspace } from './api';
import type { WorkflowKind } from './reusableWorkflows';

export type AnalysisProps = {
  workspace: Workspace | null;
  busy: boolean;
  job: Job | null;
  action: (name: string, options?: Options) => Promise<void>;
  changeView: (command: Options) => Promise<void>;
  previewRoiOpacity?: (name: string, value: number) => void;
  run: (operation: string, options?: Options) => Promise<void>;
  save: () => Promise<void>;
  openWorkflows?: (kind: WorkflowKind, draft?: Options) => void;
  editingStep?: PipelineStep | null;
};
export type DrawingProps = {
  drawOptions: Options;
  setDrawOptions: (options: Options) => void;
  drawing: boolean;
  tool: Tool;
  brushSize: number;
  label: number;
  setDrawing: (drawing: boolean) => void;
  setTool: (tool: Tool) => void;
  setBrushSize: (size: number) => void;
  setLabel: (label: number) => void;
};
