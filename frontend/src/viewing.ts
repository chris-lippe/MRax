import {
  comparisonNavigation,
  originalComparison,
  type ComparisonSettings,
  type Workspace,
} from './api';
import type { Workflow, WorkflowTask } from './WorkflowNav';

export function viewLabel(workspace: Workspace, view: string) {
  if (view === 'MRax reconstruction') return workspace.reconstruction_label;
  if (view === 'Vendor vs MRax') return 'Vendor comparison';
  if (view === 'Vendor MRax Diff') return 'Reconstruction difference';
  return view;
}

export type ViewingContext = { section: Workflow; task: WorkflowTask };
export type SavedView = {
  view: string;
  map_name: string | null;
  component: string;
  colormap: string;
  scene: { comparison: ComparisonSettings };
};

export const viewContextKey = (workspace: Workspace, context: ViewingContext) =>
  JSON.stringify([
    workspace.id,
    workspace.experiment,
    context.section,
    context.section === 'analyze' ? context.task : '',
  ]);

export function captureView(workspace: Workspace): SavedView {
  return {
    view: workspace.view,
    map_name: workspace.active_map ?? null,
    component: String(workspace.display.component ?? 'auto'),
    colormap: String(workspace.display.colormap ?? 'gray'),
    scene: { comparison: workspace.scene.comparison },
  };
}

export function reconstructionComparison(workspace: Workspace): ComparisonSettings {
  const pair = originalComparison(workspace);
  return {
    ...pair,
    mode: 'split',
    right: { ...pair.right, kind: 'reconstruction' },
  };
}

export function contextView(
  workspace: Workspace,
  context: ViewingContext,
  remembered?: SavedView,
): SavedView {
  const current = captureView(workspace);
  const image = {
    ...current,
    view: 'Image',
    component: 'auto',
    colormap: 'gray',
    scene: { comparison: { ...current.scene.comparison, enabled: false } },
  };
  if (remembered) {
    const available =
      (context.section !== 'rois' ||
        (remembered.view === 'Image' && !remembered.scene.comparison.enabled)) &&
      (remembered.view !== 'Map' ||
        workspace.maps.some((map) => map.name === remembered.map_name)) &&
      (remembered.view !== 'K-space' || workspace.has_kspace) &&
      (!['MRax reconstruction', 'Vendor vs MRax', 'Vendor MRax Diff'].includes(remembered.view) ||
        workspace.has_reconstruction) &&
      (remembered.view !== 'Registration overlay' || workspace.registrations.length > 0) &&
      (!remembered.scene.comparison.enabled ||
        [remembered.scene.comparison.left, remembered.scene.comparison.right].every((source) =>
          workspace.scene.comparison_sources.some((choice) =>
            ['experiment', 'kind', 'map_name', 'registration_key'].every(
              (key) =>
                (choice[key as keyof typeof choice] ?? null) ===
                (source[key as keyof typeof source] ?? null),
            ),
          ),
        ));
    if (available) {
      const comparison = remembered.scene.comparison;
      if (!comparison.enabled) return remembered;
      const cursor = comparisonNavigation(workspace, comparison);
      return {
        ...remembered,
        scene: {
          comparison: {
            ...comparison,
            frame: cursor.frame,
            plane: cursor.plane,
            indices: cursor.indices,
          },
        },
      };
    }
  }
  if (context.section === 'image') return current;
  if (context.section === 'reconstruct' && workspace.has_reconstruction)
    return { ...image, scene: { comparison: reconstructionComparison(workspace) } };
  const maps =
    context.section === 'fit' || (context.section === 'analyze' && context.task === 'statistics');
  if (maps && workspace.maps.length)
    return {
      ...image,
      view: 'Map',
      map_name: workspace.active_map ?? String(workspace.maps[0].name),
      colormap: 'viridis',
    };
  return image;
}
