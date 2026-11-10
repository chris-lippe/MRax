import { test, expect, type APIRequestContext, type Page, type TestInfo } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { imageZoom, loadExample, openWorkspace, selectWorkflow, zoomImage } from './ui';
import type { Workspace } from '../src/api';

async function current(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json())[0] as Workspace;
}

async function fixture(page: Page, info: TestInfo) {
  const file = info.outputPath('viewing-context.mrax');
  await mkdir(dirname(file), { recursive: true });
  execFileSync('python', [
    '-c',
    `
import sys
import numpy as np
import jax.numpy as jnp
from mrax.analysis import ROI
from mrax.data_struct import AcquisitionData, Quantity
from mrax.reco import fftnc
from mrax.gui.session import GuiSession
from mrax.gui.services.analysis_workflows import run_operation
from mrax.gui.services.pipeline import snapshot
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.workspace_io import save_workspace_bundle

image = np.linspace(1, 6, 3 * 12 * 16, dtype=np.float32).reshape(3, 12, 16)
raw = AcquisitionData(np.asarray(fftnc(jnp.asarray(image[:, None] * 2))), (12, 16), ("phase", "read"))
session = GuiSession()
session.create_empty_study("Synthetic viewing contexts")
session.add_experiment("Phantom", image, dims={"frame": 3, "x": 12, "y": 16}, kspace=raw.samples)
session.current_experiment["acquisition"] = raw
session.study.datasets[0].acquisition = raw
workspace = StudyWorkspace(key="contexts", label=session.study_name, session=session)
run_operation(workspace, "reconstruct", {"view": "mrax_reconstruction", "component": "magnitude"}, lambda *_: None)
session = workspace.session
session.replace_fitted_maps({"t2": Quantity(np.linspace(20, 80, 12 * 16).reshape(12, 16), "ms"), "amplitude": Quantity(np.full((12, 16), 100.0), "1")})
session.current_experiment["fitted_map_units"] = {"t2": "ms", "amplitude": "a.u."}
state = snapshot(session)
grid = {key: state[key] for key in ("dims", "geometry", "unit")}
workspace.fit_results_by_experiment["Phantom"] = {"fields": session.fitted_maps, "pipeline_grid": grid, "pipeline_steps": []}
session.current_experiment["fit_grid"] = grid
mask = np.zeros((12, 16), dtype=bool)
mask[3:9, 4:12] = True
session.add_roi(ROI("Tissue", mask, dims=("x", "y")))
save_workspace_bundle(workspace, sys.argv[1])
`,
    file,
  ]);
  await page.goto('/');
  await openWorkspace(page, file);
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json())
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
});

test('advanced presentation controls belong to View while navigation and intensity remain in the center', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('/');
  await loadExample(page, 'multimodal');
  const before = await current(request);
  const center = page.getByRole('region', { name: 'Image viewer', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Analysis controls', exact: true });
  await expect(inspector.getByRole('group', { name: 'Viewing mode', exact: true })).toBeVisible();
  await expect(inspector.getByLabel('Colormap', { exact: true })).toBeVisible();
  await expect(center.getByRole('group', { name: 'Viewing mode', exact: true })).toHaveCount(0);
  await expect(center.getByLabel('Colormap', { exact: true })).toHaveCount(0);
  await inspector.getByRole('button', { name: 'Overlay', exact: true }).click();
  await expect(inspector.getByLabel('Image A', { exact: true })).toBeVisible();
  await expect(inspector.getByLabel('Overlay opacity', { exact: true })).toBeVisible();
  await expect(center.getByLabel('Image A', { exact: true })).toHaveCount(0);
  for (const [workflow, task] of [
    ['Preprocess', 'Denoising'],
    ['Segment', 'Create and edit ROIs'],
    ['Fit', undefined],
    ['Reconstruct', undefined],
  ] as const) {
    await selectWorkflow(page, workflow, task);
    await expect(center.getByLabel('Plane', { exact: true })).toBeVisible();
    await expect(center.getByLabel('Brightness', { exact: true })).toBeVisible();
    await expect(page.getByRole('slider', { name: 'Echo', exact: true })).toBeVisible();
    await expect(page.getByRole('slider', { name: 'Slice', exact: true })).toBeVisible();
    await expect(
      inspector.getByRole('group', { name: 'Viewing mode', exact: true }),
    ).not.toBeVisible();
    await expect(page.getByRole('slider', { name: 'Zoom', exact: true })).toHaveCount(0);
  }
  const after = await current(request);
  expect(after.revision).toBe(before.revision);
  expect(after.pipeline).toEqual(before.pipeline);
  await selectWorkflow(page, 'View', 'Image display');
  await page.screenshot({ path: info.outputPath('view-advanced.png') });
});

test('Fit opens maps and View edits the current map without disrupting the processing workflow', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await fixture(page, info);
  const before = await current(request);
  await selectWorkflow(page, 'Fit');
  const map = page.getByLabel('Map', { exact: true });
  await expect(map).toHaveValue('t2');
  await expect(page.getByRole('group', { name: 'Map scale', exact: true })).toContainText('ms');
  await map.selectOption('amplitude');
  await selectWorkflow(page, 'View', 'Image display');
  await expect(map).toHaveValue('amplitude');
  await page.getByLabel('Colormap', { exact: true }).selectOption('hot');
  await selectWorkflow(page, 'Fit');
  await expect(map).toHaveValue('amplitude');
  expect((await current(request)).display.colormap).toBe('hot');
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await expect(page.getByLabel('Active ROI', { exact: true })).toHaveValue('Tissue');
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  expect((await current(request)).view).toBe('Image');
  await selectWorkflow(page, 'Fit');
  await expect(map).toHaveValue('amplitude');
  expect((await current(request)).display.colormap).toBe('hot');
  await map.selectOption('t2');
  await page.screenshot({ path: info.outputPath('fit-map.png') });
  const after = await current(request);
  expect(after.revision).toBe(before.revision);
  expect(after.pipeline).toEqual(before.pipeline);
  expect(after.scene.fit_rois).toEqual(before.scene.fit_rois);
});

test('Reconstruct compares protected vendor images with named reconstructions and switches to raw views without processing', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await fixture(page, info);
  const before = await current(request);
  const jobs: string[] = [];
  page.on('request', (request) => {
    if (request.url().endsWith('/tasks')) jobs.push(request.url());
  });
  await selectWorkflow(page, 'Reconstruct');
  await expect(page.getByRole('img', { name: /^Image A:/ })).toBeVisible();
  let workspace = await current(request);
  expect(workspace.scene.comparison).toMatchObject({
    enabled: true,
    left: { kind: 'original' },
    right: { kind: 'reconstruction' },
  });
  const response = await request.get('/api/workspaces/' + workspace.id + '/comparison');
  const pixels = await response.body();
  const planeLength = pixels.length / 16;
  for (let index = 0; index < planeLength; index++)
    expect(pixels.readDoubleLE((planeLength + index) * 8)).toBeCloseTo(
      pixels.readDoubleLE(index * 8) * 2,
      5,
    );
  await page.getByRole('slider', { name: 'Frame', exact: true }).focus();
  await page.keyboard.press('End');
  await expect(page.getByRole('img', { name: /^Image A:.*frame 3$/ })).toBeVisible();
  const modes = page.getByRole('group', { name: 'Reconstruction display', exact: true });
  await modes.getByRole('button', { name: workspace.reconstruction_label, exact: true }).click();
  await expect(page.getByLabel('Reconstruction component', { exact: true })).toBeVisible();
  expect((await current(request)).frame).toBe(2);
  await page.getByLabel('Reconstruction component', { exact: true }).selectOption('phase');
  await selectWorkflow(page, 'View', 'Image display');
  await expect(page.getByLabel('Component', { exact: true })).toHaveValue('phase');
  await page.getByLabel('Component', { exact: true }).selectOption('magnitude');
  await selectWorkflow(page, 'Reconstruct');
  await expect(page.getByLabel('Reconstruction component', { exact: true })).toHaveValue(
    'magnitude',
  );
  await modes.getByRole('button', { name: 'K-space', exact: true }).click();
  await expect(page.getByLabel('K-space component', { exact: true })).toBeVisible();
  await page.getByLabel('K-space component', { exact: true }).selectOption('phase');
  await modes.getByRole('button', { name: 'Difference', exact: true }).click();
  await expect.poll(async () => (await current(request)).view).toBe('Vendor MRax Diff');
  await modes.getByRole('button', { name: 'Vendor comparison', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image B:/ })).toBeVisible();
  await page.screenshot({ path: info.outputPath('reconstruct-compare.png') });
  workspace = await current(request);
  expect(workspace.revision).toBe(before.revision);
  expect(workspace.pipeline).toEqual(before.pipeline);
  expect(jobs).toEqual([]);
});

test('zoom and pan work on the image and the pair without backend requests or accidental ROI edits', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  const before = await current(request);
  const calls: string[] = [];
  page.on('request', (request) => {
    if (/\/(view|slice|comparison|tasks|rois)(?:\?|$)/.test(request.url()))
      calls.push(request.url());
  });
  await zoomImage(page, 2);
  const surface = page.getByTestId('roi-surface');
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2 + 20, { steps: 6 });
  await page.mouse.up();
  await expect
    .poll(() =>
      page
        .locator('.image-surface')
        .evaluate((element) => new DOMMatrix(getComputedStyle(element).transform).e),
    )
    .toBeCloseTo(30, 0);
  await page.locator('.image-viewport').focus();
  await page.keyboard.press('0');
  await expect.poll(() => imageZoom(page)).toBe(1);
  expect(calls).toEqual([]);
  await selectWorkflow(page, 'Segment');
  await page.getByRole('button', { name: 'Draw rectangle', exact: true }).click();
  const draw = (await surface.boundingBox())!;
  await page.mouse.move(draw.x + draw.width / 2, draw.y + draw.height / 2);
  await page.keyboard.down('Alt');
  await page.mouse.down();
  await page.mouse.move(draw.x + draw.width / 2 + 20, draw.y + draw.height / 2 + 10, { steps: 5 });
  await page.mouse.up();
  await page.keyboard.up('Alt');
  expect((await current(request)).rois).toEqual(before.rois);
  await selectWorkflow(page, 'View', 'Image display');
  await page.getByRole('button', { name: 'Overlay', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:/ })).toBeVisible();
  calls.length = 0;
  await zoomImage(page, 2.5);
  const scales = await page
    .locator('.comparison-pane canvas')
    .evaluateAll((canvases) =>
      canvases.map((canvas) => new DOMMatrix(getComputedStyle(canvas).transform).a),
    );
  expect(scales[0]).toBeCloseTo(2.5, 2);
  expect(scales[1]).toBe(scales[0]);
  expect(calls).toEqual([]);
});
