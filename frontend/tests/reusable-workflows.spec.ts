import { test, expect, type APIRequestContext, type Page, type TestInfo } from '@playwright/test';

import { mkdir, readFile } from 'node:fs/promises';

import { dirname } from 'node:path';

import { chooseFilePath, loadExample, selectExperiment, selectWorkflow } from './ui';

import type { Workspace } from '../src/api';

async function workspaces(request: APIRequestContext): Promise<Workspace[]> {
  return (await request.get('/api/workspaces')).json();
}

const dialog = (page: Page) =>
  page.getByRole('dialog', { name: 'Reusable workflows', exact: true });

async function saveWorkflow(page: Page, info: TestInfo, filename: string) {
  const path = info.outputPath(filename);
  await mkdir(dirname(path), { recursive: true });
  await dialog(page).getByRole('button', { name: 'Save workflow file', exact: true }).click();
  await chooseFilePath(page, path, 'Save');
  await expect(dialog(page).getByRole('status')).toHaveText('Workflow saved.');
  return { path, document: JSON.parse(await readFile(path, 'utf-8')) };
}

async function drawRoi(page: Page) {
  await selectWorkflow(page, 'Segment');
  const rectangle = page.getByRole('button', { name: 'Rectangle tool', exact: true });
  if ((await rectangle.getAttribute('aria-pressed')) !== 'true') await rectangle.click();
  const surface = page.getByTestId('roi-surface');
  await surface.scrollIntoViewIfNeeded();
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.65, { steps: 6 });
  await page.mouse.up();
  await expect(page.getByRole('button', { name: 'Rectangle tool', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.locator('.roi-mask .roi-outline')).toBeVisible();
}

test.afterEach(async ({ request }) => {
  const headers = { 'x-mrax-token': (await (await request.get('/api/session')).json()).token };
  for (const workspace of await workspaces(request)) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers,
      data: { revision: workspace.revision, discard: true },
    });
  }
  for (const workflow of await (await request.get('/api/workflows')).json()) {
    await request.delete('/api/workflows/' + workflow.id + '?revision=' + workflow.revision, {
      headers,
    });
  }
});

test('built-in presets preview requirements, create editable copies and check target inputs', async ({
  page,
  request,
}) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Denoise');
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  const modal = dialog(page);
  await modal.getByLabel('Search workflows', { exact: true }).fill('light spatial');
  await modal.getByRole('button', { name: /^Light spatial smoothing/ }).click();
  await expect(modal.getByRole('region', { name: 'Preset preview', exact: true })).toContainText(
    'Acquisition dimensions and slice packages stay independent',
  );
  await modal.getByRole('button', { name: 'Use preset', exact: true }).click();
  await expect(modal.getByRole('status')).toContainText('Preset copied');
  await modal.getByRole('button', { name: 'Edit definition', exact: true }).click();
  const editor = modal.getByLabel('Workflow JSON definition', { exact: true });
  const definition = JSON.parse(await editor.inputValue());
  definition.steps[0].settings.parameters.sigma = 1.2;
  await editor.fill(JSON.stringify(definition));
  await modal.getByRole('button', { name: 'Keep workflow changes', exact: true }).click();
  await modal.getByRole('button', { name: 'Check inputs', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText(
    'Inputs are ready for the selected experiments.',
  );
  expect((await workspaces(request))[0].pipeline.steps).toHaveLength(0);
  await modal.getByRole('button', { name: 'Apply workflow', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText('Workflow applied to all selected targets.');
  expect((await workspaces(request))[0].pipeline.steps).toHaveLength(1);
  const catalog = await (await request.get('/api/workflow-presets')).json();
  expect(
    catalog.find((row: { id: string }) => row.id === 'spatial-gaussian').document.steps[0].settings
      .parameters.sigma,
  ).toBe(0.6);
  await modal.getByLabel('Workflow category', { exact: true }).selectOption('registration');
  await modal.getByRole('button', { name: 'Presets', exact: true }).click();
  await modal.getByLabel('Search workflows', { exact: true }).fill('masked');
  await modal.getByRole('button', { name: /^Masked rigid alignment/ }).click();
  await modal.getByRole('button', { name: 'Use preset', exact: true }).click();
  await modal.getByRole('button', { name: 'Check inputs', exact: true }).click();
  await expect(modal.getByRole('alert')).toContainText('Resolve target compatibility');
  await expect(modal.locator('.workflow-run-report')).toContainText('Assign Reference experiment');
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
});

test('processing workflows save, edit, share across studies and batch experiments without duplicating data', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Denoise');
  let releaseTargets = () => {};
  await page.route('**/api/workflow-targets', async (route) => {
    await new Promise<void>((resolve) => {
      releaseTargets = resolve;
    });
    await route.continue().catch(() => {});
  });
  const loadingTargets = page.waitForRequest('**/api/workflow-targets');
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  await loadingTargets;
  await page.keyboard.press('Escape');
  await expect(dialog(page)).not.toBeVisible();
  releaseTargets();
  await page.unroute('**/api/workflow-targets');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  const modal = dialog(page);
  await modal.getByRole('button', { name: 'Capture current pipeline', exact: true }).click();
  await modal.getByLabel('Workflow name', { exact: true }).fill('Shared Gaussian');
  await modal.getByRole('button', { name: 'Keep workflow changes', exact: true }).click();
  const saved = await saveWorkflow(page, info, 'denoise.mflow');
  expect(saved.document).toMatchObject({
    format: 'mrax.workflow',
    kind: 'denoising',
    name: 'Shared Gaussian',
  });
  expect(saved.document.mrax_version).toMatch(/^\d+\.\d+/);
  expect(saved.document.steps[0].settings.method).toBe('gaussian');
  expect(saved.document).not.toHaveProperty('arrays');
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await loadExample(page, 'multimodal');
  const before = await workspaces(request);
  await selectWorkflow(page, 'Denoise');
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  await modal.getByRole('button', { name: /Shared Gaussian/ }).click();
  await modal
    .getByRole('checkbox', { name: 'Target T2 phantom in Synthetic T2 phantom', exact: true })
    .check();
  await modal
    .getByRole('checkbox', {
      name: 'Target Complex coils in Synthetic multimodal study',
      exact: true,
    })
    .check();
  await modal.getByRole('button', { name: 'Apply workflow', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText('Workflow applied to all selected targets.');
  await expect(modal.locator('.workflow-run-report li')).toHaveCount(2);
  expect(await workspaces(request)).toHaveLength(2);
  await page.screenshot({ path: info.outputPath('processing-workflow-targets.png') });
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await selectExperiment(page, 'Complex coils');
  const result = (await workspaces(request)).find((row) => row.id === before[1].id)!;
  expect(result.pipeline.steps.map((step) => step.method)).toEqual(['gaussian']);
  expect(result.experiments).toEqual(before[1].experiments);
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  await modal.getByRole('button', { name: /Shared Gaussian/ }).click();
  await modal.getByRole('button', { name: 'Edit definition', exact: true }).click();
  const editor = modal.getByLabel('Workflow JSON definition', { exact: true });
  await editor.fill(JSON.stringify({ ...saved.document, mrax_version: '9999.0.0' }));
  await modal.getByRole('button', { name: 'Keep workflow changes', exact: true }).click();
  await expect(modal.getByRole('alert')).toContainText(/version incompatibility|mrax_version/i);
  await modal.getByRole('button', { name: 'Close workflows', exact: true }).click();
  await expect(modal).toBeVisible();
  await modal.getByRole('button', { name: 'Discard workflow edits', exact: true }).click();
  await expect(modal.getByRole('alert')).toHaveCount(0);
  await modal.getByRole('button', { name: 'Remove from session', exact: true }).click();
  await modal.getByRole('button', { name: 'Open workflow file', exact: true }).click();
  await chooseFilePath(page, saved.path);
  await expect(modal.getByRole('heading', { name: 'Shared Gaussian', exact: true })).toBeVisible();
});

test('fitting workflow captures pending solver settings and requires ROI assignment on a new study', async ({
  page,
  request,
}, info) => {
  test.setTimeout(120000);
  await page.goto('/');
  await loadExample(page);
  await drawRoi(page);
  await selectWorkflow(page, 'Fit');
  await page.getByLabel('Fit mask ROI 1', { exact: true }).uncheck();
  await page.getByLabel('Fit mask ROI 1', { exact: true }).check();
  await page.getByLabel('Fit steps', { exact: true }).fill('333');
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  const modal = dialog(page);
  await modal.getByRole('button', { name: 'Capture current configuration', exact: true }).click();
  const saved = await saveWorkflow(page, info, 't2.mflow');
  expect(saved.document.steps[0].settings.settings.num_steps).toBe(333);
  expect(saved.document.bindings).toHaveLength(1);
  expect(JSON.stringify(saved.document.steps)).not.toContain('ROI 1');
  expect(saved.document.steps[0].settings.roi_names).toEqual([{ $binding: 'input_1' }]);
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await loadExample(page);
  await drawRoi(page);
  await selectWorkflow(page, 'Fit');
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  await modal.getByRole('button', { name: /Fitting workflow/ }).click();
  await modal.getByRole('button', { name: 'Load fitting settings', exact: true }).click();
  await expect(modal.getByRole('alert')).toContainText('Resolve target compatibility');
  await modal.getByLabel('ROI 1 for T2 phantom', { exact: true }).selectOption('ROI 1');
  await modal.getByLabel('Workflow name', { exact: true }).fill('Bound T2 fit');
  await modal.getByRole('button', { name: 'Keep workflow changes', exact: true }).click();
  await expect(modal.getByLabel('ROI 1 for T2 phantom', { exact: true })).toHaveValue('ROI 1');
  await modal.getByRole('button', { name: 'Load fitting settings', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText('Workflow applied to all selected targets.');
  let current = (await workspaces(request))[1];
  expect(current.fit_settings.num_steps).toBe(333);
  expect(current.maps).toEqual([]);
  expect(current.pipeline.steps).toHaveLength(1);
  expect(current.pipeline.steps[0].pending).toBe(true);
  const fitId = current.pipeline.steps[0].id;
  await modal
    .getByRole('checkbox', { name: 'Run fits after loading configuration', exact: true })
    .check();
  await modal.getByRole('button', { name: 'Apply workflow', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText('Workflow applied to all selected targets.', {
    timeout: 60000,
  });
  current = (await workspaces(request))[1];
  expect(current.maps.some((map) => map.name === 't2')).toBe(true);
  expect(current.pipeline.steps.map((step) => step.kind)).toEqual(['fit']);
  expect(current.pipeline.steps[0].id).toBe(fitId);
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await expect(page.getByLabel('Map', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  expect((await workspaces(request))[1].pipeline.steps).toHaveLength(1);
  expect((await workspaces(request))[1].pipeline.steps[0].id).toBe(fitId);
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  const rows = pipeline.locator('.pipeline-step');
  await expect(rows).toHaveCount(1);
  await expect(rows.first().getByRole('button', { name: /^Copy / })).toBeDisabled();
  await rows.first().getByRole('button', { name: 'Edit', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'Fit', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await expect(page.getByLabel('Fit steps', { exact: true })).toHaveValue('333');
  await page.getByLabel('Fit steps', { exact: true }).fill('334');
  await page.getByRole('button', { name: 'Apply fit configuration', exact: true }).click();
  await expect(pipeline).toContainText('Unapplied changes');
  expect((await workspaces(request))[1].fit_settings.num_steps).toBe(333);
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await expect.poll(async () => (await workspaces(request))[1].fit_settings.num_steps).toBe(334);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  current = (await workspaces(request))[1];
  expect(current.pipeline.steps).toHaveLength(1);
  expect(current.pipeline.steps[0].id).toBe(fitId);
  expect(current.maps.some((map) => map.name === 't2')).toBe(true);
  await rows.first().getByRole('button', { name: 'Edit', exact: true }).click();
  await page.getByLabel('Fit steps', { exact: true }).fill('335');
  await expect(pipeline).toContainText('Unapplied changes');
  await pipeline.getByRole('button', { name: 'Discard changes', exact: true }).click();
  expect((await workspaces(request))[1].fit_settings.num_steps).toBe(334);
  await expect(page.getByLabel('Fit steps', { exact: true })).toHaveValue('334');
  await page.screenshot({ path: info.outputPath('fitting-controls.png') });
});
