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

test('processing workflows save, edit, share across studies and batch experiments without duplicating data', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  let releaseTargets = () => {};
  await page.route('**/api/workflow-targets', async (route) => {
    await new Promise<void>((resolve) => {
      releaseTargets = resolve;
    });
    await route.continue().catch(() => {});
  });
  const loadingTargets = page.waitForRequest('**/api/workflow-targets');
  await page.getByRole('button', { name: 'Processing workflows', exact: true }).click();
  await loadingTargets;
  await page.keyboard.press('Escape');
  await expect(dialog(page)).not.toBeVisible();
  releaseTargets();
  await page.unroute('**/api/workflow-targets');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await page.getByRole('button', { name: 'Processing workflows', exact: true }).click();
  const modal = dialog(page);
  await modal.getByRole('button', { name: 'Capture current pipeline', exact: true }).click();
  await modal.getByLabel('Workflow name', { exact: true }).fill('Shared Gaussian');
  await modal.getByRole('button', { name: 'Keep workflow changes', exact: true }).click();
  const saved = await saveWorkflow(page, info, 'denoise.mpip');
  expect(saved.document).toMatchObject({
    format: 'mrax.workflow',
    format_version: 1,
    kind: 'processing',
    name: 'Shared Gaussian',
  });
  expect(saved.document.mrax_version).toMatch(/^\d+\.\d+/);
  expect(saved.document.steps[0].settings.method).toBe('gaussian');
  expect(saved.document).not.toHaveProperty('arrays');
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await loadExample(page, 'multimodal');
  const before = await workspaces(request);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByRole('button', { name: 'Processing workflows', exact: true }).click();
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
  await modal.getByRole('button', { name: 'Apply processing workflow', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText('Workflow applied to all selected targets.');
  await expect(modal.locator('.workflow-run-report li')).toHaveCount(2);
  expect(await workspaces(request)).toHaveLength(2);
  await page.screenshot({ path: info.outputPath('processing-workflow-targets.png') });
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await selectExperiment(page, 'Complex coils');
  const result = (await workspaces(request)).find((row) => row.id === before[1].id)!;
  expect(result.pipeline.steps.map((step) => step.method)).toEqual(['gaussian']);
  expect(result.experiments).toEqual(before[1].experiments);
  await page.getByRole('button', { name: 'Processing workflows', exact: true }).click();
  await modal.getByRole('button', { name: /Shared Gaussian/ }).click();
  await modal.getByRole('button', { name: 'Edit definition', exact: true }).click();
  const editor = modal.getByLabel('Workflow JSON definition', { exact: true });
  await editor.fill(JSON.stringify({ ...saved.document, format_version: 99 }));
  await modal.getByRole('button', { name: 'Keep workflow changes', exact: true }).click();
  await expect(modal.getByRole('alert')).toContainText(/Unsupported|format_version/);
  await modal.getByRole('button', { name: 'Close workflows', exact: true }).click();
  await expect(modal).toBeVisible();
  await modal.getByRole('button', { name: 'Discard workflow edits', exact: true }).click();
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
  await page.getByRole('button', { name: 'Fitting workflows', exact: true }).click();
  const modal = dialog(page);
  await modal.getByRole('button', { name: 'Capture current configuration', exact: true }).click();
  const saved = await saveWorkflow(page, info, 't2.mfit');
  expect(saved.document.steps[0].settings.settings.num_steps).toBe(333);
  expect(saved.document.bindings).toHaveLength(1);
  expect(JSON.stringify(saved.document.steps)).not.toContain('ROI 1');
  expect(saved.document.steps[0].settings.roi_names).toEqual([{ $binding: 'input_1' }]);
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await loadExample(page);
  await drawRoi(page);
  await selectWorkflow(page, 'Fit');
  await page.getByRole('button', { name: 'Fitting workflows', exact: true }).click();
  await modal.getByRole('button', { name: /Fitting workflow/ }).click();
  await modal.getByRole('button', { name: 'Apply fitting configuration', exact: true }).click();
  await expect(modal.getByRole('alert')).toContainText('Resolve target compatibility');
  await modal.getByLabel('ROI 1 for T2 phantom', { exact: true }).selectOption('ROI 1');
  await modal.getByRole('button', { name: 'Apply fitting configuration', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText('Workflow applied to all selected targets.');
  let current = (await workspaces(request))[1];
  expect(current.fit_settings.num_steps).toBe(333);
  expect(current.maps).toEqual([]);
  await modal
    .getByRole('checkbox', { name: 'Run fits after loading configuration', exact: true })
    .check();
  await modal.getByRole('button', { name: 'Run fitting workflow', exact: true }).click();
  await expect(modal.getByRole('status')).toHaveText('Workflow applied to all selected targets.', {
    timeout: 60000,
  });
  current = (await workspaces(request))[1];
  expect(current.maps.some((map) => map.name === 't2')).toBe(true);
  expect(current.pipeline.steps).toEqual([]);
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await expect(page.getByLabel('Map', { exact: true })).toBeVisible();
});
