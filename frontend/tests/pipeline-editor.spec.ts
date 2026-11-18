import { openControlGroup } from './ui';
import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

import type { Workspace } from '../src/api';

import { loadExample, openWorkspace, saveWorkspace, selectWorkflow } from './ui';

async function current(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json())[0] as Workspace;
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await (await request.get('/api/workspaces')).json())
    await request.delete('/api/workspaces/' + w.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
});

const panel = (page: Page) =>
  page.getByRole('region', { name: 'Processing pipeline', exact: true });

async function expand(page: Page) {
  const open = panel(page).getByRole('button', { name: 'Show processing steps', exact: true });
  if (await open.isVisible()) await open.click();
}

async function setup(page: Page, request: APIRequestContext, baseline = true) {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Denoise');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.6');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect.poll(async () => (await current(request)).pipeline.steps.length).toBe(1);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  if (baseline) {
    await selectWorkflow(page, 'Analyze', 'Signal and time series');
    await openControlGroup(page, 'Time-series corrections');
    await page.getByLabel('Time-series correction', { exact: true }).selectOption('baseline');
    await page.getByRole('button', { name: 'Apply baseline correction', exact: true }).click();
    await expect.poll(async () => (await current(request)).pipeline.steps.length).toBe(2);
    await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  }
  await expand(page);
}

test('chain edits stay local across navigation, can be discarded and apply in a single transaction', async ({
  page,
  request,
}, info) => {
  await setup(page, request);
  const before = await current(request);
  const image = await (await request.get('/api/workspaces/' + before.id + '/slice')).body();
  const calls: string[] = [];
  page.on('request', (request) => {
    if (request.method() === 'POST' && request.url().endsWith('/tasks'))
      calls.push(request.postData() ?? '');
  });
  const pipeline = panel(page),
    rows = pipeline.locator('.pipeline-step');
  await rows.nth(0).getByRole('button', { name: 'Copy Gaussian denoising', exact: true }).click();
  await expect(rows).toHaveCount(3);
  await rows
    .nth(1)
    .getByRole('button', { name: 'Move Gaussian denoising later', exact: true })
    .click();
  await rows.nth(0).getByRole('button', { name: 'Remove', exact: true }).click();
  await rows
    .nth(0)
    .getByRole('button', { name: 'Insert after Baseline correction', exact: true })
    .click();
  await pipeline.getByLabel('New pipeline step', { exact: true }).selectOption('median');
  await pipeline.getByRole('button', { name: 'Insert into draft', exact: true }).click();
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(1)).toHaveAttribute('aria-label', 'Median filtering');
  await expect(page.getByRole('tab', { name: 'Denoise', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  expect((await current(request)).pipeline).toEqual(before.pipeline);
  expect(await (await request.get('/api/workspaces/' + before.id + '/slice')).body()).toEqual(
    image,
  );
  expect(calls).toEqual([]);
  await selectWorkflow(page, 'View', 'Metadata');
  await selectWorkflow(page, 'View', 'Image display');
  await expand(page);
  await expect(rows).toHaveCount(3);
  await expect(pipeline).toContainText('Unapplied changes');
  await pipeline.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(rows).toHaveCount(2);
  expect((await current(request)).revision).toBe(before.revision);
  await rows.nth(0).getByRole('button', { name: 'Copy Gaussian denoising', exact: true }).click();
  const dragged = rows.nth(1);
  const handle = dragged.locator('.pipeline-drag-handle');
  const start = (await handle.boundingBox())!;
  const end = (await pipeline.locator('.pipeline-steps').boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(start.x + start.width / 2 + 10, start.y + start.height / 2, { steps: 4 });
  await page.mouse.move(end.x + end.width - 5, end.y + end.height / 2, { steps: 8 });
  await expect(pipeline.locator('.pipeline-drop-slot[data-drop-active="true"]')).toHaveAttribute(
    'data-position',
    '3',
  );
  await expect(pipeline.getByText('Drop here', { exact: true })).toBeVisible();
  await page.mouse.up();
  await expect(rows.nth(1)).toHaveAttribute('aria-label', 'Baseline correction');
  await expect(rows.nth(2)).toHaveAttribute('aria-label', 'Gaussian denoising');
  await rows.nth(2).locator('.pipeline-drag-handle').scrollIntoViewIfNeeded();
  const lastHandle = (await rows.nth(2).locator('.pipeline-drag-handle').boundingBox())!;
  await page.mouse.move(lastHandle.x + 3, lastHandle.y + 3);
  await page.mouse.down();
  await page.mouse.move(end.x + end.width / 2, end.y + end.height / 2, { steps: 8 });
  await expect(pipeline.getByText('Drop here', { exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(pipeline.getByText('Drop here', { exact: true })).toHaveCount(0);
  await expect(rows.nth(1)).toHaveAttribute('aria-label', 'Baseline correction');
  await expect(rows.nth(2)).toHaveAttribute('aria-label', 'Gaussian denoising');
  expect((await current(request)).pipeline).toEqual(before.pipeline);
  await rows.nth(2).getByRole('button', { name: 'Edit', exact: true }).click();
  await expect(page.getByLabel('Sigma (voxels)', { exact: true })).toHaveValue('0.6');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.8');
  await expect(pipeline).toContainText('Unapplied changes');
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await current(request)).pipeline.steps.length).toBe(3);
  await expect(
    pipeline.getByRole('status', { name: 'Pipeline run status', exact: true }),
  ).toHaveText('Pipeline applied');
  expect(calls).toHaveLength(1);
  expect(JSON.parse(calls[0])).toMatchObject({
    operation: 'pipeline',
    options: { action: 'replace' },
  });
  const after = await current(request);
  expect(after.pipeline.generation).toBe(before.pipeline.generation + 1);
  expect(after.experiments).toEqual(before.experiments);
  await page.screenshot({ path: info.outputPath('editable-pipeline.png') });
});

test('named pipelines retain results and drafts within one experiment and compare cached images and maps', async ({
  page,
  request,
}, info) => {
  test.setTimeout(120000);
  await setup(page, request, false);
  const pipeline = panel(page);
  const selector = pipeline.getByLabel('Active pipeline', { exact: true });
  const first = await current(request);
  const image = async () =>
    (await (await request.get('/api/workspaces/' + first.id + '/slice')).body()).toString('base64');
  const firstImage = await image();
  async function fit(steps: string) {
    await selectWorkflow(page, 'Fit');
    await page.getByLabel('Fit region', { exact: true }).selectOption('all');
    await page.getByLabel('Fit steps', { exact: true }).fill(steps);
    await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
    await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  }
  await fit('64');
  const firstMaps = (await current(request)).maps;
  await pipeline.getByRole('button', { name: 'Pipeline name and removal', exact: true }).click();
  await pipeline.getByLabel('Pipeline name', { exact: true }).fill('Reference');
  await pipeline.getByRole('button', { name: 'Rename', exact: true }).click();
  await expect(selector.locator('option:checked')).toContainText('Reference');
  await pipeline.getByRole('button', { name: 'New pipeline', exact: true }).click();
  await expect.poll(async () => (await current(request)).pipeline.id).not.toBe(first.pipeline.id);
  const secondId = (await current(request)).pipeline.id;
  expect((await current(request)).maps).toEqual([]);
  expect((await current(request)).experiments).toEqual(first.experiments);
  await selectWorkflow(page, 'Denoise');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('1.3');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect.poll(async () => (await current(request)).pipeline.steps.length).toBe(1);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  const secondImage = await image();
  expect(secondImage).not.toBe(firstImage);
  await fit('72');
  const secondMaps = (await current(request)).maps;
  const compare = pipeline.getByRole('group', { name: 'Compare pipeline results', exact: true });
  await compare
    .getByLabel('Pipeline comparison result', { exact: true })
    .selectOption({ label: 't2' });
  await compare.getByRole('button', { name: 'Split', exact: true }).click();
  await expect(
    page.getByRole('separator', { name: 'Move comparison divider', exact: true }),
  ).toBeVisible();
  const settings = (await current(request)).scene.comparison;
  expect(settings.left).toMatchObject({ pipeline_id: secondId, kind: 'map', map_name: 't2' });
  expect(settings.right).toMatchObject({
    pipeline_id: first.pipeline.id,
    kind: 'map',
    map_name: 't2',
  });
  await expect(page.getByRole('img', { name: /Image B:.*Reference.*t2/ })).toBeVisible();
  await compare.getByRole('button', { name: 'Overlay', exact: true }).click();
  await expect(page.locator('.comparison-surface')).toHaveClass(/overlay/);
  await compare.getByLabel('Pipeline comparison result', { exact: true }).selectOption('image');
  await compare.getByRole('button', { name: 'Split', exact: true }).click();
  await expect
    .poll(async () => (await current(request)).scene.comparison.left.kind)
    .toBe('processed');
  await selector.selectOption(first.pipeline.id);
  await expect(selector).toHaveValue(first.pipeline.id);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await selectWorkflow(page, 'Denoise');
  expect(await image()).toBe(firstImage);
  expect((await current(request)).maps).toEqual(firstMaps);
  expect((await current(request)).fit_settings.num_steps).toBe(64);
  await expand(page);
  await pipeline
    .locator('.pipeline-step')
    .first()
    .getByRole('button', { name: 'Edit', exact: true })
    .click();
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.9');
  await expect(pipeline).toContainText('Unapplied changes');
  await selector.selectOption(secondId);
  await expect(selector).toHaveValue(secondId);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  expect(await image()).toBe(secondImage);
  expect((await current(request)).maps).toEqual(secondMaps);
  expect((await current(request)).fit_settings.num_steps).toBe(72);
  await selector.selectOption(first.pipeline.id);
  await expect(pipeline).toContainText('Unapplied changes');
  await expect(page.getByLabel('Sigma (voxels)', { exact: true })).toHaveValue('0.9');
  await pipeline.getByRole('button', { name: 'Discard changes', exact: true }).click();
  expect((await current(request)).pipeline.steps[0].options.parameters).toMatchObject({
    sigma: 0.6,
  });
  const file = info.outputPath('named-pipelines.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  await expect(selector).toHaveValue(first.pipeline.id);
  const restored = ((await (await request.get('/api/workspaces')).json()) as Workspace[]).at(-1)!;
  expect(restored.pipeline.choices).toHaveLength(2);
  expect(restored.experiments).toEqual(first.experiments);
  await selector.selectOption(secondId);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await expect
    .poll(
      async () =>
        (await (await request.get('/api/workspaces/' + restored.id)).json()).fit_settings.num_steps,
    )
    .toBe(72);
  await page.screenshot({ path: info.outputPath('named-pipelines.png') });
});
