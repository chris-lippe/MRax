import { openControlGroup } from './ui';
import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

import type { Workspace } from '../src/api';

import { loadExample, selectWorkflow } from './ui';

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
  await rows
    .nth(0)
    .getByRole('button', { name: 'Move Gaussian denoising later', exact: true })
    .click();
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
