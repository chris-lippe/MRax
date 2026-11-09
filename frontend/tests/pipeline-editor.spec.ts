import { test, expect, type APIRequestContext, type Page } from '@playwright/test';
import type { Workspace, Job } from '../src/api';
import { loadExample, selectWorkflow, selectDataFormat, selectExperiment } from './ui';

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
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.6');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect.poll(async () => (await current(request)).pipeline.steps.length).toBe(1);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  if (baseline) {
    await selectWorkflow(page, 'Analyze', 'Time-series corrections');
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

test('inline settings and dedicated action forms stage changes and failures retain the draft', async ({
  page,
  request,
}) => {
  await setup(page, request, false);
  const before = await current(request),
    pipeline = panel(page);
  await pipeline
    .locator('.pipeline-step')
    .getByRole('button', { name: 'Edit', exact: true })
    .click();
  await pipeline.getByRole('button', { name: 'Use action controls', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'Preprocess', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('1.1');
  await page.getByRole('button', { name: 'Stage denoising settings', exact: true }).click();
  expect((await current(request)).pipeline.steps[0].version).toBe(before.pipeline.steps[0].version);
  await pipeline
    .locator('.pipeline-step')
    .getByRole('button', { name: 'Edit', exact: true })
    .click();
  await selectDataFormat(page, 'Pipeline step settings', 'Raw');
  const settings = pipeline.getByLabel('Pipeline step settings', { exact: true });
  await settings.fill(
    JSON.stringify({ ...before.pipeline.steps[0].options, parameters: { sigma: -1 } }),
  );
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect(
    pipeline.getByRole('status', { name: 'Pipeline run status', exact: true }),
  ).toContainText('previous data kept');
  expect((await current(request)).pipeline).toEqual(before.pipeline);
  await expect(
    pipeline.getByRole('button', { name: 'Discard changes', exact: true }),
  ).toBeEnabled();
  await pipeline
    .locator('.pipeline-step')
    .getByRole('button', { name: 'Edit', exact: true })
    .click();
  await selectDataFormat(page, 'Pipeline step settings', 'Raw');
  await settings.fill('{invalid');
  await expect(
    pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }),
  ).toBeDisabled();
  await pipeline.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(
    pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }),
  ).toBeDisabled();
});

test('running nodes animate with real progress states and respect reduced motion', async ({
  page,
  request,
}, info) => {
  await setup(page, request, false);
  const pipeline = panel(page);
  await pipeline
    .locator('.pipeline-step')
    .getByRole('button', { name: 'Copy Gaussian denoising', exact: true })
    .click();
  let actual: Job | null = null;
  let complete = false;
  function running(job: Job): Job {
    if (complete) return job;
    return {
      ...job,
      status: 'running',
      progress: 0.25,
      pipeline: job.pipeline
        ? {
            ...job.pipeline,
            status: 'running',
            steps: job.pipeline.steps.map((step, i) => ({
              ...step,
              status: i === 0 ? 'running' : 'pending',
              progress: i === 0 ? 0.5 : 0,
            })),
          }
        : null,
    };
  }
  await page.route('**/api/workspaces/*/tasks', async (route) => {
    const response = await route.fetch();
    const queued = (await response.json()) as Job;
    await expect
      .poll(async () => {
        actual = (await (await request.get('/api/jobs/' + queued.id)).json()) as Job;
        return actual.status;
      })
      .toBe('completed');
    await route.fulfill({ response, json: running(actual!) });
  });
  await page.route(/\/api\/jobs\/[^/]+$/, async (route) => {
    if (actual && route.request().url().endsWith(actual.id))
      await route.fulfill({ json: running(actual) });
    else await route.continue();
  });
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step[data-state="running"]')).toHaveCount(1);
  await expect(
    pipeline.getByRole('progressbar', { name: 'Gaussian denoising progress', exact: true }),
  ).toHaveAttribute('value', '0.5');
  await expect(pipeline.locator('.pipeline-step[data-state="running"]')).toHaveCSS(
    'animation-name',
    'pipeline-active',
  );
  await expect(
    pipeline.getByRole('button', { name: 'Discard changes', exact: true }),
  ).toBeDisabled();
  await page.screenshot({ path: info.outputPath('running-pipeline.png') });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(pipeline.locator('.pipeline-step[data-state="running"]')).toHaveCSS(
    'animation-name',
    'none',
  );
  complete = true;
  await expect(pipeline.locator('.pipeline-step[data-state="completed"]')).toHaveCount(2);
  await expect(
    pipeline.getByRole('status', { name: 'Pipeline run status', exact: true }),
  ).toHaveText('Pipeline applied');
});

test('removing a draft step exits its action editor without touching the confirmed step', async ({
  page,
  request,
}) => {
  await setup(page, request, false);
  const before = await current(request),
    pipeline = panel(page);
  await pipeline
    .locator('.pipeline-step')
    .getByRole('button', { name: 'Edit', exact: true })
    .click();
  await pipeline.getByRole('button', { name: 'Use action controls', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Stage denoising settings', exact: true }),
  ).toBeVisible();
  await pipeline
    .locator('.pipeline-step')
    .getByRole('button', { name: 'Remove', exact: true })
    .click();
  await expect(page.getByRole('button', { name: 'Apply denoising', exact: true })).toBeVisible();
  await expect(pipeline.locator('.pipeline-edit-status')).toHaveCount(0);
  expect((await current(request)).pipeline).toEqual(before.pipeline);
  await pipeline.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
});

test('dragged drafts stay with their experiment and cannot overwrite a changed confirmed chain', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page, 'multimodal');
  await selectExperiment(page, 'Motion reference');
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect.poll(async () => (await current(request)).pipeline.steps.length).toBe(1);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await expand(page);
  const pipeline = panel(page),
    rows = pipeline.locator('.pipeline-step');
  await rows.nth(0).getByRole('button', { name: 'Copy Gaussian denoising', exact: true }).click();
  await rows.nth(0).getByRole('button', { name: 'Copy Gaussian denoising', exact: true }).click();
  const first = await rows.nth(0).getAttribute('data-step-id');
  await rows
    .nth(0)
    .dragTo(rows.nth(2), { sourcePosition: { x: 8, y: 12 }, targetPosition: { x: 20, y: 20 } });
  await expect(rows.nth(2)).toHaveAttribute('data-step-id', first!);
  await selectExperiment(page, 'Motion shifted');
  await expect(rows).toHaveCount(0);
  await selectExperiment(page, 'Motion reference');
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(2)).toHaveAttribute('data-step-id', first!);
  const w = await current(request);
  const token = (await (await request.get('/api/session')).json()).token;
  const response = await request.post('/api/workspaces/' + w.id + '/tasks', {
    headers: { 'x-mrax-token': token },
    data: {
      revision: w.revision,
      operation: 'process',
      options: { method: 'gaussian', axes: ['x', 'y'], parameters: { sigma: 1 } },
    },
  });
  expect(response.ok()).toBe(true);
  const id = (await response.json()).id as string;
  await expect
    .poll(async () => (await (await request.get('/api/jobs/' + id)).json()).status)
    .toBe('completed');
  await page.locator('.study-item[aria-current="true"]').click();
  await expect(pipeline).toContainText('confirmed pipeline changed');
  await expect(
    pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }),
  ).toBeDisabled();
  await pipeline.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(rows).toHaveCount(1);
  expect((await current(request)).pipeline.steps[0].options.parameters).toMatchObject({ sigma: 1 });
});
