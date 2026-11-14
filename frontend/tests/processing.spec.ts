import { openControlGroup } from './ui';
import { loadExample } from './ui';

import { openRoiCoordinates } from './ui';

import { saveWorkspace, openWorkspace } from './ui';

import { test, expect, type APIRequestContext } from '@playwright/test';

import { selectWorkflow, selectTheme } from './ui';

import type { Workspace } from '../src/api';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as Workspace[];
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await opened(request)) {
    await expect
      .poll(async () => {
        const latest = (await opened(request)).find((item) => item.id === w.id);
        if (!latest) return 204;
        const response = await request.delete(`/api/workspaces/${w.id}`, {
          headers: { 'x-mrax-token': token },
          data: { revision: latest.revision, discard: true },
        });
        return response.status();
      })
      .toBe(204);
  }
});

test('processing stays in one experiment and an editable pipeline preserves sources and fits', async ({
  page,
  request,
}, info) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await selectTheme(page, 'light');
  const [initial] = await opened(request);
  const slice = async (id: string) =>
    (await (await request.get('/api/workspaces/' + id + '/slice')).body()).toString('base64');
  const original = await slice(initial.id);
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await expect(pipeline.locator('header')).not.toContainText('Original data');
  await expect(pipeline.locator('header')).not.toContainText('Protected');
  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByLabel('New ROI name', { exact: true }).fill('Tissue');
  await page.getByLabel('Label value', { exact: true }).fill('7');
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await page.getByLabel('Show Tissue', { exact: true }).uncheck();

  await selectWorkflow(page, 'Denoise');
  await expect(page.getByLabel('Filter axis frame', { exact: true })).not.toBeChecked();
  await expect(page.getByLabel('Filter axis x', { exact: true })).toBeChecked();
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('1.2');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(1);
  let [w] = await opened(request);
  expect(w.experiments).toHaveLength(initial.experiments.length);
  expect(w.experiment).toBe(initial.experiment);
  expect(w.rois[0]).toMatchObject({ name: 'Tissue', labels: [7], enabled: true, visible: false });
  expect(await slice(w.id)).not.toBe(original);
  const denoise = w.pipeline.steps[0];

  await selectWorkflow(page, 'Analyze', 'Signal and time series');
  await openControlGroup(page, 'Time-series corrections');
  await page.getByLabel('Time-series correction', { exact: true }).selectOption('baseline');
  await page.getByRole('button', { name: 'Apply baseline correction', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(2);
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Fit mask Tissue', { exact: true })).toBeChecked();
  await page.getByLabel('Fit steps', { exact: true }).fill('600');
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].maps.length, { timeout: 60000 })
    .toBeGreaterThan(0);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  [w] = await opened(request);
  const maps = w.maps;
  expect(w.pipeline.fits_stale).toBe(false);
  await page
    .getByRole('group', { name: 'Fit display', exact: true })
    .getByRole('button', { name: 'Signal', exact: true })
    .click();

  const viewer = page.getByRole('region', { name: 'Image viewer', exact: true });
  const compactHeight = (await pipeline.boundingBox())!.height;
  expect(compactHeight).toBeLessThanOrEqual(48);
  await expect(pipeline.locator('header button')).toHaveCount(1);
  await expect(pipeline.locator('header select')).toHaveCount(0);
  await expect(pipeline.getByLabel('Pipeline source', { exact: true })).not.toBeVisible();
  await expect(
    pipeline.getByRole('group', { name: 'Pipeline output', exact: true }),
  ).not.toBeVisible();
  await expect(
    pipeline.getByRole('button', { name: 'Remove all processing', exact: true }),
  ).not.toBeVisible();
  const imageHeight = (await viewer.boundingBox())!.height;
  const imagePlaneHeight = (await page.locator('.image-viewport').boundingBox())!.height;
  const imageTop = (await viewer.boundingBox())!.y;
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await expect(
    pipeline.getByRole('button', { name: 'Hide processing steps', exact: true }),
  ).toHaveAttribute('aria-expanded', 'true');
  expect((await pipeline.boundingBox())!.height).toBeGreaterThan(compactHeight);
  await expect(pipeline.getByLabel('Pipeline source', { exact: true })).toBeVisible();
  expect((await viewer.boundingBox())!.height).toBeCloseTo(imageHeight, 0);
  expect((await viewer.boundingBox())!.y).toBeGreaterThan(imageTop);
  expect(
    await page.locator('main').evaluate((element) => element.scrollHeight > element.clientHeight),
  ).toBe(true);
  await page.locator('main').hover();
  await page.mouse.wheel(0, 350);
  await expect
    .poll(() => page.locator('main').evaluate((element) => element.scrollTop))
    .toBeGreaterThan(0);
  expect((await viewer.boundingBox())!.height).toBeCloseTo(imageHeight, 0);
  const gaussian = pipeline.getByRole('listitem', { name: 'Gaussian denoising', exact: true });
  await gaussian.getByRole('button', { name: 'Edit', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Use action controls', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'Denoise', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await expect(page.getByLabel('Sigma (voxels)', { exact: true })).toHaveValue('1.2');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.6');
  await page.getByRole('button', { name: 'Stage denoising settings', exact: true }).click();
  expect((await opened(request))[0].pipeline.steps[0].version).toBe(denoise.version);
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].pipeline.steps[0].version)
    .not.toBe(denoise.version);
  [w] = await opened(request);
  expect(w.pipeline.steps).toHaveLength(2);
  expect(w.pipeline.steps[0].id).toBe(denoise.id);
  expect(w.pipeline.steps[0].options.parameters).toMatchObject({ sigma: 0.6 });
  expect(w.maps).toEqual(maps);
  await expect(pipeline).toContainText('retained from an earlier pipeline state');
  const processed = await slice(w.id);

  await pipeline.getByRole('button', { name: 'Original', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.enabled).toBe(false);
  expect(await slice(w.id)).toBe(original);
  await pipeline.getByRole('button', { name: 'Hide processing steps', exact: true }).click();
  await expect(pipeline.locator('header')).toContainText('Original');
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  let releaseRefresh = () => {};
  let observeRefresh = () => {};
  const refreshGate = new Promise<void>((resolve) => {
    releaseRefresh = resolve;
  });
  const refreshing = new Promise<void>((resolve) => {
    observeRefresh = resolve;
  });
  const refreshPath = `**/api/workspaces/${w.id}`;
  await page.route(refreshPath, async (route) => {
    if (route.request().method() !== 'GET') return route.continue();
    observeRefresh();
    await refreshGate;
    await route.continue();
  });
  try {
    await pipeline.getByRole('button', { name: 'Processed', exact: true }).click();
    await refreshing;
    await expect(pipeline.getByLabel('Enable Gaussian denoising', { exact: true })).toBeDisabled();
  } finally {
    releaseRefresh();
    await page.unrouteAll({ behavior: 'wait' });
  }
  await expect(pipeline.getByLabel('Enable Gaussian denoising', { exact: true })).toBeEnabled();
  await expect.poll(async () => (await opened(request))[0].pipeline.enabled).toBe(true);
  expect(await slice(w.id)).toBe(processed);
  await pipeline.getByLabel('Enable Gaussian denoising', { exact: true }).uncheck();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps[0].enabled).toBe(false);
  await pipeline.getByLabel('Enable Gaussian denoising', { exact: true }).check();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps[0].enabled).toBe(true);
  await expect(pipeline.getByRole('button', { name: 'Original', exact: true })).toBeEnabled();
  await expect(pipeline.locator('.pipeline-step.disabled')).toHaveCount(0);
  expect(await slice(w.id)).toBe(processed);
  await pipeline.getByRole('button', { name: 'Hide processing steps', exact: true }).click();
  await expect(
    pipeline.getByRole('button', { name: 'Show processing steps', exact: true }),
  ).toHaveAttribute('aria-expanded', 'false');
  expect((await page.locator('.image-viewport').boundingBox())!.height).toBeCloseTo(
    imagePlaneHeight,
    0,
  );
  await page.screenshot({ path: info.outputPath('process-workflow-compact.png') });
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await selectWorkflow(page, 'View', 'Image display');
  await page.screenshot({ path: info.outputPath('process-workflow.png') });

  const file = info.outputPath('pipeline-study.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  await gaussian.getByRole('button', { name: 'Remove', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect
    .poll(
      async () =>
        (await opened(request)).find((item) => item.id !== initial.id)!.pipeline.steps.length,
    )
    .toBe(1);
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
  await expect(pipeline).toContainText('Baseline correction');
  await pipeline.getByRole('button', { name: 'Remove all processing', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect
    .poll(
      async () =>
        (await opened(request)).find((item) => item.id !== initial.id)!.pipeline.steps.length,
    )
    .toBe(0);
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(0);
  const restored = (await opened(request)).find((item) => item.id !== initial.id)!;
  expect(await slice(restored.id)).toBe(original);
  expect(restored.maps).toEqual(maps);
  expect(restored.experiments).toHaveLength(initial.experiments.length);
  expect(errors).toEqual([]);
});
