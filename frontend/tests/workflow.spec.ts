import { zoomImage, imageZoom } from './ui';

import { loadExample } from './ui';

import { openRoiCoordinates } from './ui';

import { saveWorkspace, openWorkspace } from './ui';

import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

import { selectWorkflow, selectExperiment, selectTheme } from './ui';

import type { Workspace } from '../src/api';

async function current(request: APIRequestContext) {
  const studies = (await (await request.get('/api/workspaces')).json()) as Workspace[];
  return studies.at(-1)!;
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  const studies = (await (await request.get('/api/workspaces')).json()) as Workspace[];
  for (const study of studies) {
    const response = await request.delete(`/api/workspaces/${study.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: study.revision, discard: true },
    });
    expect(response.status()).toBe(204);
  }
});

async function addRoi(page: Page, name: string, column: number) {
  await openRoiCoordinates(page);
  await page.getByLabel('New ROI name', { exact: true }).fill(name);
  await page.getByLabel('ROI column', { exact: true }).fill(String(column));
  await page.getByLabel('ROI width', { exact: true }).fill('4');
  await page.getByLabel('ROI height', { exact: true }).fill('4');
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await expect(
    page.locator('.roi-card').getByRole('button', { name: new RegExp(name) }),
  ).toBeVisible();
}

test('View, Segment, Fit and Register share layers and independent ROI selections', async ({
  page,
  request,
}, info) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await loadExample(page, 'multimodal');
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await selectTheme(page, 'light');
  await selectWorkflow(page, 'Segment');
  await addRoi(page, 'Anterior', 3);
  await addRoi(page, 'Posterior', 13);

  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await page.getByLabel('Show Anterior', { exact: true }).uncheck();
  await expect(page.locator('.roi-mask')).toHaveCount(1);
  await page.getByLabel('Active ROI', { exact: true }).selectOption('Anterior');
  await expect(page.locator('.roi-mask')).toHaveCount(1);
  await selectWorkflow(page, 'Segment');
  await expect(page.locator('.roi-mask')).toHaveCount(2);
  await expect(page.locator('.roi-card.selected')).toContainText('Anterior');

  await selectWorkflow(page, 'Fit');
  await page.getByRole('separator', { name: 'Resize analysis controls', exact: true }).press('End');
  await page.getByLabel('Fit region', { exact: true }).selectOption('rois');
  await expect(page.getByLabel('Fit mask Anterior', { exact: true })).toBeChecked();
  await expect(
    page.getByText('This workflow uses all enabled ROIs in each experiment.'),
  ).toBeVisible();
  await page.getByLabel('Fit mask Posterior', { exact: true }).uncheck();
  await expect(page.getByText('This workflow uses the selected ROIs by name.')).toBeVisible();
  await page.getByLabel('Fit steps', { exact: true }).fill('600');
  await selectWorkflow(page, 'View');
  await expect(page.getByLabel('Show Anterior', { exact: true })).not.toBeChecked();
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Fit mask Posterior', { exact: true })).not.toBeChecked();
  await expect(page.getByLabel('Fit steps', { exact: true })).toHaveValue('600');
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete', {
    timeout: 60000,
  });
  expect((await current(request)).fit_info?.roi_names).toEqual(['Anterior']);

  await selectWorkflow(page, 'View');
  await page
    .getByLabel('Overlay source', { exact: true })
    .selectOption({ label: 'T2 phantom · t2' });
  await page.getByRole('button', { name: 'Add overlay', exact: true }).click();
  await expect(page.getByRole('region', { name: 'Overlay 1', exact: true })).toBeVisible();
  await page
    .getByRole('region', { name: 'Overlay 1', exact: true })
    .getByText('Layer settings', { exact: true })
    .click();
  await page.getByLabel('Overlay 1 colormap', { exact: true }).selectOption('hot');
  await selectWorkflow(page, 'View', 'Image display');
  await zoomImage(page, 2);
  const initialZoom = await imageZoom(page);
  const intensity = page.getByLabel('Minimum display value', { exact: true });
  const maximum = Number(
    await page.getByLabel('Maximum display value', { exact: true }).inputValue(),
  );
  await intensity.fill(
    String(
      Number(await intensity.inputValue()) + (maximum - Number(await intensity.inputValue())) * 0.2,
    ),
  );
  await intensity.press('Enter');
  const savedIntensity = await intensity.inputValue();

  await selectExperiment(page, 'Motion shifted');
  await selectWorkflow(page, 'Register');
  await page.getByLabel('Registration preset', { exact: true }).selectOption('different_contrasts');
  await expect(
    page.getByLabel('Registration translation initialization', { exact: true }),
  ).toBeChecked();
  await expect(page.getByLabel('Maximum registration motion', { exact: true })).toHaveValue('12');
  await page.getByLabel('Reference experiment', { exact: true }).selectOption('Motion reference');
  await page.getByLabel('Moving experiment', { exact: true }).selectOption('Motion shifted');
  await expect(
    page.getByLabel('Moving metric mask', { exact: true }).locator('option'),
  ).toHaveCount(1);
  await page.getByLabel('Moving frame', { exact: true }).fill('3');
  await page.getByLabel('Manual affine transform', { exact: true }).check();
  await page.getByLabel('Horizontal shift', { exact: true }).fill('3');
  await page.getByLabel('Moving image selection', { exact: true }).selectOption('mean');
  await page.getByRole('button', { name: 'Run registration', exact: true }).click();
  await expect.poll(async () => (await current(request)).registrations.length).toBe(1);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete', {
    timeout: 60000,
  });
  const completed = await current(request);
  expect(completed.pipeline.steps[0].options.moving_aggregation).toBe('mean');
  expect(completed.experiment).toBe('Motion shifted');
  expect(completed.registrations[0].diagnostics).toMatchObject({
    post: { mse: expect.any(Number) },
  });
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await pipeline
    .locator('.pipeline-step')
    .first()
    .getByRole('button', { name: 'Edit', exact: true })
    .click();
  await expect(page.getByLabel('Horizontal shift', { exact: true })).toHaveValue('3');
  await page.getByLabel('Horizontal shift', { exact: true }).fill('2.5');
  await expect(pipeline).toContainText('Unapplied changes');
  expect((await current(request)).pipeline.steps[0].version).toBe(
    completed.pipeline.steps[0].version,
  );
  await page.getByRole('button', { name: 'Run registration', exact: true }).click();
  await expect
    .poll(async () => (await current(request)).pipeline.steps[0].version)
    .not.toBe(completed.pipeline.steps[0].version);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  expect((await current(request)).pipeline.steps[0].id).toBe(completed.pipeline.steps[0].id);
  expect((await current(request)).pipeline.steps).toHaveLength(1);
  await page.getByRole('button', { name: 'View registered overlay', exact: true }).click();
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await expect(page.getByRole('region', { name: 'Overlay 1', exact: true })).toContainText(
    'registered',
  );
  await expect(page.getByRole('img', { name: /32 by 40 quantitative image/ })).toBeVisible();

  await selectWorkflow(page, 'Register', 'Series alignment');
  await expect(page.getByLabel('Series alignment method', { exact: true })).toHaveValue(
    'framewise',
  );
  await expect(page.getByLabel('Drift dimension frame', { exact: true })).toBeChecked();
  await expect(
    page
      .getByLabel('Series alignment preset', { exact: true })
      .locator('option[value="dynamic_contrast"]'),
  ).toHaveJSProperty('disabled', true);
  await page
    .getByLabel('Series alignment preset', { exact: true })
    .selectOption('small_translation');
  await expect(page.getByLabel('Maximum series motion', { exact: true })).toHaveValue('6');
  await page.getByLabel('Drift transform', { exact: true }).selectOption('translation');
  await page.getByRole('button', { name: 'Run series alignment', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  const series = await current(request);
  expect(series.pipeline.steps).toHaveLength(1);
  expect(series.pipeline.steps[0]).toMatchObject({
    label: 'Series alignment',
    options: { settings: { method: 'framewise', dimensions: ['frame'] } },
  });
  await page.getByRole('button', { name: 'Workflows', exact: true }).click();
  const recipes = page.getByRole('dialog', { name: 'Reusable workflows', exact: true });
  await recipes.getByRole('button', { name: 'Capture current configuration', exact: true }).click();
  await recipes.getByRole('button', { name: 'Edit definition', exact: true }).click();
  const recipe = JSON.parse(
    await recipes.getByLabel('Workflow JSON definition', { exact: true }).inputValue(),
  );
  expect(recipe).toMatchObject({ mrax_version: expect.any(String), kind: 'registration' });
  expect(recipe.steps[0]).toMatchObject({
    operation: 'drift',
    settings: { settings: { method: 'framewise', registration: { max_displacement_voxels: 6 } } },
  });
  await recipes.getByRole('button', { name: 'Done', exact: true }).click();

  await selectExperiment(page, 'T2 phantom');
  await selectWorkflow(page, 'Fit');
  await selectWorkflow(page, 'View', 'Image display');
  await expect.poll(() => imageZoom(page)).toBe(initialZoom);
  await expect(intensity).toHaveValue(savedIntensity);
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await expect(page.getByLabel('Show Anterior', { exact: true })).not.toBeChecked();
  await expect(page.getByRole('region', { name: 'Overlay 1', exact: true })).toContainText(
    'T2 phantom · t2',
  );
  const file = info.outputPath('shared-workflow.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  await expect(page.getByLabel('Show Anterior', { exact: true })).not.toBeChecked();
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Fit mask Anterior', { exact: true })).toBeChecked();
  await expect(page.getByLabel('Fit mask Posterior', { exact: true })).not.toBeChecked();
  await selectWorkflow(page, 'View');
  await page.screenshot({ path: info.outputPath('shared-workflow.png') });
  expect(errors).toEqual([]);
});
