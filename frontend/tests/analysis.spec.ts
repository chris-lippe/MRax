import { loadExample } from './ui';

import { openRoiCoordinates } from './ui';

import { test, expect, type Page, type APIRequestContext } from '@playwright/test';

import {
  selectDataFormat,
  selectWorkflow,
  openMetadataBrowser,
  closeMetadataBrowser,
  selectExperiment,
  selectTheme,
} from './ui';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as {
    id: string;
    revision: number;
  }[];
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await opened(request)) {
    const r = await request.delete(`/api/workspaces/${w.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
    expect(r.status()).toBe(204);
  }
});

async function synthetic(page: Page, multimodal = false) {
  await page.goto('/');
  await loadExample(page, multimodal ? 'multimodal' : 't2');
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await selectWorkflow(page, 'Segment');
}

test('complex views, planes, integer labels, brush, and undo use the real backend', async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await synthetic(page, true);
  await selectExperiment(page, 'Complex coils');
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Model', { exact: true })).toHaveValue('identity');
  await expect(page.getByLabel('t2 initial', { exact: true })).toHaveCount(0);
  await selectWorkflow(page, 'View');
  await page.getByLabel('Component', { exact: true }).selectOption('real');
  await page.getByLabel('frame', { exact: true }).fill('1');
  await page.getByLabel('Plane', { exact: true }).selectOption('coronal');
  await expect(page.getByRole('img', { name: /24 by 3 quantitative image/ })).toBeVisible();
  await selectWorkflow(page, 'Segment');
  await page.getByRole('button', { name: 'Ellipse tool', exact: true }).click();
  await page.getByLabel('Label value', { exact: true }).fill('7');
  await openRoiCoordinates(page);
  await page.getByLabel('ROI column', { exact: true }).fill('0');
  await page.getByLabel('ROI row', { exact: true }).fill('3');
  await page.getByLabel('ROI width', { exact: true }).fill('2');
  await page.getByLabel('ROI height', { exact: true }).fill('8');
  await page.getByRole('button', { name: 'Add ellipse by coordinates', exact: true }).click();
  await expect(page.locator('.roi-card').first().getByText('7', { exact: true })).toBeVisible();
  const count = await page.locator('.roi-card').first().locator('button span').textContent();
  await page.getByRole('button', { name: 'Brush tool', exact: true }).click();
  await page.getByLabel('Label value', { exact: true }).fill('9');
  await page.getByRole('button', { name: 'Paint labels', exact: true }).click();
  const box = (await page.getByTestId('roi-surface').boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 5 });
  await page.mouse.up();
  await expect(page.locator('.roi-card').first().locator('.label-badge')).toContainText('9');
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect(page.locator('.roi-card').first().locator('button span')).toHaveText(count!);
  await expect(page.locator('.roi-card').first().locator('.label-badge')).toHaveText('7');
  await page.getByLabel('Plane', { exact: true }).selectOption('axial');
  await selectWorkflow(page, 'Reconstruct');
  await page
    .getByRole('group', { name: 'Reconstruction display', exact: true })
    .getByRole('button', { name: 'Cartesian Fourier', exact: true })
    .click();
  await page.getByLabel('Reconstruction component', { exact: true }).selectOption('phase');
  await page.getByLabel('Coil', { exact: true }).selectOption('0');
  await page.getByLabel('Unwrap phase', { exact: true }).uncheck();
  await page
    .getByRole('group', { name: 'Reconstruction display', exact: true })
    .getByRole('button', { name: 'K-space', exact: true })
    .click();
  await page.getByLabel('K-space component', { exact: true }).selectOption('log_magnitude');
  await selectExperiment(page, 'T2 phantom');
  await openMetadataBrowser(page, 'Experiment metadata');
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  await expect(page.getByLabel('Experiment metadata', { exact: true })).toHaveValue(/"TE"/);
  await selectExperiment(page, 'Signed with missing samples');
  await openMetadataBrowser(page, 'Experiment metadata');
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  await expect(page.getByLabel('Experiment metadata', { exact: true })).not.toHaveValue(/"TE"/);
  await closeMetadataBrowser(page);
  await selectWorkflow(page, 'Segment');
  await selectTheme(page, 'dark');
  await page.screenshot({ path: testInfo.outputPath('multimodal-dark.png'), fullPage: true });
  expect(errors).toEqual([]);
});
