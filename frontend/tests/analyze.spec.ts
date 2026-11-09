import { loadExample } from './ui';
import { openRoiCoordinates } from './ui';
import { chooseFilePath } from './ui';
import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { selectWorkflow, openExport, closeExport } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('Analyze compares ROI timecourses and exports values from the study sidebar', async ({
  page,
}, testInfo) => {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment');
  for (const [name, column] of [
    ['Region A', '4'],
    ['Region B', '16'],
  ]) {
    await openRoiCoordinates(page);
    await page.getByLabel('New ROI name', { exact: true }).fill(name);
    for (const [key, value] of [
      ['column', column],
      ['row', '4'],
      ['width', '6'],
      ['height', '6'],
    ]) {
      await page.getByLabel(`ROI ${key}`, { exact: true }).fill(value);
    }
    await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
    await expect(
      page.locator('.roi-card').filter({ has: page.getByText(name, { exact: true }) }),
    ).toBeVisible();
  }
  await expect(
    page.getByRole('region', { name: 'ROI map statistics', exact: true }),
  ).not.toBeVisible();
  await selectWorkflow(page, 'Analyze');
  await expect(page.getByRole('tab', { name: 'Maps', exact: true })).toHaveCount(0);
  await expect(page.getByRole('heading', { name: 'ROI timecourses', exact: true })).toBeVisible();
  const controls = page.getByRole('region', { name: 'Timecourse analysis', exact: true });
  await controls.getByRole('button', { name: 'Clear selection', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Analyze timecourse', exact: true }),
  ).toBeDisabled();
  await controls.getByRole('button', { name: 'All ROIs', exact: true }).click();
  await page.getByRole('button', { name: 'Analyze timecourse', exact: true }).click();
  await expect(
    page.getByText('Timecourse ready: 2 ROIs, 8 samples.', { exact: true }),
  ).toBeVisible();
  const chart = page.getByRole('region', { name: 'Timecourse results', exact: true });
  await expect(chart.getByRole('img', { name: 'ROI timecourse comparison' })).toBeVisible();
  await chart.getByRole('button', { name: 'Data table', exact: true }).click();
  const table = page.getByRole('table', { name: 'Timecourse values', exact: true });
  await expect(table).toContainText('TE (ms)');
  await expect(table).toContainText('Region A');
  await expect(table).toContainText('Region B');
  await expect(table.locator('tbody tr')).toHaveCount(8);
  await expect(table.locator('tbody tr').first()).toContainText('36');
  const exportDialog = await openExport(page);
  const exportButton = page.getByRole('button', { name: 'Save timecourse CSV', exact: true });
  await expect(exportDialog.getByRole('region', { name: 'Export controls' })).toContainText(
    'Save timecourse CSV',
  );
  const file = test.info().outputPath('mrax-timecourse.csv');
  await exportButton.click();
  const saved = page.waitForResponse(
    (response) =>
      response.url().endsWith('/api/files/text') && response.request().method() === 'POST',
  );
  await chooseFilePath(page, file, 'Save');
  expect((await saved).ok()).toBe(true);
  const csv = await readFile(file, 'utf8');
  expect(csv).toContain('"TE (ms)"');
  expect(csv).toContain('"Region A valid n"');
  expect(csv.trim().split('\r\n')).toHaveLength(9);
  await closeExport(page);
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await page.getByRole('checkbox', { name: 'Show Region A', exact: true }).click();
  await selectWorkflow(page, 'Analyze');
  await expect(page.getByLabel('Analyze ROI Region A', { exact: true })).toBeChecked();
  await expect(table).toBeVisible();
  await page.getByLabel('Timecourse statistic', { exact: true }).selectOption('std');
  await openExport(page);
  await expect(exportButton).toBeDisabled();
  await closeExport(page);
  await page.getByRole('button', { name: 'Analyze timecourse', exact: true }).click();
  await expect(
    page.getByText('Timecourse ready: 2 ROIs, 8 samples.', { exact: true }),
  ).toBeVisible();
  await openExport(page);
  await expect(exportButton).toBeEnabled();
  await closeExport(page);
  await expect(chart.getByRole('heading')).toContainText('Standard deviation');
  await page.screenshot({ path: testInfo.outputPath('analyze-desktop.png') });
});

test('Analyze keeps analysis masks separate from fit exclusions and map operations', async ({
  page,
}) => {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'Fit');
  await page.getByText('Parameters and fit options', { exact: true }).click();
  await page.getByLabel('Excluded frames', { exact: true }).fill('1');
  await page.getByRole('button', { name: 'Apply frame exclusions', exact: true }).click();
  await expect(page.getByLabel('Excluded frames', { exact: true })).toHaveValue('1');
  await selectWorkflow(page, 'Analyze');
  await page.getByRole('checkbox', { name: 'Respect fit frame exclusions', exact: true }).check();
  await page.getByRole('button', { name: 'Analyze timecourse', exact: true }).click();
  const chart = page.getByRole('region', { name: 'Timecourse results', exact: true });
  await chart.getByRole('button', { name: 'Data table', exact: true }).click();
  const row = page
    .getByRole('table', { name: 'Timecourse values', exact: true })
    .locator('tbody tr')
    .nth(1);
  await expect(row).toContainText('Missing');
  await expect(row).toContainText('No');
  await expect(page.getByLabel('Denoiser', { exact: true })).not.toBeVisible();
  await selectWorkflow(page, 'Analyze', 'Derived maps');
  await expect(page.getByLabel('Map calculation', { exact: true })).toHaveValue('dti');
  await expect(
    page.getByRole('button', { name: 'Compute derived maps', exact: true }),
  ).toBeDisabled();
  await selectWorkflow(page, 'Analyze', 'ROI map statistics');
  await expect(page.getByLabel('Statistics map', { exact: true })).toBeDisabled();
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Fit mask ROI 1', { exact: true })).toBeChecked();
});
