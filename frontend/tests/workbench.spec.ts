import { clickDataControl, loadExample, showDataControls } from './ui';
import { openRoiCoordinates } from './ui';
import { chooseExportFolder, openWorkspace, saveWorkspace } from './ui';
import { stat } from 'node:fs/promises';
import { test, expect } from '@playwright/test';
import {
  selectWorkflow,
  selectExperiment,
  openMetadataBrowser,
  closeMetadataBrowser,
  selectTheme,
  openExport,
  closeExport,
} from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('one study browser and a visible workflow keep selection, metadata and actions distinct', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await loadExample(page);
  const workflow = page.getByRole('tablist', { name: 'Workflow sections' });
  await expect(workflow.getByRole('tab')).toHaveText([
    'View',
    'Reconstruct',
    'Preprocess',
    'Register',
    'Segment',
    'Fit',
    'Analyze',
  ]);
  await expect(workflow.getByRole('tab', { name: 'View', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await expect(page.getByRole('combobox', { name: 'Experiment', exact: true })).toHaveCount(0);
  await expect(page.getByRole('tab', { name: 'Data', exact: true })).toHaveCount(0);
  const files = await showDataControls(page);
  await expect(page.getByRole('button', { name: 'Import data', exact: true })).toHaveCount(1);
  const sidebar = page.getByRole('complementary', { name: 'Data & studies', exact: true });
  await expect(sidebar.getByRole('heading', { name: 'Data & studies', exact: true })).toHaveCount(
    0,
  );
  await expect(files).toHaveAttribute('aria-orientation', 'vertical');
  await expect(page.getByRole('banner').getByRole('toolbar')).toHaveCount(0);
  await expect(
    page.getByRole('banner').getByRole('tablist', { name: 'Workflow sections' }),
  ).toBeVisible();
  await expect(files.getByRole('button')).toHaveText([
    'New study',
    'Open workspace',
    'Import data',
    'Export data',
    'Save workspace',
  ]);
  const exportDialog = await openExport(page);
  await expect(
    exportDialog.getByRole('region', { name: 'Export controls', exact: true }),
  ).toBeVisible();
  await expect(exportDialog.getByLabel('Export object', { exact: true })).toBeVisible();
  await closeExport(page);
  await expect(files.getByRole('button', { name: 'Export data', exact: true })).toBeFocused();
  await showDataControls(page);
  for (const button of await files.getByRole('button').all()) {
    const border = await button.evaluate((element) => getComputedStyle(element).borderTopWidth);
    expect(parseFloat(border)).toBeGreaterThanOrEqual(1);
  }
  const newStudy = files.getByRole('button', { name: 'New study', exact: true });
  await expect(newStudy).toBeEnabled();
  await newStudy.focus();
  await page.keyboard.press('ArrowDown');
  await expect(files.getByRole('button', { name: 'Open workspace', exact: true })).toBeFocused();
  await page.keyboard.press('End');
  await expect(files.getByRole('button', { name: 'Save workspace', exact: true })).toBeFocused();
  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'Fit');
  await page.getByRole('button', { name: 'Run preview fit', exact: true }).click();
  await expect(page.getByLabel('Fitted T2', { exact: true })).toHaveText('80.00');
  for (const theme of ['light', 'dark'] as const) {
    await selectTheme(page, theme);
    for (const step of [
      'View',
      'Reconstruct',
      'Preprocess',
      'Register',
      'Segment',
      'Fit',
      'Analyze',
    ]) {
      await selectWorkflow(page, step);
      await page.screenshot({ path: testInfo.outputPath(`ux-${theme}-${step.toLowerCase()}.png`) });
    }
  }
  await openMetadataBrowser(page);
  await page.screenshot({ path: testInfo.outputPath('ux-metadata-overview.png') });
  await page.getByLabel('Study name', { exact: true }).fill('Reviewed T2');
  await page.getByRole('button', { name: 'Rename study', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Reviewed T2', exact: true })).toBeVisible();
  await closeMetadataBrowser(page);
  await expect(page.getByRole('heading', { name: 'Reviewed T2', exact: true })).toBeVisible();
  await loadExample(page, 'multimodal');
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await expect(page.locator('.job-status')).toHaveCount(0);
  await expect(
    page.getByRole('region', { name: 'Experiments', exact: true }).getByRole('button'),
  ).toHaveCount(5);
  await page.screenshot({ path: testInfo.outputPath('ux-study-browser.png') });
  await page.setViewportSize({ width: 1440, height: 720 });
  await showDataControls(page);
  const filePosition = (await files.boundingBox())!.y;
  await page.getByRole('button', { name: 'Settings', exact: true }).focus();
  await expect(files.getByRole('button', { name: 'Save workspace', exact: true })).toBeInViewport();
  expect((await files.boundingBox())!.y).toBe(filePosition);
  const order = await page.locator('.study-item strong').allTextContents();
  await page.locator('.study-item').filter({ hasText: 'Reviewed T2' }).click();
  await expect(page.getByLabel('Echo', { exact: true })).toBeEnabled();
  await page.getByLabel('Echo', { exact: true }).press('ArrowRight');
  await expect(page.getByRole('img', { name: /frame 2, slice 1/ })).toBeVisible();
  expect(await page.locator('.study-item strong').allTextContents()).toEqual(order);
});

test('one import dialog creates studies or adds experiments to the chosen study', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  const original = (await (await request.get('/api/workspaces')).json())[0];
  await clickDataControl(page, 'Import data');
  await expect(page.getByLabel('Import destination', { exact: true })).toHaveValue('new');
  await page.getByLabel('Import destination', { exact: true }).selectOption('current');
  await page.getByLabel('Import format', { exact: true }).selectOption('blank');
  await page.getByLabel('Experiment name', { exact: true }).fill('Calibration');
  await page.getByLabel('Array shape', { exact: true }).fill('3,4');
  await page.getByLabel('Dimension order', { exact: true }).fill('x,y');
  await page.getByRole('button', { name: 'Import into study', exact: true }).click();
  await expect(page.getByRole('img', { name: /3 by 4 quantitative image/ })).toBeVisible();
  const studies = await (await request.get('/api/workspaces')).json();
  expect(studies).toHaveLength(1);
  expect(studies[0].id).toBe(original.id);
  expect(studies[0].experiments).toContain('Calibration');
  await selectExperiment(page, 'T2 phantom');
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toBeVisible();
});

test('compact study panel creates, imports, saves and reopens workspaces with one dialog at a time', async ({
  page,
  request,
}, testInfo) => {
  await page.setViewportSize({ width: 700, height: 800 });
  await page.goto('/');
  const studies = page.getByRole('button', { name: 'Data & studies', exact: true });
  await studies.click();
  await clickDataControl(page, 'New study');
  await expect(page.getByRole('region', { name: 'Metadata browser', exact: true })).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await closeMetadataBrowser(page);
  await expect(page.getByRole('region', { name: 'Image viewer', exact: true })).toBeVisible();

  await studies.click();
  await clickDataControl(page, 'Import data');
  await expect(
    page.getByRole('dialog', { name: 'Import a study or experiment', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(1);
  await page.getByRole('button', { name: 'Close importer', exact: true }).click();
  await expect(studies).toBeFocused();

  await studies.click();
  await clickDataControl(page, 'Import data');
  await page.getByLabel('Import format', { exact: true }).selectOption('blank');
  await page.getByLabel('Experiment name', { exact: true }).fill('Calibration');
  await page.getByLabel('Array shape', { exact: true }).fill('3,4');
  await page.getByLabel('Dimension order', { exact: true }).fill('x,y');
  await page.getByRole('button', { name: 'Import into study', exact: true }).click();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('img', { name: /3 by 4 quantitative image/ })).toBeVisible();

  await studies.click();
  const saved = testInfo.outputPath('sidebar-workspace.mrax');
  await saveWorkspace(page, saved);
  await openWorkspace(page, saved);
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('img', { name: /3 by 4 quantitative image/ })).toBeVisible();
  await expect
    .poll(async () => (await (await request.get('/api/workspaces')).json()).length)
    .toBe(2);
  await expect(studies).toBeFocused();
  await studies.click();
  const exportDialog = await openExport(page);
  await expect(page.getByRole('dialog')).toHaveCount(1);
  await exportDialog.getByText('Packages, ROI sets and reports', { exact: true }).click();
  await exportDialog
    .getByRole('button', { name: 'Export portable study and view state', exact: true })
    .click();
  const destination = testInfo.outputPath('sidebar-state');
  await chooseExportFolder(page, destination, true);
  await expect(exportDialog.locator('.job-status strong')).toHaveText('Operation complete');
  expect((await stat(destination)).isDirectory()).toBe(true);
  await expect(page.getByRole('dialog')).toHaveCount(1);
  await closeExport(page);
  await expect(studies).toBeFocused();
});
