import { loadExample, showDataControls } from './ui';
import { chooseFilePath } from './ui';
import { readFile } from 'node:fs/promises';
import { test, expect } from '@playwright/test';
import { openExport, closeExport } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('Export sits directly below Import and keeps its form settings and direct file exports', async ({
  page,
}, testInfo) => {
  await page.goto('/');
  const files = await showDataControls(page);
  const trigger = files.getByRole('button', { name: 'Export data', exact: true });
  await expect(trigger).toBeDisabled();
  await expect(files.getByRole('button')).toHaveText([
    'New study',
    'Open workspace',
    'Import data',
    'Export data',
    'Save workspace',
  ]);
  await loadExample(page);
  const dialog = await openExport(page);
  await expect(page.getByRole('dialog')).toHaveCount(1);
  await expect(page.locator('.study-panel .study-export')).toHaveCount(0);
  await dialog
    .getByLabel('Export object', { exact: true })
    .selectOption('T2 phantom/image/mr_image');
  await dialog.getByLabel('Export format', { exact: true }).selectOption('json');
  await dialog.getByText('Packages, ROI sets and reports', { exact: true }).click();
  for (let index = 0; index < 5; index++) {
    await page.keyboard.press('Tab');
    expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
  }
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await openExport(page);
  await expect(dialog.getByLabel('Export object', { exact: true })).toHaveValue(
    'T2 phantom/image/mr_image',
  );
  await expect(dialog.getByLabel('Export format', { exact: true })).toHaveValue('json');
  await dialog.getByText('Packages, ROI sets and reports', { exact: true }).click();
  await page.screenshot({ path: testInfo.outputPath('export-dialog.png') });
  const submitted = page.waitForResponse(
    (response) => response.url().endsWith('/tasks') && response.request().method() === 'POST',
  );
  await dialog.getByRole('button', { name: 'Export selected object', exact: true }).click();
  const destination = testInfo.outputPath('export-dialog.json');
  await chooseFilePath(page, destination, 'Save');
  await submitted;
  await closeExport(page);
  await openExport(page);
  await expect(dialog.locator('.job-status .file-source')).toContainText(destination, {
    timeout: 60000,
  });
  expect(JSON.parse(await readFile(destination, 'utf8')).shape).toEqual([8, 24, 32, 3]);
  await closeExport(page);
  await expect(trigger).toBeFocused();
});
