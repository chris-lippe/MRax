import { imageZoom } from './ui';
import { clickDataControl, loadExample } from './ui';
import { openRoiCoordinates } from './ui';
import { test, expect } from '@playwright/test';
import { selectWorkflow, selectTheme, showStudies } from './ui';

test.use({ locale: 'de-DE' });

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  expect(await page.evaluate(() => navigator.language)).toBe('de-DE');
  await loadExample(page);
  await selectWorkflow(page, 'Segment');
});

test('image context menu stays English, supports the keyboard, and preserves ROIs', async ({
  page,
  request,
}) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  const surface = page.getByTestId('roi-surface');
  const viewport = page.locator('.image-viewport');
  const menu = page.getByRole('menu', { name: 'Image actions', exact: true });
  for (const theme of ['light', 'dark'] as const) {
    await selectTheme(page, theme);
    await surface.evaluate(() => {
      document.addEventListener(
        'contextmenu',
        (event) => {
          setTimeout(() => {
            document.documentElement.dataset.contextPrevented = String(event.defaultPrevented);
          }, 0);
        },
        { once: true, capture: true },
      );
    });
    await surface.click({ button: 'right' });
    await expect(menu.getByRole('menuitem')).toHaveText(['Zoom in', 'Zoom out', 'Reset display']);
    await expect(page.locator('html')).toHaveAttribute('data-context-prevented', 'true');
    const bounds = (await menu.boundingBox())!;
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(1280);
    expect(bounds.y + bounds.height).toBeLessThanOrEqual(720);
    await page.keyboard.press('ArrowDown');
    await expect(menu.getByRole('menuitem', { name: 'Reset display', exact: true })).toBeFocused();
    await page.keyboard.press('Escape');
    await expect(menu).toBeHidden();
    await expect(viewport).toBeFocused();
  }
  await selectWorkflow(page, 'View');
  await viewport.focus();
  await page.keyboard.press('Shift+F10');
  await page.keyboard.press('Enter');
  await expect.poll(() => imageZoom(page)).toBeCloseTo(1.3, 3);
  await surface.click({ button: 'right' });
  await menu.getByRole('menuitem', { name: 'Reset display', exact: true }).click();
  await expect.poll(() => imageZoom(page)).toBeCloseTo(1, 3);
  await surface.click({ button: 'right' });
  await page.getByRole('heading', { level: 1 }).click();
  await expect(menu).toBeHidden();

  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  const before = (await (await request.get('/api/workspaces')).json())[0];
  await page.getByTestId('roi-handle-0').click({ button: 'right' });
  await expect(menu).toBeVisible();
  await page.keyboard.press('Escape');
  const after = (await (await request.get('/api/workspaces')).json())[0];
  expect(after.revision).toBe(before.revision);
  expect(after.rois).toEqual(before.rois);
});

test('signal context menu uses English chart and table actions', async ({ page }) => {
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  const signal = page.getByRole('region', { name: 'ROI signal', exact: true });
  const menu = page.getByRole('menu', { name: 'Signal actions', exact: true });
  await signal.getByRole('img').click({ button: 'right' });
  await menu.getByRole('menuitem', { name: 'Show data table', exact: true }).click();
  await expect(signal.getByRole('table')).toBeVisible();
  await signal.getByRole('table').click({ button: 'right' });
  await menu.getByRole('menuitem', { name: 'Show chart', exact: true }).click();
  await expect(signal.getByRole('img')).toBeVisible();
});

test('file controls and the discard confirmation stay English', async ({ page }) => {
  let nativeDialogs = 0;
  page.on('dialog', async (dialog) => {
    nativeDialogs++;
    await dialog.dismiss();
  });
  await clickDataControl(page, 'Import data');
  const importer = page.getByRole('dialog', { name: 'Import a study or experiment', exact: true });
  await importer.getByRole('button', { name: 'Browse files and folders…', exact: true }).click();
  const chooser = page.locator('.file-system-dialog');
  await expect(chooser.getByLabel('Location', { exact: true })).toBeVisible();
  await expect(chooser.getByRole('button', { name: 'Open', exact: true })).toBeVisible();
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.getByRole('button', { name: 'Close importer', exact: true }).click();
  await selectWorkflow(page, 'Segment', 'Import masks and ROI sets');
  await expect(
    page.getByRole('button', { name: 'Open ROI set or mask…', exact: true }),
  ).toBeVisible();
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await showStudies(page);
  const close = page.getByRole('button', {
    name: 'Close study: Synthetic T2 phantom',
    exact: true,
  });
  await expect(close).toBeEnabled();
  await close.focus();
  await expect(close).toBeFocused();
  await close.press('Enter');
  const confirmation = page.getByRole('dialog', { name: 'Discard unsaved changes?', exact: true });
  await expect(confirmation.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused();
  await confirmation.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'Synthetic T2 phantom', exact: true }),
  ).toBeVisible();
  await page.locator('.study-item[aria-current="true"]').hover();
  await close.click();
  await confirmation.getByRole('button', { name: 'Discard and close', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'No study open', exact: true })).toBeVisible();
  expect(nativeDialogs).toBe(0);
});
