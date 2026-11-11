import { test, expect, type APIRequestContext } from '@playwright/test';

import type { Workspace } from '../src/api';

import { imageZoom, loadExample, openRoiCoordinates, openSettings, zoomImage } from './ui';

async function studies(request: APIRequestContext): Promise<Workspace[]> {
  return (await request.get('/api/workspaces')).json();
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await studies(request)) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('editable shortcuts reject reserved keys, avoid conflicts and respect input context', async ({
  page,
}) => {
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('shortcut-fixture')) {
      localStorage.setItem(
        'mrax.shortcuts',
        JSON.stringify({ view: 'Ctrl+W', resetImage: 'Escape' }),
      );
      sessionStorage.setItem('shortcut-fixture', 'true');
    }
  });
  await page.goto('/');
  await loadExample(page);
  const settings = await openSettings(page);
  await settings.getByRole('tab', { name: 'Keyboard shortcuts', exact: true }).click();
  const view = settings.getByLabel('View shortcut', { exact: true });
  const reconstruct = settings.getByLabel('Reconstruct shortcut', { exact: true });
  const reset = settings.getByLabel('Fit image to view shortcut', { exact: true });
  await expect(view).toHaveValue('Ctrl+Shift+1');
  await expect(reset).toHaveValue('0');
  await expect(view.locator('option[value="Ctrl+Shift+2"]')).toBeDisabled();
  const choices = await settings.locator('select option').allTextContents();
  expect(choices).not.toContain('Ctrl+W');
  expect(choices).not.toContain('Escape');
  expect(choices).not.toContain('Alt+F4');
  expect(choices).not.toContain('Tab');
  await reconstruct.selectOption('');
  await view.selectOption('Ctrl+Shift+2');
  await reset.selectOption('F');
  await settings.getByRole('button', { name: 'Done', exact: true }).focus();
  await page.keyboard.press('Control+Shift+Digit5');
  await expect(
    page.getByRole('tab', { name: 'View', exact: true, includeHidden: true }),
  ).toHaveAttribute('aria-selected', 'true');
  await settings.getByRole('button', { name: 'Done', exact: true }).click();
  await page.keyboard.press('Control+Shift+Digit5');
  await expect(page.getByRole('tab', { name: 'Segment', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await openRoiCoordinates(page);
  const name = page.getByLabel('New ROI name', { exact: true });
  await name.fill('');
  await name.press('f');
  await name.press('Control+Shift+Digit6');
  await expect(name).toHaveValue('f');
  await expect(page.getByRole('tab', { name: 'Segment', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.getByRole('tab', { name: 'Segment', exact: true }).focus();
  await page.keyboard.press('Control+Shift+Digit2');
  await expect(page.getByRole('tab', { name: 'View', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await zoomImage(page, 2);
  const image = page.locator('.image-viewport:visible');
  await image.focus();
  await page.keyboard.press('0');
  expect(await imageZoom(page)).toBeCloseTo(2, 2);
  await page.keyboard.press('f');
  await expect.poll(() => imageZoom(page)).toBeCloseTo(1, 2);
  await page.keyboard.press('Control+Shift+Digit8');
  await expect(settings).toBeVisible();
  await page.reload();
  await openSettings(page);
  await settings.getByRole('tab', { name: 'Keyboard shortcuts', exact: true }).click();
  await expect(view).toHaveValue('Ctrl+Shift+2');
  await expect(reconstruct).toHaveValue('');
  await expect(reset).toHaveValue('F');
  await settings.getByRole('button', { name: 'Reset keyboard bindings', exact: true }).click();
  await expect(view).toHaveValue('Ctrl+Shift+1');
  await expect(reset).toHaveValue('0');
});
