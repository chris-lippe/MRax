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

test('accessibility preferences persist without modifying study data', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/');
  await loadExample(page);
  const before = (await studies(request))[0];
  const settings = await openSettings(page);
  await settings.getByRole('tab', { name: 'Appearance', exact: true }).focus();
  await page.keyboard.press('ArrowDown');
  await expect(
    settings.getByRole('tab', { name: 'Keyboard shortcuts', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(settings.getByRole('tab', { name: 'Accessibility', exact: true })).toBeFocused();
  for (const title of [
    'Higher contrast',
    'Larger text',
    'Larger controls',
    'Stronger keyboard focus',
    'Reduce motion',
    'Opaque panels',
    'Always show study close buttons',
  ]) {
    await settings.getByRole('checkbox', { name: title, exact: true }).check();
  }
  await expect(page.locator('html')).toHaveAttribute('data-large-text', 'true');
  await expect(page.locator('html')).toHaveAttribute('data-reduce-motion', 'true');
  const done = settings.getByRole('button', { name: 'Done', exact: true });
  expect(
    await done.evaluate((element) => parseFloat(getComputedStyle(element).fontSize)),
  ).toBeGreaterThanOrEqual(16);
  expect((await done.boundingBox())!.height).toBeGreaterThanOrEqual(40);
  expect(await settings.evaluate((element) => getComputedStyle(element).backdropFilter)).toBe(
    'none',
  );
  await settings.getByRole('checkbox', { name: 'Stronger keyboard focus', exact: true }).focus();
  await page.screenshot({ path: info.outputPath('settings-accessibility.png') });
  await done.click();
  await page.mouse.move(700, 90);
  await expect(page.locator('.study-close')).toHaveCSS('opacity', '1');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect((await studies(request))[0]).toEqual(before);
  await page.reload();
  await openSettings(page);
  await settings.getByRole('tab', { name: 'Accessibility', exact: true }).click();
  await expect(
    settings.getByRole('checkbox', { name: 'Larger controls', exact: true }),
  ).toBeChecked();
  await expect(
    settings.getByRole('checkbox', { name: 'Opaque panels', exact: true }),
  ).toBeChecked();
  await settings.getByRole('tab', { name: 'Keyboard shortcuts', exact: true }).click();
  await expect(settings.getByLabel('View shortcut', { exact: true })).toBeVisible();
  expect(await settings.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
    true,
  );
  await page.screenshot({ path: info.outputPath('settings-keyboard.png') });
});

test('guided tutorial completes one real synthetic workflow and preserves the previous study', async ({
  page,
  request,
}, info) => {
  test.setTimeout(150000);
  await page.setViewportSize({ width: 1280, height: 720 });
  const operations: string[] = [];
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (req) => {
    if (req.method() === 'POST' && req.url().endsWith('/tasks')) {
      operations.push(req.postDataJSON().operation);
    }
  });
  await page.goto('/');
  await loadExample(page);
  const before = (await studies(request))[0];
  const settings = await openSettings(page);
  await settings.getByRole('tab', { name: 'Tutorial', exact: true }).click();
  await settings.getByRole('button', { name: 'Start tutorial', exact: true }).click();
  const guide = page.getByRole('region', { name: 'Workflow tutorial', exact: true });
  await expect(guide).toBeVisible();
  await expect(
    guide.getByRole('heading', { name: 'Your practice study', exact: true }),
  ).toBeVisible();
  const all = await studies(request);
  expect(all).toHaveLength(2);
  const practice = all.find((w) => w.id !== before.id)!;
  expect(practice.label).toBe('Tutorial — synthetic T2');
  await page.locator('.study-row[data-study-id="' + before.id + '"] .study-item').click();
  await expect(guide.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
  await guide.getByRole('button', { name: 'Return to tutorial study', exact: true }).click();
  const next = guide.getByRole('button', { name: 'Next', exact: true });
  await next.click();
  await expect(next).toBeDisabled();
  await guide.getByRole('button', { name: 'Open image controls', exact: true }).click();
  await page.getByRole('slider', { name: 'Echo', exact: true }).fill('1');
  await next.click();
  await guide.getByRole('button', { name: 'Open Denoising', exact: true }).click();
  await expect(next).toBeDisabled();
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await next.click();
  await guide.getByRole('button', { name: 'Show comparison controls', exact: true }).click();
  await page.getByRole('button', { name: 'Compare with original', exact: true }).click();
  await expect(next).toBeDisabled();
  await page.locator('.comparison-divider').focus();
  await page.keyboard.press('ArrowRight');
  await next.click();
  await guide.getByRole('button', { name: 'Open Segment', exact: true }).click();
  await expect(next).toBeDisabled();
  await page.getByRole('button', { name: 'Draw rectangle', exact: true }).click();
  const surface = page.getByTestId('roi-surface');
  await surface.scrollIntoViewIfNeeded();
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.65, { steps: 8 });
  await page.mouse.up();
  await next.click();
  await guide.getByRole('button', { name: 'Open Fit', exact: true }).click();
  await expect(next).toBeDisabled();
  await page.getByLabel('Fit steps', { exact: true }).fill('300');
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await expect(next).toBeEnabled({ timeout: 60000 });
  await next.click();
  await guide.getByRole('button', { name: 'Open map statistics', exact: true }).click();
  await expect(next).toBeDisabled();
  await page.getByLabel('Map', { exact: true }).click();
  await page.getByLabel('Map', { exact: true }).selectOption('t2');
  await page.getByLabel('Statistics map', { exact: true }).click();
  await page.getByLabel('Statistics map', { exact: true }).selectOption('t2');
  await expect(page.getByRole('table', { name: 'ROI map statistics', exact: true })).toBeVisible();
  await page.screenshot({ path: info.outputPath('tutorial-analysis.png') });
  await next.click();
  await expect(guide.getByRole('heading', { name: 'Save or export', exact: true })).toBeVisible();
  await guide.getByRole('button', { name: 'Finish tutorial', exact: true }).click();
  await expect(guide).not.toBeVisible();
  await expect(
    page.locator('.study-row[data-study-id="' + before.id + '"] .study-item'),
  ).toHaveAttribute('aria-current', 'true');
  const result = await studies(request);
  expect(result.find((w) => w.id === before.id)).toEqual(before);
  const retained = result.find((w) => w.id === practice.id)!;
  expect(retained.pipeline.steps.some((step) => step.family === 'denoise')).toBe(true);
  expect(retained.rois.some((roi) => roi.pixels > 0)).toBe(true);
  expect(retained.maps.some((map) => map.name === 't2')).toBe(true);
  expect(operations).toContain('process');
  expect(operations).toContain('full_fit');
  expect(operations).not.toContain('save');
  expect(operations).not.toContain('export');
  expect(errors).toEqual([]);
});
