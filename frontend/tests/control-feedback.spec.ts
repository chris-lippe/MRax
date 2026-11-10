import { test, expect, type Locator } from '@playwright/test';
import { loadExample, openExport, openSettings, selectTheme, showDataControls } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

async function outline(control: Locator) {
  return control.evaluate((element) => getComputedStyle(element).outlineStyle);
}

async function centeredClose(button: Locator) {
  await expect(button).toBeVisible();
  const box = (await button.boundingBox())!;
  const icon = (await button.locator('svg').boundingBox())!;
  expect(Math.abs(icon.x + icon.width / 2 - box.x - box.width / 2)).toBeLessThan(1);
  expect(Math.abs(icon.y + icon.height / 2 - box.y - box.height / 2)).toBeLessThan(1);
}

test('resize handles release their accent while retaining visible keyboard focus', async ({
  page,
}) => {
  await page.goto('/');
  await loadExample(page);
  for (const [label, key, horizontal] of [
    ['Resize study browser', 'ArrowRight', true],
    ['Resize analysis controls', 'ArrowLeft', true],
    ['Resize image viewer', 'ArrowDown', false],
    ['Resize signal preview', 'ArrowDown', false],
  ] as const) {
    const handle = page.getByRole('separator', { name: label, exact: true });
    await handle.scrollIntoViewIfNeeded();
    await handle.focus();
    await handle.press(key);
    await expect.poll(() => outline(handle)).toBe('solid');
    const box = (await handle.boundingBox())!;
    const x = box.x + box.width / 2;
    const y = Math.min(box.y + box.height / 2, page.viewportSize()!.height - 60);
    await page.mouse.move(x, y);
    await page.mouse.down();
    await expect(handle).toHaveAttribute('data-dragging', 'true');
    const active = await handle
      .locator('span')
      .evaluate((element) => getComputedStyle(element).backgroundColor);
    await page.mouse.move(x + (horizontal ? 24 : 0), y + (horizontal ? 0 : 24), { steps: 4 });
    await page.mouse.up();
    await expect(handle).not.toHaveAttribute('data-dragging');
    await expect(handle).toBeFocused();
    await expect.poll(() => outline(handle)).toBe('none');
    await expect
      .poll(() =>
        handle.locator('span').evaluate((element) => getComputedStyle(element).backgroundColor),
      )
      .not.toBe(active);
    await handle.press(key);
    await expect.poll(() => outline(handle)).toBe('solid');
    await handle.press('Home');
  }
});

test('image sliders clear pointer focus and keep keyboard access with stronger focus enabled', async ({
  page,
}) => {
  await page.goto('/');
  await loadExample(page);
  const settings = await openSettings(page);
  await settings.getByRole('tab', { name: 'Accessibility', exact: true }).click();
  await settings.getByRole('checkbox', { name: 'Stronger keyboard focus', exact: true }).check();
  await settings.getByRole('button', { name: 'Close settings', exact: true }).click();
  for (const label of ['Brightness', 'Display contrast', 'Echo', 'Slice']) {
    const control = page.getByRole('slider', { name: label, exact: true });
    await control.scrollIntoViewIfNeeded();
    await control.focus();
    await control.press('ArrowRight');
    await expect.poll(() => outline(control)).toBe('solid');
    const before = await control.inputValue();
    const box = (await control.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.85, box.y + box.height / 2, { steps: 4 });
    await page.mouse.up();
    await expect(control).toBeFocused();
    await expect.poll(() => outline(control)).toBe('none');
    expect(await control.inputValue()).not.toBe(before);
    await control.press('ArrowLeft');
    await expect.poll(() => outline(control)).toBe('solid');
  }
  await page.getByRole('button', { name: 'Original vs processed', exact: true }).click();
  const comparison = page.locator('.comparison-divider');
  await expect(comparison).toBeVisible();
  await comparison.focus();
  await comparison.press('ArrowRight');
  await expect.poll(() => outline(comparison.locator('span'))).toBe('solid');
  await comparison.click();
  await expect.poll(() => outline(comparison.locator('span'))).toBe('none');
});

test('the subtle header separator follows the study browser divider in both themes', async ({
  page,
}, info) => {
  await page.goto('/');
  const line = page.locator('.header-panel-divider');
  const handle = page.getByRole('separator', { name: 'Resize study browser', exact: true });
  for (const theme of ['light', 'dark'] as const) {
    await selectTheme(page, theme);
    for (const key of ['Home', 'Shift+ArrowRight']) {
      await handle.focus();
      await handle.press(key);
      const box = (await line.boundingBox())!;
      const divider = (await handle.boundingBox())!;
      expect(Math.abs(box.x + box.width / 2 - divider.x - divider.width / 2)).toBeLessThan(1);
      expect(box.width).toBe(1);
      expect(box.height).toBeLessThan((await page.locator('.app-header').boundingBox())!.height);
      expect(
        await line.evaluate((element) => Number(getComputedStyle(element).opacity)),
      ).toBeLessThan(0.7);
    }
    await page.screenshot({ path: info.outputPath('header-' + theme + '.png') });
  }
});

test('dialog close icons stay centered with normal and enlarged accessibility controls', async ({
  page,
}, info) => {
  await page.goto('/');
  await loadExample(page);
  const settings = await openSettings(page);
  const closeSettings = settings.getByRole('button', { name: 'Close settings', exact: true });
  await centeredClose(closeSettings);
  await settings.getByRole('tab', { name: 'Accessibility', exact: true }).click();
  await settings.getByRole('checkbox', { name: 'Larger controls', exact: true }).check();
  await settings.getByRole('checkbox', { name: 'Larger text', exact: true }).check();
  await centeredClose(closeSettings);
  await page.screenshot({ path: info.outputPath('settings-close.png') });
  await closeSettings.click();
  await openExport(page);
  const closeExport = page.getByRole('button', { name: 'Close export dialog', exact: true });
  await centeredClose(closeExport);
  await closeExport.click();
  await (
    await showDataControls(page)
  )
    .getByRole('button', { name: 'Import data', exact: true })
    .click();
  const closeImport = page.getByRole('button', { name: 'Close importer', exact: true });
  await centeredClose(closeImport);
  await closeImport.click();
  await (
    await showDataControls(page)
  )
    .getByRole('button', { name: 'Open workspace', exact: true })
    .click();
  const closeFile = page.getByRole('button', { name: 'Close file dialog', exact: true });
  await centeredClose(closeFile);
  await closeFile.click();
});
