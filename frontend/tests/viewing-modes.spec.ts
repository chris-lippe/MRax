import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

import { loadExample, selectTheme, saveWorkspace, openWorkspace } from './ui';

import type { Workspace } from '../src/api';

async function current(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json())[0] as Workspace;
}

async function pair(page: Page) {
  await page.getByRole('button', { name: 'Overlay', exact: true }).click();
  await page
    .getByLabel('Image A', { exact: true })
    .selectOption({ label: 'Motion reference: Original' });
  await expect(page.getByRole('img', { name: /^Image A: Motion reference:/ })).toBeVisible();
  await page
    .getByLabel('Image B', { exact: true })
    .selectOption({ label: 'Motion shifted: Processed' });
  await expect(page.getByRole('img', { name: /^Image B: Motion shifted:/ })).toBeVisible();
  await expect(page.getByLabel('Overlay opacity', { exact: true })).toBeEnabled();
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json())
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
});

test('Image, Overlay and Split view are peers and share arbitrary experiment sources across modes and saves', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('/');
  await loadExample(page, 'multimodal');
  const before = await current(request);
  const original = await (await request.get('/api/workspaces/' + before.id + '/slice')).body();
  const modes = page.getByRole('group', { name: 'Viewing mode', exact: true });
  await expect(modes.getByRole('button')).toHaveText(['Image', 'Overlay', 'Split view']);
  await pair(page);
  const imageA = page.getByLabel('Image A', { exact: true }),
    imageB = page.getByLabel('Image B', { exact: true });
  await expect(imageA.locator('optgroup[label="Motion reference"]')).toHaveCount(1);
  const sources = [await imageA.inputValue(), await imageB.inputValue()];
  await expect(page.locator('.comparison-surface')).toHaveClass(/overlay/);
  await expect(
    page.getByRole('separator', { name: 'Move comparison divider', exact: true }),
  ).toHaveCount(0);
  await page.getByText('Image options', { exact: true }).click();
  await page.getByLabel('Image A frame', { exact: true }).selectOption('0');
  await page.getByLabel('Image B frame', { exact: true }).selectOption({ value: '2' });
  await page.getByLabel('Image B palette', { exact: true }).selectOption('hot');
  await page.getByText('Image options', { exact: true }).click();
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(
    page.getByRole('separator', { name: 'Move comparison divider', exact: true }),
  ).toBeVisible();
  expect([await imageA.inputValue(), await imageB.inputValue()]).toEqual(sources);
  await page.getByRole('button', { name: 'Side by side', exact: true }).click();
  await page.getByRole('button', { name: 'Overlay', exact: true }).click();
  await expect(page.locator('.comparison-surface')).toHaveClass(/overlay/);
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(page.locator('.comparison-surface')).toHaveClass(/side-by-side/);
  await page.getByRole('button', { name: 'Overlay', exact: true }).click();
  const after = await current(request);
  expect(after.experiment).toBe(before.experiment);
  expect(after.experiments).toEqual(before.experiments);
  expect(after.pipeline).toEqual(before.pipeline);
  expect(after.revision).toBe(before.revision);
  expect(after.scene.comparison).toMatchObject({
    mode: 'overlay',
    layout: 'side-by-side',
    left: { experiment: 'Motion reference', frame: 0 },
    right: { experiment: 'Motion shifted', frame: 2, colormap: 'hot' },
  });
  expect(await (await request.get('/api/workspaces/' + before.id + '/slice')).body()).toEqual(
    original,
  );
  const file = info.outputPath('image-pair.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  await expect(modes.getByRole('button', { name: 'Overlay', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(
    page.getByRole('img', { name: /^Image B: Motion shifted:.*frame 3$/ }),
  ).toBeVisible();
  expect([await imageA.inputValue(), await imageB.inputValue()]).toEqual(sources);
  await expect(page.getByRole('slider', { name: 'Frame', exact: true })).toHaveCount(0);
  await expect(page.getByRole('slider', { name: 'Slice', exact: true })).toHaveCount(0);
  await page.locator('.viewer-controls > summary').click();
  for (const theme of ['light', 'dark'] as const) {
    await selectTheme(page, theme);
    await page.screenshot({ path: info.outputPath(`overlay-pair-${theme}.png`) });
  }
});
