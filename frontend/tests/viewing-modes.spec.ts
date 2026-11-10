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

test('pair navigation uses the selected experiments and preserves the active analysis cursor', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page, 'multimodal');
  const before = await current(request);
  await pair(page);
  const frame = page.getByRole('slider', { name: 'Frame', exact: true });
  await expect(frame).toHaveAttribute('max', '3');
  await expect(page.getByRole('slider', { name: 'Slice', exact: true })).toHaveCount(0);
  await expect(
    page.getByLabel('Plane', { exact: true }).getByRole('option', { name: 'coronal', exact: true }),
  ).toHaveJSProperty('disabled', true);
  await frame.focus();
  await frame.press('End');
  await expect(page.getByRole('img', { name: /^Image A:.*frame 4$/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /^Image B:.*frame 4$/ })).toBeVisible();
  const after = await current(request);
  expect(after.scene.comparison.frame).toBe(3);
  expect(after.experiment).toBe(before.experiment);
  expect(after.frame).toBe(before.frame);
  expect(after.indices).toEqual(before.indices);
  expect(after.plane).toBe(before.plane);
  await page.getByRole('button', { name: 'Image', exact: true }).click();
  await expect(page.getByRole('slider', { name: 'Echo', exact: true })).toHaveValue(
    String(before.frame),
  );
});

test('pair frame and slice controls stay responsive through slow image updates', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  const before = await current(request);
  await page.getByRole('button', { name: 'Overlay', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:/ })).toBeVisible();
  await page.route('**/api/workspaces/*/navigate', async (route) => {
    if (route.request().postDataJSON().commit)
      await new Promise((resolve) => setTimeout(resolve, 250));
    await route.continue();
  });
  for (const name of ['Frame', 'Slice']) {
    const slider = page.getByRole('slider', { name, exact: true });
    await slider.scrollIntoViewIfNeeded();
    const element = await slider.elementHandle();
    const box = (await slider.boundingBox())!;
    const y = box.y + box.height / 2;
    await page.mouse.move(box.x + 8, y);
    await page.mouse.down();
    try {
      await page.mouse.move(box.x + box.width / 2, y, { steps: 8 });
      await expect(page.locator('.comparison-surface[aria-busy="false"]')).toBeVisible();
      expect(await element!.evaluate((input) => input.isConnected)).toBe(true);
      await expect(slider).toBeEnabled();
      await page.mouse.move(box.x + box.width - 8, y, { steps: 8 });
      await expect(slider).toHaveValue((await slider.getAttribute('max'))!);
    } finally {
      await page.mouse.up();
    }
  }
  await expect(page.getByRole('img', { name: /^Image A:.*frame 8$/ })).toBeVisible();
  await expect.poll(async () => (await current(request)).scene.comparison.indices?.z).toBe(2);
  const after = await current(request);
  expect(after.frame).toBe(before.frame);
  expect(after.indices).toEqual(before.indices);
});

test('overlay opacity previews continuously without image requests and probes both image values', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page, 'multimodal');
  await pair(page);
  const calls: string[] = [];
  page.on('request', (request) => {
    if (/\/(view|slice|comparison|tasks)(?:\?|$)/.test(request.url())) calls.push(request.url());
  });
  const opacity = page.getByLabel('Overlay opacity', { exact: true });
  await opacity.scrollIntoViewIfNeeded();
  const handle = await opacity.elementHandle();
  const box = (await opacity.boundingBox())!;
  const x = (value: number) => box.x + 5 + (box.width - 10) * value;
  await page.mouse.move(x(0.5), box.y + box.height / 2);
  await page.mouse.down();
  try {
    for (const value of [0.25, 0.75]) {
      await page.mouse.move(x(value), box.y + box.height / 2, { steps: 8 });
      await expect.poll(async () => Number(await opacity.inputValue())).toBeCloseTo(value, 1);
      await expect
        .poll(() =>
          page
            .locator('.comparison-right canvas')
            .evaluate((canvas) => Number(getComputedStyle(canvas).opacity)),
        )
        .toBeCloseTo(value, 1);
      expect(await handle!.evaluate((element) => element.isConnected)).toBe(true);
      expect(calls).toEqual([]);
    }
  } finally {
    await page.mouse.up();
  }
  await expect
    .poll(async () => (await current(request)).scene.comparison.opacity)
    .toBeCloseTo(0.75, 1);
  expect(calls.filter((url) => url.endsWith('/view'))).toHaveLength(1);
  expect(calls.filter((url) => !url.endsWith('/view'))).toEqual([]);
  const canvas = page.locator('.comparison-right canvas');
  await canvas.scrollIntoViewIfNeeded();
  const image = (await canvas.boundingBox())!;
  await page.mouse.move(image.x + image.width / 2, image.y + image.height / 2);
  await expect(page.locator('.comparison-left .comparison-pixel')).toContainText(/\(\d+, \d+\)/);
  await expect(page.locator('.comparison-right .comparison-pixel')).toContainText(/\(\d+, \d+\)/);
});

test('incompatible geometry never blends and remains inspectable side by side', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page, 'multimodal');
  await pair(page);
  await page
    .getByLabel('Image B', { exact: true })
    .selectOption({ label: 'T2 phantom: Processed' });
  await expect(page.getByRole('img', { name: /^Image B: T2 phantom:/ })).toBeVisible();
  await expect(page.locator('.comparison-surface')).toHaveClass(/side-by-side/);
  await expect(page.locator('.comparison-view > .hint').first()).toContainText(
    'Overlay unavailable',
  );
  await expect(page.locator('.comparison-view > .hint').first()).toContainText(/register/i);
  await expect(page.getByLabel('Overlay opacity', { exact: true })).toBeDisabled();
  const workspace = await current(request);
  const response = await request.get('/api/workspaces/' + workspace.id + '/comparison');
  const metadata = JSON.parse(response.headers()['x-comparison']);
  expect(metadata.geometry_compatible).toBe(false);
  expect(metadata.planes[0].shape).not.toEqual(metadata.planes[1].shape);
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await page.getByRole('button', { name: 'Side by side', exact: true }).click();
  await expect(page.locator('.comparison-surface')).toHaveClass(/side-by-side/);
  await page.getByRole('button', { name: 'Image', exact: true }).click();
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  expect((await current(request)).experiment).toBe(workspace.experiment);
});
