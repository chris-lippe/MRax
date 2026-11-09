import { zoomImage, imageZoom } from './ui';
import { test, expect, type APIRequestContext, type Locator, type Page } from '@playwright/test';
import { loadExample, selectTheme, selectWorkflow } from './ui';
import type { Workspace } from '../src/api';

async function current(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json())[0] as Workspace;
}
async function canvas(page: Page, selector = '.image-surface canvas') {
  return page.locator(selector).evaluate((element) => (element as HTMLCanvasElement).toDataURL());
}
async function pointer(slider: Locator, fraction: number) {
  const box = (await slider.boundingBox())!;
  return { x: box.x + 5 + (box.width - 10) * fraction, y: box.y + box.height / 2 };
}
async function drag(
  page: Page,
  slider: Locator,
  start: number,
  middle: number,
  end: number,
  check: () => Promise<void>,
) {
  await slider.scrollIntoViewIfNeeded();
  const element = await slider.elementHandle();
  const first = await pointer(slider, start);
  await page.mouse.move(first.x, first.y);
  await page.mouse.down();
  try {
    const mid = await pointer(slider, middle);
    await page.mouse.move(mid.x, mid.y, { steps: 8 });
    await check();
    const last = await pointer(slider, end);
    await page.mouse.move(last.x, last.y, { steps: 8 });
    expect(await element!.evaluate((input) => input.isConnected)).toBe(true);
    await check();
  } finally {
    await page.mouse.up();
  }
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json())
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
});

test('combined intensity controls replace the marked labels and preserve scientific data across workflows', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await loadExample(page);
  const before = await current(request);
  const raw = await (await request.get('/api/workspaces/' + before.id + '/slice')).body();
  const original = await canvas(page);
  const controls = page.getByRole('region', { name: 'Image intensity', exact: true });
  await expect(controls).toBeInViewport();
  await expect(page.locator('.workspace-footer, .pipeline-empty')).toHaveCount(0);
  await expect(page.locator('.pipeline-panel header')).not.toContainText('Protected');
  await page.getByLabel('Brightness', { exact: true }).fill('20');
  await expect.poll(() => canvas(page)).not.toBe(original);
  await page.getByLabel('Display contrast', { exact: true }).fill('1.3');
  for (const workflow of ['Preprocess', 'Register', 'Segment', 'Fit', 'Analyze', 'View']) {
    await selectWorkflow(page, workflow);
    await expect(controls).toBeVisible();
    await expect(page.getByLabel('Brightness', { exact: true })).toHaveValue('20');
    await expect(page.getByLabel('Display contrast', { exact: true })).toHaveValue('1.3');
  }
  await selectWorkflow(page, 'View', 'Image display');
  await zoomImage(page, 2);
  const zoomBeforeIntensityReset = await imageZoom(page);
  const lower = page.getByLabel('Minimum display value', { exact: true });
  const upper = page.getByLabel('Maximum display value', { exact: true });
  const low = Number(await lower.inputValue());
  const high = Number(await upper.inputValue());
  await lower.fill(String(low + (high - low) * 0.1));
  await lower.press('Enter');
  await expect(page.getByLabel('Brightness', { exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Auto intensity', exact: true }).click();
  await expect(page.getByLabel('Brightness', { exact: true })).toHaveValue('0');
  await expect(page.getByLabel('Display contrast', { exact: true })).toHaveValue('1');
  await expect.poll(() => imageZoom(page)).toBe(zoomBeforeIntensityReset);
  await expect.poll(() => canvas(page)).toBe(original);
  expect(await (await request.get('/api/workspaces/' + before.id + '/slice')).body()).toEqual(raw);
  const after = await current(request);
  expect(after.revision).toBe(before.revision);
  expect(after.pipeline).toEqual(before.pipeline);
  expect(after.dirty).toBe(before.dirty);
  for (const theme of ['light', 'dark'] as const) {
    await selectTheme(page, theme);
    await page.screenshot({ path: info.outputPath(`intensity-${theme}.png`) });
  }
});

test('brightness, contrast and both value handles remain reactive throughout a drag without data requests', async ({
  page,
}) => {
  await page.goto('/');
  await loadExample(page);
  const calls: string[] = [];
  page.on('request', (request) => {
    if (/\/(view|slice|comparison|tasks)(?:\?|$)/.test(request.url())) calls.push(request.url());
  });
  const brightness = page.getByLabel('Brightness', { exact: true });
  await drag(page, brightness, 0.5, 0.6, 0.7, async () => {
    expect(Number(await brightness.inputValue())).toBeGreaterThan(10);
  });
  await page.getByRole('button', { name: 'Auto intensity', exact: true }).click();
  const contrast = page.getByLabel('Display contrast', { exact: true });
  await drag(page, contrast, 1 / 3, 0.5, 1, async () => {
    expect(Number(await contrast.inputValue())).toBeGreaterThan(1);
  });
  await page.getByRole('button', { name: 'Auto intensity', exact: true }).click();
  const lower = page.getByLabel('Lower display bound', { exact: true });
  const upper = page.getByLabel('Upper display bound', { exact: true });
  const minimum = Number(await lower.inputValue());
  const maximum = Number(await upper.inputValue());
  await drag(page, lower, 0, 0.2, 0.4, async () => {
    expect(Number(await lower.inputValue())).toBeGreaterThan(minimum);
  });
  await drag(page, upper, 1, 0.8, 0.6, async () => {
    expect(Number(await upper.inputValue())).toBeLessThan(maximum);
  });
  expect(Number(await upper.inputValue())).toBeGreaterThan(Number(await lower.inputValue()));
  const previous = Number(await lower.inputValue());
  await lower.press('ArrowRight');
  expect(Number(await lower.inputValue())).toBeGreaterThan(previous);
  expect(Number(await lower.inputValue())).toBeLessThan(previous + (maximum - minimum) * 0.01);
  expect(calls).toEqual([]);
});

test('view selectors and intensity form one collapsible group above the image without shrinking it', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/');
  await loadExample(page);
  const group = page.locator('.viewer-controls');
  const summary = group.locator('summary').first();
  const brightness = group.getByLabel('Brightness', { exact: true });
  await expect(group).toHaveAttribute('open', '');
  await expect(page.getByRole('combobox', { name: 'View', exact: true })).toBeVisible();
  await expect(group.getByLabel('View', { exact: true })).toHaveCount(0);
  await expect(group.getByLabel('Plane', { exact: true })).toBeVisible();
  await brightness.fill('15');
  const image = page.getByTestId('roi-surface');
  const before = (await image.boundingBox())!;
  expect(before.height).toBeGreaterThan(150);
  expect((await brightness.boundingBox())!.y).toBeLessThan(before.y);
  const calls: string[] = [];
  page.on('request', (request) => {
    if (/\/(view|slice|comparison|tasks)(?:\?|$)/.test(request.url())) calls.push(request.url());
  });
  await summary.click();
  await expect(brightness).toBeHidden();
  await expect(page.getByRole('combobox', { name: 'View', exact: true })).toBeVisible();
  await expect(group.getByLabel('Plane', { exact: true })).toBeHidden();
  await expect.poll(async () => (await image.boundingBox())!.y).toBeLessThan(before.y - 50);
  expect((await image.boundingBox())!.height).toBeCloseTo(before.height, 0);
  await summary.press('Enter');
  await expect(brightness).toHaveValue('15');
  expect(calls).toEqual([]);
  await summary.click();
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:/ })).toBeVisible();
  await expect(brightness).toBeHidden();
  await summary.click();
  await expect(page.getByLabel('Image A', { exact: true })).toBeVisible();
  await expect(group.getByLabel('Image A', { exact: true })).toHaveCount(0);
  await expect(group.getByLabel('Adjust comparison intensity', { exact: true })).toBeVisible();
  await summary.click();
  await page.reload();
  await expect(page.getByRole('img', { name: /^Image A:/ })).toBeVisible();
  await expect(brightness).toBeHidden();
  await expect(summary).toBeVisible();
  await page.screenshot({ path: info.outputPath('view-controls-collapsed.png') });
});

test('split intensity previews immediately and commits linked or independent windows on release', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:/ })).toBeVisible();
  const before = await current(request);
  await expect(page.getByLabel('Adjust comparison intensity', { exact: true })).toBeDisabled();
  const original = await canvas(page, '.comparison-left canvas');
  const calls: string[] = [];
  page.on('request', (request) => {
    if (/\/(view|slice|comparison|tasks)(?:\?|$)/.test(request.url())) calls.push(request.url());
  });
  const brightness = page.getByLabel('Brightness', { exact: true });
  await drag(page, brightness, 0.5, 0.6, 0.7, async () => {
    await expect.poll(() => canvas(page, '.comparison-left canvas')).not.toBe(original);
    expect(calls).toEqual([]);
  });
  await expect.poll(async () => (await current(request)).scene.comparison.left.window).toBeTruthy();
  const linked = await current(request);
  expect(linked.scene.comparison.left.window).toEqual(linked.scene.comparison.right.window);
  expect(calls.filter((url) => url.endsWith('/view'))).toHaveLength(1);
  expect(calls.filter((url) => !url.endsWith('/view'))).toEqual([]);
  await page.getByRole('checkbox', { name: 'Link intensity windows', exact: true }).uncheck();
  await page.getByLabel('Adjust comparison intensity', { exact: true }).selectOption('right');
  await page.getByLabel('Brightness', { exact: true }).fill('0');
  await page.getByLabel('Brightness', { exact: true }).press('Tab');
  await expect
    .poll(async () => (await current(request)).scene.comparison.right.window?.center)
    .not.toBe(linked.scene.comparison.right.window?.center);
  const independent = await current(request);
  expect(independent.scene.comparison.left.window).toEqual(linked.scene.comparison.left.window);
  await page.getByRole('button', { name: 'Auto intensity', exact: true }).click();
  await expect.poll(async () => (await current(request)).scene.comparison.right.window).toBeNull();
  expect((await current(request)).scene.comparison.left.window).toEqual(
    linked.scene.comparison.left.window,
  );
  expect((await current(request)).revision).toBe(before.revision);
});

for (const [label, values] of [
  ['constant', [5, 5, 5, 5]],
  ['negative', [-8, -4, 0, Number.NaN]],
  ['small', [1e-12, 2e-12, 4e-12, Number.POSITIVE_INFINITY]],
  ['nonfinite', [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]],
] as const)
  test(`value bounds support ${label} image values`, async ({ page }) => {
    await page.route('**/api/workspaces/*/slice', async (route) => {
      const response = await route.fetch();
      const data = new Float64Array(24 * 32);
      for (let index = 0; index < data.length; index++) data[index] = values[index % values.length];
      const body = Buffer.from(data.buffer);
      await route.fulfill({
        response,
        headers: {
          ...response.headers(),
          'x-dtype': 'float64-le',
          'content-length': String(body.length),
        },
        body,
      });
    });
    await page.goto('/');
    await loadExample(page);
    const lower = page.getByLabel('Minimum display value', { exact: true });
    const upper = page.getByLabel('Maximum display value', { exact: true });
    const low = Number(await lower.inputValue()),
      high = Number(await upper.inputValue());
    expect(Number.isFinite(low) && Number.isFinite(high)).toBe(true);
    expect(high).toBeGreaterThan(low);
    await lower.fill(String(low + (high - low) / 4));
    await lower.press('Enter');
    await upper.fill(String(low + (3 * (high - low)) / 4));
    await upper.press('Enter');
    expect(Number(await upper.inputValue()) - Number(await lower.inputValue())).toBeGreaterThan(0);
    await page.getByRole('button', { name: 'Auto intensity', exact: true }).click();
    await expect(lower).toHaveValue(String(low));
    await expect(upper).toHaveValue(String(high));
  });
