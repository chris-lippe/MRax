import { test, expect, type Locator } from '@playwright/test';
import { loadExample, selectWorkflow } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

async function documentTop(locator: Locator) {
  return locator.evaluate(
    (element) => element.getBoundingClientRect().top + (element.closest('main')?.scrollTop ?? 0),
  );
}

test('enlarging the image moves the signal lower and fits the physical image aspect ratio to the width', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await loadExample(page);
  const listed = (await (await request.get('/api/workspaces')).json())[0];
  const original = await (await request.get('/api/workspaces/' + listed.id)).json();
  const aspect =
    (original.shape[1] * original.spacing[1]) / (original.shape[0] * original.spacing[0]);
  const viewer = page.getByRole('region', { name: 'Image viewer', exact: true });
  const surface = viewer.locator('.image-surface');
  const preview = page.locator('.image-preview');
  const signal = page.locator('.signal-preview');
  const divider = page.getByRole('separator', { name: 'Resize image viewer', exact: true });
  await expect(surface).toBeVisible();
  const startHeight = (await preview.boundingBox())!.height;
  const startWidth = (await surface.boundingBox())!.width;
  const signalTop = await documentTop(signal);
  const signalHeight = (await signal.boundingBox())!.height;
  const calls: string[] = [];
  page.on('request', (request) => {
    if (/\/(slice|tasks|view)(?:\?|$)/.test(request.url())) calls.push(request.url());
  });
  const handle = (await divider.boundingBox())!;
  const startY = handle.y + handle.height / 2;
  const targetY = Math.min(startY + 180, page.viewportSize()!.height - 12);
  const distance = targetY - startY;
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
  await page.mouse.down();
  await page.mouse.move(handle.x + handle.width / 2, targetY, {
    steps: 18,
  });
  await page.mouse.up();
  await expect
    .poll(async () => (await preview.boundingBox())!.height)
    .toBeGreaterThan(startHeight + distance - 12);
  const grownHeight = (await preview.boundingBox())!.height;
  expect(
    Math.abs((await documentTop(signal)) - signalTop - (grownHeight - startHeight)),
  ).toBeLessThan(3);
  expect((await signal.boundingBox())!.height).toBeCloseTo(signalHeight, 0);
  expect((await surface.boundingBox())!.width).toBeGreaterThan(startWidth + 60);
  let geometry = (await surface.boundingBox())!;
  expect(geometry.width / geometry.height).toBeCloseTo(aspect, 2);
  await divider.focus();
  await divider.press('End');
  await expect(divider).toHaveAttribute('aria-valuenow', '2400');
  geometry = (await surface.boundingBox())!;
  expect(geometry.width).toBeCloseTo(
    (await viewer.locator('.image-viewport').boundingBox())!.width,
    0,
  );
  expect(geometry.width / geometry.height).toBeCloseTo(aspect, 2);
  expect((await signal.boundingBox())!.height).toBeCloseTo(signalHeight, 0);
  expect(calls).toEqual([]);
  const after = await (await request.get('/api/workspaces/' + original.id)).json();
  expect(after.revision).toBe(original.revision);
  expect(after.view_revision).toBe(original.view_revision);
  await divider.press('Home');
  await page.screenshot({ path: info.outputPath('image-and-studies-layout.png') });
});

test('comparison views enlarge with the same handle without stretching either image', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await loadExample(page);
  await page.getByRole('button', { name: 'Original vs processed', exact: true }).click();
  const comparison = page.locator('.comparison-surface[aria-busy="false"]');
  await expect(comparison).toBeVisible();
  const before = (await comparison.boundingBox())!.height;
  const canvases = comparison.locator('canvas');
  await expect(canvases).toHaveCount(2);
  const ratios = await Promise.all(
    (await canvases.all()).map(async (canvas) => {
      const box = (await canvas.boundingBox())!;
      return box.width / box.height;
    }),
  );
  const divider = page.getByRole('separator', { name: 'Resize image viewer', exact: true });
  await divider.focus();
  await divider.press('Shift+ArrowDown');
  await expect
    .poll(async () => (await comparison.boundingBox())!.height)
    .toBeGreaterThan(before + 60);
  for (let index = 0; index < ratios.length; index++) {
    await expect
      .poll(async () => {
        const box = (await canvases.nth(index).boundingBox())!;
        return box.width / box.height;
      })
      .toBeCloseTo(ratios[index], 2);
  }
});

test('image height persists, can be cancelled or reset, and survives action tab changes', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await loadExample(page);
  const preview = page.locator('.image-preview');
  const divider = page.getByRole('separator', { name: 'Resize image viewer', exact: true });
  const initial = Math.round((await preview.boundingBox())!.height);
  await divider.focus();
  await divider.press('ArrowDown');
  await divider.press('Shift+ArrowDown');
  await expect(divider).toHaveAttribute('aria-valuenow', String(initial + 80));
  await divider.scrollIntoViewIfNeeded();
  const handle = (await divider.boundingBox())!;
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
  await page.mouse.down();
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2 + 48, {
    steps: 5,
  });
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(divider).toHaveAttribute('aria-valuenow', String(initial + 80));
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await expect(divider).toHaveAttribute('aria-valuenow', String(initial + 80));
  await page.reload();
  await expect(divider).toHaveAttribute('aria-valuenow', String(initial + 80));
  await divider.focus();
  await divider.press('Home');
  await expect.poll(() => page.evaluate(() => localStorage.getItem('mrax.imageHeight'))).toBeNull();
  expect((await preview.boundingBox())!.height).toBeLessThan(initial + 80);
  await divider.press('ArrowDown');
  await divider.dblclick();
  await expect.poll(() => page.evaluate(() => localStorage.getItem('mrax.imageHeight'))).toBeNull();
});
