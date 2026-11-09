import { loadExample } from './ui';
import { test, expect, type APIRequestContext, type Page } from '@playwright/test';
import { originalComparison, type Workspace } from '../src/api';
import { selectExperiment, selectTheme, selectWorkflow } from './ui';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as Workspace[];
}
test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await opened(request)) {
    const response = await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
    expect(response.status()).toBe(204);
  }
});

test('original comparison uses the current experiment and its protected source', async ({
  request,
}) => {
  const token = (await (await request.get('/api/session')).json()).token;
  const response = await request.post('/api/workspaces/synthetic', {
    headers: { 'x-mrax-token': token },
  });
  const original = (await response.json()) as Workspace;
  expect(originalComparison(original).left.experiment).toBe(original.experiment);
  expect(originalComparison(original).left.kind).toBe('original');
  expect(originalComparison(original).right.experiment).toBe(original.experiment);
  expect(originalComparison(original).right.kind).toBe('processed');
  const current = {
    ...original,
    pipeline: { ...original.pipeline, steps: [{} as Workspace['pipeline']['steps'][number]] },
  };
  expect(originalComparison(current).left.experiment).toBe(current.experiment);
  expect(originalComparison(current).right.experiment).toBe(current.experiment);
});

async function study(page: Page) {
  await page.setViewportSize({ width: 1600, height: 1080 });
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
}

async function drag(page: Page, fraction: number, release = true) {
  const divider = page.getByRole('separator', { name: 'Move comparison divider', exact: true });
  const box = (await divider.boundingBox())!;
  const surface = (await page.locator('.comparison-surface').boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(surface.x + surface.width * fraction, box.y + box.height / 2, {
    steps: 16,
  });
  if (release) await page.mouse.up();
}

test('pipeline shortcut compares original and processed with a responsive divider and no pixel reloads', async ({
  page,
  request,
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await study(page);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('1.2');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(1);
  const [before] = await opened(request);
  await pipeline.getByRole('button', { name: 'Compare', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:.*Original/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /^Image B:.*Processed/ })).toBeVisible();
  const divider = page.getByRole('separator', { name: 'Move comparison divider', exact: true });
  await expect(divider).toHaveAttribute('aria-valuenow', '50');
  const requests: string[] = [];
  page.on('request', (request) => {
    if (/\/(comparison|slice|tasks)(?:\?|$)/.test(request.url())) requests.push(request.url());
  });
  await drag(page, 0.72, false);
  await expect(divider).toHaveAttribute('aria-valuenow', '72');
  expect(requests).toEqual([]);
  await page.mouse.up();
  await expect
    .poll(async () => (await opened(request))[0].scene.comparison.position)
    .toBeCloseTo(0.72, 2);
  expect(requests).toEqual([]);
  await divider.press('ArrowLeft');
  await expect(divider).toHaveAttribute('aria-valuenow', '70');
  await expect
    .poll(async () => Math.round((await opened(request))[0].scene.comparison.position * 100))
    .toBe(70);
  await drag(page, 0.32, false);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(divider).toHaveAttribute('aria-valuenow', '70');
  await divider.dblclick();
  await expect(divider).toHaveAttribute('aria-valuenow', '50');
  await selectTheme(page, 'light');
  await page.screenshot({ path: info.outputPath('split-swipe-light.png') });
  await page.getByRole('button', { name: 'Side by side', exact: true }).click();
  await expect(page.locator('.comparison-surface')).toHaveClass(/side-by-side/);
  const left = page.locator('.comparison-left');
  const initialWidth = (await left.boundingBox())!.width;
  await drag(page, 0.66);
  await expect
    .poll(async () => (await left.boundingBox())!.width)
    .toBeGreaterThan(initialWidth * 1.25);
  await page.getByText('Image options', { exact: true }).click();
  await page.getByLabel('Image B palette', { exact: true }).selectOption('hot');
  await page.getByLabel('Minimum display value', { exact: true }).fill('500000');
  await page.getByLabel('Minimum display value', { exact: true }).press('Enter');
  await expect(page.getByLabel('Minimum display value', { exact: true })).toHaveValue('500000');
  await page.getByLabel('Maximum display value', { exact: true }).fill('700000');
  await page.getByLabel('Maximum display value', { exact: true }).press('Enter');
  await expect
    .poll(async () => (await opened(request))[0].scene.comparison.left.window?.center)
    .toBe(600000);
  expect(requests).toEqual([]);
  const [after] = await opened(request);
  expect(after.pipeline).toEqual(before.pipeline);
  expect(after.revision).toBe(before.revision);
  expect(after.experiments).toEqual(before.experiments);
  await selectTheme(page, 'dark');
  await page.getByText('Image options', { exact: true }).click();
  await page.screenshot({ path: info.outputPath('split-side-by-side-dark.png') });
  await page.reload();
  await expect(page.getByRole('img', { name: /^Image B:.*Processed/ })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Side by side', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(divider).toHaveAttribute('aria-valuenow', '66');
  await page.getByRole('button', { name: 'Image', exact: true }).click();
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await page.getByRole('button', { name: 'Original vs processed', exact: true }).click();
  await expect(divider).toHaveAttribute('aria-valuenow', '50');
  await selectWorkflow(page, 'Segment');
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Draw rectangle', exact: true })).toBeEnabled();
  expect(errors).toEqual([]);
});

test('split compares arbitrary frames and follows shared frame and plane navigation while retaining overlays', async ({
  page,
  request,
}) => {
  await study(page);
  const token = (await (await request.get('/api/session')).json()).token;
  const [w] = await opened(request);
  const layer = { id: 'echo', experiment: w.experiment, kind: 'image', frame: 2, opacity: 0.3 };
  const updated = await request.patch(`/api/workspaces/${w.id}/view`, {
    headers: { 'x-mrax-token': token },
    data: { revision: w.revision, scene: { layers: [layer] } },
  });
  expect(updated.ok()).toBe(true);
  await page.reload();
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:.*frame 1$/ })).toBeVisible();
  const slider = page.getByRole('slider', { name: 'Frame', exact: true });
  await slider.focus();
  await slider.press('ArrowRight');
  await expect(page.getByRole('img', { name: /^Image A:.*frame 2$/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /^Image B:.*frame 2$/ })).toBeVisible();
  await page.getByText('Image options', { exact: true }).click();
  await page.getByLabel('Image A frame', { exact: true }).selectOption('0');
  await expect(page.getByRole('img', { name: /^Image A:.*frame 1$/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /^Image B:.*frame 2$/ })).toBeVisible();
  await page.getByRole('button', { name: 'Swap images', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:.*frame 2$/ })).toBeVisible();
  await expect(page.getByRole('img', { name: /^Image B:.*frame 1$/ })).toBeVisible();
  await page.getByLabel('Plane', { exact: true }).selectOption('coronal');
  await expect(page.locator('.comparison-surface[aria-busy="false"]')).toBeVisible();
  const [after] = await opened(request);
  expect(after.scene.comparison.plane).toBe('coronal');
  expect(after.plane).toBe(w.plane);
  expect(after.frame).toBe(w.frame);
  expect(after.scene.layers).toMatchObject([layer]);
  await page.getByRole('button', { name: 'Image', exact: true }).click();
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  expect((await opened(request))[0].scene.layers).toEqual(after.scene.layers);
});

test('different source grids remain available side by side with an explicit swipe explanation', async ({
  page,
  request,
}, info) => {
  const token = (await (await request.get('/api/session')).json()).token;
  const response = await request.post('/api/workspaces/synthetic?kind=multimodal', {
    headers: { 'x-mrax-token': token },
  });
  expect(response.ok()).toBe(true);
  await page.setViewportSize({ width: 1600, height: 1080 });
  await page.goto('/');
  await selectExperiment(page, 'Complex coils');
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(page.getByRole('img', { name: /^Image A:/ })).toBeVisible();
  await page
    .getByLabel('Image B', { exact: true })
    .selectOption({ label: 'Motion reference: Original' });
  const geometryNotice = page.locator('.comparison-view > .hint').first();
  await expect(geometryNotice).toContainText('Different image grids: shown side by side.');
  await expect(geometryNotice).toContainText(/register/i);
  await expect(page.locator('.comparison-surface')).toHaveClass(/side-by-side/);
  const dimensions = await page
    .locator('.comparison-pane canvas')
    .evaluateAll((canvases) =>
      canvases.map((canvas) => [
        (canvas as HTMLCanvasElement).width,
        (canvas as HTMLCanvasElement).height,
      ]),
    );
  expect(dimensions[0]).not.toEqual(dimensions[1]);
  await page.getByRole('button', { name: 'Side by side', exact: true }).click();
  await page.screenshot({ path: info.outputPath('split-native-grids.png') });
  await page
    .getByLabel('Image B', { exact: true })
    .selectOption({ label: 'Complex coils: Processed' });
  await page.getByRole('button', { name: 'Swipe', exact: true }).click();
  await expect(page.locator('.comparison-surface')).toHaveClass(/swipe/);
  await expect(page.getByText(/Different image grids:/)).toHaveCount(0);
});
