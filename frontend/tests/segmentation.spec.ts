import { test, expect, type Page, type APIRequestContext } from '@playwright/test';
import { readFile, readdir } from 'node:fs/promises';

import type { Workspace } from '../src/api';

import {
  loadExample,
  selectWorkflow,
  saveWorkspace,
  openWorkspace,
  chooseFilePath,
  chooseExportFolder,
} from './ui';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as Workspace[];
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await opened(request)) {
    await request.delete('/api/workspaces/' + w.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
  }
});

async function draw(page: Page, start: number[], end: number[], preview?: () => Promise<void>) {
  const surface = page.getByTestId('roi-surface');
  await surface.scrollIntoViewIfNeeded();
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + (box.width * start[0]) / 32, box.y + (box.height * start[1]) / 24);
  await page.mouse.down();
  await page.mouse.move(box.x + (box.width * end[0]) / 32, box.y + (box.height * end[1]) / 24, {
    steps: 5,
  });
  await preview?.();
  await page.mouse.up();
}

test('ROI editing and file exchange preserve binary and indexed masks', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment');
  const imageReads: string[] = [];
  page.on('request', (request) => {
    if (/\/slice(?:\?|$)/.test(request.url())) imageReads.push(request.url());
  });
  const rectangle = page.getByRole('button', { name: 'Rectangle tool', exact: true });
  await expect(rectangle).toHaveAttribute('aria-pressed', 'false');
  await rectangle.click();
  await expect(page.locator('.image-surface')).toHaveClass(/drawing/);
  await rectangle.click();
  await expect(page.locator('.image-surface')).not.toHaveClass(/drawing/);
  await rectangle.click();
  await expect(page.getByRole('button', { name: 'Pan image tool', exact: true })).toHaveCount(0);
  for (const [index, name] of ['A', 'B'].entries()) {
    await draw(page, [2.1 + index * 2, 2.1 + index * 2], [5.9 + index * 2, 5.9 + index * 2]);
    await expect(page.locator('.roi-card.selected strong')).toHaveText('ROI ' + (index + 1));
    await page.getByLabel('Rename selected ROI', { exact: true }).fill(name);
    await page.getByRole('button', { name: 'Rename ROI', exact: true }).click();
    await expect(page.locator('.roi-card.selected strong')).toHaveText(name);
    await expect(page.locator('.image-surface')).toHaveClass(/drawing/);
  }
  await page.getByRole('slider', { name: 'Opacity for B', exact: true }).press('Home');
  await expect.poll(async () => (await opened(request))[0].scene.roi_styles.B?.opacity).toBe(0);
  await expect(
    page
      .locator('.roi-mask')
      .filter({ has: page.getByRole('img', { name: 'B · label 1', exact: true }) })
      .locator('.roi-outline'),
  ).toHaveAttribute('stroke-opacity', '1');
  expect(imageReads).toEqual([]);
  await page.getByRole('button', { name: 'Lasso tool', exact: true }).click();
  const surface = page.getByTestId('roi-surface');
  await surface.scrollIntoViewIfNeeded();
  const image = (await surface.boundingBox())!;
  await page.mouse.move(image.x + image.width * 0.4, image.y + image.height * 0.2);
  await page.mouse.down();
  for (const [x, y] of [
    [0.6, 0.2],
    [0.6, 0.4],
    [0.4, 0.4],
    [0.4, 0.2],
  ])
    await page.mouse.move(image.x + image.width * x, image.y + image.height * y, { steps: 3 });
  await page.mouse.up();
  await expect(page.locator('.roi-card.selected strong')).toHaveText('ROI 3');
  await expect(page.getByRole('img', { name: 'ROI 3 · label 1', exact: true })).toBeVisible();
  await expect(page.locator('.image-surface')).toHaveClass(/drawing/);
  const lasso = (await opened(request))[0].rois.find((roi) => roi.name === 'ROI 3')!;
  await page.getByRole('button', { name: 'Brush tool', exact: true }).click();
  const mode = page.getByLabel('Drawing mode', { exact: true });
  await expect(mode).toBeVisible();
  await mode.selectOption('add');
  await page.getByLabel('Brush size', { exact: true }).fill('3');
  await draw(page, [15, 6], [22, 8], async () => {
    await expect(surface.locator('.roi-brush-draft')).toBeVisible();
    await expect(surface.locator('.roi-draft')).toHaveCount(0);
  });
  await expect
    .poll(async () => (await opened(request))[0].rois.find((roi) => roi.name === 'ROI 3')!.pixels)
    .toBeGreaterThan(lasso.pixels);
  const painted = (await opened(request))[0].rois.find((roi) => roi.name === 'ROI 3')!.pixels;
  await page.getByRole('button', { name: 'Eraser tool', exact: true }).click();
  await expect(mode).toBeVisible();
  await expect(mode).toHaveValue('subtract');
  await draw(page, [15, 6], [22, 8]);
  await expect
    .poll(async () => (await opened(request))[0].rois.find((roi) => roi.name === 'ROI 3')!.pixels)
    .toBeLessThan(painted);
  for (let i = 0; i < 2; i++)
    await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((roi) => roi.name === 'ROI 3')!.mask)
    .toEqual(lasso.mask);
  await page.getByRole('button', { name: 'Brush tool', exact: true }).click();
  await mode.selectOption('new');
  await draw(page, [24, 16], [29, 17]);
  await expect(page.locator('.roi-card.selected strong')).toHaveText('ROI 4');
  expect((await opened(request))[0].rois.find((roi) => roi.name === 'ROI 4')!.metadata.kind).toBe(
    'mask',
  );
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect(page.locator('.roi-card')).toHaveCount(3);
  await page
    .getByRole('group', { name: 'ROI tool', exact: true })
    .getByRole('button', { name: 'Otsu threshold', exact: true })
    .click();
  const otsu = page.getByRole('form', { name: 'Otsu segmentation', exact: true });
  await expect(otsu).toBeVisible();
  await expect(page.locator('.roi-card')).toHaveCount(3);
  await expect(page.locator('.image-surface')).not.toHaveClass(/drawing/);
  await otsu.getByLabel('Otsu foreground', { exact: true }).selectOption('dark');
  await otsu.getByLabel('Otsu regions', { exact: true }).selectOption('all');
  await otsu.getByLabel('Histogram bins', { exact: true }).fill('64');
  await otsu.getByRole('checkbox', { name: 'Fill holes', exact: true }).uncheck();
  await expect(page.locator('.roi-card')).toHaveCount(3);
  await otsu.getByRole('button', { name: 'Apply Otsu', exact: true }).click();
  await expect(page.locator('.roi-card.selected strong')).toHaveText('ROI 4');
  expect((await opened(request))[0].rois.at(-1)!.metadata).toMatchObject({
    foreground: 'dark',
    largest_component: false,
    fill_holes: false,
    bins: 64,
  });
  await expect(page.locator('.image-surface')).not.toHaveClass(/drawing/);
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect(page.locator('.roi-card')).toHaveCount(3);
  await rectangle.click();
  await expect(otsu).not.toBeVisible();
  await page.getByRole('separator', { name: 'Resize analysis controls', exact: true }).press('End');
  await selectWorkflow(page, 'Segment', 'Edit ROIs and masks');
  await expect(page.getByRole('region', { name: 'Combine ROIs', exact: true })).toBeVisible();
  await expect(
    page.getByRole('region', { name: 'Expand across slices', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('form', { name: 'ROI coordinate editor' })).not.toBeVisible();
  await page.getByLabel('Combine method', { exact: true }).selectOption('subtract');
  await page.getByLabel('Subtract from', { exact: true }).selectOption('B');
  await page
    .getByRole('group', { name: 'ROIs to subtract', exact: true })
    .getByLabel('Include A', { exact: true })
    .check();
  await page.getByRole('button', { name: 'Subtract A from B', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'B')?.pixels)
    .toBe(12);
  expect((await opened(request))[0].rois.find((r) => r.name === 'A')?.pixels).toBe(16);
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'B')?.pixels)
    .toBe(16);
  await page.getByLabel('Drawing mode', { exact: true }).selectOption('subtract');
  await expect(
    page.getByLabel('Drawing mode', { exact: true }).locator('option:checked'),
  ).toHaveText('Subtract drawn region from ROI 3');
  await page.getByLabel('Drawing mode', { exact: true }).selectOption('new');
  await page.getByLabel('Combine method', { exact: true }).selectOption('binary');
  await page.getByLabel('Include A', { exact: true }).check();
  await page.getByLabel('Include B', { exact: true }).check();
  await page.getByLabel('Combined ROI name', { exact: true }).fill('Joint');
  await page.getByRole('button', { name: 'Create joint binary ROI', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Joint')?.pixels)
    .toBe(28);
  await page.getByLabel('Combine method', { exact: true }).selectOption('indexed');
  await page.getByLabel('Combined ROI name', { exact: true }).fill('Labels');
  await page.getByRole('button', { name: 'Create indexed label mask', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Labels')?.labels)
    .toEqual([1, 2]);
  const labelOne = page.getByRole('img', { name: 'Labels · label 1', exact: true });
  const labelTwo = page.getByRole('img', { name: 'Labels · label 2', exact: true });
  await page
    .getByLabel('Color for label 1 in Labels', { exact: true })
    .filter({ visible: true })
    .fill('#cc3355');
  await expect(labelOne).toHaveCSS('fill', 'rgb(204, 51, 85)');
  const labelOpacity = page.getByRole('slider', {
    name: 'Opacity for label 2 in Labels',
    exact: true,
  });
  await labelOpacity.fill('0.25');
  await expect(labelTwo).toHaveCSS('fill-opacity', '0.25');
  await expect(labelOne).toHaveCSS('fill-opacity', '0.35');
  await labelOpacity.dispatchEvent('pointerup');
  await expect
    .poll(async () => (await opened(request))[0].scene.roi_styles.Labels?.label_opacities['2'])
    .toBe(0.25);
  await page.getByRole('checkbox', { name: 'Show label 2 in Labels', exact: true }).uncheck();
  await expect(labelTwo).toHaveCount(0);
  await expect(labelOne).toBeVisible();
  expect((await opened(request))[0].rois.find((r) => r.name === 'Labels')?.pixels).toBe(28);
  await page.getByRole('button', { name: 'Use label 2', exact: true }).click();
  await expect(labelTwo).toBeVisible();
  await expect(
    page.getByRole('checkbox', { name: 'Keep other labels when painting 2', exact: true }),
  ).toBeChecked();
  await page
    .getByRole('button', { name: 'Reset appearance for label 2 in Labels', exact: true })
    .click();
  await expect(labelTwo).toHaveCSS('fill-opacity', '0.35');
  await expect(labelOne).toHaveCSS('fill', 'rgb(204, 51, 85)');
  await expect(page.getByRole('button', { name: 'Brush tool', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(
    page.getByRole('checkbox', { name: 'Auto subtract from other ROIs', exact: true }),
  ).toBeVisible();
  await page.getByLabel('Replacement label', { exact: true }).fill('7');
  await page.getByRole('button', { name: 'Relabel selected label', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Use label 7', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Use label 7', exact: true }).click();
  await page.getByRole('button', { name: 'Eraser tool', exact: true }).click();
  await page.getByRole('checkbox', { name: 'Erase only label 7', exact: true }).check();
  await draw(page, [5, 5], [6, 6]);
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Labels')?.pixels)
    .toBeLessThan(28);
  const edited = (await opened(request))[0].rois.find((r) => r.name === 'Labels')!;
  expect(edited.mask.flat().filter((value) => value === 1)).toHaveLength(16);
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Labels')?.pixels)
    .toBe(28);
  await page.getByRole('button', { name: 'Copy current slice', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Labels')?.pixels)
    .toBe(84);
  await page.getByRole('button', { name: 'Grow ROI in 3D', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Labels')?.pixels)
    .toBeGreaterThan(84);
  const before = (await opened(request))[0];
  expect(before.rois.slice(0, 2).map((r) => r.pixels)).toEqual([16, 16]);
  const file = info.outputPath('label-masks.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  const restored = (await opened(request)).find((w) => w.id !== before.id)!;
  expect(restored.rois).toEqual(before.rois);

  await selectWorkflow(page, 'Segment', 'Import and export ROIs');
  await expect(page.getByRole('group', { name: 'ROI tool', exact: true })).not.toBeVisible();
  await expect(page.getByRole('region', { name: 'Export ROIs and masks' })).toBeVisible();
  await page.getByLabel('Export ROI', { exact: true }).selectOption('roi/Labels');
  const filename = page.getByLabel('ROI export file name', { exact: true });
  await expect(filename).toHaveValue('Labels.npz');
  await page.getByLabel('Export format', { exact: true }).selectOption('npy');
  await expect(filename).toHaveValue('Labels.npy');
  await filename.fill('lesion-mask.npy');
  const maskFile = info.outputPath('lesion-mask.npy');
  await page.getByRole('button', { name: 'Export selected mask', exact: true }).click();
  await expect(
    page.locator('.file-system-dialog').getByLabel('File name', { exact: true }),
  ).toHaveValue('lesion-mask.npy');
  await chooseFilePath(page, maskFile, 'Save');
  await expect(page.locator('#analysis-controls .job-status .file-source')).toContainText(maskFile);
  expect((await readFile(maskFile)).subarray(0, 6).toString('latin1')).toBe('\x93NUMPY');
  await page.getByLabel('Export format', { exact: true }).selectOption('nifti');
  await expect(filename).toHaveValue('lesion-mask.nii');
  await filename.fill('../outside.nii');
  await expect(
    page.getByRole('button', { name: 'Export selected mask', exact: true }),
  ).toBeDisabled();
  await filename.fill('all-regions.npz');
  const roiFolder = info.outputPath('ROI set');
  await page.getByRole('button', { name: 'Save ROI set', exact: true }).click();
  await chooseExportFolder(page, roiFolder, true);
  await expect(page.locator('#analysis-controls .job-status .file-source')).toContainText(
    roiFolder,
  );
  expect((await readdir(roiFolder)).sort()).toEqual(['all-regions.json', 'all-regions.npz']);
  const priorIds = new Set((await opened(request)).map((w) => w.id));
  await loadExample(page);
  const target = (await opened(request)).find((w) => !priorIds.has(w.id))!;
  await selectWorkflow(page, 'Segment', 'Import and export ROIs');
  await page.getByRole('button', { name: 'Open ROI set or mask…', exact: true }).click();
  await chooseFilePath(page, roiFolder + '/all-regions.npz');
  await expect
    .poll(async () => (await opened(request)).find((w) => w.id === target.id)?.rois)
    .toEqual(before.rois);
  await selectWorkflow(page, 'Segment', 'Edit ROIs and masks');
  await expect(page.locator('.roi-card')).toHaveCount(before.rois.length);
  const masked = (await opened(request)).at(-1)!;
  const token = (await (await request.get('/api/session')).json()).token;
  const seeded = await request.post('/api/workspaces/' + masked.id + '/actions', {
    headers: { 'x-mrax-token': token },
    data: {
      revision: masked.revision,
      view_revision: masked.view_revision,
      action: 'visual.update',
      options: { show_original: false, mask_enabled: true, mask_roi: 'A' },
    },
  });
  expect(seeded.ok()).toBe(true);
  await page.reload();
  await selectWorkflow(page, 'Segment');
  await page.locator('.roi-card').getByRole('button', { name: /^A / }).click();
  await expect(page.getByRole('button', { name: 'Brush tool', exact: true })).toBeDisabled();
  const maskedPixels = await (await request.get('/api/workspaces/' + masked.id + '/slice')).body();
  const readsBefore = imageReads.length;
  await page.getByRole('button', { name: 'Delete ROI', exact: true }).click();
  await expect.poll(() => imageReads.length).toBeGreaterThan(readsBefore);
  await expect
    .poll(async () =>
      (await opened(request)).find((w) => w.id === masked.id)?.rois.some((roi) => roi.name === 'A'),
    )
    .toBe(false);
  expect(await (await request.get('/api/workspaces/' + masked.id + '/slice')).body()).not.toEqual(
    maskedPixels,
  );
});
