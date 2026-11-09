import { test, expect, type APIRequestContext, type Page } from '@playwright/test';
import type { Workspace } from '../src/api';
import {
  loadExample,
  selectWorkflow,
  openRoiCoordinates,
  saveWorkspace,
  openWorkspace,
} from './ui';
import { mkdir } from 'node:fs/promises';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as Workspace[];
}
test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await opened(request))
    await request.delete('/api/workspaces/' + w.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
});
async function create(page: Page) {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await openRoiCoordinates(page);
  await page.getByLabel('New ROI name', { exact: true }).fill('Anterior');
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await expect(page.locator('.roi-card.selected strong')).toHaveText('Anterior');
}

test('individual ROI colors and opacity update across workflows and survive workspace reload', async ({
  page,
  request,
}, info) => {
  await create(page);
  const before = (await opened(request))[0];
  const original = await (await request.get('/api/workspaces/' + before.id + '/slice')).body();
  const appearance = page.getByRole('group', { name: 'ROI appearance', exact: true });
  await appearance.getByLabel('Color for Anterior', { exact: true }).fill('#c462ee');
  await expect(
    page.getByTestId('roi-surface').getByRole('img', { name: 'Anterior · label 1', exact: true }),
  ).toHaveCSS('fill', 'rgb(196, 98, 238)');
  const slider = appearance.getByLabel('Opacity for Anterior', { exact: true });
  await slider.focus();
  await slider.press('End');
  await slider.press('ArrowLeft');
  await expect(page.locator('.roi-mask')).toHaveCSS('opacity', '0.95');
  const after = (await opened(request))[0];
  expect(after.revision).toBe(before.revision);
  expect(after.rois[0].mask).toEqual(before.rois[0].mask);
  expect(await (await request.get('/api/workspaces/' + before.id + '/slice')).body()).toEqual(
    original,
  );
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await expect(appearance.getByLabel('Color for Anterior', { exact: true })).toHaveValue('#c462ee');
  await expect(appearance.getByLabel('Opacity for Anterior', { exact: true })).toHaveValue('0.95');
  const output = info.outputPath('saved');
  await mkdir(output, { recursive: true });
  await saveWorkspace(page, output + '/appearance.mrax');
  await openWorkspace(page, output + '/appearance.mrax');
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await expect(appearance.getByLabel('Color for Anterior', { exact: true })).toHaveValue('#c462ee');
  await expect(appearance.getByLabel('Opacity for Anterior', { exact: true })).toHaveValue('0.95');
  await appearance.getByLabel('Opacity for Anterior', { exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: info.outputPath('roi-appearance.png') });
  await page.getByRole('button', { name: 'Reset ROI appearance', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Reset ROI appearance', exact: true }),
  ).toBeDisabled();
});

test('shape conversion preserves voxels, enables brush editing and has an exact native export preview', async ({
  page,
  request,
}, info) => {
  await create(page);
  const before = (await opened(request))[0];
  const mask = before.rois[0].mask;
  await expect(page.getByTestId('roi-handle-0')).toBeVisible();
  await page.getByRole('button', { name: 'Preview export mask', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Export mask preview', exact: true });
  await expect(dialog).toBeVisible();
  const image = dialog.getByRole('img', { name: 'Native export mask for Anterior', exact: true });
  await expect(image.locator('path')).toHaveCount(1);
  await image.focus();
  for (let i = 0; i < 5; i++) await image.press('ArrowDown');
  for (let i = 0; i < 5; i++) await image.press('ArrowRight');
  await expect(dialog.getByRole('status', { name: 'Mask label value', exact: true })).toContainText(
    'Row 5, column 5 · Label 1',
  );
  await page.screenshot({ path: info.outputPath('native-export-mask.png') });
  await dialog.getByLabel('Mask preview plane', { exact: true }).selectOption('x,z');
  await expect(image).toHaveAttribute('viewBox', '0 0 3 24');
  await dialog.getByRole('button', { name: 'Close preview', exact: true }).click();
  expect((await opened(request))[0].revision).toBe(before.revision);
  await page.getByRole('button', { name: 'Convert to pixel mask', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Brush tool', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByTestId('roi-handle-0')).toBeHidden();
  expect((await opened(request))[0].rois[0].mask).toEqual(mask);
  const surface = page.getByTestId('roi-surface');
  await surface.scrollIntoViewIfNeeded();
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + (box.width * 23) / 32, box.y + (box.height * 18) / 24);
  await page.mouse.down();
  await page.mouse.move(box.x + (box.width * 25) / 32, box.y + (box.height * 18) / 24, {
    steps: 3,
  });
  await page.mouse.up();
  await expect
    .poll(async () => (await opened(request))[0].rois[0].pixels)
    .toBeGreaterThan(before.rois[0].pixels);
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  expect((await opened(request))[0].rois[0].mask).toEqual(mask);
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Convert to pixel mask', exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: info.outputPath('roi-appearance-controls.png') });
});

test('indexed label colors stay distinct and export preview remains visible at zero overlay opacity', async ({
  page,
  request,
}) => {
  await create(page);
  await page.getByLabel('Drawing mode', { exact: true }).selectOption('add');
  await page.getByLabel('Label value', { exact: true }).fill('2');
  await page.getByLabel('ROI column', { exact: true }).fill('20');
  await page.getByLabel('ROI row', { exact: true }).fill('5');
  await page.getByLabel('ROI width', { exact: true }).fill('5');
  await page.getByLabel('ROI height', { exact: true }).fill('5');
  await page.getByRole('button', { name: 'Add rectangle round', exact: true }).click();
  const before = (await opened(request))[0];
  expect(before.rois[0].labels).toEqual([1, 2]);
  const appearance = page.getByRole('group', { name: 'ROI appearance', exact: true });
  await appearance.getByText('Label colors', { exact: true }).click();
  await appearance.getByLabel('Color for label 2 in Anterior', { exact: true }).fill('#22bb55');
  await expect(
    page.getByTestId('roi-surface').getByRole('img', { name: 'Anterior · label 2', exact: true }),
  ).toHaveCSS('fill', 'rgb(34, 187, 85)');
  await appearance.getByLabel('Opacity for Anterior', { exact: true }).press('Home');
  await expect(page.locator('.roi-mask')).toHaveCSS('opacity', '0');
  await page.getByRole('button', { name: 'Preview export mask', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Export mask preview', exact: true });
  const mask = dialog.getByRole('img', { name: 'Native export mask for Anterior', exact: true });
  await expect(mask.locator('path[fill="#22bb55"]')).toBeVisible();
  await expect(mask).toHaveCSS('opacity', '1');
  expect((await opened(request))[0].rois[0].mask).toEqual(before.rois[0].mask);
});
