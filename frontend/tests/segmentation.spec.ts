import { test, expect, type Page, type APIRequestContext } from '@playwright/test';

import type { Workspace } from '../src/api';

import {
  loadExample,
  selectWorkflow,
  openRoiCoordinates,
  saveWorkspace,
  openWorkspace,
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

async function create(page: Page, name: string, bounds: number[]) {
  await openRoiCoordinates(page);
  await page.getByLabel('New ROI name', { exact: true }).fill(name);
  for (const [index, label] of ['column', 'row', 'width', 'height'].entries())
    await page.getByLabel('ROI ' + label, { exact: true }).fill(String(bounds[index]));
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await expect(page.locator('.roi-card.selected strong')).toHaveText(name);
}

async function draw(page: Page, start: number[], end: number[]) {
  const surface = page.getByTestId('roi-surface');
  await surface.scrollIntoViewIfNeeded();
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + (box.width * start[0]) / 32, box.y + (box.height * start[1]) / 24);
  await page.mouse.down();
  await page.mouse.move(box.x + (box.width * end[0]) / 32, box.y + (box.height * end[1]) / 24, {
    steps: 5,
  });
  await page.mouse.up();
}

test('binary and indexed masks stay editable across 3D expansion and workspace save/load', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment');
  await create(page, 'A', [2, 2, 4, 4]);
  await create(page, 'B', [4, 4, 4, 4]);
  await page.getByRole('separator', { name: 'Resize analysis controls', exact: true }).press('End');
  await selectWorkflow(page, 'Segment', 'Combine ROIs');
  await page.getByLabel('Include A', { exact: true }).check();
  await page.getByLabel('Include B', { exact: true }).check();
  await page.getByLabel('Combined ROI name', { exact: true }).fill('Joint');
  await page.getByRole('button', { name: 'Create joint binary ROI', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Joint')?.pixels)
    .toBe(28);
  await page.getByLabel('Combined mask type', { exact: true }).selectOption('indexed');
  await page.getByLabel('Combined ROI name', { exact: true }).fill('Labels');
  await page.getByRole('button', { name: 'Create indexed label mask', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].rois.find((r) => r.name === 'Labels')?.labels)
    .toEqual([1, 2]);
  await page.screenshot({ path: info.outputPath('segment-combine.png') });
  await selectWorkflow(page, 'Segment', 'Edit label masks');
  await expect(page.getByRole('button', { name: 'Paint labels', exact: true })).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'Auto subtract', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Use label 2', exact: true }).click();
  await page.getByLabel('Replacement label', { exact: true }).fill('7');
  await page.getByRole('button', { name: 'Relabel selected label', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Use label 7', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Use label 7', exact: true }).click();
  await page.getByRole('checkbox', { name: 'Erase only label 7', exact: true }).check();
  await page.getByRole('button', { name: 'Eraser tool', exact: true }).click();
  await page.getByRole('button', { name: 'Erase labels', exact: true }).click();
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
  await page.screenshot({ path: info.outputPath('segment-label-mask.png') });
  await selectWorkflow(page, 'Segment', '3D expansion');
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
  await page.screenshot({ path: info.outputPath('segment-3d.png') });
  const file = info.outputPath('label-masks.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  const restored = (await opened(request)).find((w) => w.id !== before.id)!;
  expect(restored.rois).toEqual(before.rois);
});
