import { zoomImage } from './ui';
import { clickDataControl, loadExample } from './ui';
import { openRoiCoordinates } from './ui';
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { chooseFilePath, saveWorkspace, openWorkspace } from './ui';
import { test, expect, type Page, type APIRequestContext } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import {
  selectDataFormat,
  selectWorkflow,
  openMetadataBrowser,
  closeMetadataBrowser,
  selectExperiment,
  selectTheme,
  openExport,
} from './ui';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as {
    id: string;
    revision: number;
  }[];
}
test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await opened(request)) {
    const r = await request.delete(`/api/workspaces/${w.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
    expect(r.status()).toBe(204);
  }
});
async function synthetic(page: Page, multimodal = false) {
  await page.goto('/');
  await loadExample(page, multimodal ? 'multimodal' : 't2');
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await selectWorkflow(page, 'Segment');
}
async function waitComplete(page: Page) {
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete', {
    timeout: 60000,
  });
}

test('complex views, planes, integer labels, brush, and undo use the real backend', async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await synthetic(page, true);
  await selectExperiment(page, 'Complex coils');
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Model', { exact: true })).toHaveValue('identity');
  await expect(page.getByLabel('t2 initial', { exact: true })).toHaveCount(0);
  await selectWorkflow(page, 'View');
  await page.getByLabel('Component', { exact: true }).selectOption('real');
  await page.getByLabel('frame', { exact: true }).fill('1');
  await page.getByLabel('Plane', { exact: true }).selectOption('coronal');
  await expect(page.getByRole('img', { name: /24 by 3 quantitative image/ })).toBeVisible();
  await selectWorkflow(page, 'Segment');
  await page.getByRole('button', { name: 'Ellipse tool', exact: true }).click();
  await page.getByLabel('Label value', { exact: true }).fill('7');
  await openRoiCoordinates(page);
  await page.getByLabel('ROI column', { exact: true }).fill('0');
  await page.getByLabel('ROI row', { exact: true }).fill('3');
  await page.getByLabel('ROI width', { exact: true }).fill('2');
  await page.getByLabel('ROI height', { exact: true }).fill('8');
  await page.getByRole('button', { name: 'Add ellipse by coordinates', exact: true }).click();
  await expect(page.locator('.roi-card').first().getByText('7', { exact: true })).toBeVisible();
  const count = await page.locator('.roi-card').first().locator('button span').textContent();
  await page.getByRole('button', { name: 'Brush tool', exact: true }).click();
  await page.getByLabel('Label value', { exact: true }).fill('9');
  await page.getByRole('button', { name: 'Draw brush', exact: true }).click();
  const box = (await page.getByTestId('roi-surface').boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 5 });
  await page.mouse.up();
  await expect(page.locator('.roi-card').first().locator('.label-badge')).toContainText('9');
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect(page.locator('.roi-card').first().locator('button span')).toHaveText(count!);
  await expect(page.locator('.roi-card').first().locator('.label-badge')).toHaveText('7');
  await page.getByLabel('Plane', { exact: true }).selectOption('axial');
  await selectWorkflow(page, 'Reconstruct');
  await page
    .getByRole('group', { name: 'Reconstruction display', exact: true })
    .getByRole('button', { name: 'Cartesian Fourier', exact: true })
    .click();
  await page.getByLabel('Reconstruction component', { exact: true }).selectOption('phase');
  await page.getByLabel('Coil', { exact: true }).selectOption('0');
  await page.getByLabel('Unwrap phase', { exact: true }).uncheck();
  await page
    .getByRole('group', { name: 'Reconstruction display', exact: true })
    .getByRole('button', { name: 'K-space', exact: true })
    .click();
  await page.getByLabel('K-space component', { exact: true }).selectOption('log_magnitude');
  await selectExperiment(page, 'T2 phantom');
  await openMetadataBrowser(page, 'Experiment metadata');
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  await expect(page.getByLabel('Experiment metadata', { exact: true })).toHaveValue(/"TE"/);
  await selectExperiment(page, 'Signed with missing samples');
  await openMetadataBrowser(page, 'Experiment metadata');
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  await expect(page.getByLabel('Experiment metadata', { exact: true })).not.toHaveValue(/"TE"/);
  await closeMetadataBrowser(page);
  await selectWorkflow(page, 'Segment');
  await selectTheme(page, 'dark');
  await page.screenshot({ path: testInfo.outputPath('multimodal-dark.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('raw-file import declares shape, byte order and dimensions and preserves signed data', async ({
  page,
  request,
}, testInfo) => {
  await page.goto('/');
  await clickDataControl(page, 'Import data');
  const values = Buffer.alloc(4 * 6 * 2);
  for (let i = 0; i < 24; i++) values.writeInt16BE(i - 12, i * 2);
  const source = testInfo.outputPath('signed.raw');
  await mkdir(dirname(source), { recursive: true });
  await writeFile(source, values);
  await page.getByRole('button', { name: 'Browse files and folders…', exact: true }).click();
  await chooseFilePath(page, source);
  await expect(page.locator('.file-source')).toContainText('signed.raw');
  await page.getByLabel('Import format', { exact: true }).selectOption('raw');
  await page.getByLabel('Array shape', { exact: true }).fill('4,6');
  await page.getByLabel('Dimension order', { exact: true }).fill('x,y');
  await page.getByLabel('Data type', { exact: true }).selectOption('int16');
  await page.getByLabel('Byte order', { exact: true }).selectOption('big');
  await page.getByRole('button', { name: 'Import study', exact: true }).click();
  await waitComplete(page);
  await expect(page.getByRole('img', { name: /4 by 6 quantitative image/ })).toBeVisible();
  const [w] = await opened(request);
  const response = await request.get(`/api/workspaces/${w.id}/slice`);
  const buffer = await response.body();
  const pixels = new Float32Array(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
  expect(Array.from(pixels)).toEqual(Array.from({ length: 24 }, (_, i) => i - 12));
});

test('geometry handles resize in native coordinates under zoom and retain labels', async ({
  page,
}) => {
  await synthetic(page);
  await page.getByLabel('Label value', { exact: true }).fill('7');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await expect(page.getByText('120 voxels', { exact: true })).toBeVisible();
  await selectWorkflow(page, 'View');
  await zoomImage(page, 2);
  await selectWorkflow(page, 'Segment');
  const handle = (await page.getByTestId('roi-handle-1').boundingBox())!,
    box = (await page.getByTestId('roi-surface').boundingBox())!;
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + (box.width * 18.75) / 32, box.y + (box.height * 14.75) / 24, {
    steps: 6,
  });
  await page.mouse.up();
  await expect(page.getByText('140 voxels', { exact: true })).toBeVisible();
  await expect(page.locator('.label-badge').first()).toHaveText('7');
  await page.getByRole('button', { name: 'Undo ROI edit', exact: true }).click();
  await expect(page.getByText('120 voxels', { exact: true })).toBeVisible();
});

test('frame selection, full map fit and NumPy export preserve quantitative maps', async ({
  page,
}, testInfo) => {
  await synthetic(page);
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'Fit');
  await page.getByText('Parameters and fit options', { exact: true }).click();
  await page.getByLabel('Fit steps', { exact: true }).fill('600');
  await page.getByLabel('Excluded frames', { exact: true }).fill('7');
  let finishUpdate = () => {};
  const updatePending = new Promise<void>((resolve) => {
    finishUpdate = resolve;
  });
  await page.route('**/api/workspaces/*/actions', async (route) => {
    if (route.request().postDataJSON().action === 'frames.exclude') await updatePending;
    await route.continue();
  });
  await page.getByRole('button', { name: 'Apply frame exclusions', exact: true }).click();
  try {
    await expect(page.getByLabel('Fit steps', { exact: true })).toBeDisabled();
    await expect(page.getByLabel('Step size', { exact: true })).toBeDisabled();
  } finally {
    finishUpdate();
  }
  await page.getByLabel('Fit steps', { exact: true }).fill('600');
  const fitting = page.waitForRequest(
    (request) =>
      request.url().endsWith('/tasks') && request.postDataJSON().operation === 'full_fit',
  );
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  expect((await fitting).postDataJSON().options.settings.num_steps).toBe(600);
  await waitComplete(page);
  await page
    .getByRole('group', { name: 'Fit display', exact: true })
    .getByRole('button', { name: 'Parameter map', exact: true })
    .click();
  await expect(page.getByRole('alert')).not.toBeVisible();
  await page.getByLabel('Map', { exact: true }).selectOption('t2');
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toBeVisible();
  await openMetadataBrowser(page);
  await page.getByLabel('Active experiment name', { exact: true }).fill('Renamed T2');
  await page.getByRole('button', { name: 'Rename experiment', exact: true }).click();
  await closeMetadataBrowser(page);
  const exportDialog = await openExport(page);
  await page
    .getByLabel('Export object', { exact: true })
    .selectOption({ label: 'Renamed T2 / map / t2 [ms]' });
  await page.getByLabel('Export format', { exact: true }).selectOption('npy');
  const file = testInfo.outputPath('t2-export.npy');
  await page.getByRole('button', { name: 'Export selected object', exact: true }).click();
  await chooseFilePath(page, file, 'Save');
  await waitComplete(page);
  await expect(exportDialog.locator('.file-source')).toContainText(file);
  const result = JSON.parse(
    execFileSync(
      'python',
      [
        '-c',
        'import json,numpy as np,sys;a=np.load(sys.argv[1],allow_pickle=False);print(json.dumps({"median":float(np.median(a[5:15,5:17,0])),"shape":list(a.shape)}))',
        file,
      ],
      { encoding: 'utf8' },
    ),
  );
  expect(Math.abs(result.median - 80)).toBeLessThan(0.1);
  expect(result.shape).toEqual([24, 32, 3]);
});

test('manual registration improves alignment and round trips transforms and labels', async ({
  page,
  request,
}, testInfo) => {
  await synthetic(page, true);
  await selectExperiment(page, 'Motion shifted');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'Register');
  await page.getByLabel('Reference experiment', { exact: true }).selectOption('Motion reference');
  await page.getByLabel('Moving experiment', { exact: true }).selectOption('Motion shifted');
  await page.getByLabel('Moving frame', { exact: true }).fill('3');
  await page.getByLabel('Manual affine transform', { exact: true }).check();
  await page.getByLabel('Horizontal shift', { exact: true }).fill('3');
  await page.getByRole('button', { name: 'Run registration', exact: true }).click();
  await waitComplete(page);
  await page.getByLabel('Stored transform', { exact: true }).selectOption({ index: 1 });
  await page.getByRole('button', { name: 'Transfer labelled ROI', exact: true }).click();
  await page.getByRole('button', { name: 'View registered overlay', exact: true }).click();
  await expect(page.getByRole('img', { name: /32 by 40 quantitative image/ })).toBeVisible();
  const [w] = await opened(request);
  const data = await (await request.get(`/api/workspaces/${w.id}`)).json();
  expect(data.registrations[0].diagnostics.post.mse).toBeLessThan(1e-10);
  expect(data.rois.some((r: { name: string }) => r.name.endsWith('transformed'))).toBe(true);
  const file = testInfo.outputPath('registered.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  await expect(page.getByLabel('Stored transform', { exact: true }).locator('option')).toHaveCount(
    2,
  );
});

test('study constants survive editing and importing the first experiment', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await clickDataControl(page, 'New study');
  await openMetadataBrowser(page, 'Study constants');
  await selectDataFormat(page, 'Study constants', 'Raw');
  await page
    .getByLabel('Study constants', { exact: true })
    .fill(JSON.stringify({ field_strength_mhz: { values: 123.4, unit: 'MHz' } }));
  await page.getByRole('button', { name: 'Apply study constants', exact: true }).click();
  await closeMetadataBrowser(page);
  await clickDataControl(page, 'Import data');
  await page.getByLabel('Import format', { exact: true }).selectOption('blank');
  await page.getByLabel('Array shape', { exact: true }).fill('2,3');
  await page.getByLabel('Dimension order', { exact: true }).fill('x,y');
  await page.getByRole('button', { name: 'Import into study', exact: true }).click();
  await waitComplete(page);
  await expect(page.getByRole('img', { name: /2 by 3 quantitative image/ })).toBeVisible();
  const [w] = await opened(request);
  const data = await (await request.get(`/api/workspaces/${w.id}/inspect`)).json();
  expect(data.study_globals.field_strength_mhz.unit).toBe('MHz');
  expect(data.study_globals.field_strength_mhz.values).toBeCloseTo(123.4, 3);
});
