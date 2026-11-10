import { loadExample } from './ui';
import { openRoiCoordinates } from './ui';
import { saveWorkspace, openWorkspace } from './ui';
import { execFileSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { test, expect, type APIRequestContext } from '@playwright/test';
import { selectWorkflow, selectExperiment, selectTheme } from './ui';
import type { Workspace } from '../src/api';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as Workspace[];
}
test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await opened(request)) {
    await expect
      .poll(async () => {
        const latest = (await opened(request)).find((item) => item.id === w.id);
        if (!latest) return 204;
        const response = await request.delete(`/api/workspaces/${w.id}`, {
          headers: { 'x-mrax-token': token },
          data: { revision: latest.revision, discard: true },
        });
        return response.status();
      })
      .toBe(204);
  }
});

test('processing stays in one experiment and an editable pipeline preserves sources and fits', async ({
  page,
  request,
}, info) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await selectTheme(page, 'light');
  const [initial] = await opened(request);
  const slice = async (id: string) =>
    (await (await request.get('/api/workspaces/' + id + '/slice')).body()).toString('base64');
  const original = await slice(initial.id);
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await expect(pipeline.locator('header')).not.toContainText('Original data');
  await expect(pipeline.locator('header')).not.toContainText('Protected');
  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByLabel('New ROI name', { exact: true }).fill('Tissue');
  await page.getByLabel('Label value', { exact: true }).fill('7');
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await page.getByLabel('Show Tissue', { exact: true }).uncheck();

  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await expect(page.getByLabel('Filter axis frame', { exact: true })).not.toBeChecked();
  await expect(page.getByLabel('Filter axis x', { exact: true })).toBeChecked();
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('1.2');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(1);
  let [w] = await opened(request);
  expect(w.experiments).toHaveLength(initial.experiments.length);
  expect(w.experiment).toBe(initial.experiment);
  expect(w.rois[0]).toMatchObject({ name: 'Tissue', labels: [7], enabled: true, visible: false });
  expect(await slice(w.id)).not.toBe(original);
  const denoise = w.pipeline.steps[0];

  await selectWorkflow(page, 'Analyze', 'Time-series corrections');
  await page.getByLabel('Time-series correction', { exact: true }).selectOption('baseline');
  await page.getByRole('button', { name: 'Apply baseline correction', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(2);
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Fit mask Tissue', { exact: true })).toBeChecked();
  await page.getByLabel('Fit steps', { exact: true }).fill('600');
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].maps.length, { timeout: 60000 })
    .toBeGreaterThan(0);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  [w] = await opened(request);
  const maps = w.maps;
  expect(w.pipeline.fits_stale).toBe(false);
  await page
    .getByRole('group', { name: 'Fit display', exact: true })
    .getByRole('button', { name: 'Signal', exact: true })
    .click();

  const viewer = page.getByRole('region', { name: 'Image viewer', exact: true });
  const compactHeight = (await pipeline.boundingBox())!.height;
  expect(compactHeight).toBeLessThanOrEqual(48);
  await expect(pipeline.locator('header button')).toHaveCount(1);
  await expect(pipeline.locator('header select')).toHaveCount(0);
  await expect(pipeline.getByLabel('Pipeline source', { exact: true })).not.toBeVisible();
  await expect(
    pipeline.getByRole('group', { name: 'Pipeline output', exact: true }),
  ).not.toBeVisible();
  await expect(
    pipeline.getByRole('button', { name: 'Remove all processing', exact: true }),
  ).not.toBeVisible();
  const imageHeight = (await viewer.boundingBox())!.height;
  const imagePlaneHeight = (await page.locator('.image-viewport').boundingBox())!.height;
  const imageTop = (await viewer.boundingBox())!.y;
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await expect(
    pipeline.getByRole('button', { name: 'Hide processing steps', exact: true }),
  ).toHaveAttribute('aria-expanded', 'true');
  expect((await pipeline.boundingBox())!.height).toBeGreaterThan(compactHeight);
  await expect(pipeline.getByLabel('Pipeline source', { exact: true })).toBeVisible();
  expect((await viewer.boundingBox())!.height).toBeCloseTo(imageHeight, 0);
  expect((await viewer.boundingBox())!.y).toBeGreaterThan(imageTop);
  expect(
    await page.locator('main').evaluate((element) => element.scrollHeight > element.clientHeight),
  ).toBe(true);
  await page.locator('main').hover();
  await page.mouse.wheel(0, 350);
  await expect
    .poll(() => page.locator('main').evaluate((element) => element.scrollTop))
    .toBeGreaterThan(0);
  expect((await viewer.boundingBox())!.height).toBeCloseTo(imageHeight, 0);
  const gaussian = pipeline.getByRole('listitem', { name: 'Gaussian denoising', exact: true });
  await gaussian.getByRole('button', { name: 'Edit', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Use action controls', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'Preprocess', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await expect(page.getByLabel('Sigma (voxels)', { exact: true })).toHaveValue('1.2');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.6');
  await page.getByRole('button', { name: 'Stage denoising settings', exact: true }).click();
  expect((await opened(request))[0].pipeline.steps[0].version).toBe(denoise.version);
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].pipeline.steps[0].version)
    .not.toBe(denoise.version);
  [w] = await opened(request);
  expect(w.pipeline.steps).toHaveLength(2);
  expect(w.pipeline.steps[0].id).toBe(denoise.id);
  expect(w.pipeline.steps[0].options.parameters).toMatchObject({ sigma: 0.6 });
  expect(w.maps).toEqual(maps);
  await expect(pipeline).toContainText('retained from an earlier pipeline state');
  const processed = await slice(w.id);

  await pipeline.getByRole('button', { name: 'Original', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.enabled).toBe(false);
  expect(await slice(w.id)).toBe(original);
  await pipeline.getByRole('button', { name: 'Hide processing steps', exact: true }).click();
  await expect(pipeline.locator('header')).toContainText('Original');
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  await pipeline.getByRole('button', { name: 'Processed', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.enabled).toBe(true);
  expect(await slice(w.id)).toBe(processed);
  await pipeline.getByLabel('Enable Gaussian denoising', { exact: true }).uncheck();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps[0].enabled).toBe(false);
  await pipeline.getByLabel('Enable Gaussian denoising', { exact: true }).check();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps[0].enabled).toBe(true);
  await expect(pipeline.getByRole('button', { name: 'Original', exact: true })).toBeEnabled();
  await expect(pipeline.locator('.pipeline-step.disabled')).toHaveCount(0);
  expect(await slice(w.id)).toBe(processed);
  await pipeline.getByRole('button', { name: 'Hide processing steps', exact: true }).click();
  await expect(
    pipeline.getByRole('button', { name: 'Show processing steps', exact: true }),
  ).toHaveAttribute('aria-expanded', 'false');
  expect((await page.locator('.image-viewport').boundingBox())!.height).toBeCloseTo(
    imagePlaneHeight,
    0,
  );
  await page.screenshot({ path: info.outputPath('process-workflow-compact.png') });
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await selectWorkflow(page, 'View', 'Image display');
  await page.screenshot({ path: info.outputPath('process-workflow.png') });

  const file = info.outputPath('pipeline-study.mrax');
  await saveWorkspace(page, file);
  await openWorkspace(page, file);
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  await gaussian.getByRole('button', { name: 'Remove', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect
    .poll(
      async () =>
        (await opened(request)).find((item) => item.id !== initial.id)!.pipeline.steps.length,
    )
    .toBe(1);
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
  await expect(pipeline).toContainText('Baseline correction');
  await pipeline.getByRole('button', { name: 'Remove all processing', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect
    .poll(
      async () =>
        (await opened(request)).find((item) => item.id !== initial.id)!.pipeline.steps.length,
    )
    .toBe(0);
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(0);
  const restored = (await opened(request)).find((item) => item.id !== initial.id)!;
  expect(await slice(restored.id)).toBe(original);
  expect(restored.maps).toEqual(maps);
  expect(restored.experiments).toHaveLength(initial.experiments.length);
  expect(errors).toEqual([]);
});

test('denoising and registration replace their steps without duplicating experiments or losing labels', async ({
  page,
  request,
}) => {
  test.setTimeout(120000);
  await page.goto('/');
  await loadExample(page, 'multimodal');
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  const [initial] = await opened(request);
  await selectExperiment(page, 'Motion shifted');
  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByLabel('New ROI name', { exact: true }).fill('Motion mask');
  await page.getByLabel('Label value', { exact: true }).fill('17');
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.5');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.7');
  const first = (await opened(request))[0].pipeline.steps[0];
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].pipeline.steps[0].version)
    .not.toBe(first.version);
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
  await selectWorkflow(page, 'Register', 'Image alignment');
  await page.getByLabel('Reference experiment', { exact: true }).selectOption('Motion reference');
  await page.getByLabel('Moving experiment', { exact: true }).selectOption('Motion shifted');
  await page.getByLabel('Moving frame', { exact: true }).fill('3');
  await page.getByLabel('Manual affine transform', { exact: true }).check();
  await page.getByLabel('Horizontal shift', { exact: true }).fill('3');
  await page.getByRole('button', { name: 'Run registration', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  const registration = pipeline.getByRole('listitem', { name: 'Registration', exact: true });
  await registration.getByRole('button', { name: 'Edit', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Use action controls', exact: true }).click();
  await expect(page.getByLabel('Horizontal shift', { exact: true })).toHaveValue('3');
  await page.getByLabel('Horizontal shift', { exact: true }).fill('2');
  const previous = (await opened(request))[0].pipeline.steps[1].version;
  await page.getByRole('button', { name: 'Stage registration settings', exact: true }).click();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect
    .poll(async () => (await opened(request))[0].pipeline.steps[1].version)
    .not.toBe(previous);
  const [w] = await opened(request);
  expect(w.experiments).toHaveLength(initial.experiments.length);
  expect(w.experiment).toBe('Motion shifted');
  expect(w.rois.some((roi) => roi.enabled && roi.labels.includes(17))).toBe(true);
  expect(w.pipeline.steps).toHaveLength(2);
  await page.getByRole('button', { name: 'View registered overlay', exact: true }).click();
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await expect(page.getByRole('region', { name: 'Overlay 1', exact: true })).toContainText(
    'Motion shifted',
  );
  await expect(page.getByRole('img', { name: /32 by 40 quantitative image/ })).toBeVisible();
});

test('normalization and time-series corrections share one persistent experiment pipeline', async ({
  page,
  request,
}) => {
  test.setTimeout(120000);
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  const [initial] = await opened(request);
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await selectWorkflow(page, 'Preprocess', 'Signal normalization');
  await page.getByLabel('Normalization method', { exact: true }).selectOption('normalize');
  await page.getByRole('button', { name: 'Normalize signal', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(1);
  await selectWorkflow(page, 'Analyze', 'Time-series corrections');
  await page.getByLabel('Time-series correction', { exact: true }).selectOption('baseline');
  await page.getByRole('button', { name: 'Apply baseline correction', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(2);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(3);
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(3);
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await expect(pipeline).toContainText('Reference normalization / Z spectra');
  await expect(pipeline).toContainText('Baseline correction');
  await expect(pipeline).toContainText('Gaussian denoising');
  const [w] = await opened(request);
  expect(w.experiments).toHaveLength(initial.experiments.length);
  expect(w.experiment).toBe(initial.experiment);
});

test('reconstruction is an explicit processing action and views reuse its result', async ({
  page,
  request,
}, info) => {
  const source = info.outputPath('synthetic-kspace.mrax');
  await mkdir(dirname(source), { recursive: true });
  execFileSync('python', [
    '-c',
    `
import sys
import numpy as np
import jax.numpy as jnp
from mrax.data_struct import AcquisitionData
from mrax.reco import fftnc
from mrax.gui.session import GuiSession
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.persistence import save_workspace_bundle

image = np.arange(20, dtype=np.float32).reshape((1, 4, 5)) / 20 + 1
acquisition = AcquisitionData(np.asarray(fftnc(jnp.asarray(image[:, None] * 2))), (4, 5), ("phase", "read"))
session = GuiSession()
session.create_empty_study("Synthetic raw data")
session.add_experiment("Raw image", image, dims={"frame": 1, "x": 4, "y": 5}, kspace=acquisition.samples)
session.current_experiment["acquisition"] = acquisition
session.study.datasets[0].acquisition = acquisition
save_workspace_bundle(StudyWorkspace(key="raw", label=session.study_name, session=session), sys.argv[1])
`,
    source,
  ]);
  await page.goto('/');
  await openWorkspace(page, source);
  const [before] = await opened(request);
  expect(before).toMatchObject({
    has_kspace: true,
    has_reconstruction: false,
    can_reconstruct: true,
  });
  const slice = async (id = before.id) => {
    const response = await request.get('/api/workspaces/' + id + '/slice');
    const bytes = await response.body();
    const size = response.headers()['x-dtype'] === 'float64-le' ? 8 : 4;
    return Array.from({ length: bytes.length / size }, (_, index) =>
      size === 8 ? bytes.readDoubleLE(index * size) : bytes.readFloatLE(index * size),
    );
  };
  const original = await slice();
  const pipeline = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  const input = pipeline.getByLabel('Pipeline source', { exact: true });
  await expect(input).not.toBeVisible();
  await pipeline.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await expect(input).toBeVisible();
  await expect(input).toHaveValue('vendor');
  await expect(
    input.getByRole('option', { name: 'Reconstruction (.mrec)', exact: true }),
  ).toHaveJSProperty('disabled', false);
  const jobs: string[] = [];
  page.on('request', (request) => {
    if (request.url().endsWith('/tasks')) jobs.push(request.postDataJSON().operation);
  });
  const view = page.getByRole('combobox', { name: 'View', exact: true });
  await expect(
    view.getByRole('option', { name: 'Cartesian Fourier', exact: true }),
  ).toHaveAttribute('disabled', '');
  await selectWorkflow(page, 'Reconstruct');
  await page.getByRole('button', { name: 'Apply reconstruction', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await expect(
    page
      .getByRole('group', { name: 'Reconstruction display', exact: true })
      .getByRole('button', { name: 'Vendor comparison', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true');
  const reconstructed = await slice();
  expect(reconstructed).toHaveLength(original.length);
  reconstructed.forEach((value, index) => expect(value).toBeCloseTo(original[index] * 2, 5));
  await expect(input).toHaveValue('mrax');
  await expect(
    page.getByRole('button', { name: 'Apply reconstruction', exact: true }),
  ).toBeEnabled();
  await expect(
    page.getByRole('region', { name: 'Processing pipeline', exact: true }),
  ).toContainText('Cartesian Fourier');
  expect((await opened(request))[0].pipeline.steps).toHaveLength(1);
  await selectWorkflow(page, 'View', 'Image display');
  await page.getByRole('button', { name: 'Image', exact: true }).click();
  await view.selectOption('Image');
  await expect(view).toHaveValue('Image');
  await view.selectOption('MRax reconstruction');
  await expect(view).toHaveValue('MRax reconstruction');
  expect(jobs).toEqual(['reconstruct']);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('0.6');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(2);
  await expect(input).toHaveValue('mrax');
  let workspace = (await opened(request))[0];
  const stepIds = workspace.pipeline.steps.map((step) => step.id);
  const mrax = await slice();
  await input.selectOption('vendor');
  await expect(input).toHaveValue('vendor');
  expect(await slice()).toEqual(mrax);
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps[0].enabled).toBe(false);
  const vendor = await slice();
  vendor.forEach((value, index) => expect(value * 2).toBeCloseTo(mrax[index], 5));
  expect(vendor).not.toEqual(original);
  workspace = (await opened(request))[0];
  expect(workspace.pipeline.steps.map((step) => step.id)).toEqual(stepIds);
  expect(workspace.pipeline.steps[0].enabled).toBe(false);
  await input.selectOption('mrax');
  await expect(input).toHaveValue('mrax');
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps[0].enabled).toBe(true);
  (await slice()).forEach((value, index) => expect(value).toBeCloseTo(mrax[index], 5));
  await pipeline.getByRole('button', { name: 'Original', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.enabled).toBe(false);
  expect(await slice()).toEqual(original);
  await expect(input).toHaveValue('mrax');
  await pipeline.getByRole('button', { name: 'Processed', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.enabled).toBe(true);
  await expect(pipeline.getByRole('button', { name: 'Processed', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await pipeline
    .getByRole('listitem', { name: 'Cartesian Fourier', exact: true })
    .getByRole('button', { name: 'Remove', exact: true })
    .click();
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(1);
  await expect(input).toHaveValue('vendor');
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(1);
  await input.selectOption('mrax');
  await expect(input).toHaveValue('mrax');
  await expect(pipeline.locator('.pipeline-step')).toHaveCount(2);
  await pipeline.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(2);
  expect((await opened(request))[0].pipeline.steps[1].id).toBe(stepIds[1]);
  (await slice()).forEach((value, index) => expect(value).toBeCloseTo(mrax[index], 5));
  const saved = info.outputPath('pipeline-source.mrax');
  await saveWorkspace(page, saved);
  await openWorkspace(page, saved);
  await expect(input).toHaveValue('mrax');
  const restored = (await opened(request)).find((item) => item.id !== before.id)!;
  (await slice(restored.id)).forEach((value, index) => expect(value).toBeCloseTo(mrax[index], 5));
  expect(restored.experiments).toHaveLength(1);
  expect(jobs.filter((operation) => operation === 'reconstruct')).toHaveLength(1);
  expect(jobs.filter((operation) => operation === 'pipeline')).toHaveLength(6);
  expect(jobs.filter((operation) => operation === 'process')).toHaveLength(1);
  await pipeline.getByRole('button', { name: 'Hide processing steps', exact: true }).click();
  await expect(input).not.toBeVisible();
  await expect(pipeline.locator('header')).toContainText('Processed');
  await page.screenshot({ path: info.outputPath('pipeline-source.png') });
});
