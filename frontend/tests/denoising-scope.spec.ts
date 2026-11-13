import { execFileSync } from 'node:child_process';

import { mkdir } from 'node:fs/promises';

import { dirname } from 'node:path';

import { test, expect } from '@playwright/test';

import type { Workspace } from '../src/api';

import { openWorkspace, selectWorkflow } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('denoising dimensions and run targets follow the selected slice package', async ({
  page,
  request,
}, info) => {
  const source = info.outputPath('independent-packages.mrax');
  await mkdir(dirname(source), { recursive: true });
  execFileSync('python', [
    '-c',
    `
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'tests'))
from test_bruker_geometry_workflows import _scan
from mrax.gui.session import GuiSession
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.persistence import save_workspace_bundle
source = Path(sys.argv[1])
_scan(source.parent / 'synthetic', counts=(3, 2, 1))
session = GuiSession()
session.import_bruker_study(source.parent / 'synthetic', field_strength_mhz=100, load_raw_kspace=False)
save_workspace_bundle(StudyWorkspace('test', 'Independent packages', session), source)
`,
    source,
  ]);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  let catalogs = 0;
  page.on('request', (r) => {
    if (r.method() === 'GET' && /\/processing(?:\?|$)/.test(r.url())) catalogs++;
  });
  await page.goto('/');
  await openWorkspace(page, source);
  const workspace = async () =>
    (await (await request.get('/api/workspaces')).json())[0] as Workspace;
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  const scope = page.getByRole('combobox', { name: 'Denoising scope', exact: true });
  await expect(scope).toHaveValue('package');
  const packages = page.getByRole('combobox', { name: 'Slice package', exact: true });
  const dims = page.locator('.processing-axes .axis-options');
  for (const [index, labels] of [
    ['x (7)', 'y (5)', 'z (3)'],
    ['x (7)', 'z (5)', 'y (2)'],
    ['y (7)', 'z (5)', 'x (1)'],
  ].entries()) {
    await packages.selectOption(String(index));
    for (const label of labels) await expect(dims.getByText(label, { exact: true })).toBeVisible();
    await expect(page.getByLabel('Denoising target', { exact: true })).toHaveText(
      `Package ${index + 1}`,
    );
  }
  await packages.selectOption('1');
  const denoiser = page.getByRole('combobox', { name: 'Denoiser', exact: true });
  const bmAvailable = async (method: string) =>
    (await denoiser.locator(`option[value="${method}"]`).getAttribute('disabled')) === null;
  if (await bmAvailable('bm3d')) {
    await denoiser.selectOption('bm3d');
    await expect(dims.locator('input:checked')).toHaveCount(2);
    await expect(page.getByLabel('Filter axis y', { exact: true })).toBeDisabled();
    await page.getByLabel('Filter axis x', { exact: true }).uncheck();
    await expect(page.getByLabel('Filter axis y', { exact: true })).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Apply denoising', exact: true })).toBeDisabled();
    await expect(page.locator('.processing-axes').getByRole('status')).toContainText(
      'Select exactly 2 dimensions (1 selected).',
    );
    await page.getByLabel('Filter axis x', { exact: true }).check();
    await expect(page.getByLabel('Filter axis y', { exact: true })).toBeDisabled();
  }
  if (await bmAvailable('bm4d')) {
    await denoiser.selectOption('bm4d');
    await expect(dims.locator('input:checked')).toHaveCount(3);
    await expect(page.getByLabel('Filter axis frame', { exact: true })).toBeDisabled();
    await page.getByLabel('Filter axis x', { exact: true }).uncheck();
    await expect(page.getByRole('button', { name: 'Apply denoising', exact: true })).toBeDisabled();
    await expect(page.locator('.processing-axes').getByRole('status')).toContainText(
      'Select exactly 3 dimensions (2 selected).',
    );
    await page.getByLabel('Filter axis x', { exact: true }).check();
    await scope.selectOption('image');
    await expect(scope).toHaveValue('image');
    await expect(page.getByRole('button', { name: 'Apply denoising', exact: true })).toBeDisabled();
    await expect(page.getByText(/BM4D.*requires 3 filter dimensions/)).toBeVisible();
  }
  await denoiser.selectOption('gaussian');
  await scope.selectOption('image');
  await expect(dims.locator('input:checked')).toHaveCount(2);
  await expect(page.getByLabel('Filter axis y', { exact: true })).toBeDisabled();
  await expect(page.getByLabel('Denoising target', { exact: true })).toContainText('Package 2');
  const initialCatalogs = catalogs;
  const frame = page.getByRole('slider', { name: 'Frame', exact: true });
  if (await frame.count()) await frame.fill('1');
  await page.getByRole('slider', { name: 'Slice', exact: true }).fill('1');
  await expect.poll(async () => (await workspace()).indices.z).toBe(1);
  expect(catalogs).toBe(initialCatalogs);
  await page.screenshot({ path: info.outputPath('coronal-denoising-controls.png') });
  const selectedPlane = (await workspace()).plane_dims;
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  const step = (await workspace()).pipeline.steps[0];
  expect(step.options).toMatchObject({
    scope: 'image',
    indices: { slice_package: 1, z: 1 },
    plane_dims: selectedPlane,
  });
  await packages.selectOption('0');
  await scope.selectOption('package');
  await expect(page.getByLabel('Denoising target', { exact: true })).toHaveText('Package 1');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await expect.poll(async () => (await workspace()).pipeline.steps.length).toBe(2);
  expect((await workspace()).pipeline.steps[0].options).toEqual(step.options);
  expect((await workspace()).experiments).toEqual(['1']);
  expect(errors).toEqual([]);
});
