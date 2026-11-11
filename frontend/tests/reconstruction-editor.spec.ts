import { execFileSync } from 'node:child_process';

import { mkdir, readFile, writeFile } from 'node:fs/promises';

import { dirname } from 'node:path';

import { test, expect, type APIRequestContext, type Page, type TestInfo } from '@playwright/test';

import { chooseFilePath, openWorkspace, selectWorkflow } from './ui';

import type { Options, Workspace } from '../src/api';

const opened = async (request: APIRequestContext): Promise<Workspace[]> =>
  (await request.get('/api/workspaces')).json();

async function fixture(page: Page, info: TestInfo, sampled = false) {
  const path = info.outputPath('independent-acquisition.mrax');
  await mkdir(dirname(path), { recursive: true });
  execFileSync('python', [
    '-c',
    `
import sys
import numpy as np
from mrax.data_struct import AcquisitionData
from mrax.gui.session import GuiSession
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.persistence import save_workspace_bundle
from mrax.reco import fftnc
x, y = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 16), indexing='ij')
image = np.stack([1 + np.exp(-6 * (x*x + y*y)), 1 + .7*np.exp(-6 * (x*x+y*y))])
image = image * np.exp(1j * (.15*x + .1*y))
coils = np.stack([image, .7 * image * np.exp(.3j)], axis=1)
samples = np.asarray(fftnc(coils, axes=(-2, -1)))
sampled = sys.argv[2] == 'sampled'
trajectory = np.stack(np.meshgrid((np.arange(12)-6)/12, (np.arange(16)-8)/16, indexing='ij'), -1).reshape(-1, 2).astype(np.float32)
raw = AcquisitionData(samples.reshape(2, 2, -1) if sampled else samples, (12, 16), ('x', 'y'),
                      kind='sampled' if sampled else 'cartesian', trajectory=trajectory if sampled else None)
session = GuiSession()
session.create_empty_study('Independent reconstruction study', field_strength_mhz=100)
session.add_experiment('Raw acquisition', np.abs(image).astype(np.float32) * .5,
                       dims={'frame': 2, 'x': 12, 'y': 16}, kspace=raw.samples, model_name='identity')
session.current_experiment['acquisition'] = raw
session.study.datasets[0].acquisition = raw
save_workspace_bundle(StudyWorkspace('independent', session.study_name, session), sys.argv[1])
`,
    path,
    sampled ? 'sampled' : 'cartesian',
  ]);
  await page.goto('/');
  await openWorkspace(page, path);
  await selectWorkflow(page, 'Reconstruct');
  await expect(page.getByLabel('Reconstruction preset', { exact: true })).toBeEnabled();
}

async function pixels(request: APIRequestContext) {
  const [w] = await opened(request);
  const response = await request.get('/api/workspaces/' + w.id + '/slice');
  const bytes = await response.body();
  const size = response.headers()['x-dtype'] === 'float64-le' ? 8 : 4;
  return Array.from({ length: bytes.length / size }, (_, i) =>
    size === 8 ? bytes.readDoubleLE(i * size) : bytes.readFloatLE(i * size),
  );
}

test.afterEach(async ({ request }) => {
  const headers = { 'x-mrax-token': (await (await request.get('/api/session')).json()).token };
  for (const w of await opened(request))
    await request.delete('/api/workspaces/' + w.id, {
      headers,
      data: { revision: w.revision, discard: true },
    });
  for (const entry of await (await request.get('/api/workflows')).json())
    await request.delete('/api/workflows/' + entry.id + '?revision=' + entry.revision, { headers });
});

test('stages insert, edit, remove and discard. Named reconstructions save and reopen as mrec without running', async ({
  page,
  request,
}, info) => {
  await fixture(page, info);
  const original = await pixels(request);
  const chain = page.getByRole('region', { name: 'Reconstruction stages', exact: true });
  const window = chain.getByRole('listitem', { name: 'K-space window', exact: true });
  await window.getByRole('button', { name: 'K-space window settings', exact: true }).click();
  await window.getByRole('button', { name: 'Remove stage', exact: true }).click();
  await expect(window).toHaveCount(0);
  expect(await pixels(request)).toEqual(original);
  await page.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await expect(window).toHaveCount(1);
  await window.getByRole('button', { name: 'K-space window settings', exact: true }).click();
  await window.getByRole('button', { name: 'Remove stage', exact: true }).click();
  await chain.getByLabel('Add reconstruction stage', { exact: true }).selectOption('window');
  await chain.getByRole('button', { name: 'Insert stage', exact: true }).click();
  await expect(window.getByLabel('Window (all encoding axes)', { exact: true })).toHaveValue(
    'hann',
  );
  await window.getByRole('button', { name: 'Move K-space window down', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveText(
    'K-space corrections precede the solver. Geometry and coil combination follow it.',
  );
  await expect(
    page.getByRole('button', { name: 'Apply reconstruction', exact: true }),
  ).toBeDisabled();
  expect(await pixels(request)).toEqual(original);
  await window.getByRole('button', { name: 'Move K-space window up', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.getByLabel('Reconstruction pipeline name', { exact: true }).fill('Windowed Fourier');
  expect((await opened(request))[0].pipeline.steps).toHaveLength(0);
  await page.getByRole('button', { name: 'Apply reconstruction', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await expect
    .poll(async () => (await opened(request))[0].reconstruction_label)
    .toBe('Windowed Fourier');
  expect(await pixels(request)).not.toEqual(original);
  const [applied] = await opened(request);
  expect(applied.pipeline.steps[0].label).toBe('Windowed Fourier');
  expect(
    (applied.pipeline.steps[0].options.workflow as { steps: { settings: Options }[] }).steps[0]
      .settings.pipeline,
  ).toMatchObject({ name: 'Windowed Fourier' });
  await expect(page.getByRole('button', { name: 'Vendor comparison', exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Reconstruction workflows', exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Reusable workflows', exact: true });
  await modal.getByRole('button', { name: 'Capture current configuration', exact: true }).click();
  const file = info.outputPath('windowed.mrec');
  await modal.getByRole('button', { name: 'Save workflow file', exact: true }).click();
  await chooseFilePath(page, file, 'Save');
  await expect(modal.getByRole('status')).toHaveText('Workflow saved.');
  const document = JSON.parse(await readFile(file, 'utf8'));
  expect(document.kind).toBe('reconstruction');
  expect(document.steps[0].settings.pipeline.name).toBe('Windowed Fourier');
  document.steps[0].settings.pipeline.name = 'Imported Fourier';
  const imported = info.outputPath('imported.mrec');
  await writeFile(imported, JSON.stringify(document, null, 2));
  await modal.getByRole('button', { name: 'Close workflows', exact: true }).click();
  await page.getByRole('button', { name: 'Open .mrec', exact: true }).click();
  await chooseFilePath(page, imported, 'Open');
  await expect(page.getByLabel('Reconstruction pipeline name', { exact: true })).toHaveValue(
    'Imported Fourier',
  );
  expect((await opened(request))[0].reconstruction_label).toBe('Windowed Fourier');
  await page.screenshot({ path: info.outputPath('reconstruction-stages.png') });
});

test('variational settings run and a changed regularizer recomputes a cached reconstruction', async ({
  page,
  request,
}, info) => {
  await fixture(page, info);
  await page.getByLabel('Reconstruction preset', { exact: true }).selectOption('variational');
  const solver = page
    .getByRole('region', { name: 'Reconstruction stages', exact: true })
    .getByRole('listitem', { name: 'Variational reconstruction', exact: true });
  await solver
    .getByRole('button', { name: 'Variational reconstruction settings', exact: true })
    .click();
  await solver.getByLabel('Iterations', { exact: true }).fill('3');
  await solver.getByLabel('Regularizer', { exact: true }).selectOption('none');
  await solver.getByLabel('Regularization weight', { exact: true }).fill('0');
  await page.getByRole('button', { name: 'Apply reconstruction', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  const unregularized = await pixels(request);
  await expect(
    page.getByRole('button', { name: 'Apply reconstruction', exact: true }),
  ).toBeEnabled();
  await solver.getByLabel('Regularizer', { exact: true }).selectOption('l2');
  await solver.getByLabel('Regularization weight', { exact: true }).fill('.1');
  await page.getByRole('button', { name: 'Apply reconstruction', exact: true }).click();
  await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
  await expect
    .poll(async () => {
      const plan = (
        (await opened(request))[0].pipeline.steps[0].options.workflow as {
          steps: { settings: Options }[];
        }
      ).steps[0].settings.pipeline as {
        stages: { kind: string; options: Record<string, unknown> }[];
      };
      return plan.stages.find((stage) => stage.kind === 'variational')?.options.regularizer_weight;
    })
    .toBe(0.1);
  expect(await pixels(request)).not.toEqual(unregularized);
  const [w] = await opened(request);
  const plan = (w.pipeline.steps[0].options.workflow as { steps: { settings: Options }[] }).steps[0]
    .settings.pipeline as {
    stages: { kind: string; options: Record<string, unknown> }[];
  };
  expect(plan.stages.find((s) => s.kind === 'variational')?.options).toMatchObject({
    regularizer: 'l2',
    regularizer_weight: 0.1,
  });
});
