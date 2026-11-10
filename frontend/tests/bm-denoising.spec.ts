import { execFileSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { test, expect } from '@playwright/test';
import { openWorkspace, selectWorkflow } from './ui';
import type { Workspace } from '../src/api';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

for (const method of ['bm3d', 'bm4d']) {
  test(`${method}: automatic and manual noise levels visibly denoise a padded MRI-scale image and retain its source`, async ({
    page,
    request,
  }, info) => {
    const source = info.outputPath('synthetic-noisy-phantom.mrax');
    await mkdir(dirname(source), { recursive: true });
    execFileSync('python', [
      '-c',
      `
import sys
import numpy as np
from mrax.analysis import ROI
from mrax.gui.session import GuiSession
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.persistence import save_workspace_bundle
truth = np.zeros((1, 32, 32, 12))
truth[:, 8:24, 8:24, 3:9] = 1e6
noisy = np.where(truth != 0, truth + np.random.default_rng(42).normal(0, 80000, truth.shape), 0)
session = GuiSession()
session.create_empty_study('Synthetic denoising')
session.add_experiment('Padded phantom', noisy, dims={'frame': 1, 'x': 32, 'y': 32, 'z': 12}, model_name='identity')
session.add_roi(ROI('Uniform signal', truth[0] != 0, dims=('x', 'y', 'z')))
save_workspace_bundle(StudyWorkspace(key='synthetic', label=session.study_name, session=session), sys.argv[1])
`,
      source,
    ]);
    await page.goto('/');
    await openWorkspace(page, source);
    await page.getByRole('slider', { name: 'Slice', exact: true }).fill('6');
    const opened = async () => (await (await request.get('/api/workspaces')).json()) as Workspace[];
    const id = (await opened())[0].id;
    const slice = async () => {
      const response = await request.get('/api/workspaces/' + id + '/slice');
      const bytes = await response.body();
      const size = response.headers()['x-dtype'] === 'float64-le' ? 8 : 4;
      return Array.from({ length: bytes.length / size }, (_, index) =>
        size === 8 ? bytes.readDoubleLE(index * size) : bytes.readFloatLE(index * size),
      );
    };
    const mse = (values: number[]) =>
      values.reduce((sum, value, index) => {
        const x = index % 32,
          y = Math.floor(index / 32);
        const truth = x >= 8 && x < 24 && y >= 8 && y < 24 ? 1e6 : 0;
        return sum + (value - truth) ** 2;
      }, 0) / values.length;
    const original = await slice();
    await selectWorkflow(page, 'Preprocess', 'Denoising');
    const denoiser = page.getByRole('combobox', { name: 'Denoiser', exact: true });
    test.skip(
      (await denoiser.locator(`option[value="${method}"]`).getAttribute('disabled')) !== null,
      'Optional denoiser is unavailable',
    );
    await denoiser.selectOption(method);
    const sigma = page.getByLabel('Noise sigma (automatic if empty)', { exact: true });
    await expect(sigma).toHaveValue('');
    await page.getByRole('button', { name: 'Estimate noise', exact: true }).click();
    const estimated = page.getByLabel('Estimated noise sigma', { exact: true });
    await expect(estimated).toBeVisible();
    const estimate = Number((await estimated.innerText()).replaceAll(',', ''));
    expect(estimate).toBeGreaterThan(56000);
    expect(estimate).toBeLessThan(104000);
    expect((await opened())[0].pipeline.steps).toHaveLength(0);
    for (const [index, level] of ['', '', '80000'].entries()) {
      if (index === 1) {
        await page.getByRole('checkbox', { name: 'Estimate noise from ROIs', exact: true }).check();
        await expect(
          page.getByRole('button', { name: 'Apply denoising', exact: true }),
        ).toBeDisabled();
        await page
          .getByRole('checkbox', { name: 'Estimate noise in Uniform signal', exact: true })
          .check();
        await page
          .getByRole('combobox', { name: 'Noise estimation model', exact: true })
          .selectOption('gaussian');
        await page.getByRole('button', { name: 'Estimate noise', exact: true }).click();
        await expect(estimated).toBeVisible();
        await expect(estimated).toBeInViewport();
        await expect
          .poll(() =>
            estimated.evaluate((element) => {
              const rect = element.getBoundingClientRect();
              return (
                document
                  .elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2)
                  ?.closest('.noise-estimate-result') === element.closest('.noise-estimate-result')
              );
            }),
          )
          .toBe(true);
        await page.screenshot({ path: info.outputPath(method + '-noise-roi.png') });
      }
      await sigma.fill(level);
      await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
      await expect(page.locator('.job-status strong')).toHaveText('Operation complete');
      const result = await slice();
      expect(mse(result)).toBeLessThan(mse(original) * 0.25);
      expect(result).not.toEqual(original);
      const w = (await opened())[0];
      expect(w.experiments).toEqual(['Padded phantom']);
      expect(w.pipeline.steps).toHaveLength(1);
      expect(w.pipeline.steps[0].options.parameters).toMatchObject({ sigma: level ? 80000 : null });
      if (!level) {
        expect(w.pipeline.steps[0].noise_estimate?.sigma).toBeGreaterThan(56000);
        expect(w.pipeline.steps[0].noise_estimate?.scope).toBe(index === 1 ? 'roi' : 'block');
        await expect(estimated).toBeVisible();
      }
      if (index === 1)
        expect(w.pipeline.steps[0].options.noise_roi_names).toEqual(['Uniform signal']);
    }
    await page.getByRole('button', { name: 'Show processing steps', exact: true }).click();
    const output = page.getByRole('group', { name: 'Pipeline output', exact: true });
    await output.getByRole('button', { name: 'Original', exact: true }).click();
    await expect.poll(async () => (await opened())[0].pipeline.enabled).toBe(false);
    expect(await slice()).toEqual(original);
    await output.getByRole('button', { name: 'Processed', exact: true }).click();
    await expect.poll(async () => (await opened())[0].pipeline.enabled).toBe(true);
    expect(mse(await slice())).toBeLessThan(mse(original) * 0.25);
    await page.screenshot({ path: info.outputPath(method + '-denoised.png') });
  });
}
