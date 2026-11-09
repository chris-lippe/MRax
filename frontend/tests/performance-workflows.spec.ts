import { clickDataControl, loadExample } from './ui';
import { test, expect, type APIRequestContext } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { chooseFilePath } from './ui';

async function cleanup(request: APIRequestContext) {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json())
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
}
test.afterEach(async ({ request }) => cleanup(request));

test('Bruker import reviews headers and imports only selected experiments and reconstruction', async ({
  page,
  request,
}, info) => {
  const source = info.outputPath('Synthetic selected study');
  await mkdir(source, { recursive: true });
  execFileSync('python', [
    '-c',
    'import sys,shutil,numpy as np; from pathlib import Path; sys.path.insert(0,str(Path.cwd().parent/"tests")); from _synthetic import _write_minimal_bruker_scan; p=Path(sys.argv[1]); _write_minimal_bruker_scan(p); shutil.copytree(p/"1",p/"2"); shutil.copytree(p/"2"/"pdata"/"1",p/"2"/"pdata"/"2"); (np.arange(4,dtype=np.int16)+100).tofile(p/"2"/"pdata"/"2"/"2dseq")',
    source,
  ]);
  await page.goto('/');
  await clickDataControl(page, 'Import data');
  await page.getByRole('button', { name: 'Browse files and folders…', exact: true }).click();
  await chooseFilePath(page, source, 'Open', false, true);
  const selection = page.getByRole('region', { name: 'Bruker import selection', exact: true });
  await expect(selection).toBeVisible();
  expect(await (await request.get('/api/workspaces')).json()).toHaveLength(0);
  await selection.getByRole('button', { name: 'Select none', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Import study', exact: true })).toBeDisabled();
  await selection.getByLabel('Reconstruction for 2', { exact: true }).selectOption('2');
  await selection.getByLabel('Import experiment 2', { exact: true }).check();
  await selection.getByLabel('Import frequency (MHz)', { exact: true }).fill('400');
  await expect(
    selection.getByRole('checkbox', { name: 'Load raw k-space', exact: true }),
  ).not.toBeChecked();
  await page.screenshot({ path: info.outputPath('bruker-selection.png') });
  const sent = page.waitForRequest(
    (request) => request.url().endsWith('/api/import') && request.method() === 'POST',
  );
  await page.getByRole('button', { name: 'Import study', exact: true }).click();
  const options = (await sent).postDataJSON().options;
  expect(options.selected_experiments).toEqual(['2']);
  expect(options.pdata_index).toEqual({ '2': '2' });
  expect(options.load_raw_kspace).toBe(false);
  await expect(page.getByRole('img', { name: /2 by 2 quantitative image/ })).toBeVisible({
    timeout: 60000,
  });
  const [workspace] = await (await request.get('/api/workspaces')).json();
  expect(workspace.experiments).toEqual(['2']);
  const data = await (await request.get('/api/workspaces/' + workspace.id + '/slice')).body();
  const values = new Float32Array(
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
  );
  expect([...values].sort((a, b) => a - b)).toEqual([210, 212, 214, 216]);
});

test('returning to recently viewed frames uses cached planes', async ({ page }) => {
  await page.goto('/');
  await loadExample(page);
  const image = page.getByRole('img', { name: /24 by 32 quantitative image/ });
  await expect(image).toHaveAttribute('aria-label', /frame 1, slice 1/);
  let slices = 0;
  page.on('request', (request) => {
    if (request.url().includes('/slice')) slices++;
  });
  const slider = page.getByRole('slider', { name: 'Echo', exact: true });
  await slider.focus();
  await slider.press('ArrowRight');
  await expect(image).toHaveAttribute('aria-label', /frame 2, slice 1/);
  expect(slices).toBe(1);
  await slider.press('ArrowLeft');
  await expect(image).toHaveAttribute('aria-label', /frame 1, slice 1/);
  await slider.press('ArrowRight');
  await expect(image).toHaveAttribute('aria-label', /frame 2, slice 1/);
  expect(slices).toBe(1);
  await expect(slider).toBeFocused();
});

test('large import reviews page experiments without losing selections', async ({ page }, info) => {
  const source = info.outputPath('Synthetic headers');
  await mkdir(source, { recursive: true });
  const experiments = Array.from({ length: 63 }, (_, index) => ({
    scan: String(index + 1),
    method: 'MSME',
    protocol: 'Synthetic',
    frame_count: 8,
    detected_type: 'mse',
    detected_model: 't2_se',
    field_strength_mhz: 400,
    pdata_index: '1',
    reconstructions: ['1'],
    importable: true,
    has_raw: false,
  }));
  await page.route('**/api/files/preview', (route) =>
    route.fulfill({ json: { kind: 'bruker', experiments } }),
  );
  await page.goto('/');
  await clickDataControl(page, 'Import data');
  await page.getByRole('button', { name: 'Browse files and folders…', exact: true }).click();
  await chooseFilePath(page, source, 'Open', false, true);
  const selection = page.getByRole('region', { name: 'Bruker import selection', exact: true });
  await expect(selection.getByRole('checkbox', { name: /^Import experiment / })).toHaveCount(50);
  await selection.getByRole('button', { name: 'Select none', exact: true }).click();
  await selection.getByRole('button', { name: 'Next experiments', exact: true }).click();
  await expect(selection.getByRole('checkbox', { name: /^Import experiment / })).toHaveCount(13);
  await selection.getByLabel('Import experiment 63', { exact: true }).check();
  await selection.getByRole('button', { name: 'Previous experiments', exact: true }).click();
  await expect(selection.getByLabel('Import experiment 1', { exact: true })).not.toBeChecked();
  await expect(selection.getByRole('status')).toHaveText('1 of 63 experiments selected');
});
