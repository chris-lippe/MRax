import { test, expect, type APIRequestContext, type Page } from '@playwright/test';

import { readFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';

import type { Workspace } from '../src/api';

import {
  chooseFilePath,
  loadExample,
  openExport,
  openRoiCoordinates,
  openWorkspace,
  selectTheme,
  selectWorkflow,
} from './ui';

async function current(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json())[0] as Workspace;
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await (await request.get('/api/workspaces')).json())
    await request.delete('/api/workspaces/' + w.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
});

async function study(page: Page) {
  await page.goto('/');
  await loadExample(page);
  await expect(
    page.getByRole('img', { name: 'Whole image mean signal', exact: true }),
  ).toBeVisible();
  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  for (const [name, col] of [
    ['Anterior', '5'],
    ['Posterior', '16'],
  ] as const) {
    await page.getByLabel('New ROI name', { exact: true }).fill(name);
    await page.getByLabel('ROI column', { exact: true }).fill(col);
    await page.getByLabel('ROI row', { exact: true }).fill('5');
    await page.getByLabel('ROI width', { exact: true }).fill('6');
    await page.getByLabel('ROI height', { exact: true }).fill('8');
    await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
    await expect(page.getByRole('region', { name: 'ROIs', exact: true })).toContainText(name);
  }
}

test('scientific viewer inspects samples, customizes axes, summarizes ranges and exports normal files without altering data', async ({
  page,
  request,
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await study(page);
  await selectWorkflow(page, 'View', 'Signal curves');
  const plot = page.getByRole('img', { name: 'Scientific signal curve plot', exact: true });
  await expect(plot).toBeVisible();
  const before = await current(request),
    raw = await (await request.get('/api/workspaces/' + before.id + '/slice')).body();
  await plot.focus();
  await plot.press('End');
  await expect(
    page.getByRole('status', { name: 'Signal sample values', exact: true }),
  ).toContainText('Sample 8 / 8');
  await page.getByLabel('Signal X min', { exact: true }).fill('20');
  await page.getByLabel('Signal X max', { exact: true }).fill('100');
  await page.getByRole('tab', { name: 'Statistics', exact: true }).click();
  await expect(
    page.getByRole('table', { name: 'Signal range statistics', exact: true }),
  ).toContainText('Posterior');
  await expect(
    page
      .getByRole('table', { name: 'Signal range statistics', exact: true })
      .locator('tbody tr')
      .first()
      .locator('td')
      .first(),
  ).toHaveText('4');
  await page.getByRole('tab', { name: 'Samples', exact: true }).click();
  await expect(
    page.getByRole('table', { name: 'Scientific signal samples', exact: true }).locator('tbody tr'),
  ).toHaveCount(8);
  await page.getByRole('tab', { name: 'Plot', exact: true }).click();
  await page.getByLabel('Signal Y scale', { exact: true }).selectOption('log');
  await page.getByLabel('Signal Y min', { exact: true }).fill('0');
  await expect(page.getByRole('alert')).toContainText('positive');
  await page.getByLabel('Signal Y min', { exact: true }).fill('');
  await page.getByText('Figure appearance', { exact: true }).click();
  await page.getByLabel('Signal figure theme', { exact: true }).selectOption('light');
  await page.getByLabel('Signal figure title', { exact: true }).fill('Independent T2 signal');
  await page.getByLabel('Signal Y title', { exact: true }).fill('Signal intensity');
  await expect(plot).toContainText('Independent T2 signal');
  await page.getByRole('button', { name: 'Reset axes', exact: true }).click();
  await page.getByRole('button', { name: 'Select X range', exact: true }).click();
  const box = (await plot.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.5, { steps: 5 });
  await page.mouse.up();
  await expect(page.getByLabel('Signal X min', { exact: true })).not.toHaveValue('');
  await page.getByRole('button', { name: 'Reset axes', exact: true }).click();
  const after = await current(request);
  expect(after.revision).toBe(before.revision);
  expect(after.pipeline).toEqual(before.pipeline);
  expect(after.fit_settings).toEqual(before.fit_settings);
  expect(await (await request.get('/api/workspaces/' + before.id + '/slice')).body()).toEqual(raw);
  const output = info.outputPath('exports');
  await mkdir(output, { recursive: true });
  const dialog = await openExport(page);
  for (const [button, name] of [
    ['Save signal CSV', 'signal.csv'],
    ['Save signal figure SVG', 'signal.svg'],
  ] as const) {
    await dialog.getByRole('button', { name: button, exact: true }).click();
    await chooseFilePath(page, output + '/' + name, 'Save');
    await expect
      .poll(async () => readFile(output + '/' + name, 'utf8').catch(() => ''))
      .not.toBe('');
  }
  const csv = await readFile(output + '/signal.csv', 'utf8');
  expect(csv).toContain('Included in fit');
  expect(csv).toContain('Posterior');
  const svg = await readFile(output + '/signal.svg', 'utf8');
  expect(svg).toContain('Independent T2 signal');
  expect(svg).toContain('xmlns="http://www.w3.org/2000/svg"');
  expect(svg).not.toContain('data-plot-inspection');
  expect(svg).not.toMatch(/var\(--plot-/);
  expect(svg).not.toContain('aria-hidden="true"');
  expect(svg).not.toContain('aria-describedby=');
  const rendered = await page.evaluate(async (source) => {
    const document = new DOMParser().parseFromString(source, 'image/svg+xml');
    if (document.querySelector('parsererror')) throw new Error('Invalid exported SVG');
    const image = new Image();
    image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(source);
    await image.decode();
    const canvas = window.document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d')!;
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let ink = 0;
    for (let i = 0; i < pixels.length; i += 4)
      if (pixels[i] < 240 || pixels[i + 1] < 240 || pixels[i + 2] < 240) ink++;
    return { width: canvas.width, height: canvas.height, ink };
  }, svg);
  expect(rendered.width).toBeGreaterThan(300);
  expect(rendered.height).toBeGreaterThanOrEqual(420);
  expect(rendered.ink).toBeGreaterThan(1000);
  await page.getByRole('button', { name: 'Close export', exact: true }).click();
  await selectWorkflow(page, 'Segment');
  await selectWorkflow(page, 'View', 'Signal curves');
  await expect(page.getByLabel('Signal figure title', { exact: true })).toHaveValue(
    'Independent T2 signal',
  );
  await page.screenshot({ path: info.outputPath('scientific-signal-light.png') });
  await selectTheme(page, 'dark');
  await page.getByLabel('Signal figure theme', { exact: true }).selectOption('auto');
  await page.screenshot({ path: info.outputPath('scientific-signal-dark.png') });
  const spectrumFile = info.outputPath('spectrum.mrax');
  execFileSync(
    'python',
    [
      '-c',
      `
import sys
import numpy as np
from mrax.gui import GuiSession
from mrax.data_struct import Quantity
session = GuiSession()
session.create_empty_study('Synthetic spectroscopy', field_strength_mhz=100)
frequency = np.linspace(-200, 200, 65)
values = 2 + 10 * np.exp(-((frequency - 35) / 25) ** 2)
session.add_experiment('Spectrum', np.stack([values, 2 * values])[..., None],
    dims={'frame': 2, 'x': 65, 'y': 1}, exp_type='spectroscopy', non_imaging=True,
    data_role='spectrum', observables={'frequency': Quantity(frequency, 'Hz')})
session.save_imported_study(sys.argv[1])
`,
      spectrumFile,
    ],
    { cwd: '..' },
  );
  await selectWorkflow(page, 'View', 'Image display');
  await openWorkspace(page, spectrumFile);
  await expect(page.getByRole('region', { name: 'Image viewer', exact: true })).not.toBeVisible();
  await expect(plot).toBeVisible();
  await expect(plot).toContainText('frequency (Hz)');
  await expect(page.getByRole('heading', { name: 'Signal display', exact: true })).toBeVisible();
  await expect(page.getByLabel('Signal profile axis', { exact: true })).toBeVisible();
  const spectrum = (await (await request.get('/api/workspaces')).json()).find(
    (w: Workspace) => w.experiment === 'Spectrum',
  ) as Workspace;
  expect(spectrum.non_imaging).toBe(true);
  expect(spectrum.timecourse?.x).toHaveLength(65);
  const first = spectrum.timecourse!.observed.map((v) => 2 * v!);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await expect(plot).toBeVisible();
  const frame = page.locator('.signal-navigation input[type="range"]');
  await expect(frame).toHaveCount(1);
  await frame.fill('1');
  await expect
    .poll(async () => (await (await request.get('/api/workspaces/' + spectrum.id)).json()).frame)
    .toBe(1);
  const updated = (await (await request.get('/api/workspaces/' + spectrum.id)).json()) as Workspace;
  updated.timecourse!.observed.forEach((value, i) => expect(value).toBeCloseTo(first[i], 5));
  await plot.focus();
  await plot.press('End');
  await expect(
    page.getByRole('status', { name: 'Signal sample values', exact: true }),
  ).toContainText('Sample 65 / 65');
  await page.screenshot({ path: info.outputPath('spectroscopy-signal.png') });
  expect(errors).toEqual([]);
});
