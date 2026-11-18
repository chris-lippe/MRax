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
  selectExperiment,
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
  const echo = page.getByRole('slider', { name: 'Echo', exact: true });
  const canvas = page.locator('.image-surface canvas');
  const brightestPixel = () =>
    canvas.evaluate((element: HTMLCanvasElement) => {
      const pixels = element.getContext('2d')!.getImageData(0, 0, element.width, element.height);
      let brightest = 0;
      for (let i = 0; i < pixels.data.length; i += 4)
        brightest = Math.max(brightest, pixels.data[i]);
      return brightest;
    });
  await expect.poll(brightestPixel).toBeGreaterThan(100);
  const firstPeak = await brightestPixel();
  await echo.press('End');
  await expect(canvas).toHaveAttribute('aria-label', /frame 8, slice 1/);
  await expect.poll(brightestPixel).toBeLessThan(firstPeak / 4);
  const scaling = page.getByLabel('Intensity scaling', { exact: true });
  await scaling.selectOption('frame');
  await expect.poll(brightestPixel).toBe(255);
  await scaling.selectOption('fixed');
  await echo.press('Home');
  await expect.poll(brightestPixel).toBe(firstPeak);
  await expect(
    page.getByRole('img', { name: 'Whole image mean signal', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Signal ROI', exact: true })).toHaveValue('');
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
  const preview = page.locator('.signal-preview .curve-panel:visible');
  const axis = preview.getByRole('combobox', { name: 'Signal x-axis', exact: true });
  const roi = preview.getByRole('combobox', { name: 'Signal ROI', exact: true });
  const initialProfile = (await current(request)).timecourse!;
  await expect(roi).toHaveValue('Posterior');
  await roi.selectOption('Anterior');
  await expect.poll(async () => (await current(request)).timecourse?.roi_name).toBe('Anterior');
  await roi.selectOption('');
  await expect.poll(async () => (await current(request)).timecourse?.roi_name).toBeNull();
  await expect(
    preview.getByRole('img', { name: 'Whole image mean signal', exact: true }),
  ).toBeVisible();
  const whole = await current(request);
  expect(whole.selected_roi).toBe('Anterior');
  expect(whole.rois).toHaveLength(2);
  await selectWorkflow(page, 'Denoise');
  await expect(roi).toHaveValue('');
  await selectWorkflow(page, 'View', 'Signal curves');
  await expect(page.getByRole('combobox', { name: 'Signal ROI', exact: true })).toHaveValue('');
  await expect(
    page.getByRole('img', { name: 'Scientific signal curve plot', exact: true }),
  ).toContainText('Whole image');
  await page.getByRole('combobox', { name: 'Signal ROI', exact: true }).selectOption('Posterior');
  await expect.poll(async () => (await current(request)).timecourse?.roi_name).toBe('Posterior');
  await selectWorkflow(page, 'Segment');
  await expect(roi).toHaveValue('Posterior');
  await axis.selectOption('x');
  await expect.poll(async () => (await current(request)).timecourse?.dimension).toBe('x');
  await preview.getByRole('button', { name: 'Data table', exact: true }).click();
  await expect(preview.locator('tbody tr')).toHaveCount((await current(request)).dimensions.x);
  await selectWorkflow(page, 'Denoise');
  await expect(axis).toBeVisible();
  await expect(axis).toHaveValue('x');
  await axis.selectOption('auto');
  await expect
    .poll(async () => (await current(request)).timecourse?.dimension)
    .toBe(initialProfile.dimension);
  await preview.getByRole('button', { name: 'Chart', exact: true }).click();
  await selectWorkflow(page, 'View', 'Signal curves');
  await expect(page.getByLabel('Signal profile axis', { exact: true })).toHaveValue('auto');
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
offsets = np.asarray([0, -3, 3, -1, 1, -2, 2], dtype=np.float32)
session.add_experiment('Reference', np.ones((4,3),dtype=np.float32), dims={'x':4,'y':3},
    exp_type='sat_trans', observables={'sat_b1':Quantity(0,'uT')})
detuning = offsets * 100 * 2 * np.pi
rf = 2 * np.pi * 42.58 * 4
z = np.abs(1.55 - 2 * rf**2/(rf**2 + detuning**2) * np.sin(np.sqrt(rf**2 + detuning**2)*0.005/2)**2) + 1e-6
signal = np.broadcast_to(z[None,None,:,None,None],(60,1,7,4,3)).copy()
signal *= (1+0.05*np.sin(np.arange(60)/10))[:,None,None,None,None]
session.add_experiment('Saturation sweep', signal,
    dims={'t':60,'b1':1,'offset':7,'x':4,'y':3}, exp_type='sat_trans', model_name='wasabi',
    observables={'TR':Quantity(np.linspace(300,360,7),'ms'),
                 'freq_offsets':Quantity(offsets,'ppm'), 'sat_b1':Quantity([4],'uT'),
                 'sat_time':Quantity(5,'ms')})
ti = np.asarray([50,200,800,2500],dtype=np.float32)
ir = np.broadcast_to(np.abs(1e6-2e6*np.exp(-ti/1400))[None,:,None,None,None],(2,4,2,4,3)).copy()
session.add_experiment('IR sweeps',ir, dims={'slice_package':2,'ti':4,'fair':2,'x':4,'y':3},
    exp_type='fair_asl',model_name='t1_ir',observables={'TR':Quantity(np.linspace(900,1000,16),'ms'),
    'TI':Quantity(ti,'ms'),'fair_label':Quantity(np.asarray([0,1]),'1')})
session.select_experiment('Spectrum')
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
  await page.getByLabel('Signal component', { exact: true }).selectOption('imag');
  await expect
    .poll(
      async () =>
        (await (await request.get('/api/workspaces/' + spectrum.id)).json()).timecourse.observed[0],
    )
    .toBe(0);
  let releaseComponent!: () => void;
  const componentGate = new Promise<void>((resolve) => (releaseComponent = resolve));
  await page.route(
    `**/api/workspaces/${spectrum.id}/view`,
    async (route) => {
      const result = await route.fetch();
      if (route.request().postDataJSON()?.component === 'real') await componentGate;
      await route.fulfill({ response: result });
    },
    { times: 1 },
  );
  await page.getByLabel('Signal component', { exact: true }).selectOption('real');
  await expect
    .poll(
      async () =>
        (await (await request.get('/api/workspaces/' + spectrum.id)).json()).timecourse.observed[0],
    )
    .toBeCloseTo(spectrum.timecourse!.observed[0]!, 5);
  try {
    await selectWorkflow(page, 'Denoise');
  } finally {
    releaseComponent();
  }
  await expect(plot).toBeVisible();
  const frame = page.locator('.signal-navigation input[type="range"]');
  await expect(frame).toHaveCount(1);
  await frame.fill('1');
  await expect
    .poll(async () => (await (await request.get('/api/workspaces/' + spectrum.id)).json()).frame)
    .toBe(1);
  const updated = (await (await request.get('/api/workspaces/' + spectrum.id)).json()) as Workspace;
  expect(updated.timecourse!.observed[0]).toBeCloseTo(2 * spectrum.timecourse!.observed[0]!, 5);
  await plot.focus();
  await plot.press('End');
  await expect(
    page.getByRole('status', { name: 'Signal sample values', exact: true }),
  ).toContainText('Sample 65 / 65');
  await selectExperiment(page, 'Saturation sweep');
  await selectWorkflow(page, 'View', 'Image display');
  await page.getByRole('combobox', { name: 'Curve axis', exact: true }).selectOption('offset');
  await expect(axis).toHaveValue('offset');
  await expect(preview).toContainText('freq_offsets (ppm)');
  const saturation = (await (await request.get('/api/workspaces')).json()).find(
    (w: Workspace) => w.experiment === 'Saturation sweep',
  ) as Workspace;
  expect(saturation.timecourse!.x).toEqual([0, -3, 3, -1, 1, -2, 2]);
  expect(saturation.timecourse!.x_unit).toBe('ppm');
  await axis.selectOption('b1');
  await expect(preview).toContainText('sat_b1 (uT)');
  await axis.selectOption('offset');
  await selectWorkflow(page, 'View', 'Signal curves');
  await expect(page.getByLabel('Signal profile axis', { exact: true })).toHaveValue('offset');
  await expect(plot).toContainText('freq_offsets (ppm)');
  await selectWorkflow(page, 'Fit');
  const b1Parameter = page.locator('fieldset.fit-parameter').filter({
    has: page.getByLabel('b1 initial', { exact: true }),
  });
  await b1Parameter.getByText('Dimension settings', { exact: true }).click();
  await b1Parameter
    .getByRole('button', { name: 'Add b1 spatial regularizer', exact: true })
    .click();
  await b1Parameter
    .getByRole('button', { name: 'Add b1 temporal regularizer', exact: true })
    .click();
  await expect(b1Parameter.getByLabel('b1 regularizer 2', { exact: true })).toHaveValue(
    'smoothness',
  );
  await expect(
    b1Parameter.getByLabel('b1 regularization dimension 2 t', { exact: true }),
  ).toBeChecked();
  await b1Parameter.getByLabel('b1 regularization weight 2', { exact: true }).fill('0.001');
  await page.getByLabel('Loss RMS scale', { exact: true }).selectOption('full');
  await page.getByLabel('Preview steps', { exact: true }).fill('2');
  await page.getByRole('button', { name: 'Run preview fit', exact: true }).click();
  await expect(
    preview.getByRole('img', { name: 'Observed signal and fitted curve', exact: true }),
  ).toBeVisible();
  await expect(preview).toContainText('freq_offsets (ppm)');
  const fittedPath = await preview.locator('path[stroke-dasharray]').getAttribute('d');
  const plottedX = [...fittedPath!.matchAll(/[ML](-?[\d.]+)/g)].map((match) => Number(match[1]));
  expect(plottedX).toHaveLength(7);
  expect(plottedX.every((value, index) => index === 0 || value >= plottedX[index - 1])).toBe(true);
  await page.screenshot({ path: info.outputPath('wasabi-preview.png') });
  const dynamicFit = (await (
    await request.get('/api/workspaces/' + saturation.id)
  ).json()) as Workspace;
  expect(
    (dynamicFit.fit_settings.regularizer_configs as Record<string, unknown[]>).b1,
  ).toHaveLength(2);
  expect(dynamicFit.fit_settings.data_normalization).toBe('full');
  await page.getByRole('slider', { name: 't', exact: true }).fill('59');
  await expect
    .poll(
      async () => (await (await request.get('/api/workspaces/' + saturation.id)).json()).indices.t,
    )
    .toBe(59);
  await expect(preview).toContainText('7 samples. freq_offsets (ppm)');
  await preview.getByRole('button', { name: 'Open signal curves', exact: true }).click();
  await expect(plot).toContainText('freq_offsets (ppm)');
  for (const path of await plot.locator('[data-signal-trace] path').all()) {
    const d = await path.getAttribute('d');
    const xs = [...d!.matchAll(/[ML](-?[\d.]+)/g)].map((match) => Number(match[1]));
    expect(xs.every((value, index) => index === 0 || value >= xs[index - 1])).toBe(true);
  }
  await page.screenshot({ path: info.outputPath('dynamic-signal-viewer.png') });
  await selectWorkflow(page, 'Fit');
  await axis.selectOption('b1');
  await expect(preview).toContainText('1 samples. sat_b1 (uT)');
  await expect(preview.locator('circle')).toHaveCount(2);
  await selectExperiment(page, 'IR sweeps');
  await axis.selectOption('ti');
  await page.getByLabel('Preview steps', { exact: true }).fill('400');
  await page.getByRole('button', { name: 'Run preview fit', exact: true }).click();
  await expect(
    preview.getByRole('img', { name: 'Observed signal and fitted curve', exact: true }),
  ).toBeVisible();
  await expect(preview).toContainText('4 samples. TI (ms)');
  expect(
    (await (await request.get('/api/workspaces/' + saturation.id)).json()).preview.relative_rmse,
  ).toBeLessThan(0.01);
  await page.getByLabel('Slice package', { exact: true }).selectOption('1');
  await expect(preview).toContainText('4 samples. TI (ms)');
  await page.getByLabel('Fit steps', { exact: true }).fill('400');
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await expect(page.getByLabel('Map', { exact: true })).toBeVisible();
  await expect(preview).toContainText('4 samples. TI (ms)');
  await page.getByLabel('Map', { exact: true }).selectOption('t1');
  await expect(page.getByRole('group', { name: 'Map scale', exact: true })).toContainText(
    't1 (ms)',
  );
  await page.screenshot({ path: info.outputPath('ir-preview-map.png') });
  expect(errors).toEqual([]);
});
