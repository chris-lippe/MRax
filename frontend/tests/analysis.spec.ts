import { loadExample } from './ui';

import { openRoiCoordinates } from './ui';

import { test, expect, type Page, type APIRequestContext } from '@playwright/test';
import type { Workspace } from '../src/api';

import {
  selectDataFormat,
  selectWorkflow,
  openMetadataBrowser,
  closeMetadataBrowser,
  selectExperiment,
  selectTheme,
} from './ui';

async function opened(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()) as Workspace[];
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

test('complex views, planes, integer labels, brush, and undo use the real backend', async ({
  page,
  request,
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
  const imageBounds = (await page.getByTestId('roi-surface').boundingBox())!;
  await page.mouse.move(
    imageBounds.x + imageBounds.width * 0.43,
    imageBounds.y + imageBounds.height * 0.41,
  );
  const readout = page.locator('.image-readout > span').first();
  const beforeFrame = await readout.textContent();
  await page.getByLabel('frame', { exact: true }).fill('1');
  await expect(readout).not.toHaveText(beforeFrame!);
  const position = (await readout.textContent())!.match(/Row (\d+), column (\d+)/)!;
  const current = (await opened(request))[0];
  const info = await (await request.get('/api/workspaces/' + current.id)).json();
  const slice = await request.get('/api/workspaces/' + current.id + '/slice');
  const header = slice.headers(),
    data = await slice.body();
  const bytes = header['x-dtype'] === 'float64-le' ? 8 : 4;
  const offset =
    (Number(position[1]) * JSON.parse(header['x-shape'])[1] + Number(position[2])) *
    info.channels *
    bytes;
  const value = bytes === 8 ? data.readDoubleLE(offset) : data.readFloatLE(offset);
  await expect(readout).toContainText(value.toPrecision(6));
  await page.getByLabel('Plane', { exact: true }).selectOption('coronal');
  await expect(readout).toHaveText('Pixel value: —');
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

test('fitting shows only the inputs used by the selected model', async ({ page, request }) => {
  await synthetic(page);
  await selectWorkflow(page, 'Fit');
  const modelPanel = page.getByRole('region', { name: 'Model and solver', exact: true });
  await expect(modelPanel).toBeVisible();
  const checkFit = page.getByRole('button', { name: 'Check fit readiness', exact: true });
  const readiness = page.getByRole('status', { name: 'Fit readiness', exact: true });
  await expect(page.getByLabel('Fit region', { exact: true })).toHaveValue('all');
  await expect(page.getByLabel('Fit ROI', { exact: true })).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Run preview fit', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Run full map fit', exact: true })).toBeEnabled();
  await checkFit.click();
  await expect(readiness).toContainText('Ready to fit');
  await page.getByLabel('Preview steps', { exact: true }).fill('40');
  await page.getByRole('button', { name: 'Run preview fit', exact: true }).click();
  await expect(page.getByLabel('Preview fit results', { exact: true })).toBeVisible();
  expect((await opened(request))[0].preview?.roi_name).toBeNull();
  await page.getByLabel('Fit steps', { exact: true }).fill('301');
  await expect(readiness).toHaveCount(0);
  await expect(modelPanel.getByLabel('Reference scan', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('T1 prior map', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Derived maps', { exact: true })).toHaveCount(0);

  let current = (await opened(request))[0];
  const path = '/api/workspaces/' + current.id;
  const general = page.getByRole('region', { name: 'General parameters', exact: true });
  const t2 = general
    .locator('fieldset.fit-parameter')
    .filter({ has: page.getByLabel('t2 initial', { exact: true }) });
  await t2.getByText('Dimension settings', { exact: true }).click();
  await t2.getByRole('button', { name: 'Add t2 spatial regularizer', exact: true }).click();
  await t2.getByLabel('t2 regularizer', { exact: true }).selectOption('smoothness');
  await t2.getByRole('checkbox', { name: 't2 regularization dimension y', exact: true }).uncheck();
  await t2.getByRole('checkbox', { name: 't2 regularization dimension z', exact: true }).uncheck();
  await t2.getByRole('checkbox', { name: 't2 constant dimension z', exact: true }).check();
  const frames = page.getByRole('group', { name: 'Frames for fitting', exact: true });
  await frames.locator('summary').first().click();
  await expect(frames.getByRole('list', { name: 'Frames for fitting' })).toContainText('TE:');
  await frames.getByRole('checkbox', { name: 'Include frame 1', exact: true }).uncheck();
  await frames.getByRole('button', { name: 'View frame 3', exact: true }).click();
  await expect(
    frames.getByRole('checkbox', { name: 'Include frame 1', exact: true }),
  ).not.toBeChecked();
  await checkFit.click();
  await expect(readiness).toContainText('Ready to fit');
  await page.getByRole('button', { name: 'Apply fit configuration', exact: true }).click();
  await expect
    .poll(async () => (await (await request.get(path)).json()).frame_exclusions)
    .toEqual([0]);
  current = (await opened(request))[0];
  expect(current.fit_settings.parameter_constant_dims).toEqual({ t2: ['z'] });
  const choices = await (await request.get(path + '/fit-options')).json();
  const xAxis = choices.field_dimensions.find(
    (dimension: { name: string }) => dimension.name === 'x',
  ).axis;
  expect(
    (current.fit_settings.regularizer_configs as Record<string, { axes: number[] }[]>).t2[0].axes,
  ).toEqual([xAxis]);
  const inspected = await (await request.get(path + '/inspect')).json();
  const token = (await (await request.get('/api/session')).json()).token;
  const response = await request.post(path + '/actions', {
    headers: { 'x-mrax-token': token },
    data: {
      revision: current.revision,
      action: 'metadata.update',
      options: {
        exp_type: 'sat_trans',
        observables: {
          ...inspected.metadata.observables,
          freq_offsets: { values: [-3, -2, -1, -0.5, 0.5, 1, 2, 3], unit: 'ppm' },
          sat_time: { values: 5, unit: 'ms' },
          sat_b1: { values: 1, unit: 'uT' },
        },
      },
    },
  });
  expect(response.status()).toBe(200);
  await page.reload();
  await selectWorkflow(page, 'Fit');
  await page.getByLabel('Model', { exact: true }).selectOption('wasabi');
  await expect(modelPanel.getByLabel('Reference scan', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Derived maps', { exact: true })).toHaveCount(1);
  for (const kind of ['T1', 'B0', 'B1'])
    await expect(modelPanel.getByLabel(kind + ' prior map', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Fit region', { exact: true })).toHaveValue('all');
  await checkFit.click();
  await expect(readiness).toContainText('No matching single-frame saturation-transfer reference');
  await expect(readiness).not.toContainText('ROI');
  await expect(page.getByRole('button', { name: 'Run preview fit', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Run full map fit', exact: true })).toBeEnabled();

  await page.getByLabel('Model', { exact: true }).selectOption('ab_mt_n');
  for (const kind of ['T1', 'B0', 'B1'])
    await expect(modelPanel.getByLabel(kind + ' prior map', { exact: true })).toBeVisible();
  await expect(modelPanel.getByLabel('Reference scan', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Derived maps', { exact: true })).toHaveCount(0);

  await page.getByLabel('Model', { exact: true }).selectOption('multipool_lorentzian');
  await expect(modelPanel.getByLabel('Reference scan', { exact: true })).toBeVisible();
  for (const kind of ['T1', 'B0', 'B1'])
    await expect(modelPanel.getByLabel(kind + ' prior map', { exact: true })).toHaveCount(0);

  await page.getByLabel('Pool template', { exact: true }).selectOption('glu');
  await page.getByLabel('New pool name', { exact: true }).fill('mobile_glu');
  await page.getByRole('button', { name: 'Add pool', exact: true }).click();
  await expect(page.getByLabel('Name for pool mobile_glu', { exact: true })).toBeVisible();
  const parameters = page.getByRole('group', {
    name: 'Parameters for pool mobile_glu',
    exact: true,
  });
  await expect(parameters.getByLabel('amp_mobile_glu initial', { exact: true })).toBeHidden();
  await parameters.locator('summary').first().click();
  await parameters.getByLabel('amp_mobile_glu initial', { exact: true }).fill('0.12');
  await parameters.getByLabel('amp_mobile_glu lower bound', { exact: true }).fill('0.02');
  await parameters.getByLabel('amp_mobile_glu upper bound', { exact: true }).fill('0.25');
  await parameters.getByRole('checkbox', { name: 'Fix amp_mobile_glu', exact: true }).check();
  await parameters.getByRole('checkbox', { name: 'Fix amp_mobile_glu', exact: true }).uncheck();
  await expect(parameters.getByLabel('amp_mobile_glu lower bound', { exact: true })).toHaveValue(
    '0.02',
  );
  await parameters.getByRole('checkbox', { name: 'Fix amp_mobile_glu', exact: true }).check();
  const water = page.getByRole('group', { name: 'Parameters for pool water', exact: true });
  await water.locator('summary').first().click();
  await expect(water.getByRole('checkbox', { name: 'Fix dw_water', exact: true })).toBeChecked();
  await water.getByLabel('dw_water initial', { exact: true }).fill('0.15');
  await page.getByRole('button', { name: 'Apply fit configuration', exact: true }).click();
  await expect
    .poll(async () => {
      const settings = (await (await request.get(path)).json()).fit_settings;
      return [settings.pool_overrides.mobile_glu.amp.value, settings.pool_overrides.water.dw.value];
    })
    .toEqual([0.12, 0.15]);
  await expect(parameters.getByLabel('amp_mobile_glu lower bound', { exact: true })).toHaveValue(
    '0.02',
  );
  await page.getByLabel('Name for pool mobile_glu', { exact: true }).fill('amide');
  await page.getByRole('button', { name: 'Rename pool mobile_glu', exact: true }).click();
  await expect(page.getByLabel('Name for pool amide', { exact: true })).toBeVisible();
  const renamedParameters = page.getByRole('group', {
    name: 'Parameters for pool amide',
    exact: true,
  });
  await renamedParameters.locator('summary').first().click();
  await expect(renamedParameters.getByLabel('amp_amide initial', { exact: true })).toHaveValue(
    '0.12',
  );
  await expect(
    renamedParameters.getByRole('checkbox', { name: 'Fix amp_amide', exact: true }),
  ).toBeChecked();
  await expect(page.getByLabel('amp_amide initial', { exact: true })).toHaveCount(1);
  await expect(general.getByLabel('baseline initial', { exact: true })).toBeVisible();
  expect(
    await general.evaluate((element) =>
      Boolean(
        element.compareDocumentPosition(document.querySelector('section[aria-label="Pools"]')!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
      ),
    ),
  ).toBe(true);
  const additional = page.getByRole('group', { name: 'Additional fit settings', exact: true });
  await additional.locator('summary').first().click();
  await selectDataFormat(page, 'Additional fit settings', 'Raw');
  const extras = JSON.parse(
    await additional
      .getByRole('textbox', { name: 'Additional fit settings', exact: true })
      .inputValue(),
  );
  expect(extras).not.toHaveProperty('num_steps');
  expect(extras).not.toHaveProperty('param_overrides');
  expect(extras).not.toHaveProperty('regularizer_configs');
  await checkFit.click();
  await expect(readiness).toContainText('Needs attention');
  await expect(readiness).toContainText('reference');
  await page.getByRole('button', { name: 'Remove pool amide', exact: true }).click();
  await expect(page.getByLabel('Name for pool amide', { exact: true })).toHaveCount(0);
  await expect(readiness).toHaveCount(0);
  await page.getByLabel('New pool name', { exact: true }).fill('mobile_glu');
  await page.getByRole('button', { name: 'Add pool', exact: true }).click();
  await expect(page.getByLabel('Name for pool mobile_glu', { exact: true })).toHaveValue(
    'mobile_glu',
  );
  await page.getByRole('button', { name: 'Remove pool mobile_glu', exact: true }).click();

  await page.getByLabel('Model', { exact: true }).selectOption('t2_se');
  await expect(modelPanel.getByLabel('Reference scan', { exact: true })).toHaveCount(0);
  await expect(page.getByLabel('Derived maps', { exact: true })).toHaveCount(0);
});
