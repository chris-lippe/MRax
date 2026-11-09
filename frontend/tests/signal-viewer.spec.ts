import { test, expect, type APIRequestContext, type Page } from '@playwright/test';
import { readFile, mkdir } from 'node:fs/promises';
import type { Workspace } from '../src/api';
import {
  chooseFilePath,
  loadExample,
  openExport,
  openRoiCoordinates,
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
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
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

test('compact preview enlarges downward, persists height and keeps image geometry and keyboard access', async ({
  page,
  request,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await study(page);
  await expect(
    page.getByRole('img', { name: 'Observed ROI signal across echoes', exact: true }),
  ).toBeVisible();
  const image = page.getByRole('region', { name: 'Image viewer', exact: true });
  const height = (await image.boundingBox())!.height;
  const divider = page.getByRole('separator', { name: 'Resize signal preview', exact: true });
  const before = Number(await divider.getAttribute('aria-valuenow'));
  const calls: string[] = [];
  page.on('request', (r) => {
    if (/\/(slice|tasks|view)(?:\?|$)/.test(r.url())) calls.push(r.url());
  });
  await divider.focus();
  await divider.press('ArrowDown');
  await divider.press('Shift+ArrowDown');
  await expect(divider).toHaveAttribute('aria-valuenow', String(before + 80));
  expect((await image.boundingBox())!.height).toBe(height);
  const box = (await divider.boundingBox())!;
  await divider.scrollIntoViewIfNeeded();
  const start = (await divider.boundingBox())!;
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2 + 35, { steps: 4 });
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(divider).toHaveAttribute('aria-valuenow', String(before + 80));
  expect(box.width).toBeGreaterThan(200);
  expect(calls).toEqual([]);
  await page.reload();
  await expect(divider).toHaveAttribute('aria-valuenow', String(before + 80));
  const preview = page.getByRole('region', { name: 'ROI signal', exact: true });
  await preview.getByRole('button', { name: 'Data table', exact: true }).click();
  await expect(page.getByRole('table', { name: 'ROI signal values', exact: true })).toBeVisible();
  await preview.getByRole('button', { name: 'Open signal curves', exact: true }).click();
  await expect(
    page.getByRole('region', { name: 'Scientific signal viewer', exact: true }),
  ).toBeVisible();
  expect((await current(request)).experiments).toHaveLength(1);
});

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
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await selectWorkflow(page, 'View', 'Signal curves');
  await expect(page.getByLabel('Signal figure title', { exact: true })).toHaveValue(
    'Independent T2 signal',
  );
  await page.screenshot({ path: info.outputPath('scientific-signal-light.png') });
  await selectTheme(page, 'dark');
  await page.getByLabel('Signal figure theme', { exact: true }).selectOption('auto');
  await page.screenshot({ path: info.outputPath('scientific-signal-dark.png') });
  expect(errors).toEqual([]);
});

test('dense scientific signals retain exact samples while limiting plot and table elements', async ({
  page,
  request,
}) => {
  await study(page);
  const workspace = await current(request);
  const x = Array.from({ length: 2001 }, (_, i) => i);
  const observed = x.map((v) => (v >= 50 && v < 60 ? null : Math.sin(v / 30) * 100));
  await page.route(/\/api\/workspaces(?:\/[^/?]+(?:\/view)?)?$/, async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const replace = (value: Workspace) =>
      value.id !== workspace.id
        ? value
        : {
            ...value,
            timecourse: {
              ...workspace.timecourse,
              x,
              observed,
              included: x.map(() => true),
              x_name: 'Time',
              x_unit: 's',
            },
          };
    await route.fulfill({
      response,
      json: Array.isArray(body) ? body.map(replace) : replace(body),
    });
  });
  await page.reload();
  await selectWorkflow(page, 'View', 'Signal curves');
  const plot = page.getByRole('img', { name: 'Scientific signal curve plot', exact: true });
  await expect(plot).toBeVisible();
  expect(await plot.locator('circle').count()).toBeLessThanOrEqual(1601);
  await plot.focus();
  await plot.press('End');
  await expect(
    page.getByRole('status', { name: 'Signal sample values', exact: true }),
  ).toContainText('Sample 2001 / 2001');
  await page.getByRole('tab', { name: 'Samples', exact: true }).click();
  const rows = page
    .getByRole('table', { name: 'Scientific signal samples', exact: true })
    .locator('tbody tr');
  await expect(rows).toHaveCount(100);
  await page.getByRole('button', { name: 'Next samples', exact: true }).click();
  await expect(rows.first().getByRole('rowheader')).toHaveText('101');
  await page.getByRole('tab', { name: 'Statistics', exact: true }).click();
  await expect(
    page
      .getByRole('table', { name: 'Signal range statistics', exact: true })
      .locator('tbody tr td')
      .first(),
  ).toHaveText('1991');
  await page.getByRole('tab', { name: 'Plot', exact: true }).click();
  await page.getByLabel('Signal Y scale', { exact: true }).selectOption('log');
  await expect(
    page.getByRole('region', { name: 'Scientific signal viewer', exact: true }),
  ).toContainText('non-positive values');
  expect((await current(request)).revision).toBe(workspace.revision);
});

test('saved fit inputs, residual diagnostics and ROI-volume curves share the scientific viewer', async ({
  page,
  request,
}) => {
  await study(page);
  await selectWorkflow(page, 'Fit');
  await page.getByRole('button', { name: 'Run preview fit', exact: true }).click();
  await expect(page.getByLabel('Fitted T2', { exact: true })).toBeVisible();
  await page
    .getByRole('region', { name: 'ROI signal', exact: true })
    .getByRole('button', { name: 'Open signal curves', exact: true })
    .click();
  await expect(page.getByLabel('Signal source', { exact: true })).toHaveValue('fit');
  await expect(page.locator('[data-signal-trace]')).toHaveCount(2);
  await page.getByLabel('Signal fit display', { exact: true }).selectOption('residual');
  await expect(
    page.getByRole('img', { name: 'Scientific signal residual plot', exact: true }),
  ).toBeVisible();
  await page.getByRole('tab', { name: 'Statistics', exact: true }).click();
  await expect(
    page.getByRole('region', { name: 'Signal fit diagnostics', exact: true }),
  ).toContainText('RMSE');
  const before = await current(request);
  await page.getByLabel('Signal source', { exact: true }).selectOption('volume');
  await page.getByRole('button', { name: 'All ROIs', exact: true }).click();
  await page.getByRole('button', { name: 'Plot ROI curves', exact: true }).click();
  await expect(
    page.getByRole('table', { name: 'Signal range statistics', exact: true }),
  ).toContainText('Anterior');
  await expect(
    page.getByRole('table', { name: 'Signal range statistics', exact: true }),
  ).toContainText('Posterior');
  await page.getByRole('tab', { name: 'Plot', exact: true }).click();
  await expect(page.locator('[data-signal-trace]')).toHaveCount(2);
  await page.getByLabel('Plot Anterior', { exact: true }).uncheck();
  await expect(page.locator('[data-signal-trace]')).toHaveCount(1);
  await selectWorkflow(page, 'Analyze', 'ROI timecourses');
  await expect(page.getByRole('region', { name: 'Timecourse results', exact: true })).toContainText(
    'Anterior',
  );
  await selectWorkflow(page, 'View', 'Signal curves');
  await expect(page.getByLabel('Signal source', { exact: true })).toHaveValue('volume');
  await expect(page.getByLabel('Plot Anterior', { exact: true })).not.toBeChecked();
  expect((await current(request)).revision).toBe(before.revision);
});
