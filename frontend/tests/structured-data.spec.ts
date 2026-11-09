import { clickDataControl, loadExample } from './ui';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import { closeMetadataBrowser, openMetadataBrowser, selectDataFormat, selectTheme } from './ui';

async function inspection(request: APIRequestContext) {
  const workspaces = await (await request.get('/api/workspaces')).json();
  return (await request.get(`/api/workspaces/${workspaces.at(-1).id}/inspect`)).json();
}

async function openMetadata(page: Page, load = true) {
  if (load) {
    await page.goto('/');
    await loadExample(page);
  }
  await openMetadataBrowser(page, 'Experiment metadata');
  const editor = page.getByRole('form', { name: 'Experiment metadata editor', exact: true });
  await expect(editor).toBeVisible();
  return editor;
}

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    const response = await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
    expect(response.status()).toBe(204);
  }
});

test('metadata defaults to editable tables and preserves typed quantities and nested records through Raw', async ({
  page,
  request,
}, info) => {
  const editor = await openMetadata(page);
  const original = (await inspection(request)).metadata;
  await expect(editor.getByRole('button', { name: 'Table', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  const unit = editor.getByRole('textbox', {
    name: 'Experiment metadata / observables / TE / unit',
    exact: true,
  });
  await expect(unit).toHaveValue(original.observables.TE.unit);
  await editor
    .getByLabel('Expand Experiment metadata / observables / TE / values', { exact: true })
    .click();
  await editor
    .getByRole('textbox', {
      name: 'Experiment metadata / observables / TE / values / 0',
      exact: true,
    })
    .fill('12.5');
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  const raw = editor.getByRole('textbox', { name: 'Experiment metadata', exact: true });
  const payload = JSON.parse(await raw.inputValue());
  expect(payload.observables.TE.values[0]).toBe(12.5);
  expect(payload.observables.TE.unit).toBe(original.observables.TE.unit);
  payload.metadata = {
    'coil/label': 'Receiver A',
    enabled: true,
    missing: null,
    records: [
      { phase: 'baseline', value: 1 },
      { phase: 'delayed', value: 2.5 },
    ],
    mixed: [1, '1', false, null],
  };
  await raw.fill(JSON.stringify(payload));
  await selectDataFormat(page, 'Experiment metadata', 'Table');
  await editor
    .getByRole('textbox', { name: 'Experiment metadata / metadata / coil/label', exact: true })
    .fill('Receiver B');
  await editor
    .getByRole('checkbox', { name: 'Experiment metadata / metadata / enabled', exact: true })
    .uncheck();
  await editor
    .getByLabel('Expand Experiment metadata / metadata / records', { exact: true })
    .click();
  const records = editor.getByRole('table', {
    name: 'Experiment metadata / metadata / records table',
    exact: true,
  });
  await expect(records.getByRole('columnheader')).toHaveText(['Index', 'phase', 'value']);
  await editor
    .getByRole('textbox', {
      name: 'Experiment metadata / metadata / records / 1 / value',
      exact: true,
    })
    .fill('2.75');
  expect((await inspection(request)).metadata).toEqual(original);
  await editor.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect
    .poll(async () => (await inspection(request)).metadata.metadata)
    .toMatchObject({
      'coil/label': 'Receiver B',
      enabled: false,
      missing: null,
      records: [
        { phase: 'baseline', value: 1 },
        { phase: 'delayed', value: 2.75 },
      ],
      mixed: [1, '1', false, null],
    });
  const saved = (await inspection(request)).metadata;
  expect(saved.observables.TE.values[0]).toBeCloseTo(12.5, 5);
  expect(saved.observables.TE.unit).toBe(original.observables.TE.unit);
  for (const theme of ['light', 'dark'] as const) {
    await closeMetadataBrowser(page);
    await selectTheme(page, theme);
    await openMetadata(page, false);
    await expect(unit).toHaveValue(original.observables.TE.unit);
    await page.screenshot({ path: info.outputPath(`metadata-${theme}.png`) });
  }
});

test('failed conversions and incomplete numeric edits retain the exact raw draft and cannot be applied', async ({
  page,
  request,
}) => {
  const editor = await openMetadata(page);
  const original = (await inspection(request)).metadata;
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  const raw = editor.getByRole('textbox', { name: 'Experiment metadata', exact: true });
  const broken = '{"metadata": {"value": 2,}}';
  await raw.fill(broken);
  await editor.getByRole('button', { name: 'Table', exact: true }).click();
  await expect(editor.getByRole('alert')).toContainText('Table view unavailable');
  await expect(raw).toHaveValue(broken);
  await expect(editor.getByRole('button', { name: 'Raw', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await editor.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  expect((await inspection(request)).metadata).toEqual(original);
  const payload = { ...original, metadata: { value: 2, note: 'Keep this edit' } };
  await raw.fill(JSON.stringify(payload));
  await selectDataFormat(page, 'Experiment metadata', 'Table');
  const number = editor.getByRole('textbox', {
    name: 'Experiment metadata / metadata / value',
    exact: true,
  });
  await number.fill('1e');
  await expect(number).toHaveValue('1e');
  await expect(number).toHaveAttribute('aria-invalid', 'true');
  await editor.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect(raw).toHaveValue(/"value": 1e/);
  await expect(raw).toHaveValue(/Keep this edit/);
  await expect(editor.getByRole('alert')).toBeVisible();
  expect((await inspection(request)).metadata).toEqual(original);
  payload.metadata.value = 0.001;
  await raw.fill(JSON.stringify(payload));
  await selectDataFormat(page, 'Experiment metadata', 'Table');
  await editor.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect
    .poll(async () => (await inspection(request)).metadata.metadata)
    .toMatchObject(payload.metadata);
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  const invalidShape = JSON.parse(await raw.inputValue());
  invalidShape.observables.TE.values = [1, 2];
  await raw.fill(JSON.stringify(invalidShape));
  await editor.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect(editor.getByRole('alert')).toContainText('TE');
  expect(JSON.parse(await raw.inputValue()).observables.TE.values).toEqual([1, 2]);
  expect((await inspection(request)).metadata.observables).toEqual(original.observables);
});

test('study constants expose values and units as separate editable fields', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await clickDataControl(page, 'New study');
  await openMetadataBrowser(page, 'Study constants');
  await selectDataFormat(page, 'Study constants', 'Raw');
  await page.getByRole('textbox', { name: 'Study constants', exact: true }).fill(
    JSON.stringify({
      field_strength_mhz: { values: 123.4, unit: 'MHz' },
    }),
  );
  await selectDataFormat(page, 'Study constants', 'Table');
  await page
    .getByRole('textbox', { name: 'Study constants / field_strength_mhz / values', exact: true })
    .fill('124.5');
  await expect(
    page.getByRole('textbox', { name: 'Study constants / field_strength_mhz / unit', exact: true }),
  ).toHaveValue('MHz');
  await page.getByRole('button', { name: 'Apply study constants', exact: true }).click();
  await expect
    .poll(async () => (await inspection(request)).study_globals.field_strength_mhz.values)
    .toBeCloseTo(124.5, 3);
  expect((await inspection(request)).study_globals.field_strength_mhz.unit).toBe('MHz');
});

test('long record tables keep off-page values and reports offer read-only tables with Raw access', async ({
  page,
}) => {
  const editor = await openMetadata(page);
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  const raw = editor.getByRole('textbox', { name: 'Experiment metadata', exact: true });
  const payload = JSON.parse(await raw.inputValue());
  payload.metadata = {
    samples: Array.from({ length: 120 }, (_, index) => ({ name: `Frame ${index}`, value: index })),
  };
  await raw.fill(JSON.stringify(payload));
  await selectDataFormat(page, 'Experiment metadata', 'Table');
  await editor
    .getByLabel('Expand Experiment metadata / metadata / samples', { exact: true })
    .click();
  const next = editor.getByRole('button', {
    name: 'Next rows for Experiment metadata / metadata / samples',
    exact: true,
  });
  await next.click();
  await next.click();
  await expect(next).toBeDisabled();
  await editor
    .getByRole('textbox', {
      name: 'Experiment metadata / metadata / samples / 105 / value',
      exact: true,
    })
    .fill('105.5');
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  const updated = JSON.parse(await raw.inputValue());
  expect(updated.metadata.samples).toHaveLength(120);
  expect(updated.metadata.samples[0].value).toBe(0);
  expect(updated.metadata.samples[105].value).toBe(105.5);
  expect(updated.metadata.samples[119].value).toBe(119);
  await openMetadataBrowser(page, 'Study overview');
  const report = page.getByRole('region', { name: 'Study summary report', exact: true });
  await expect(
    report.getByRole('table', { name: 'Study summary table', exact: true }),
  ).toBeVisible();
  await expect(report.getByRole('textbox')).toHaveCount(0);
  await selectDataFormat(page, 'Study summary', 'Raw');
  const source = JSON.parse(
    await report.getByLabel('Study summary raw', { exact: true }).innerText(),
  );
  expect(Array.isArray(source)).toBe(true);
  expect(source.length).toBeGreaterThan(0);
  await selectDataFormat(page, 'Study summary', 'Table');
  await expect(
    report.getByRole('table', { name: 'Study summary table', exact: true }),
  ).toBeVisible();
});

test('unsupported table depth automatically falls back to Raw without breaking the dialog', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  let nested: unknown = 'Original value';
  for (let index = 0; index < 20; index++) nested = { nested };
  await page.route('**/api/workspaces/*/inspect', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.metadata = nested;
    body.summary = nested;
    await route.fulfill({ response, json: body });
  });
  const editor = await openMetadata(page);
  await expect(editor.getByRole('button', { name: 'Raw', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(editor.getByRole('alert')).toContainText('deeply nested');
  const raw = editor.getByRole('textbox', { name: 'Experiment metadata', exact: true });
  expect(JSON.parse(await raw.inputValue())).toEqual(nested);
  await raw.fill('{"metadata": {"recovered": true}}');
  await selectDataFormat(page, 'Experiment metadata', 'Table');
  await expect(
    editor.getByRole('checkbox', {
      name: 'Experiment metadata / metadata / recovered',
      exact: true,
    }),
  ).toBeChecked();
  await expect(editor.getByRole('alert')).toHaveCount(0);
  await openMetadataBrowser(page, 'Study overview');
  const report = page.getByRole('region', { name: 'Study summary report', exact: true });
  await expect(report.getByRole('alert')).toContainText('deeply nested');
  expect(
    JSON.parse(await report.getByLabel('Study summary raw', { exact: true }).innerText()),
  ).toEqual(nested);
  expect(errors).toEqual([]);
});
