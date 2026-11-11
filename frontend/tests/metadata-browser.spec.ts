import { zoomImage, imageZoom } from './ui';

import { loadExample } from './ui';

import { writeFile, mkdir } from 'node:fs/promises';

import { dirname } from 'node:path';

import { chooseFilePath } from './ui';

import { expect, test, type APIRequestContext, type Page } from '@playwright/test';

import { openMetadataBrowser, selectDataFormat, selectTheme, selectWorkflow } from './ui';

async function current(request: APIRequestContext) {
  return (await (await request.get('/api/workspaces')).json()).at(-1);
}

async function inspect(request: APIRequestContext) {
  const workspace = await current(request);
  return (await request.get(`/api/workspaces/${workspace.id}/inspect`)).json();
}

async function load(page: Page, multimodal = false) {
  await page.goto('/');
  await loadExample(page, multimodal ? 'multimodal' : 't2');
  return openMetadataBrowser(page, 'Experiment metadata');
}

async function add(page: Page, name: string, value: string) {
  await page
    .getByRole('region', { name: 'Metadata browser', exact: true })
    .getByRole('button', { name: 'Add field', exact: true })
    .click();
  const form = page.getByRole('dialog', { name: 'Add metadata field', exact: true });
  await form.getByLabel('Metadata field name', { exact: true }).fill(name);
  await form.getByLabel('New metadata value', { exact: true }).fill(value);
  await form.getByRole('button', { name: 'Add to draft', exact: true }).click();
  await expect(form).not.toBeVisible();
  await expect(
    page.getByRole('textbox', { name: `Experiment metadata / metadata / ${name}`, exact: true }),
  ).toBeFocused();
}

async function rawMetadata(page: Page) {
  await selectDataFormat(page, 'Experiment metadata', 'Raw');
  const result = JSON.parse(
    await page.getByRole('textbox', { name: 'Experiment metadata', exact: true }).inputValue(),
  );
  await selectDataFormat(page, 'Experiment metadata', 'Table');
  return result;
}

async function importFile(page: Page, name: string, contents: string) {
  await page
    .getByRole('region', { name: 'Metadata browser', exact: true })
    .getByRole('button', { name: 'Import from file', exact: true })
    .click();
  const dialog = page.getByRole('dialog', { name: 'Import metadata from file', exact: true });
  const file = test.info().outputPath(name);
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, contents);
  await dialog.getByRole('button', { name: 'Open metadata file…', exact: true }).click();
  await chooseFilePath(page, file);
  await expect(
    dialog.getByRole('button', { name: 'Open metadata file…', exact: true }),
  ).toBeEnabled();
  return dialog;
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

test('metadata browser edits study and experiment fields and inspects acquisition metadata', async ({
  page,
  request,
}, info) => {
  await page.goto('/');
  await loadExample(page);
  await zoomImage(page, 2);
  const initialZoom = await imageZoom(page);
  await selectWorkflow(page, 'View', 'Metadata');
  const browser = page.getByRole('region', { name: 'Metadata browser', exact: true });
  await expect(browser).toBeVisible();
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  await add(page, 'review_note', 'Needs review');
  const field = browser.getByRole('textbox', {
    name: 'Experiment metadata / metadata / review_note',
    exact: true,
  });
  await field.fill('Reviewed');
  expect((await inspect(request)).metadata.metadata.review_note).toBeUndefined();
  await browser.getByLabel('Search metadata', { exact: true }).fill('review_note');
  await browser
    .getByRole('button', { name: 'Open Experiment metadata /metadata/review_note', exact: true })
    .click();
  await expect(field).toHaveValue('Reviewed');
  await browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect
    .poll(async () => (await inspect(request)).metadata.metadata.review_note)
    .toBe('Reviewed');
  await browser.getByLabel('Search metadata', { exact: true }).fill('');
  await browser.getByRole('button', { name: 'Browse /observables', exact: true }).click();
  await browser.getByRole('button', { name: 'Add field', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Add metadata field', exact: true });
  await expect(dialog.getByLabel('Metadata field type', { exact: true })).toHaveValue('quantity');
  await dialog.getByLabel('Metadata field name', { exact: true }).fill('TR');
  await dialog.getByLabel('New metadata value', { exact: true }).fill('1000');
  await dialog.getByLabel('New metadata unit', { exact: true }).fill('ms');
  await dialog.getByRole('button', { name: 'Add to draft', exact: true }).click();
  await browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect
    .poll(async () => (await inspect(request)).metadata.observables.TR)
    .toMatchObject({ values: 1000, unit: 'ms' });
  await openMetadataBrowser(page, 'Study overview');
  await page.getByLabel('Study name', { exact: true }).fill('Metadata study');
  await page.getByRole('button', { name: 'Rename study', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Metadata study', exact: true })).toBeVisible();
  await openMetadataBrowser(page, 'Arrays');
  await expect(
    browser.getByRole('region', { name: 'Array details report', exact: true }),
  ).toBeVisible();
  await expect(browser.getByRole('button', { name: 'Add field', exact: true })).toHaveCount(0);
  await openMetadataBrowser(page, 'Diagnostics');
  await expect(
    browser.getByRole('region', { name: 'Diagnostics and provenance report', exact: true }),
  ).toBeVisible();
  await selectWorkflow(page, 'View', 'Image display');
  await expect.poll(() => imageZoom(page)).toBe(initialZoom);
  await selectWorkflow(page, 'View', 'Metadata');
  await openMetadataBrowser(page, 'Experiment metadata');
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  for (const theme of ['light', 'dark'] as const) {
    await selectTheme(page, theme);
    await page.screenshot({ path: info.outputPath(`metadata-browser-${theme}.png`) });
  }
});

test('JSON, quoted CSV and TSV imports preview and merge typed data before Apply', async ({
  page,
  request,
}) => {
  const browser = await load(page);
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  let dialog = await importFile(
    page,
    'metadata.json',
    JSON.stringify({
      sequence: { description: 'Original description' },
      'receiver/gain': 3,
      reviewed: true,
    }),
  );
  await expect(dialog.getByRole('button', { name: 'Import to draft', exact: true })).toBeEnabled();
  expect((await inspect(request)).metadata.metadata.sequence).toBeUndefined();
  await dialog.getByRole('button', { name: 'Import to draft', exact: true }).click();
  const csv =
    '\uFEFFpath,value,type,unit\r\n/sequence/description,Changed,string,\r\n/sequence/TR,1500,number,ms\r\n/sequence/text,"Line 1\nLine 2",string,\r\n/receiver~1gain,99,number,\r\n';
  dialog = await importFile(page, 'metadata.csv', csv);
  await dialog.getByLabel('Metadata import mode', { exact: true }).selectOption('add');
  await expect(dialog.getByRole('button', { name: 'Import to draft', exact: true })).toBeEnabled();
  await dialog.getByRole('button', { name: 'Import to draft', exact: true }).click();
  let draft = await rawMetadata(page);
  expect(draft.metadata.sequence).toEqual({
    description: 'Original description',
    TR: { values: 1500, unit: 'ms' },
    text: 'Line 1\nLine 2',
  });
  expect(draft.metadata['receiver/gain']).toBe(3);
  dialog = await importFile(
    page,
    'metadata.tsv',
    'name\tvalue\ttype\tunit\nidentifier\t001\tstring\t\nreviewed\tfalse\tboolean\t\ntext\t"tab\tinside"\tstring\t\n',
  );
  await expect(dialog.getByRole('button', { name: 'Import to draft', exact: true })).toBeEnabled();
  await dialog.getByRole('button', { name: 'Import to draft', exact: true }).click();
  draft = await rawMetadata(page);
  expect(draft.metadata.identifier).toBe('001');
  expect(draft.metadata.reviewed).toBe(false);
  expect(draft.metadata.text).toBe('tab\tinside');
  expect((await inspect(request)).metadata.metadata.identifier).toBeUndefined();
  await browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect.poll(async () => (await inspect(request)).metadata.metadata.identifier).toBe('001');
  expect((await inspect(request)).metadata.metadata.sequence.TR).toEqual({
    values: 1500,
    unit: 'ms',
  });
});
