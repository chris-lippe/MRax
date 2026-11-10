import { zoomImage, imageZoom } from './ui';
import { loadExample } from './ui';
import { writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { chooseFilePath } from './ui';
import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import {
  openMetadataBrowser,
  selectDataFormat,
  selectExperiment,
  selectTheme,
  selectWorkflow,
} from './ui';

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

test('View Metadata replaces study details with editable paths, names and read-only inspection', async ({
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
  await expect(page.getByRole('button', { name: 'Study details', exact: true })).toHaveCount(0);
  await expect(
    page
      .getByRole('complementary', { name: 'Data & Studies', exact: true })
      .getByRole('button', { name: 'Metadata', exact: true }),
  ).toHaveCount(0);
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('region', { name: 'Image viewer', exact: true })).not.toBeVisible();
  await expect(
    page.getByRole('complementary', { name: 'Analysis controls', exact: true }),
  ).not.toBeVisible();
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

test('invalid imports remain editable and cannot partially change the draft', async ({
  page,
  request,
}) => {
  const browser = await load(page);
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  const original = await rawMetadata(page);
  const dialog = await importFile(
    page,
    'metadata.csv',
    'name,value,type\ngood,1,number\nbad,no,number\n',
  );
  await expect(dialog.getByRole('alert')).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Import to draft', exact: true })).toBeDisabled();
  const source = dialog.getByLabel('Metadata import source', { exact: true });
  await source.fill('path,value,type\n/a,1,number\n/a,2,number\n');
  await expect(dialog.getByRole('alert')).toContainText('repeats');
  await source.fill('path,value,type\n/a,{},object\n/a/b,2,number\n');
  await expect(dialog.getByRole('alert')).toContainText('overlaps');
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
  expect(await rawMetadata(page)).toEqual(original);
  await importFile(page, 'metadata.json', '{"recovered":');
  const jsonDialog = page.getByRole('dialog', { name: 'Import metadata from file', exact: true });
  await expect(jsonDialog.getByRole('alert')).toBeVisible();
  await jsonDialog.getByLabel('Metadata import source', { exact: true }).fill('{"recovered":42}');
  await expect(
    jsonDialog.getByRole('button', { name: 'Import to draft', exact: true }),
  ).toBeEnabled();
  await jsonDialog.getByRole('button', { name: 'Import to draft', exact: true }).click();
  expect((await rawMetadata(page)).metadata.recovered).toBe(42);
  expect((await inspect(request)).metadata.metadata.recovered).toBeUndefined();
  await browser.getByRole('button', { name: 'Revert draft', exact: true }).click();
  expect(await rawMetadata(page)).toEqual(original);
});

test('drafts follow experiments and require conflict resolution when values change elsewhere', async ({
  page,
  request,
}) => {
  const browser = await load(page, true);
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  await add(page, 'pending_note', 'Keep this draft');
  await selectExperiment(page, 'Complex coils');
  await expect(
    browser.getByRole('textbox', {
      name: 'Experiment metadata / metadata / pending_note',
      exact: true,
    }),
  ).toHaveCount(0);
  await selectExperiment(page, 'T2 phantom');
  await expect(
    browser.getByRole('textbox', {
      name: 'Experiment metadata / metadata / pending_note',
      exact: true,
    }),
  ).toHaveValue('Keep this draft');
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await openMetadataBrowser(page, 'Experiment metadata');
  await expect(
    browser.getByRole('textbox', {
      name: 'Experiment metadata / metadata / pending_note',
      exact: true,
    }),
  ).toHaveValue('Keep this draft');
  const workspace = await current(request);
  const payload = (await inspect(request)).metadata;
  payload.metadata.external_note = 'Applied elsewhere';
  const token = (await (await request.get('/api/session')).json()).token;
  const response = await request.post(`/api/workspaces/${workspace.id}/actions`, {
    headers: { 'x-mrax-token': token },
    data: { revision: workspace.revision, action: 'metadata.update', options: payload },
  });
  expect(response.status()).toBe(200);
  await page
    .getByRole('navigation', { name: 'Study and experiment browser', exact: true })
    .getByRole('button', { name: /^Synthetic multimodal study/ })
    .click();
  await expect(browser.getByRole('alert')).toContainText('Metadata changed elsewhere');
  await expect(
    browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }),
  ).toBeDisabled();
  await browser.getByRole('button', { name: 'Keep draft', exact: true }).click();
  await browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect
    .poll(async () => (await inspect(request)).metadata.metadata.pending_note)
    .toBe('Keep this draft');
  expect((await inspect(request)).metadata.metadata.external_note).toBe('Applied elsewhere');
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  await add(page, 'rename_note', 'Keep through rename');
  await openMetadataBrowser(page, 'Study overview');
  await page.getByLabel('Active experiment name', { exact: true }).fill('Renamed phantom');
  await page.getByRole('button', { name: 'Rename experiment', exact: true }).click();
  await expect(
    page
      .getByRole('region', { name: /^Experiments in / })
      .getByRole('button', { name: 'Renamed phantom', exact: true }),
  ).toHaveAttribute('aria-current', 'true');
  await openMetadataBrowser(page, 'Experiment metadata');
  await expect(
    browser.getByRole('textbox', {
      name: 'Experiment metadata / metadata / rename_note',
      exact: true,
    }),
  ).toHaveValue('Keep through rename');
  await browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
  await expect
    .poll(async () => (await inspect(request)).metadata.metadata.rename_note)
    .toBe('Keep through rename');
});

test('a delayed apply refresh cannot replace the metadata of another experiment', async ({
  page,
}) => {
  const browser = await load(page, true);
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  await add(page, 'delayed_note', 'Saved in T2');
  let hold = false;
  let held = 0;
  let release: () => void = () => undefined;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route('**/api/workspaces/*/inspect', async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    if (hold && body.metadata.model_name === 't2_se') {
      held++;
      await gate;
    }
    await route.fulfill({ response, json: body });
  });
  try {
    hold = true;
    await browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }).click();
    await expect.poll(() => held).toBeGreaterThan(0);
    await selectExperiment(page, 'Complex coils');
    await expect(
      browser.getByRole('textbox', { name: 'Experiment metadata / model_name', exact: true }),
    ).toHaveValue('identity');
    release();
    await expect(
      browser.getByRole('button', { name: 'Apply experiment metadata', exact: true }),
    ).toBeEnabled();
    await expect(
      browser.getByRole('textbox', { name: 'Experiment metadata / model_name', exact: true }),
    ).toHaveValue('identity');
    await expect(
      browser.getByRole('textbox', {
        name: 'Experiment metadata / metadata / delayed_note',
        exact: true,
      }),
    ).toHaveCount(0);
  } finally {
    release();
  }
});

test('search results paginate and metadata stays usable in compact desktop windows', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1000, height: 760 });
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'View', 'Metadata');
  const browser = page.getByRole('region', { name: 'Metadata browser', exact: true });
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await browser.getByRole('button', { name: 'Browse /metadata', exact: true }).click();
  const dialog = await importFile(
    page,
    'metadata.json',
    JSON.stringify(
      Object.fromEntries(
        Array.from({ length: 120 }, (_, index) => [`channel${index}`, `value ${index}`]),
      ),
    ),
  );
  await expect(dialog.getByRole('button', { name: 'Import to draft', exact: true })).toBeEnabled();
  await dialog.getByRole('button', { name: 'Import to draft', exact: true }).click();
  const search = browser.getByLabel('Search metadata', { exact: true });
  await search.fill('channel');
  await expect(browser.getByRole('status').filter({ hasText: '120 matches' })).toBeVisible();
  await browser.getByRole('button', { name: 'Next matches', exact: true }).click();
  await browser.getByRole('button', { name: 'Next matches', exact: true }).click();
  await expect(browser.getByRole('button', { name: 'Next matches', exact: true })).toBeDisabled();
  await search.fill('channel119');
  await browser
    .getByRole('button', { name: 'Open Experiment metadata /metadata/channel119', exact: true })
    .click();
  await expect(
    browser.getByRole('textbox', {
      name: 'Experiment metadata / metadata / channel119',
      exact: true,
    }),
  ).toHaveValue('value 119');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1000);
});
