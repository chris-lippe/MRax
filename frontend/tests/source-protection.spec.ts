import { test, expect, type Page } from '@playwright/test';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { loadExample, openExport, closeExport, chooseExportFolder } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

async function selectExisting(page: Page, path: string) {
  const chooser = page.locator('.file-system-dialog');
  await expect(chooser.getByRole('button', { name: 'Go', exact: true })).toBeEnabled();
  const parent = path.slice(0, path.lastIndexOf('/'));
  await chooser.getByLabel('Location', { exact: true }).fill(parent);
  await chooser.getByRole('button', { name: 'Go', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(parent);
  await chooser
    .locator('input[aria-label="File name"], input[aria-label="Folder name"]')
    .fill(path.split('/').at(-1)!);
  await expect(
    chooser.locator('input[aria-label="File name"], input[aria-label="Folder name"]'),
  ).toHaveValue(path.split('/').at(-1)!);
  await chooser.getByRole('button', { name: 'Save', exact: true }).click();
  return chooser;
}

test('image replacement waits for confirmation and cancelling leaves the file unchanged', async ({
  page,
}, info) => {
  const parent = info.outputPath('Existing files');
  await mkdir(parent, { recursive: true });
  const path = parent + '/image.npy';
  await writeFile(path, 'independent existing input');
  await page.goto('/');
  await loadExample(page);
  const exporter = await openExport(page);
  await exporter
    .getByLabel('Export object', { exact: true })
    .selectOption('T2 phantom/image/mr_image');
  await exporter.getByLabel('Export format', { exact: true }).selectOption('npy');
  await exporter.getByRole('button', { name: 'Export selected object', exact: true }).click();
  const chooser = await selectExisting(page, path);
  await expect(chooser.getByRole('alert')).toContainText('will be overwritten');
  expect(await readFile(path, 'utf8')).toBe('independent existing input');
  await chooser.getByRole('button', { name: 'Keep existing', exact: true }).click();
  expect(await readFile(path, 'utf8')).toBe('independent existing input');
  await chooser.getByRole('button', { name: 'Save', exact: true }).click();
  await chooser.getByRole('button', { name: 'Replace', exact: true }).click();
  await expect(chooser).not.toBeVisible();
  await expect(exporter.locator('.job-status .file-source')).toContainText(path, {
    timeout: 60000,
  });
  expect((await readFile(path)).subarray(0, 6).toString('latin1')).toBe('\x93NUMPY');
  expect(await readdir(parent)).toEqual(['image.npy']);
  await closeExport(page);
});

test('metadata exports require a new filename without offering replacement', async ({
  page,
}, info) => {
  const parent = info.outputPath('Metadata');
  await mkdir(parent, { recursive: true });
  const path = parent + '/metadata.json';
  await writeFile(path, '{"keep": true}');
  await page.goto('/');
  await loadExample(page);
  const exporter = await openExport(page);
  await exporter
    .getByLabel('Export object', { exact: true })
    .selectOption('T2 phantom/image/mr_image');
  await exporter.getByLabel('Export format', { exact: true }).selectOption('json');
  await exporter.getByRole('button', { name: 'Export selected object', exact: true }).click();
  const chooser = await selectExisting(page, path);
  await expect(chooser.getByRole('alert')).toContainText('Choose a new name');
  await expect(chooser.getByRole('button', { name: 'Replace', exact: true })).toHaveCount(0);
  expect(await readFile(path, 'utf8')).toBe('{"keep": true}');
  expect(await readdir(parent)).toEqual(['metadata.json']);
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await closeExport(page);
});

test('multi-file exports retain existing contents and confirm only matching filenames', async ({
  page,
}, info) => {
  const path = info.outputPath('Existing results');
  await mkdir(path + '/unrelated', { recursive: true });
  await writeFile(path + '/keep.txt', 'keep');
  await writeFile(path + '/unrelated/keep.txt', 'nested keep');
  const originalDirectory = await stat(path);
  const originalFile = await stat(path + '/keep.txt');
  await page.goto('/');
  await loadExample(page);
  const exporter = await openExport(page);
  await exporter
    .getByLabel('Export object', { exact: true })
    .selectOption('T2 phantom/image/mr_image');
  await exporter.getByLabel('Export format', { exact: true }).selectOption('nifti');
  await exporter.getByRole('button', { name: 'Export selected object', exact: true }).click();
  await chooseExportFolder(page, path);
  await expect(exporter.locator('.job-status .file-source')).toContainText(path, {
    timeout: 60000,
  });
  const names = (await readdir(path)).filter((name) => name.endsWith('.nii'));
  expect(names.length).toBe(8);
  const first = await readFile(path + '/' + names[0]);
  await exporter.getByRole('button', { name: 'Export selected object', exact: true }).click();
  await chooseExportFolder(page, path);
  const review = exporter.getByRole('alert', { name: 'Existing output files', exact: true });
  await expect(review).toBeVisible({ timeout: 60000 });
  await expect(review.getByRole('listitem')).toHaveCount(8);
  await expect(review).toContainText('Only the listed files');
  await expect(exporter.locator('.job-status .file-source')).toHaveCount(0);
  await page.screenshot({ path: info.outputPath('export-file-confirmation.png') });
  expect(await readFile(path + '/' + names[0])).toEqual(first);
  await review.getByRole('button', { name: 'Cancel export', exact: true }).click();
  await expect(review).not.toBeVisible();
  await expect(exporter.locator('.job-status')).toContainText('Export cancelled');
  await writeFile(path + '/' + names[0], 'previous image');
  await exporter.getByRole('button', { name: 'Export selected object', exact: true }).click();
  await chooseExportFolder(page, path);
  await expect(review).toBeVisible({ timeout: 60000 });
  await review.getByRole('button', { name: 'Replace listed files', exact: true }).click();
  await expect(exporter.locator('.job-status .file-source')).toContainText(path, {
    timeout: 60000,
  });
  await expect(review).not.toBeVisible();
  expect(await readFile(path + '/' + names[0])).toEqual(first);
  expect(await readFile(path + '/keep.txt', 'utf8')).toBe('keep');
  expect(await readFile(path + '/unrelated/keep.txt', 'utf8')).toBe('nested keep');
  expect((await stat(path)).ino).toBe(originalDirectory.ino);
  expect((await stat(path + '/keep.txt')).mtimeMs).toBe(originalFile.mtimeMs);
  expect((await readdir(path)).sort()).toEqual([...names, 'keep.txt', 'unrelated'].sort());
  await closeExport(page);
});
