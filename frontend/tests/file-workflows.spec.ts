import { clickDataControl, loadExample } from './ui';
import { openRoiCoordinates } from './ui';
import { test, expect } from '@playwright/test';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import {
  chooseFilePath,
  chooseExportFolder,
  saveWorkspace,
  openWorkspace,
  selectTheme,
  selectWorkflow,
  openExport,
  closeExport,
} from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('mounted-folder shortcuts open hidden files in their mounted location', async ({
  page,
  request,
}, info) => {
  const mounted = info.outputPath('.mounted studies');
  await mkdir(mounted, { recursive: true });
  await writeFile(mounted + '/.metadata.json', '{}');
  const places = await (await request.get('/api/files/roots')).json();
  const home = places.find((place: { id: string }) => place.id === 'home');
  const working = places.find((place: { id: string }) => place.id === 'working');
  expect(home).toBeTruthy();
  expect(working).toBeTruthy();
  await page.route('**/api/files/roots', (route) =>
    route.fulfill({
      json: [
        { ...home, mounted: false },
        { ...working, label: '.mounted studies', path: mounted, mounted: true },
      ],
    }),
  );
  await page.goto('/');
  await clickDataControl(page, 'Open workspace');
  const chooser = page.locator('.file-system-dialog');
  await expect(
    chooser.getByRole('heading', { name: 'Mounted folders', exact: true }),
  ).toBeVisible();
  const shortcut = chooser.getByRole('button', { name: '.mounted studies', exact: true });
  await expect(shortcut).toBeEnabled();
  await shortcut.click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(mounted);
  await expect(
    chooser.getByRole('button', { name: '.metadata.json 2 B', exact: true }),
  ).toBeVisible();
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
});

test('ordinary Save and Open preserve a study without browser transfers', async ({
  page,
  request,
}, info) => {
  const transfers: string[] = [];
  page.on('download', () => transfers.push('download'));
  page.on('filechooser', () => transfers.push('upload'));
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByText(/Backend connected|Connecting to|Local by design/)).toHaveCount(0);
  await expect(page.locator('input[type=file]')).toHaveCount(0);
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toBeVisible();
  const before = (await (await request.get('/api/workspaces')).json())[0];
  await clickDataControl(page, 'Save workspace');
  const chooser = page.locator('.file-system-dialog');
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  const cancelled = await (await request.get('/api/workspaces/' + before.id)).json();
  expect(cancelled.dirty).toBe(before.dirty);
  const file = info.outputPath('Study Ω.mrax');
  await saveWorkspace(page, file);
  expect((await stat(file)).size).toBeGreaterThan(0);
  const original = await readFile(file);
  await clickDataControl(page, 'Save workspace');
  await expect(chooser.getByRole('button', { name: 'Go', exact: true })).toBeEnabled();
  await chooser.getByLabel('Location', { exact: true }).fill(file);
  await chooser.getByRole('button', { name: 'Go', exact: true }).click();
  await expect(chooser.getByLabel('File name', { exact: true })).toHaveValue('Study Ω.mrax');
  await chooser.getByRole('button', { name: 'Save', exact: true }).click();
  await expect(chooser.getByRole('button', { name: 'Replace', exact: true })).toBeVisible();
  await chooser.getByRole('button', { name: 'Keep existing', exact: true }).click();
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  expect(await readFile(file)).toEqual(original);
  await openWorkspace(page, file);
  const restored = (await (await request.get('/api/workspaces')).json()).at(-1);
  expect(restored.experiments).toEqual(before.experiments);
  expect(restored.shape).toEqual(before.shape);
  expect(transfers).toEqual([]);
  await clickDataControl(page, 'Open workspace');
  await expect(chooser.getByRole('button', { name: 'Go', exact: true })).toBeEnabled();
  await page.screenshot({ path: info.outputPath('file-chooser-light.png') });
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await selectTheme(page, 'dark');
  await clickDataControl(page, 'Open workspace');
  await expect(chooser.getByRole('button', { name: 'Go', exact: true })).toBeEnabled();
  await page.screenshot({ path: info.outputPath('file-chooser-dark.png') });
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
});

test('Bruker studies open directly from their acquisition folder', async ({ page }, info) => {
  const source = info.outputPath('Synthetic Bruker study');
  await mkdir(source, { recursive: true });
  execFileSync('python', [
    '-c',
    'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd().parent/"tests")); from _synthetic import _write_minimal_bruker_scan; _write_minimal_bruker_scan(Path(sys.argv[1]))',
    source,
  ]);
  await page.goto('/');
  await clickDataControl(page, 'Import data');
  await page.getByRole('button', { name: 'Browse files and folders…', exact: true }).click();
  await chooseFilePath(page, source, 'Open', false, true);
  await page.getByLabel('Import format', { exact: true }).selectOption('bruker');
  await page.getByText('Import options and observables', { exact: true }).click();
  await page
    .getByLabel('Import options', { exact: true })
    .fill('{"field_strength_mhz":400,"load_raw_kspace":false}');
  await page.getByRole('button', { name: 'Import study', exact: true }).click();
  await expect(page.getByRole('img', { name: /2 by 2 quantitative image/ })).toBeVisible({
    timeout: 60000,
  });
  await expect(page.getByRole('status', { name: 'Workspace status' })).not.toContainText('upload');
});

test('file navigation supports folders, search and a new export directory', async ({
  page,
}, info) => {
  const parent = info.outputPath('Folder navigation');
  await mkdir(parent, { recursive: true });
  await writeFile(parent + '/.metadata.json', '{}');
  await page.goto('/');
  await clickDataControl(page, 'Open workspace');
  const chooser = page.locator('.file-system-dialog');
  await expect(chooser.getByRole('button', { name: 'Go', exact: true })).toBeEnabled();
  await chooser.getByLabel('Location', { exact: true }).fill(parent);
  await chooser.getByRole('button', { name: 'Go', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(parent);
  await expect(chooser.getByRole('checkbox', { name: 'Hidden files', exact: true })).toHaveCount(0);
  await expect(
    chooser.getByRole('button', { name: '.metadata.json 2 B', exact: true }),
  ).toBeVisible();
  await chooser.getByRole('button', { name: 'New folder', exact: true }).click();
  await chooser.getByLabel('New folder name', { exact: true }).fill('Results Ω');
  await chooser.getByRole('button', { name: 'Create folder', exact: true }).click();
  const entry = chooser.getByRole('button', { name: 'Results Ω Folder', exact: true });
  await expect(entry).toBeVisible();
  await entry.dblclick();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(parent + '/Results Ω');
  await chooser.getByRole('button', { name: 'Parent folder', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(parent);
  await chooser.getByLabel('Find in folder', { exact: true }).fill('does-not-exist');
  await chooser.getByRole('button', { name: 'Find', exact: true }).click();
  await expect(chooser.getByText('No matching files or folders.', { exact: true })).toBeVisible();
  await chooser.getByLabel('Location', { exact: true }).fill(parent + '/missing-folder');
  await chooser.getByRole('button', { name: 'Go', exact: true }).click();
  await expect(chooser.getByRole('alert')).toHaveText('The selected file or folder does not exist');
  await expect(chooser.getByRole('button', { name: 'Open', exact: true })).toBeDisabled();
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
});

test('ROI sets export and reopen from a folder in the segmentation workflow', async ({
  page,
}, info) => {
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toBeVisible();
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  const destination = info.outputPath('ROI set');
  const exporter = await openExport(page);
  await exporter.getByText('Packages, ROI sets and reports', { exact: true }).click();
  await exporter.getByRole('button', { name: 'Save ROI set', exact: true }).click();
  await chooseExportFolder(page, destination, true);
  await expect(exporter.locator('.job-status .file-source')).toContainText(destination, {
    timeout: 60000,
  });
  await closeExport(page);
  await loadExample(page);
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toBeVisible();
  await selectWorkflow(page, 'Segment', 'Import masks and ROI sets');
  await page.getByRole('button', { name: 'Open ROI set or mask…', exact: true }).click();
  await chooseFilePath(page, destination, 'Open', false, true);
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await expect(page.getByRole('checkbox', { name: 'Show ROI 1', exact: true })).toBeVisible();
});
