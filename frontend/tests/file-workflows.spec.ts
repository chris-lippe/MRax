import { clickDataControl, loadExample } from './ui';

import { test, expect } from '@playwright/test';

import { readFile, stat } from 'node:fs/promises';

import { saveWorkspace, openWorkspace, selectTheme } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
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
  let releaseListing!: () => void;
  const listingGate = new Promise<void>((resolve) => (releaseListing = resolve));
  await page.route(
    '**/api/files/list?**',
    async (route) => {
      await listingGate;
      await route.continue();
    },
    { times: 1 },
  );
  await clickDataControl(page, 'Save workspace');
  const chooser = page.locator('.file-system-dialog');
  try {
    await expect(chooser.getByLabel('Location', { exact: true })).not.toBeEditable();
  } finally {
    releaseListing();
  }
  await expect(chooser.getByLabel('Location', { exact: true })).toBeEditable();
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
  await expect(chooser.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
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
