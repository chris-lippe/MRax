import { test, expect, type Page, type APIRequestContext } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { clickDataControl, selectTheme } from './ui';

async function folders(page: Page, request: APIRequestContext, base: string) {
  const names = ['Desktop', 'Documents', 'Downloads', 'Pictures', 'Music', 'Videos'];
  for (const name of [...names, 'Acquisitions', 'Acquisitions/Scan Ω', 'Results'])
    await mkdir(base + '/' + name, { recursive: true });
  await writeFile(base + '/.metadata.json', '{}');
  await writeFile(base + '/notes.txt', 'Independent synthetic file-browser fixture.');
  const roots = await (await request.get('/api/files/roots')).json();
  const home = roots.find((item: { id: string }) => item.id === 'home');
  expect(home).toBeTruthy();
  const places = [
    { ...home, label: 'Home', path: base, mounted: false },
    ...names.map((name) => ({
      ...home,
      id: name.toLowerCase(),
      label: name,
      path: base + '/' + name,
      mounted: false,
    })),
    { ...home, id: 'drive0', label: 'File system', path: '/', mounted: false },
  ];
  await page.route('**/api/files/roots', (route) => route.fulfill({ json: places }));
  // Exercise real directory reads with synthetic standard-folder aliases on every OS.
  await page.route(/\/api\/files\/(list|stat)\?/, async (route) => {
    const url = new URL(route.request().url());
    const key = url.searchParams.get('root');
    url.searchParams.set('root', home.id);
    const response = await route.fetch({ url: url.toString() });
    await route.fulfill({ response, json: { ...(await response.json()), root: key } });
  });
  await page.goto('/');
  await clickDataControl(page, 'Open workspace');
  const chooser = page.locator('.file-system-dialog');
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base);
  return chooser;
}

test('Explorer-style selection, keyboard navigation and history keep folder changes predictable', async ({
  page,
  request,
}, info) => {
  const base = info.outputPath('Folder browser Ω');
  const chooser = await folders(page, request, base);
  const locations = chooser.getByRole('navigation', { name: 'File locations', exact: true });
  for (const name of [
    'Home',
    'Desktop',
    'Documents',
    'Downloads',
    'Pictures',
    'Music',
    'Videos',
    'File system',
  ])
    await expect(locations.getByRole('button', { name, exact: true })).toBeVisible();
  await expect(chooser.getByRole('button', { name: 'Back', exact: true })).toBeDisabled();
  await expect(chooser.getByRole('button', { name: 'Forward', exact: true })).toBeDisabled();
  const contents = chooser.getByRole('list', { name: 'Files and folders', exact: true });
  await expect(contents).not.toContainText('▸');
  const acquisitions = contents.getByRole('button', { name: 'Acquisitions Folder', exact: true });
  await acquisitions.click();
  await expect(acquisitions).toHaveAttribute('aria-pressed', 'true');
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base);
  await expect(chooser.getByLabel('File name', { exact: true })).toHaveValue('Acquisitions');
  await acquisitions.press('Enter');
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Acquisitions');
  await expect(chooser.getByLabel('File name', { exact: true })).toBeEmpty();
  await chooser.getByRole('button', { name: 'Scan Ω Folder', exact: true }).dblclick();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(
    base + '/Acquisitions/Scan Ω',
  );
  await chooser.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Acquisitions');
  await chooser.getByRole('button', { name: 'Forward', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(
    base + '/Acquisitions/Scan Ω',
  );
  await chooser.press('Alt+ArrowUp');
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Acquisitions');
  await chooser.press('Alt+ArrowLeft');
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(
    base + '/Acquisitions/Scan Ω',
  );
  await chooser.press('Alt+ArrowRight');
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Acquisitions');
  await locations.getByRole('button', { name: 'Documents', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Documents');
  await expect(chooser.getByRole('button', { name: 'Forward', exact: true })).toBeDisabled();
  await locations.getByRole('button', { name: 'Home', exact: true }).click();
  await expect(
    chooser.getByRole('button', { name: '.metadata.json 2 B', exact: true }),
  ).toBeVisible();
  await acquisitions.click();
  await acquisitions.press('ArrowDown');
  await expect(contents.getByRole('button', { name: 'Desktop Folder', exact: true })).toBeFocused();
  await chooser.press('Control+l');
  await expect(chooser.getByLabel('Location', { exact: true })).toBeFocused();
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Open workspace', exact: true })).toBeFocused();
});

test('favorite shortcuts persist across dialogs and reloads and can be removed without changing folders', async ({
  page,
  request,
}, info) => {
  const base = info.outputPath('Favorite folders Ω');
  const chooser = await folders(page, request, base);
  const locations = chooser.getByRole('navigation', { name: 'File locations', exact: true });
  const favorites = locations.getByRole('region', { name: 'Favorite folders', exact: true });
  await locations.getByRole('button', { name: 'Documents', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Documents');
  await chooser.getByRole('button', { name: 'Add to favorites', exact: true }).click();
  await expect(favorites.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await expect(
    chooser.getByRole('button', { name: 'Remove favorite', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true');
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await clickDataControl(page, 'Open workspace');
  await expect(favorites.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.reload();
  await clickDataControl(page, 'Open workspace');
  await favorites.getByRole('button', { name: 'Documents', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Documents');
  await expect(
    chooser.getByRole('button', { name: 'Remove favorite', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true');
  await favorites
    .getByRole('button', { name: 'Remove Documents from favorites', exact: true })
    .click();
  await expect(favorites.getByRole('button', { name: 'Documents', exact: true })).toHaveCount(0);
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Documents');
  await expect(
    chooser.getByRole('button', { name: 'Add to favorites', exact: true }),
  ).toBeVisible();
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.reload();
  await clickDataControl(page, 'Open workspace');
  await expect(favorites.getByRole('button', { name: 'Documents', exact: true })).toHaveCount(0);
  await locations.getByRole('button', { name: 'Documents', exact: true }).click();
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base + '/Documents');
});

test('file dialog stays readable in both themes and a shorter desktop window', async ({
  page,
  request,
}, info) => {
  const base = info.outputPath('Synthetic studies');
  const chooser = await folders(page, request, base);
  await chooser.getByRole('button', { name: 'Add to favorites', exact: true }).click();
  await page.screenshot({ path: info.outputPath('file-browser-light.png') });
  await chooser.getByRole('button', { name: 'Cancel', exact: true }).click();
  await selectTheme(page, 'dark');
  await clickDataControl(page, 'Open workspace');
  await expect(chooser.getByLabel('Location', { exact: true })).toHaveValue(base);
  await page.screenshot({ path: info.outputPath('file-browser-dark.png') });
  await page.setViewportSize({ width: 900, height: 650 });
  await expect(chooser.getByRole('button', { name: 'Open', exact: true })).toBeInViewport();
  await expect(chooser.getByRole('button', { name: 'Cancel', exact: true })).toBeInViewport();
  expect(await chooser.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
    true,
  );
  await page.setViewportSize({ width: 600, height: 750 });
  await expect(chooser.getByLabel('Location', { exact: true })).toBeInViewport();
  await expect(chooser.getByRole('button', { name: 'Open', exact: true })).toBeInViewport();
  expect(await chooser.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
    true,
  );
});
