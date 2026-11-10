import { test, expect } from '@playwright/test';
import { loadExample, openSettings } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('the permanent Data & Studies header aligns with the browser and workflow columns', async ({
  page,
}) => {
  await page.goto('/');
  const header = page.getByRole('banner', { name: 'MRax', exact: true });
  const label = header.getByRole('heading', { name: 'Data & Studies', exact: true });
  await expect(label).toBeVisible();
  await expect(label.locator('svg')).toBeVisible();
  await expect(
    header.getByRole('button', { name: /pin study browser|Data controls/i }),
  ).toHaveCount(0);
  const bounds = (await header.boundingBox())!;
  const studies = page.getByRole('complementary', { name: 'Data & Studies', exact: true });
  expect(bounds.x).toBe(0);
  expect(bounds.width).toBe(page.viewportSize()!.width);
  expect((await studies.boundingBox())!.y).toBe(bounds.height);
  expect(
    (await page.getByRole('separator', { name: 'Resize study browser' }).boundingBox())!.y,
  ).toBe(bounds.height);
  expect((await page.locator('.header-workflows').boundingBox())!.x).toBe(
    (await page.locator('main').boundingBox())!.x,
  );
  await expect(header.getByRole('img')).toHaveCount(0);
});

test('old collapsed preferences cannot hide studies and width remains adjustable across reloads', async ({
  page,
  request,
}) => {
  await page.addInitScript(() => localStorage.setItem('mrax.studyPanel', 'collapsed'));
  await page.goto('/');
  await loadExample(page);
  const original = (await (await request.get('/api/workspaces')).json())[0];
  const studies = page.getByRole('complementary', { name: 'Data & Studies', exact: true });
  await expect(studies).toBeVisible();
  const divider = page.getByRole('separator', { name: 'Resize study browser', exact: true });
  const before = Number(await divider.getAttribute('aria-valuenow'));
  await divider.focus();
  await divider.press('Shift+ArrowRight');
  await expect(divider).toHaveAttribute('aria-valuenow', String(before + 64));
  await page.reload();
  await expect(studies).toBeVisible();
  await expect(divider).toHaveAttribute('aria-valuenow', String(before + 64));
  await expect(page.getByRole('complementary', { name: 'Study browser shortcuts' })).toHaveCount(0);
  await expect(page.getByRole('dialog', { name: 'Data & Studies', exact: true })).toHaveCount(0);
  const current = (await (await request.get('/api/workspaces')).json())[0];
  expect(current.id).toBe(original.id);
  expect(current.revision).toBe(original.revision);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
});

test('studies and settings stay accessible in compact windows without pin or drawer controls', async ({
  page,
}) => {
  await page.setViewportSize({ width: 700, height: 800 });
  await page.goto('/');
  const studies = page.getByRole('complementary', { name: 'Data & Studies', exact: true });
  await expect(studies).toBeVisible();
  const settings = await openSettings(page);
  await settings.getByRole('button', { name: 'Done', exact: true }).click();
  await expect(studies.getByRole('button', { name: 'Settings', exact: true })).toBeFocused();
  await page.setViewportSize({ width: 1440, height: 1080 });
  await expect(studies).toBeVisible();
  await expect(page.getByRole('button', { name: /pin study browser/i })).toHaveCount(0);
});
