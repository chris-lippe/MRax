import { test, expect, type Page } from '@playwright/test';
import {
  loadExample,
  showDataControls,
  openNewStudy,
  openSettings,
  showStudies,
  selectWorkflow,
  openRoiCoordinates,
  zoomImage,
  imageZoom,
} from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    const response = await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
    expect(response.status()).toBe(204);
  }
});

async function dragBrowser(page: Page, width: number, release = true) {
  const divider = page.getByRole('separator', { name: 'Resize study browser', exact: true });
  const box = (await divider.boundingBox())!;
  const current = Number(await divider.getAttribute('aria-valuenow'));
  await page.mouse.move(box.x + box.width / 2, box.y + 80);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + width - current, box.y + 80, { steps: 16 });
  if (release) await page.mouse.up();
}

test('hover close targets its study and preserves the active view and keyboard focus', async ({
  page,
  request,
}, info) => {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await loadExample(page, 'multimodal');
  await zoomImage(page, 2);
  const studies = await showStudies(page);
  const row = studies.locator('.study-row').filter({
    has: page.getByRole('button', { name: 'Close study: Synthetic T2 phantom', exact: true }),
  });
  const close = row.getByRole('button', { name: 'Close study: Synthetic T2 phantom', exact: true });
  await page.getByRole('banner').hover();
  await expect(close).toHaveCSS('opacity', '0');
  await row.hover();
  await expect(close).toHaveCSS('opacity', '1');
  await page.screenshot({ path: info.outputPath('study-close-hover.png') });
  await close.click();
  const confirmation = page.getByRole('dialog', { name: 'Discard unsaved changes?', exact: true });
  await expect(confirmation).toContainText('Synthetic T2 phantom');
  await confirmation.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'Synthetic multimodal study', exact: true }),
  ).toBeVisible();
  await expect(close).toBeFocused();
  expect(await imageZoom(page)).toBeCloseTo(2);
  const rows = await (await request.get('/api/workspaces')).json();
  const active = rows.find((w: { label: string }) => w.label === 'Synthetic multimodal study');
  let imageRequests = 0;
  page.on('request', (r) => {
    if (r.url().includes('/api/workspaces/' + active.id + '/slice')) imageRequests++;
  });
  await close.click();
  await confirmation.getByRole('button', { name: 'Discard and close', exact: true }).click();
  await expect(row).toHaveCount(0);
  await expect
    .poll(async () => (await (await request.get('/api/workspaces')).json()).length)
    .toBe(1);
  await expect(
    page.getByRole('heading', { name: 'Synthetic multimodal study', exact: true }),
  ).toBeVisible();
  expect(await imageZoom(page)).toBeCloseTo(2);
  expect(imageRequests).toBe(0);
  const activeTitle = studies.locator('.study-item[aria-current="true"]');
  await expect(activeTitle).toBeFocused();
  await page.keyboard.press('Tab');
  const activeClose = studies.getByRole('button', {
    name: 'Close study: Synthetic multimodal study',
    exact: true,
  });
  await expect(activeClose).toBeFocused();
  await expect(activeClose).toHaveCSS('opacity', '1');
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'No study open', exact: true })).toBeVisible();
  await expect(
    page.getByRole('dialog', { name: 'Discard unsaved changes?', exact: true }),
  ).toHaveCount(0);
  await expect(studies.getByRole('button', { name: 'New study', exact: true })).toBeFocused();
});

test('open studies fill the panel, long names reflow, and Settings stays fixed while scrolling', async ({
  page,
  request,
}, info) => {
  await page.setViewportSize({ width: 1600, height: 800 });
  const token = (await (await request.get('/api/session')).json()).token;
  for (let index = 1; index <= 8; index++) {
    const created = await request.post('/api/workspaces/synthetic', {
      headers: { 'x-mrax-token': token },
    });
    const workspace = await created.json();
    const response = await request.post('/api/workspaces/' + workspace.id + '/actions', {
      headers: { 'x-mrax-token': token },
      data: {
        revision: workspace.revision,
        view_revision: workspace.view_revision,
        action: 'study.rename',
        options: {
          name: `Independent synthetic longitudinal study ${index}: repeated acquisitions and relaxation measurements`,
        },
      },
    });
    expect(response.ok()).toBe(true);
  }
  await page.goto('/');
  const studies = page.getByRole('complementary', { name: 'Data & studies', exact: true });
  await expect(studies.locator('.study-item')).toHaveCount(8);
  await expect(studies.getByRole('heading', { name: 'Data controls', exact: true })).toBeVisible();
  await expect(studies.getByRole('button', { name: 'Import data', exact: true })).toBeVisible();
  const browser = studies.locator('.study-browser-body');
  const files = (await studies
    .getByRole('toolbar', { name: 'Workspace files', exact: true })
    .boundingBox())!;
  expect((await browser.boundingBox())!.y).toBeGreaterThanOrEqual(files.y + files.height);
  const title = studies.locator('.study-item[aria-current="true"] strong');
  const narrowHeight = (await title.boundingBox())!.height;
  const footer = studies.getByRole('button', { name: 'Settings', exact: true });
  const footerTop = (await footer.boundingBox())!.y;
  const browserBox = (await browser.boundingBox())!;
  expect(footerTop - (browserBox.y + browserBox.height)).toBeLessThan(20);
  const requests: string[] = [];
  page.on('request', (request) => {
    if (request.url().includes('/api/')) requests.push(request.url());
  });
  await dragBrowser(page, 520, false);
  const divider = page.getByRole('separator', { name: 'Resize study browser', exact: true });
  await expect(divider).toHaveAttribute('aria-valuenow', '520');
  expect(await page.evaluate(() => localStorage.getItem('mrax.studyWidth'))).toBeNull();
  await page.mouse.up();
  expect(await page.evaluate(() => localStorage.getItem('mrax.studyWidth'))).toBe('520');
  expect((await title.boundingBox())!.height).toBeLessThan(narrowHeight);
  expect(await title.textContent()).toContain('relaxation measurements');
  expect(await studies.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
    true,
  );
  const main = (await page.locator('main').boundingBox())!;
  const viewer = (await page
    .getByRole('region', { name: 'Image viewer', exact: true })
    .boundingBox())!;
  expect(viewer.x + viewer.width).toBeLessThanOrEqual(main.x + main.width);
  const list = studies.getByRole('navigation', {
    name: 'Study and experiment browser',
    exact: true,
  });
  await list.hover();
  await page.mouse.wheel(0, 10000);
  await expect.poll(() => list.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);
  await expect(footer).toBeInViewport({ ratio: 1 });
  expect((await footer.boundingBox())!.y).toBeCloseTo(footerTop, 0);
  const settings = await openSettings(page);
  await settings.getByRole('button', { name: 'Dark theme', exact: true }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await settings.getByRole('button', { name: 'Done', exact: true }).click();
  await expect(footer).toBeFocused();
  expect(requests).toEqual([]);
  await page.screenshot({ path: info.outputPath('study-panel-wide-dark.png') });
  await page.reload();
  await expect(divider).toHaveAttribute('aria-valuenow', '520');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(studies.getByRole('button', { name: 'Import data', exact: true })).toBeVisible();
  await openSettings(page);
  await page.screenshot({ path: info.outputPath('settings-dark.png') });
});

test('both panel widths respect the workspace, support keyboard and cancel, and reset independently', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('/');
  await loadExample(page);
  const divider = page.getByRole('separator', { name: 'Resize study browser', exact: true });
  const inspector = page.getByRole('separator', { name: 'Resize analysis controls', exact: true });
  await divider.focus();
  await divider.press('ArrowRight');
  await expect(divider).toHaveAttribute('aria-valuenow', '264');
  await divider.press('Shift+ArrowRight');
  await expect(divider).toHaveAttribute('aria-valuenow', '328');
  await divider.press('ArrowLeft');
  await expect(divider).toHaveAttribute('aria-valuenow', '312');
  await divider.press('End');
  await expect(divider).toHaveAttribute('aria-valuenow', '720');
  await page.setViewportSize({ width: 1120, height: 800 });
  await expect(divider).toHaveAttribute('aria-valuenow', '408');
  await expect(inspector).toHaveAttribute('aria-valuenow', '288');
  expect((await page.locator('main').boundingBox())!.width).toBeGreaterThanOrEqual(408);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1120);
  await page.setViewportSize({ width: 1600, height: 1000 });
  await expect(divider).toHaveAttribute('aria-valuenow', '720');
  await page.setViewportSize({ width: 1120, height: 800 });
  const settings = await openSettings(page);
  await settings.getByRole('button', { name: 'Reset panel widths', exact: true }).click();
  await settings.getByRole('button', { name: 'Done', exact: true }).click();
  await expect(divider).toHaveAttribute('aria-valuenow', '248');
  await expect(inspector).toHaveAttribute('aria-valuenow', '360');
  expect(await page.evaluate(() => localStorage.getItem('mrax.inspectorWidth'))).toBe('360');
  await dragBrowser(page, 350, false);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(divider).toHaveAttribute('aria-valuenow', '248');
  await page.setViewportSize({ width: 1600, height: 1000 });
  await dragBrowser(page, 500);
  await divider.dblclick();
  await expect(divider).toHaveAttribute('aria-valuenow', '248');
});

test('New study keeps examples in its form and data actions stay visible under their heading', async ({
  page,
  request,
}, info) => {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Load synthetic study', exact: true })).toHaveCount(
    0,
  );
  const files = await showDataControls(page);
  const newStudy = files.getByRole('button', { name: 'New study', exact: true });
  await expect(files.getByText('Example studies', { exact: true })).toHaveCount(0);
  await newStudy.focus();
  await page.keyboard.press('ArrowDown');
  await expect(files.getByRole('button', { name: 'Open workspace', exact: true })).toBeFocused();
  const form = await openNewStudy(page);
  await expect(form.getByRole('form', { name: 'New study form', exact: true })).toBeVisible();
  await expect(form.getByRole('region', { name: 'Example studies', exact: true })).toBeVisible();
  expect(await (await request.get('/api/workspaces')).json()).toHaveLength(0);
  await page.screenshot({ path: info.outputPath('new-study-form.png') });
  await form.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(newStudy).toBeFocused();
  expect(await (await request.get('/api/workspaces')).json()).toHaveLength(0);
  await loadExample(page);
  expect(await (await request.get('/api/workspaces')).json()).toHaveLength(1);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await expect(files).toBeVisible();
  await showDataControls(page);
  const actions = files.locator(':scope > button');
  await expect(actions).toHaveText([
    'New study',
    'Open workspace',
    'Import data',
    'Export data',
    'Save workspace',
  ]);
  const imports = (await files
    .getByRole('button', { name: 'Import data', exact: true })
    .boundingBox())!;
  const exports = (await files
    .getByRole('button', { name: 'Export data', exact: true })
    .boundingBox())!;
  expect(exports.y).toBeGreaterThan(imports.y);
  const appearance = await page
    .getByRole('button', { name: 'Settings', exact: true })
    .evaluate((element) => {
      const style = getComputedStyle(element);
      return {
        background: style.backgroundColor,
        border: style.borderTopWidth,
        shadow: style.boxShadow,
      };
    });
  expect(appearance).toEqual({ background: 'rgba(0, 0, 0, 0)', border: '0px', shadow: 'none' });
  const settings = await openSettings(page);
  await expect(
    settings.getByRole('group', { name: 'Theme', exact: true }).getByRole('button'),
  ).toHaveCount(3);
  await expect(
    settings.getByRole('heading', { name: 'Workspace layout', exact: true }),
  ).toBeVisible();
  await expect(settings.getByText(/backend|connection|server/i)).toHaveCount(0);
  await page.screenshot({ path: info.outputPath('settings-light.png') });
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Settings', exact: true })).toBeFocused();
});

test('compact study drawers retain the Settings footer and become a single sidebar on resize', async ({
  page,
}) => {
  await page.setViewportSize({ width: 700, height: 600 });
  await page.goto('/');
  const studies = await showStudies(page);
  const settingsButton = studies.getByRole('button', { name: 'Settings', exact: true });
  await expect(settingsButton).toBeInViewport({ ratio: 1 });
  const settings = await openSettings(page);
  await settings.getByRole('button', { name: 'Dark theme', exact: true }).click();
  await settings.getByRole('button', { name: 'Done', exact: true }).click();
  await expect(settingsButton).toBeFocused();
  await expect(page.getByRole('dialog', { name: 'Data & studies', exact: true })).toBeVisible();
  await page.setViewportSize({ width: 1280, height: 720 });
  await expect(page.getByRole('dialog', { name: 'Data & studies', exact: true })).toHaveCount(0);
  await expect(
    page.getByRole('complementary', { name: 'Data & studies', exact: true }),
  ).toHaveCount(1);
  await expect(
    page.getByRole('separator', { name: 'Resize study browser', exact: true }),
  ).toBeVisible();
  await expect(settingsButton).toBeInViewport({ ratio: 1 });
});
