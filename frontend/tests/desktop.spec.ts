import { zoomImage } from './ui';
import { clickDataControl, loadExample, openSettings } from './ui';
import { openRoiCoordinates } from './ui';
import { test, expect } from '@playwright/test';
import {
  selectWorkflow,
  selectTheme,
  openExport,
  closeExport,
  openMetadataBrowser,
  closeMetadataBrowser,
} from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

for (const viewport of [
  { width: 1280, height: 720 },
  { width: 1440, height: 900 },
  { width: 1920, height: 1080 },
]) {
  test(`desktop ${viewport.width}×${viewport.height}: collapsed view controls keep image, signal and tools in view`, async ({
    page,
  }, testInfo) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.setViewportSize(viewport);
    await page.goto('/');
    await loadExample(page);
    await page.locator('.viewer-controls > summary').click();
    await expect(page.getByLabel('Lower display bound', { exact: true })).toBeHidden();
    const image = page.getByTestId('roi-surface');
    await expect(image).toBeVisible();
    for (const tab of await page
      .getByRole('tablist', { name: 'Workflow sections' })
      .getByRole('tab')
      .all()) {
      await expect(tab).toBeInViewport({ ratio: 1 });
    }
    for (const theme of ['light', 'dark'] as const) {
      await selectTheme(page, theme);
      for (const name of ['Image viewer', 'ROI signal', 'Data & Studies', 'Analysis controls']) {
        const region = page.getByRole(
          name === 'Data & Studies' || name === 'Analysis controls' ? 'complementary' : 'region',
          { name, exact: true },
        );
        const box = await region.boundingBox();
        expect(box, name).not.toBeNull();
        expect(box!.y, name).toBeGreaterThanOrEqual(0);
        expect(box!.y + box!.height, name).toBeLessThanOrEqual(viewport.height + 1);
      }
      const box = (await image.boundingBox())!;
      expect(box.height).toBeGreaterThan(100);
      expect(box.width / box.height).toBeCloseTo((32 * 0.9) / (24 * 0.7), 2);
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
        viewport.width,
      );
      expect(await page.evaluate(() => window.scrollY)).toBe(0);
    }
    await selectWorkflow(page, 'Segment');
    await page.getByRole('button', { name: 'Draw rectangle', exact: true }).click();
    const box = (await image.boundingBox())!;
    await page.mouse.move(box.x + (box.width * 8.25) / 32, box.y + (box.height * 6.25) / 24);
    await page.mouse.down();
    await page.mouse.move(box.x + (box.width * 15.75) / 32, box.y + (box.height * 17.75) / 24, {
      steps: 6,
    });
    await page.mouse.up();
    await expect(page.getByText('96 voxels', { exact: true })).toBeVisible();
    await expect(
      page.getByRole('img', { name: 'Observed ROI signal across echoes' }),
    ).toBeInViewport();
    await selectWorkflow(page, 'View');
    await zoomImage(page, 2);
    await page.locator('.image-viewport').focus();
    await page.keyboard.press('0');
    await selectWorkflow(page, 'Segment');
    await expect(page.getByText('96 voxels', { exact: true })).toBeVisible();
    for (const theme of ['light', 'dark'] as const) {
      await selectTheme(page, theme);
      await page.screenshot({ path: testInfo.outputPath(`desktop-${theme}.png`) });
    }
    expect(errors).toEqual([]);
  });
}

test('Settings defaults to the system appearance and preserves explicit overrides', async ({
  page,
}) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/');
  await openSettings(page);
  const themes = page.getByRole('group', { name: 'Theme', exact: true });
  await expect(themes.getByRole('button')).toHaveCount(3);
  await expect(themes.getByRole('button')).toHaveText(['System', 'Light', 'Dark']);
  await expect(themes.getByRole('button', { name: 'System theme', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.emulateMedia({ colorScheme: 'light' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await selectTheme(page, 'dark');
  await page.reload();
  await openSettings(page);
  await expect(themes.getByRole('button', { name: 'Dark theme', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.emulateMedia({ colorScheme: 'light' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await selectTheme(page, 'light');
  await page.reload();
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await openSettings(page);
  await themes.getByRole('button', { name: 'Dark theme', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(themes.getByRole('button', { name: 'Dark theme', exact: true })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});

test('one compact header keeps workflow navigation reachable in narrower desktop windows', async ({
  page,
}) => {
  await page.goto('/');
  await page.evaluate(() => {
    document.documentElement.dataset.desktop = 'true';
    document.documentElement.style.setProperty('--window-controls-width', '138px');
  });
  for (const width of [760, 1120, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    const header = page.getByRole('banner', { name: 'MRax', exact: true });
    const bounds = await header.boundingBox();
    expect(bounds!.height).toBeLessThanOrEqual(50);
    const navigation = header.getByRole('tablist', { name: 'Workflow sections' });
    const analyze = navigation.getByRole('tab', { name: 'Analyze', exact: true });
    await analyze.focus();
    await expect(analyze).toBeFocused();
    await expect(analyze).toBeInViewport();
    const analyzeBounds = await analyze.boundingBox();
    expect(analyzeBounds!.x + analyzeBounds!.width).toBeLessThanOrEqual(width - 138);
    await navigation.getByRole('tab', { name: 'View', exact: true }).focus();
    await expect(navigation.getByRole('tab', { name: 'View', exact: true })).toBeFocused();
  }
});

test('workflow navigation exposes scientific tasks directly and preserves drafts', async ({
  page,
}) => {
  await page.goto('/');
  const header = page.getByRole('banner', { name: 'MRax', exact: true });
  await expect(header).toBeVisible();
  await expect(header.getByRole('img')).toHaveCount(0);
  const headerBounds = await header.boundingBox();
  expect(headerBounds!.height).toBeLessThanOrEqual(50);
  const studyBounds = await page
    .getByRole('complementary', { name: 'Data & Studies', exact: true })
    .boundingBox();
  const mainBounds = await page.locator('main').boundingBox();
  expect(studyBounds!.y).toBe(headerBounds!.height);
  expect(headerBounds!.x).toBe(0);
  expect((await page.locator('.header-workflows').boundingBox())!.x).toBe(mainBounds!.x);
  expect(
    (await header.getByRole('heading', { name: 'Data & Studies', exact: true }).boundingBox())!.y,
  ).toBeLessThan(headerBounds!.height);
  await loadExample(page);
  const navigation = page.getByRole('tablist', { name: 'Workflow sections' });
  await expect(navigation.getByRole('tab')).toHaveText([
    'View',
    'Reconstruct',
    'Preprocess',
    'Register',
    'Segment',
    'Fit',
    'Analyze',
  ]);
  await expect(page.locator('body')).not.toContainText(
    /Local by design|From selection to|Inspect, define|A clearer view/,
  );

  await selectWorkflow(page, 'Segment');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'Fit');
  await expect(page.getByRole('button', { name: 'Run preview fit', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run full map fit', exact: true })).toBeVisible();
  await expect(page.getByLabel('Fit ROI', { exact: true })).toHaveValue('ROI 1');
  await page.getByLabel('Fit steps', { exact: true }).fill('777');
  await selectWorkflow(page, 'Segment', 'Import masks and ROI sets');
  await expect(
    page.getByRole('button', { name: 'Open ROI set or mask…', exact: true }),
  ).toBeVisible();
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Fit steps', { exact: true })).toHaveValue('777');
  await page.getByRole('tab', { name: 'Fit', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Analyze', exact: true })).toBeFocused();
  await expect(page.getByRole('heading', { name: 'ROI timecourses', exact: true })).toBeVisible();
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Register', exact: true })).toBeFocused();
  await expect(page.getByLabel('Reference experiment', { exact: true })).toBeVisible();
  await page.keyboard.press('End');
  await expect(page.getByRole('tab', { name: 'Analyze', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.keyboard.press('Home');
  await expect(page.getByRole('tab', { name: 'View', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await expect(page.getByRole('tab', { name: 'Export', exact: true })).toHaveCount(0);
  const exportDialog = await openExport(page);
  await expect(exportDialog.getByLabel('Export object', { exact: true })).toBeVisible();
  await closeExport(page);
  await openMetadataBrowser(page);
  await expect(page.getByLabel('Active experiment name', { exact: true })).toBeVisible();
  await closeMetadataBrowser(page);
  await clickDataControl(page, 'Import data');
  await expect(page.getByRole('dialog', { name: 'Import a study or experiment' })).toBeVisible();
  await page.getByRole('button', { name: 'Close importer', exact: true }).click();

  const fonts = await page
    .locator('body,button,input,select,textarea,h1,h2,h3,svg text,pre')
    .evaluateAll((elements) => [
      ...new Set(elements.map((element) => getComputedStyle(element).fontFamily)),
    ]);
  expect(fonts).toEqual(['Arial, sans-serif']);
});
