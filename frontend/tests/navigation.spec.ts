import { zoomImage, imageZoom } from './ui';
import { loadExample } from './ui';
import { openRoiCoordinates } from './ui';
import { expect, test } from '@playwright/test';
import { selectTheme, selectWorkflow } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  const workspaces = await (await request.get('/api/workspaces')).json();
  for (const workspace of workspaces) {
    const response = await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
    expect(response.status()).toBe(204);
  }
});

test('workflow menus expose distinct tasks with keyboard navigation and predictable dismissal', async ({
  page,
}, info) => {
  await page.goto('/');
  await loadExample(page);
  const analyze = page.getByRole('tab', { name: 'Analyze', exact: true });
  const menu = page.getByRole('menu', { name: 'Analyze tasks', exact: true });
  const timecourse = menu.getByRole('menuitemradio', { name: 'ROI timecourses', exact: true });
  const series = menu.getByRole('menuitemradio', { name: 'Time-series corrections', exact: true });
  const statistics = menu.getByRole('menuitemradio', { name: 'ROI map statistics', exact: true });
  const maps = menu.getByRole('menuitemradio', { name: 'Derived maps', exact: true });

  for (const name of ['View', 'Preprocess', 'Register', 'Segment', 'Analyze'])
    await expect(page.getByRole('tab', { name, exact: true })).toHaveAttribute(
      'aria-haspopup',
      'menu',
    );
  for (const name of ['Reconstruct', 'Fit'])
    await expect(page.getByRole('tab', { name, exact: true })).not.toHaveAttribute('aria-haspopup');

  await analyze.click();
  await expect(analyze).toHaveAttribute('aria-expanded', 'true');
  await expect(timecourse).toBeFocused();
  await expect(page.getByRole('tab', { name: 'View', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.keyboard.press('ArrowDown');
  await expect(series).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(statistics).toBeFocused();
  await page.keyboard.press('End');
  await expect(maps).toBeFocused();
  await page.keyboard.press('Home');
  await expect(timecourse).toBeFocused();
  await page.keyboard.press('d');
  await expect(maps).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(menu).not.toBeVisible();
  await expect(analyze).toBeFocused();
  await expect(analyze).toHaveAttribute('aria-expanded', 'false');

  await page.keyboard.press('ArrowUp');
  await expect(maps).toBeFocused();
  await page.keyboard.press('ArrowUp');
  await page.keyboard.press('Enter');
  await expect(
    page.getByRole('heading', { name: 'ROI map statistics', level: 2, exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel('Timecourse input', { exact: true })).not.toBeVisible();
  await expect(page.getByLabel('Denoiser', { exact: true })).not.toBeVisible();
  await analyze.click();
  await expect(statistics).toHaveAttribute('aria-checked', 'true');
  await page.keyboard.press('Escape');
  await selectWorkflow(page, 'Fit');
  await analyze.click();
  await expect(menu.locator('[aria-checked="true"]')).toHaveCount(0);
  await expect(page.locator('.workflow-check')).toHaveCount(0);
  await expect(statistics).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(menu).not.toBeVisible();
  await expect(page.getByRole('button', { name: 'New study', exact: true })).toBeFocused();
  await expect(page.getByRole('tab', { name: 'Fit', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );

  await analyze.click();
  await page.getByRole('heading', { level: 1 }).click();
  await expect(menu).not.toBeVisible();
  for (const theme of ['light', 'dark'] as const) {
    await selectTheme(page, theme);
    await analyze.click();
    const bounds = (await menu.boundingBox())!;
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(page.viewportSize()!.width);
    await page.screenshot({ path: info.outputPath(`workflow-menu-${theme}.png`) });
    await page.keyboard.press('Escape');
  }
});

test('submenus preserve display, overlay, ROI, analysis and processing drafts across workflows', async ({
  page,
}) => {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add rectangle by coordinates', exact: true }).click();
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await page.getByLabel('Show ROI 1', { exact: true }).uncheck();
  await selectWorkflow(page, 'View', 'Image display');
  await zoomImage(page, 2);
  const initialZoom = await imageZoom(page);
  await selectWorkflow(page, 'Preprocess', 'Signal normalization');
  await page.getByLabel('Normalization method', { exact: true }).selectOption('dilution');
  await page.getByLabel('Dilution smoothing (samples)', { exact: true }).fill('2');
  await selectWorkflow(page, 'Analyze', 'Time-series corrections');
  await page.getByLabel('Time-series correction', { exact: true }).selectOption('despike');
  await page.getByLabel('Spike ratio threshold', { exact: true }).fill('7');
  await selectWorkflow(page, 'Register', 'Image alignment');
  await page.getByLabel('Reference frame', { exact: true }).fill('2');
  await selectWorkflow(page, 'Register', 'Drift correction');
  await expect(
    page.getByRole('button', { name: 'Run drift correction', exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel('Reference experiment', { exact: true })).not.toBeVisible();
  await selectWorkflow(page, 'Register', 'Image alignment');
  await expect(page.getByLabel('Reference frame', { exact: true })).toHaveValue('2');
  await selectWorkflow(page, 'Segment', 'Import masks and ROI sets');
  await expect(
    page.getByRole('button', { name: 'Open ROI set or mask…', exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel('ROI tool', { exact: true })).not.toBeVisible();
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await expect(page.locator('.roi-card.selected')).toContainText('ROI 1');
  await selectWorkflow(page, 'Fit');
  await page.getByLabel('Fit steps', { exact: true }).fill('777');
  await selectWorkflow(page, 'Analyze', 'ROI timecourses');
  await page.getByLabel('Timecourse statistic', { exact: true }).selectOption('median');
  await selectWorkflow(page, 'Analyze', 'ROI map statistics');
  await expect(page.getByLabel('Statistics map', { exact: true })).toBeVisible();
  await expect(page.getByLabel('Timecourse statistic', { exact: true })).not.toBeVisible();
  await selectWorkflow(page, 'Analyze', 'Derived maps');
  await expect(page.getByLabel('Map calculation', { exact: true })).toHaveValue('dti');
  await expect(page.getByLabel('Statistics map', { exact: true })).not.toBeVisible();
  await selectWorkflow(page, 'Analyze', 'ROI timecourses');
  await expect(page.getByLabel('Timecourse statistic', { exact: true })).toHaveValue('median');
  await selectWorkflow(page, 'Analyze', 'Time-series corrections');
  await expect(page.getByLabel('Time-series correction', { exact: true })).toHaveValue('despike');
  await expect(page.getByLabel('Spike ratio threshold', { exact: true })).toHaveValue('7');
  await selectWorkflow(page, 'Preprocess', 'Signal normalization');
  await expect(page.getByLabel('Normalization method', { exact: true })).toHaveValue('dilution');
  await expect(page.getByLabel('Dilution smoothing (samples)', { exact: true })).toHaveValue('2');
  await selectWorkflow(page, 'View', 'Image and ROI overlays');
  await expect(page.getByLabel('Show ROI 1', { exact: true })).not.toBeChecked();
  await selectWorkflow(page, 'View', 'Image display');
  await expect.poll(() => imageZoom(page)).toBe(initialZoom);
  await selectWorkflow(page, 'Fit');
  await expect(page.getByLabel('Fit steps', { exact: true })).toHaveValue('777');
  await expect(page.getByLabel('Fit mask ROI 1', { exact: true })).toBeChecked();
});

test('viewing and scientific actions have distinct menus without starting processing', async ({
  page,
}) => {
  await page.goto('/');
  await loadExample(page);
  const jobs: string[] = [];
  page.on('request', (request) => {
    if (request.url().endsWith('/tasks')) jobs.push(request.postDataJSON().operation);
  });
  const menus = {
    View: ['Image display', 'Image and ROI overlays', 'Signal curves', 'Metadata'],
    Preprocess: ['Denoising', 'Signal normalization'],
    Register: ['Image alignment', 'Drift correction'],
    Segment: [
      'Create and edit ROIs',
      'Edit label masks',
      '3D expansion',
      'Combine ROIs',
      'Import masks and ROI sets',
    ],
    Analyze: ['ROI timecourses', 'Time-series corrections', 'ROI map statistics', 'Derived maps'],
  };
  for (const [name, tasks] of Object.entries(menus)) {
    await page.getByRole('tab', { name, exact: true }).click();
    const menu = page.getByRole('menu', { name: name + ' tasks', exact: true });
    await expect(menu.getByRole('menuitemradio')).toHaveText(tasks);
    await expect(menu.locator('[aria-checked="true"]')).toHaveCount(name === 'View' ? 1 : 0);
    await expect(menu.locator('.app-symbol')).toHaveCount(tasks.length);
    for (const icon of await menu.locator('.app-symbol').all()) {
      await expect(icon).toBeVisible();
      await expect(icon).toHaveAttribute('aria-hidden', 'true');
    }
    await page.keyboard.press('Escape');
  }
  await selectWorkflow(page, 'Analyze', 'Time-series corrections');
  const correction = page.getByLabel('Time-series correction', { exact: true });
  await expect(correction.locator('option')).toHaveText([
    'Baseline correction',
    'Temporal Gaussian smoothing',
    'Isolated-spike correction',
  ]);
  await selectWorkflow(page, 'Preprocess', 'Signal normalization');
  const normalization = page.getByLabel('Normalization method', { exact: true });
  await expect(normalization.locator('option')).toHaveText([
    'Reference normalization / Z spectra',
    'Dilution correction',
  ]);
  await selectWorkflow(page, 'Reconstruct');
  await expect(page.getByRole('tab', { name: 'Reconstruct', exact: true })).not.toHaveAttribute(
    'aria-haspopup',
  );
  await expect(
    page.getByRole('button', { name: 'Apply reconstruction', exact: true }),
  ).toBeDisabled();
  await selectWorkflow(page, 'Register', 'Image alignment');
  await expect(page.getByRole('button', { name: 'Run registration', exact: true })).toBeVisible();
  await selectWorkflow(page, 'View', 'Image display');
  await expect(correction).not.toBeVisible();
  await expect(
    page.getByRole('button', { name: 'Run registration', exact: true }),
  ).not.toBeVisible();
  await expect(
    page.getByRole('button', { name: 'Apply reconstruction', exact: true }),
  ).not.toBeVisible();
  await page.getByLabel('Plane', { exact: true }).selectOption('coronal');
  await expect(page.getByLabel('Plane', { exact: true })).toHaveValue('coronal');
  expect(jobs).toEqual([]);
});

test('compact windows choose a submenu before opening the controls drawer', async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 760 });
  await page.goto('/');
  await loadExample(page);
  const analyze = page.getByRole('tab', { name: 'Analyze', exact: true });
  const menu = page.getByRole('menu', { name: 'Analyze tasks', exact: true });
  const drawer = page.getByRole('dialog', { name: 'Analysis controls', exact: true });
  await analyze.click();
  await expect(menu).toBeVisible();
  await expect(drawer).not.toBeVisible();
  await page.keyboard.press('Escape');
  await expect(analyze).toBeFocused();
  await selectWorkflow(page, 'Analyze', 'ROI timecourses');
  await expect(drawer).toBeVisible();
  await expect(drawer.getByRole('heading', { name: 'ROI timecourses', exact: true })).toBeVisible();
  await drawer.getByRole('button', { name: 'Close panel', exact: true }).click();
  await analyze.click();
  await expect(
    menu.getByRole('menuitemradio', { name: 'ROI timecourses', exact: true }),
  ).toHaveAttribute('aria-checked', 'true');
  await page.keyboard.press('Escape');
});
