import { loadExample } from './ui';
import { test, expect, type Page } from '@playwright/test';
import { openRoiCoordinates, selectWorkflow, selectTheme } from './ui';

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

async function study(page: Page) {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
}

async function resize(page: Page, width: number) {
  const divider = page.getByRole('separator', { name: 'Resize analysis controls', exact: true });
  const box = (await divider.boundingBox())!;
  const current = Number(await divider.getAttribute('aria-valuenow'));
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + current - width, box.y + 80, { steps: 12 });
  await page.mouse.up();
  await expect(divider).toHaveAttribute('aria-valuenow', String(width));
}

test('divider reflows processing groups, preserves drafts and saves width without data requests', async ({
  page,
}, info) => {
  await study(page);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  const controls = page.getByRole('complementary', { name: 'Analysis controls', exact: true });
  await page.getByLabel('Sigma (voxels)', { exact: true }).fill('1.6');
  const operation = controls.getByRole('region', { name: 'Denoiser and input', exact: true });
  const settings = controls.getByRole('region', { name: 'Settings', exact: true });
  const narrow = {
    operation: (await operation.boundingBox())!,
    settings: (await settings.boundingBox())!,
  };
  expect(narrow.settings.y).toBeGreaterThan(narrow.operation.y);
  const requests: string[] = [];
  page.on('request', (request) => {
    if (request.method() !== 'GET') requests.push(request.url());
  });
  await resize(page, 680);
  const wide = {
    operation: (await operation.boundingBox())!,
    settings: (await settings.boundingBox())!,
  };
  expect(wide.settings.y).toBeCloseTo(wide.operation.y, 0);
  expect(wide.settings.x).toBeGreaterThan(wide.operation.x);
  await expect(page.getByLabel('Sigma (voxels)', { exact: true })).toHaveValue('1.6');
  const divider = page.getByRole('separator', { name: 'Resize analysis controls', exact: true });
  const handle = (await divider.boundingBox())!;
  await page.mouse.move(handle.x + handle.width / 2, handle.y + 80);
  await page.mouse.down();
  await page.mouse.move(handle.x + 180, handle.y + 80, { steps: 5 });
  await page.keyboard.press('Escape');
  await page.mouse.up();
  await expect(divider).toHaveAttribute('aria-valuenow', '680');
  await expect(
    controls.getByRole('button', { name: 'Apply denoising', exact: true }),
  ).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1600);
  expect(requests).toEqual([]);
  await selectTheme(page, 'light');
  await page.screenshot({ path: info.outputPath('denoise-wide-light.png') });
  await selectTheme(page, 'dark');
  await page.screenshot({ path: info.outputPath('denoise-wide-dark.png') });
  await resize(page, 360);
  await expect(page.getByLabel('Sigma (voxels)', { exact: true })).toHaveValue('1.6');
  expect((await settings.boundingBox())!.y).toBeGreaterThan((await operation.boundingBox())!.y);
  expect(await controls.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
    true,
  );
  await selectTheme(page, 'light');
  await page.screenshot({ path: info.outputPath('denoise-narrow.png') });
  await resize(page, 680);
  await page.reload();
  await expect(
    page.getByRole('separator', { name: 'Resize analysis controls', exact: true }),
  ).toHaveAttribute('aria-valuenow', '680');
});

test('divider supports keyboard adjustment, viewport limits and resetting', async ({ page }) => {
  await study(page);
  const divider = page.getByRole('separator', { name: 'Resize analysis controls', exact: true });
  await divider.focus();
  await divider.press('ArrowLeft');
  await expect(divider).toHaveAttribute('aria-valuenow', '376');
  await divider.press('Shift+ArrowLeft');
  await expect(divider).toHaveAttribute('aria-valuenow', '440');
  await divider.press('ArrowRight');
  await expect(divider).toHaveAttribute('aria-valuenow', '424');
  await divider.press('End');
  await expect(divider).toHaveAttribute('aria-valuenow', '900');
  await page.setViewportSize({ width: 1200, height: 800 });
  await expect(divider).toHaveAttribute('aria-valuenow', '528');
  const image = (await page
    .getByRole('region', { name: 'Image viewer', exact: true })
    .boundingBox())!;
  expect(image.width).toBeGreaterThan(350);
  await page.setViewportSize({ width: 1600, height: 1000 });
  await expect(divider).toHaveAttribute('aria-valuenow', '900');
  await divider.dblclick();
  await expect(divider).toHaveAttribute('aria-valuenow', '360');
  await page.setViewportSize({ width: 1000, height: 800 });
  await expect(divider).toHaveCount(0);
  await page.getByRole('button', { name: 'Controls', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Analysis controls', exact: true })).toBeVisible();
});

test('each task exposes its own groups and keeps advanced controls accessible', async ({
  page,
}, info) => {
  await study(page);
  await resize(page, 680);
  const controls = page.getByRole('complementary', { name: 'Analysis controls', exact: true });
  const tasks = [
    ['View', 'Image display', ['Presentation', 'Appearance']],
    ['Reconstruct', undefined, ['Reconstruction pipeline', 'Output']],
    ['Preprocess', 'Denoising', ['Denoiser and input', 'Settings', 'Region']],
    ['Register', 'Image alignment', ['Reference image', 'Moving image', 'Alignment']],
    ['Register', 'Drift correction', ['Motion model', 'Reference and solver']],
    ['Segment', 'Create and edit ROIs', ['Draw', 'ROIs']],
    ['Fit', undefined, ['Model and solver', 'Preview', 'Map fitting region']],
    ['Analyze', 'ROI timecourses', ['Signal and statistic', 'ROIs and frames']],
    ['Analyze', 'ROI map statistics', ['Map selection', 'Statistics']],
  ] as const;
  for (const [workflow, task, groups] of tasks) {
    await selectWorkflow(page, workflow, task);
    for (const name of groups)
      await expect(controls.getByRole('region', { name, exact: true })).toBeVisible();
    if (workflow === 'Segment' || (workflow === 'Register' && task === 'Image alignment'))
      await page.screenshot({
        path: info.outputPath(workflow === 'Segment' ? 'segment-wide.png' : 'register-wide.png'),
      });
  }
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  const coordinates = page.getByRole('form', { name: 'ROI coordinate editor', exact: true });
  await expect(coordinates).not.toBeVisible();
  const ellipse = controls.getByRole('button', { name: 'Ellipse tool', exact: true });
  await ellipse.focus();
  await ellipse.press('Space');
  await expect(ellipse).toHaveAttribute('aria-pressed', 'true');
  await openRoiCoordinates(page);
  await page.getByRole('button', { name: 'Add ellipse by coordinates', exact: true }).click();
  await expect(controls.locator('.roi-card')).toHaveCount(1);
  await selectWorkflow(page, 'Fit');
  await selectTheme(page, 'light');
  await page.screenshot({ path: info.outputPath('fit-wide.png') });
});
