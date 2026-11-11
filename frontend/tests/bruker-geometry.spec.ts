import { test, expect } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { chooseFilePath, clickDataControl, selectWorkflow } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

test('Bruker packages retain their anatomical planes through viewing and lazy reconstruction', async ({
  page,
  request,
}, info) => {
  test.setTimeout(240000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const source = info.outputPath('Independent multi-package phantom');
  await mkdir(source, { recursive: true });
  execFileSync('python', [
    '-c',
    'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd().parent/"tests")); from test_bruker_geometry_workflows import _scan; _scan(Path(sys.argv[1]), counts=(3,2,1))',
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
    .fill('{"field_strength_mhz":100,"load_raw_kspace":false}');
  await page.getByRole('button', { name: 'Import study', exact: true }).click();
  await expect(page.getByRole('img', { name: /5 by 7 quantitative image/ })).toBeVisible({
    timeout: 60000,
  });
  await expect
    .poll(async () =>
      page.getByRole('img', { name: /5 by 7 quantitative image/ }).evaluate((element) => {
        const canvas = element as HTMLCanvasElement;
        const pixels = canvas
          .getContext('2d')!
          .getImageData(0, 0, canvas.width, canvas.height).data;
        const centre = (2 * canvas.width + 3) * 4;
        return (
          pixels[centre + 4] - pixels[centre] > pixels[centre + canvas.width * 4] - pixels[centre]
        );
      }),
    )
    .toBe(true);
  const selected = async () => (await (await request.get('/api/workspaces')).json())[0];
  const imported = await selected();
  expect(imported.experiments).toEqual(['1']);
  expect(imported.dimensions.slice_package).toBe(3);
  const packages = page.getByRole('combobox', { name: 'Slice package', exact: true });
  await expect(packages).toBeVisible();
  await expect(packages.getByRole('option')).toHaveText([
    'Package 1 · Axial · 3 slices',
    'Package 2 · Coronal · 2 slices',
    'Package 3 · Sagittal · 1 slice',
  ]);
  const controls = page.locator('details.viewer-controls');
  const expand = async () => {
    if ((await controls.getAttribute('open')) === null) await controls.locator('summary').click();
  };
  await expand();
  await page.getByLabel('Plane', { exact: true }).selectOption('coronal');
  await expect(page.getByRole('img', { name: /3 by 7 quantitative image/ })).toBeVisible();
  await page.getByLabel('Plane', { exact: true }).selectOption('sagittal');
  await expect(page.getByRole('img', { name: /3 by 5 quantitative image/ })).toBeVisible();
  await page.getByLabel('Plane', { exact: true }).selectOption('axial');
  for (const [index, plane] of ['axial', 'coronal', 'sagittal'].entries()) {
    await packages.selectOption(String(index));
    await expand();
    await expect(page.getByLabel('Plane', { exact: true })).toHaveValue(plane);
    await expect(page.locator('.orientation.top')).toHaveText(plane === 'sagittal' ? 'y' : 'x');
    await expect(page.locator('.orientation.left')).toHaveText(plane === 'axial' ? 'y' : 'z');
    await expect(page.locator('.orientation.top')).toHaveAttribute('aria-label', /increases left;/);
    await expect(page.locator('.orientation.left')).toHaveAttribute(
      'aria-label',
      plane === 'axial' ? /increases up;/ : /increases down;/,
    );
    await expect(page.getByRole('img', { name: /5 by 7 quantitative image/ })).toBeVisible();
    await page.screenshot({ path: info.outputPath(`axes-${plane}.png`) });
    if (plane === 'sagittal') {
      const selector = page.getByLabel('Plane', { exact: true });
      await expect(
        selector.getByRole('option', { name: 'sagittal', exact: true }),
      ).toHaveJSProperty('disabled', false);
      await expect(selector.getByRole('option', { name: 'axial', exact: true })).toHaveJSProperty(
        'disabled',
        true,
      );
      await expect(selector.getByRole('option', { name: 'coronal', exact: true })).toHaveJSProperty(
        'disabled',
        true,
      );
    }
    await page.getByRole('tab', { name: 'Reconstruct', exact: true }).click();
    if (index === 0) {
      await page
        .getByRole('combobox', { name: 'Reconstruction preset', exact: true })
        .selectOption('cartesian-fourier');
      await page.getByRole('button', { name: 'Apply reconstruction', exact: true }).click();
      await expect
        .poll(async () => (await selected()).has_reconstruction, { timeout: 60000 })
        .toBe(true);
    }
    const compare = page.getByRole('group', { name: 'Reconstruction display', exact: true });
    await compare.getByRole('button', { name: 'Vendor comparison', exact: true }).click();
    await expect.poll(async () => (await selected()).scene.comparison.enabled).toBe(true);
    await expect(page.locator('.comparison-view .orientation.top')).toHaveCount(1);
    await expect(page.locator('.comparison-view .orientation.top')).toBeInViewport();
    if (index === 0) await page.screenshot({ path: info.outputPath('comparison-axes.png') });
    expect((await selected()).plane).toBe(plane);
    await compare.getByRole('button', { name: 'Experiment', exact: true }).click();
  }
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await page.getByRole('button', { name: 'Draw rectangle', exact: true }).click();
  const surface = page.getByTestId('roi-surface');
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.25, box.y + box.height * 0.25);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.7, { steps: 6 });
  await page.mouse.up();
  await expect.poll(async () => (await selected()).rois.length).toBe(1);
  expect((await selected()).rois[0].pixels).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});
