import { loadExample } from './ui';
import { selectWorkflow } from './ui';
import { test, expect, type Locator, type Page } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await loadExample(page);
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await selectWorkflow(page, 'Segment');
});

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete(`/api/workspaces/${workspace.id}`, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

async function point(slider: Locator, fraction: number) {
  const bounds = (await slider.boundingBox())!;
  return { x: bounds.x + 8 + (bounds.width - 16) * fraction, y: bounds.y + bounds.height / 2 };
}

async function move(page: Page, slider: Locator, fraction: number) {
  const position = await point(slider, fraction);
  await page.mouse.move(position.x, position.y, { steps: 8 });
}

test('frame and slice sliders keep one drag across slow image updates', async ({ page }) => {
  const failures: string[] = [];
  page.on('pageerror', (error) => failures.push(error.message));
  await page.route('**/api/workspaces/*/navigate', async (route) => {
    if (route.request().postDataJSON().commit)
      await new Promise((resolve) => setTimeout(resolve, 350));
    await route.continue();
  });
  const image = page.getByTestId('roi-surface');
  const originalImage = await image.elementHandle();
  for (const [name, middle, maximum, caption] of [
    ['Echo', 3, 7, /frame 8, slice 1/],
    ['Slice', 1, 2, /frame 8, slice 3/],
  ] as const) {
    const slider = page.getByLabel(name, { exact: true });
    const originalSlider = await slider.elementHandle();
    const start = await point(slider, 0);
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    try {
      await move(page, slider, middle / maximum);
      await expect(slider).toBeEnabled();
      await expect(slider).toHaveValue(String(middle));
      await expect(
        page.getByRole('img', {
          name: name === 'Echo' ? /frame 4, slice 1/ : /frame 8, slice 2/,
        }),
      ).toBeVisible();
      expect(await originalSlider!.evaluate((element) => element.isConnected)).toBe(true);
      expect(await originalImage!.evaluate((element) => element.isConnected)).toBe(true);
      await move(page, slider, 1);
      await expect(slider).toHaveValue(String(maximum));
      await expect(page.getByRole('img', { name: caption })).toBeVisible();
    } finally {
      await page.mouse.up();
    }
  }
  expect(failures).toEqual([]);
});

test('rapid keyboard navigation keeps the latest frame and slice without parallel updates', async ({
  page,
  request,
}) => {
  let active = 0,
    maximumActive = 0,
    updates = 0;
  let started!: () => void;
  const firstRequest = new Promise<void>((resolve) => {
    started = resolve;
  });
  await page.route('**/api/workspaces/*/navigate', async (route) => {
    if (!route.request().postDataJSON().commit) {
      await route.continue();
      return;
    }
    active++;
    updates++;
    maximumActive = Math.max(maximumActive, active);
    started();
    await new Promise((resolve) => setTimeout(resolve, 200));
    const response = await route.fetch();
    await route.fulfill({ response });
    active--;
  });
  const before = (await (await request.get('/api/workspaces')).json())[0];
  const frame = page.getByLabel('Echo', { exact: true });
  const slice = page.getByLabel('Slice', { exact: true });
  await frame.press('End');
  await firstRequest;
  await expect(frame).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Draw rectangle', exact: true })).toBeDisabled();
  await slice.press('End');
  await frame.press('Home');
  for (let i = 0; i < 5; i++) await frame.press('ArrowRight');
  await expect(frame).toHaveValue('5');
  await expect(slice).toHaveValue('2');
  await expect(page.getByRole('region', { name: 'Image viewer', exact: true })).toHaveAttribute(
    'aria-busy',
    'false',
  );
  await expect(page.getByRole('img', { name: /frame 6, slice 3/ })).toBeVisible();
  const after = (await (await request.get('/api/workspaces')).json())[0];
  expect(after.frame).toBe(5);
  expect(after.z).toBe(2);
  expect(after.revision).toBe(before.revision);
  expect(maximumActive).toBe(1);
  expect(updates).toBeLessThan(8);
});

test('intensity bound tracks a continuous drag without backend requests', async ({ page }) => {
  let viewRequests = 0;
  page.on('request', (request) => {
    if (request.method() === 'PATCH' && request.url().endsWith('/view')) viewRequests++;
  });
  await selectWorkflow(page, 'View');
  {
    const slider = page.getByLabel('Lower display bound', { exact: true });
    await slider.scrollIntoViewIfNeeded();
    const element = await slider.elementHandle();
    const min = Number(await slider.getAttribute('min'));
    const max = Number(await slider.getAttribute('max'));
    const start = await point(slider, (Number(await slider.inputValue()) - min) / (max - min));
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    try {
      await move(page, slider, 0.5);
      expect(Number(await slider.inputValue())).toBeGreaterThan(min + (max - min) * 0.45);
      await move(page, slider, 0.8);
      expect(Number(await slider.inputValue())).toBeGreaterThan(min + (max - min) * 0.75);
      expect(await element!.evaluate((input) => input.isConnected)).toBe(true);
    } finally {
      await page.mouse.up();
    }
  }
  expect(viewRequests).toBe(0);
});

test('a failed navigation keeps the displayed image and allows another slider change', async ({
  page,
}) => {
  let fail = true;
  await page.route('**/api/workspaces/*/navigate', async (route) => {
    if (fail && route.request().postDataJSON().commit) {
      fail = false;
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'View update unavailable' }),
      });
    } else await route.continue();
  });
  const slider = page.getByLabel('Echo', { exact: true });
  const originalImage = await page.getByTestId('roi-surface').elementHandle();
  await slider.press('End');
  await expect(page.getByRole('alert')).toContainText('View update unavailable');
  await expect(slider).toBeEnabled();
  await expect(slider).toHaveValue('0');
  expect(await originalImage!.evaluate((image) => image.isConnected)).toBe(true);
  await expect(page.getByRole('img', { name: /frame 1, slice 1/ })).toBeVisible();
  await slider.press('End');
  await expect(slider).toHaveValue('7');
  await expect(page.getByRole('img', { name: /frame 8, slice 1/ })).toBeVisible();
  await expect(page.getByRole('alert')).toBeHidden();
});
