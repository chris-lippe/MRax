import { test, expect, type APIRequestContext, type Page } from '@playwright/test';
import { loadExample, selectWorkflow } from './ui';
import {
  decodeNavigation,
  NavigationCache,
  navigationKey,
  type NavigationFrame,
} from '../src/navigation';
import type { Workspace } from '../src/api';

const opened = async (request: APIRequestContext): Promise<Workspace[]> =>
  (await request.get('/api/workspaces')).json();
test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await opened(request))
    await request.delete('/api/workspaces/' + w.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
});

async function warmFrame(page: Page) {
  const warmed = page.waitForResponse(
    (response) =>
      response.url().endsWith('/navigate') &&
      response.ok() &&
      response.request().postDataJSON().commit === false &&
      response.request().postDataJSON().indices.frame === 1,
  );
  return async () => {
    await (await warmed).finished();
    await page.evaluate(
      () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
    );
  };
}

test('cached neighbours paint while a slow navigation confirmation is still pending', async ({
  page,
  request,
}, info) => {
  await page.goto('/');
  const warm = await warmFrame(page);
  await loadExample(page);
  await warm();
  const upper = page.getByLabel('Maximum display value', { exact: true });
  await upper.fill(String(Number(await upper.inputValue()) * 1.1));
  await upper.press('Enter');
  let confirmed = false;
  await page.route('**/navigate', async (route) => {
    if (route.request().postDataJSON().commit) {
      expect(route.request().postDataJSON().include_images).toBe(false);
      await new Promise((resolve) => setTimeout(resolve, 450));
      confirmed = true;
    }
    await route.continue();
  });
  const image = page.getByRole('img', { name: /24 by 32 quantitative image/ });
  const before = await image.evaluate((canvas: HTMLCanvasElement) =>
    Array.from(canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data),
  );
  const start = Date.now();
  await page.getByRole('slider', { name: 'Echo', exact: true }).fill('1');
  await expect(image).toHaveAttribute('aria-label', /frame 2, slice 1/, { timeout: 250 });
  expect(confirmed).toBe(false);
  const paintMs = Date.now() - start;
  console.log(`Cached frame paint: ${paintMs} ms; confirmation delayed by 450 ms`);
  const after = await image.evaluate((canvas: HTMLCanvasElement) =>
    Array.from(canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data),
  );
  expect(after).not.toEqual(before);
  await expect.poll(async () => (await opened(request))[0].frame).toBe(1);
  await info.attach('cached-navigation-latency', {
    body: JSON.stringify({ cachedPaintMs: paintMs, simulatedConfirmationDelayMs: 450 }),
    contentType: 'application/json',
  });
});

test('rapid frame and slice changes coalesce, deliver one packet, and never flash obsolete frames', async ({
  page,
  request,
}) => {
  await page.route('**/navigate', async (route) => {
    if (!route.request().postDataJSON().commit) await route.abort();
    else await route.continue();
  });
  await page.goto('/');
  await loadExample(page);
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => {
    release = resolve;
  });
  const packets: Record<string, number>[] = [];
  let notify!: () => void;
  const started = new Promise<void>((resolve) => {
    notify = resolve;
  });
  await page.route('**/navigate', async (route) => {
    const body = route.request().postDataJSON();
    if (!body.commit) {
      await route.abort();
      return;
    }
    packets.push(body.indices);
    if (packets.length === 1) {
      notify();
      await blocked;
    }
    await route.continue();
  });
  const legacy: string[] = [];
  page.on('request', (request) => {
    if (
      /\/(slice|comparison)$/.test(request.url()) ||
      (request.method() === 'PATCH' && request.url().endsWith('/view'))
    )
      legacy.push(request.url());
  });
  await page.evaluate(() => {
    const state = window as typeof window & { navigationLabels: string[] };
    state.navigationLabels = [];
    const image = document.querySelector('canvas[role="img"]')!;
    new MutationObserver(() =>
      state.navigationLabels.push(image.getAttribute('aria-label') ?? ''),
    ).observe(image, { attributes: true, attributeFilter: ['aria-label'] });
  });
  const echo = page.getByRole('slider', { name: 'Echo', exact: true });
  await echo.fill('2');
  await started;
  await echo.fill('4');
  await echo.fill('7');
  await page.getByRole('slider', { name: 'Slice', exact: true }).fill('2');
  release();
  const image = page.getByRole('img', { name: /24 by 32 quantitative image/ });
  await expect(image).toHaveAttribute('aria-label', /frame 8, slice 3/);
  await expect.poll(async () => (await opened(request))[0].frame).toBe(7);
  expect((await opened(request))[0].z).toBe(2);
  expect(packets).toHaveLength(2);
  expect(packets[1]).toMatchObject({ frame: 7, z: 2 });
  expect(legacy).toEqual([]);
  const labels = await page.evaluate(
    () => (window as typeof window & { navigationLabels: string[] }).navigationLabels,
  );
  expect(labels.every((label) => label.includes('frame 8, slice 3'))).toBe(true);
});

test('the frame slider keeps tracking throughout one continuous drag', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  await page.route('**/navigate', async (route) => {
    if (route.request().postDataJSON().commit)
      await new Promise((resolve) => setTimeout(resolve, 80));
    await route.continue();
  });
  const slider = page.getByRole('slider', { name: 'Echo', exact: true });
  const box = (await slider.boundingBox())!;
  const thumb = (fraction: number) => ({
    x: box.x + 10 + (box.width - 20) * fraction,
    y: box.y + box.height / 2,
  });
  const start = thumb(0);
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  try {
    const middle = thumb(0.5);
    await page.mouse.move(middle.x, middle.y, { steps: 12 });
    await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(0);
    const end = thumb(1);
    await page.mouse.move(end.x, end.y, { steps: 12 });
    await expect(slider).toHaveValue('7');
  } finally {
    await page.mouse.up();
  }
  await expect.poll(async () => (await opened(request))[0].frame).toBe(7);
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toHaveAttribute(
    'aria-label',
    /frame 8, slice 1/,
  );
});

test('rejected navigation restores the confirmed image and slider', async ({ page }) => {
  await page.goto('/');
  const warm = await warmFrame(page);
  await loadExample(page);
  await warm();
  await page.route('**/navigate', async (route) => {
    if (route.request().postDataJSON().commit)
      await route.fulfill({ status: 400, json: { detail: 'Independent navigation rejection' } });
    else await route.continue();
  });
  const echo = page.getByRole('slider', { name: 'Echo', exact: true });
  await echo.fill('1');
  await expect(page.getByRole('alert')).toContainText('Independent navigation rejection');
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toHaveAttribute(
    'aria-label',
    /frame 1, slice 1/,
  );
  await expect(echo).toHaveValue('0');
  await expect(echo).toBeEnabled();
});

test('a lost confirmation resynchronizes the cursor before further navigation', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  let loseFirst = true;
  await page.route('**/navigate', async (route) => {
    if (route.request().postDataJSON().commit && loseFirst) {
      loseFirst = false;
      await route.fetch();
      await route.abort();
    } else await route.continue();
  });
  const echo = page.getByRole('slider', { name: 'Echo', exact: true });
  await echo.fill('1');
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.getByRole('img', { name: /24 by 32 quantitative image/ })).toHaveAttribute(
    'aria-label',
    /frame 2, slice 1/,
  );
  await expect(echo).toBeEnabled();
  await echo.fill('2');
  await expect.poll(async () => (await opened(request))[0].frame).toBe(2);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('comparison navigation receives both planes together without a follow-up image request', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'View', 'Image display');
  await page.getByRole('button', { name: 'Original vs processed', exact: true }).click();
  const frame = page.getByRole('slider', { name: 'Frame', exact: true });
  await expect(frame).toBeVisible();
  const oldRequests: string[] = [];
  page.on('request', (request) => {
    if (request.url().endsWith('/comparison')) oldRequests.push(request.url());
  });
  await frame.fill('2');
  await expect.poll(async () => (await opened(request))[0].scene.comparison.frame).toBe(2);
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(oldRequests).toEqual([]);
});

test('navigation codec preserves float precision, nonfinite samples and mixed-plane alignment', async ({
  request,
}) => {
  const token = (await (await request.get('/api/session')).json()).token;
  const w: Workspace = await (
    await request.post('/api/workspaces/synthetic', { headers: { 'x-mrax-token': token } })
  ).json();
  const arrays = [new Float32Array([1, NaN, Infinity]), new Float64Array([Math.PI, -0, 1e100])];
  const metadata = new TextEncoder().encode(
    JSON.stringify({
      workspace: w,
      planes: arrays.map((_array, index) => ({
        id: index ? 'layer:float64' : 'image',
        dtype: index ? 'float64-le' : 'float32-le',
        shape: [1, 3],
        length: 3,
      })),
      comparison: null,
    }),
  );
  const start = 8 + Math.ceil(metadata.length / 8) * 8;
  const buffer = new ArrayBuffer(start + 16 + 24);
  const prefix = new DataView(buffer);
  prefix.setUint32(0, 1, true);
  prefix.setUint32(4, metadata.length, true);
  new Uint8Array(buffer, 8, metadata.length).set(metadata);
  new Float32Array(buffer, start, 3).set(arrays[0]);
  new Float64Array(buffer, start + 16, 3).set(arrays[1]);
  const decoded = decodeNavigation(buffer);
  expect(Number.isNaN(decoded.values![1])).toBe(true);
  expect(decoded.values![2]).toBe(Infinity);
  expect(decoded.layers.float64[0]).toBe(Math.PI);
  expect(Object.is(decoded.layers.float64[1], -0)).toBe(true);
  expect(decoded.layers.float64[2]).toBe(1e100);
  expect(() => decodeNavigation(buffer.slice(0, buffer.byteLength - 1))).toThrow();
  const entry: NavigationFrame = { workspace: w, values: new Float32Array(4), layers: {} };
  const bytes = JSON.stringify(w).length * 2 + 16;
  const cache = new NavigationCache(bytes * 2 + 1024);
  cache.set(entry);
  const next = (frame: number): NavigationFrame => ({
    ...entry,
    workspace: { ...w, indices: { ...w.indices, frame }, frame },
  });
  cache.set(next(1));
  cache.get(navigationKey(w));
  cache.set(next(2));
  expect(cache.get(navigationKey(w, { frame: 1 }))).toBeUndefined();
  expect(cache.get(navigationKey(w))).toBe(entry);
  cache.clear(w.id);
  expect(cache.bytes).toBe(0);
});
