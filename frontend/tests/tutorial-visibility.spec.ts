import { test, expect, type Locator, type Page } from '@playwright/test';
import { openSettings, selectWorkflow } from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const workspace of await (await request.get('/api/workspaces')).json()) {
    await request.delete('/api/workspaces/' + workspace.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: workspace.revision, discard: true },
    });
  }
});

async function startTutorial(page: Page) {
  await page.setViewportSize({ width: 1280, height: 640 });
  await page.goto('/');
  const settings = await openSettings(page);
  await settings.getByRole('tab', { name: 'Tutorial', exact: true }).click();
  await settings.getByRole('button', { name: 'Start tutorial', exact: true }).click();
  const guide = page.getByRole('region', { name: 'Workflow tutorial', exact: true });
  await expect(guide).toBeVisible();
  await guide.getByRole('button', { name: 'Next', exact: true }).click();
  return guide;
}

async function matchingSpotlight(page: Page, target: Locator) {
  const spotlight = page.locator('.tutorial-spotlight');
  await expect(spotlight).toBeVisible();
  await expect
    .poll(async () => {
      const box = (await spotlight.boundingBox())!;
      const control = (await target.boundingBox())!;
      return (
        box.x <= control.x + 1 &&
        box.y <= control.y + 1 &&
        box.x + box.width >= control.x + control.width - 1 &&
        box.y + box.height >= control.y + control.height - 1
      );
    })
    .toBe(true);
}

test('offscreen image controls get a scroll arrow instead of a misplaced box and track scrolling and resizing', async ({
  page,
}, info) => {
  await startTutorial(page);
  const main = page.locator('main');
  const controls = page.locator('.dimension-controls');
  const arrow = page.getByRole('button', {
    name: 'Scroll down to the tutorial control',
    exact: true,
  });
  await expect(arrow).toBeVisible();
  await expect(page.locator('.tutorial-spotlight')).toHaveCount(0);
  expect(await main.evaluate((element) => element.scrollTop)).toBe(0);
  const area = (await main.boundingBox())!;
  const cue = (await arrow.boundingBox())!;
  expect(cue.x).toBeGreaterThanOrEqual(area.x);
  expect(cue.x + cue.width).toBeLessThanOrEqual(area.x + area.width);
  expect(cue.y + cue.height).toBeLessThanOrEqual(area.y + area.height);
  await page.screenshot({ path: info.outputPath('tutorial-scroll-down.png') });
  await arrow.click();
  await expect(arrow).not.toBeVisible();
  await matchingSpotlight(page, controls);
  await main.evaluate((element) => {
    element.scrollTop = 0;
  });
  await expect(arrow).toBeVisible();
  await expect(page.locator('.tutorial-spotlight')).toHaveCount(0);
  await page.setViewportSize({ width: 1280, height: 1200 });
  await expect(arrow).not.toBeVisible();
  await matchingSpotlight(page, controls);
});

test('sticky control-panel targets stay aligned while scrolling and targets above the image view get an up arrow', async ({
  page,
}, info) => {
  const guide = await startTutorial(page);
  await guide.getByRole('button', { name: 'Open image controls', exact: true }).click();
  await page.getByRole('slider', { name: 'Echo', exact: true }).fill('1');
  await guide.getByRole('button', { name: 'Next', exact: true }).click();
  await guide.getByRole('button', { name: 'Skip step', exact: true }).click();
  await guide.getByRole('button', { name: 'Skip step', exact: true }).click();
  await guide.getByRole('button', { name: 'Open Denoising', exact: true }).click();
  const panel = page.locator('#workflow-panel');
  const apply = page.getByRole('button', { name: 'Apply denoising', exact: true });
  await matchingSpotlight(page, apply);
  await panel.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  await matchingSpotlight(page, apply);
  await selectWorkflow(page, 'Preprocess', 'Signal normalization');
  await matchingSpotlight(page, page.locator('#workflow-preprocess'));
  await guide.getByRole('button', { name: 'Open Denoising', exact: true }).click();
  await matchingSpotlight(page, apply);
  await page.screenshot({ path: info.outputPath('tutorial-control-target.png') });
  await apply.click();
  await expect(guide.getByRole('button', { name: 'Next', exact: true })).toBeEnabled();
  await guide.getByRole('button', { name: 'Next', exact: true }).click();
  await guide.getByRole('button', { name: 'Show pipeline controls', exact: true }).click();
  await page.locator('main').evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  const up = page.getByRole('button', { name: 'Scroll up to the tutorial control', exact: true });
  await expect(up).toBeVisible();
  await expect(page.locator('.tutorial-spotlight')).toHaveCount(0);
  await up.click();
  await matchingSpotlight(
    page,
    page.getByRole('button', { name: 'Show processing steps', exact: true }),
  );
});
