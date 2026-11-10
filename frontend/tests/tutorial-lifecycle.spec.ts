import { test, expect } from '@playwright/test';
import { tutorialSteps } from '../src/tutorialSteps';
import { startWorkflowTutorial } from './tutorial';
import {
  loadExample,
  openSettings,
  selectDataFormat,
  selectWorkflow,
  zoomImage,
  imageZoom,
} from './ui';

test.afterEach(async ({ request }) => {
  const token = (await (await request.get('/api/session')).json()).token;
  for (const w of await (await request.get('/api/workspaces')).json())
    await request.delete('/api/workspaces/' + w.id, {
      headers: { 'x-mrax-token': token },
      data: { revision: w.revision, discard: true },
    });
});

test('skipping steps performs no processing and preserves an existing pipeline draft', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  const chain = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  await chain.getByRole('button', { name: 'Show processing steps', exact: true }).click();
  await chain.locator('[data-tutorial="pipeline-denoise-edit"]').click();
  await selectDataFormat(page, 'Pipeline step settings', 'Raw');
  const editor = chain.getByRole('textbox', { name: 'Pipeline step settings', exact: true });
  const draft = JSON.parse(await editor.inputValue());
  draft.parameters.sigma = 1.25;
  await editor.fill(JSON.stringify(draft));
  await expect(chain.getByText('Unapplied changes', { exact: true })).toBeVisible();
  const [before] = await (await request.get('/api/workspaces')).json();
  const operations: string[] = [];
  page.on('request', (r) => {
    if (r.method() === 'POST' && r.url().endsWith('/tasks'))
      operations.push(r.postDataJSON().operation);
  });
  const guide = await startWorkflowTutorial(page, openSettings);
  for (const [index, step] of tutorialSteps.entries()) {
    await expect(guide.getByRole('heading', { name: step.title, exact: true })).toBeVisible();
    if (index === 2) {
      await guide.getByRole('button', { name: 'Back', exact: true }).click();
      await expect(guide.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
      await guide.getByRole('button', { name: 'Skip step', exact: true }).click();
    }
    if (index < tutorialSteps.length - 1)
      await guide.getByRole('button', { name: 'Skip step', exact: true }).click();
  }
  await guide.getByRole('button', { name: 'Finish tutorial', exact: true }).click();
  await expect(page.locator('.study-item[aria-current="true"]')).toContainText(before.label);
  const all = await (await request.get('/api/workspaces')).json();
  expect(all.find((w: { id: string }) => w.id === before.id)).toEqual(before);
  await expect(chain.getByText('Unapplied changes', { exact: true })).toBeVisible();
  expect(operations).toEqual([]);
});

test('Settings pauses the guide and Resume keeps the same practice study and step', async ({
  page,
  request,
}) => {
  await page.goto('/');
  const guide = await startWorkflowTutorial(page, openSettings);
  await guide.getByRole('button', { name: 'Next', exact: true }).click();
  const [practice] = await (await request.get('/api/workspaces')).json();
  const settings = await openSettings(page);
  await expect(guide).toHaveCount(0);
  await expect(page.locator('.tutorial-spotlight')).toHaveCount(0);
  await settings.getByRole('tab', { name: 'Tutorial', exact: true }).click();
  await settings.getByRole('button', { name: 'Resume tutorial', exact: true }).click();
  await expect(guide.getByRole('heading', { name: 'Navigate images', exact: true })).toBeVisible();
  const all = await (await request.get('/api/workspaces')).json();
  expect(all.map((w: { id: string }) => w.id)).toEqual([practice.id]);
  await guide.getByRole('button', { name: 'End tutorial', exact: true }).click();
  await expect(guide).toHaveCount(0);
  await expect(page.locator('.study-item[aria-current="true"]')).toContainText(practice.label);
});

test('ending early restores the previous study, action and image zoom', async ({
  page,
  request,
}) => {
  await page.goto('/');
  await loadExample(page);
  await selectWorkflow(page, 'Segment', 'Create and edit ROIs');
  await zoomImage(page, 2);
  const [before] = await (await request.get('/api/workspaces')).json();
  const guide = await startWorkflowTutorial(page, openSettings);
  await guide.getByRole('button', { name: 'Skip step', exact: true }).click();
  await guide.getByRole('button', { name: 'End tutorial', exact: true }).click();
  await expect(page.locator('.study-item[aria-current="true"]')).toContainText(before.label);
  await expect(page.getByRole('tab', { name: 'Segment', exact: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  expect(await imageZoom(page)).toBeCloseTo(2);
  const all = await (await request.get('/api/workspaces')).json();
  expect(all).toHaveLength(2);
  expect(all.find((w: { id: string }) => w.id === before.id)).toEqual(before);
});
