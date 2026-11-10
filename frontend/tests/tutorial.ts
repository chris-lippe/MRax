import { expect, type Page } from '@playwright/test';
import type * as UI from './ui';

export async function startWorkflowTutorial(page: Page, openSettings: typeof UI.openSettings) {
  const settings = await openSettings(page);
  await settings.getByRole('tab', { name: 'Tutorial', exact: true }).click();
  await settings.getByRole('button', { name: 'Start tutorial', exact: true }).click();
  const guide = page.getByRole('region', { name: 'Workflow tutorial', exact: true });
  await expect(
    guide.getByRole('heading', { name: 'Your practice study', exact: true }),
  ).toBeVisible();
  return guide;
}

/** Exercise the real controls; no file save/export or measurement data. */
export async function completeTutorialWorkflow(
  page: Page,
  selectDataFormat: typeof UI.selectDataFormat,
  checkpoint?: (name: string) => Promise<void>,
) {
  const guide = page.getByRole('region', { name: 'Workflow tutorial', exact: true });
  const next = guide.getByRole('button', { name: 'Next', exact: true });
  const advance = async (heading: string) => {
    await expect(next).toBeEnabled({ timeout: 60000 });
    await next.click();
    await expect(guide.getByRole('heading', { name: heading, exact: true })).toBeVisible();
  };
  const open = async (name: string) => guide.getByRole('button', { name, exact: true }).click();
  await advance('Navigate images');
  await expect(next).toBeDisabled();
  await open('Open image controls');
  await page.getByRole('slider', { name: 'Echo', exact: true }).fill('1');
  await advance('Inspect metadata');
  await open('Open Metadata');
  await expect(page.getByRole('region', { name: 'Metadata browser', exact: true })).toBeVisible();
  await advance('Reconstruct from k-space');
  await open('Open Reconstruct');
  await expect(next).toBeDisabled();
  await page
    .getByRole('combobox', { name: 'Reconstruction preset', exact: true })
    .selectOption('cartesian-fourier');
  await page.getByRole('button', { name: 'Apply reconstruction', exact: true }).click();
  await advance('Denoise the current images');
  await open('Open Denoising');
  await expect(next).toBeDisabled();
  await page.getByRole('combobox', { name: 'Denoiser', exact: true }).selectOption('gaussian');
  await page.getByRole('button', { name: 'Apply denoising', exact: true }).click();
  await advance('Edit the processing pipeline');
  await open('Show pipeline controls');
  const chain = page.getByRole('region', { name: 'Processing pipeline', exact: true });
  const expand = chain.getByRole('button', { name: 'Show processing steps', exact: true });
  if (await expand.count()) await expand.click();
  await expect(chain.getByRole('listitem', { name: 'Pipeline input', exact: true })).toContainText(
    'Acquired k-space',
  );
  await expect(next).toBeDisabled();
  await chain.locator('[data-tutorial="pipeline-denoise-edit"]').click();
  await selectDataFormat(page, 'Pipeline step settings', 'Raw');
  const editor = chain.getByRole('textbox', { name: 'Pipeline step settings', exact: true });
  const settings = JSON.parse(await editor.inputValue());
  settings.parameters.sigma = 0.75;
  await editor.fill(JSON.stringify(settings));
  await expect(next).toBeDisabled();
  await chain.getByRole('button', { name: 'Discard changes', exact: true }).click();
  await advance('Compare images');
  await guide.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(next).toBeEnabled();
  await advance('Compare images');
  await open('Open comparison controls');
  await page.getByRole('button', { name: 'Original vs processed', exact: true }).click();
  await expect(next).toBeDisabled();
  await page.locator('.comparison-divider').focus();
  await page.keyboard.press('ArrowRight');
  await advance('Align acquisitions');
  await open('Open Register');
  await expect(page.getByRole('region', { name: 'Reference image', exact: true })).toBeVisible();
  await advance('Create an ROI');
  await open('Open Segment');
  await expect(next).toBeDisabled();
  await page.getByRole('button', { name: 'Draw rectangle', exact: true }).click();
  const surface = page.getByTestId('roi-surface');
  await surface.scrollIntoViewIfNeeded();
  const box = (await surface.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.35);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.65, { steps: 8 });
  await page.mouse.up();
  await advance('Edit a pixel mask');
  await open('Open mask editing');
  await page.getByRole('button', { name: 'Convert to pixel mask', exact: true }).click();
  await expect(next).toBeEnabled();
  await page.getByRole('button', { name: 'Preview export mask', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(guide).toHaveCount(0);
  await expect(page.locator('.tutorial-spotlight')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(guide).toBeVisible();
  await advance('Fit parameter maps');
  await open('Open Fit');
  await expect(next).toBeDisabled();
  await page.getByLabel('Fit steps', { exact: true }).fill('300');
  await page.getByRole('button', { name: 'Run full map fit', exact: true }).click();
  await advance('Analyze fitted maps');
  await open('Open map statistics');
  await page.getByLabel('Map', { exact: true }).selectOption('t2');
  await page.getByLabel('Statistics map', { exact: true }).selectOption('t2');
  await expect(page.getByRole('table', { name: 'ROI map statistics', exact: true })).toBeVisible();
  await checkpoint?.('tutorial-map-statistics');
  await advance('Inspect scientific signal curves');
  await open('Open signal curves');
  await expect(
    page.getByRole('region', { name: 'Scientific signal viewer', exact: true }),
  ).toBeVisible();
  await guide.getByRole('button', { name: 'Collapse tutorial guidance', exact: true }).click();
  await expect.poll(async () => (await guide.boundingBox())!.height).toBeLessThan(80);
  await checkpoint?.('tutorial-signal-curves');
  await guide.getByRole('button', { name: 'Expand tutorial guidance', exact: true }).click();
  await advance('Reuse a workflow');
  await open('Show processing workflows');
  await page.getByRole('button', { name: 'Processing workflows', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(guide).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(guide).toBeVisible();
  await advance('Save and export');
  await expect(page.getByRole('toolbar', { name: 'Workspace files', exact: true })).toBeVisible();
  await guide.getByRole('button', { name: 'Finish tutorial', exact: true }).click();
  await expect(guide).toHaveCount(0);
}
