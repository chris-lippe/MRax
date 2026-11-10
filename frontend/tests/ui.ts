import { expect, type Page } from '@playwright/test';

export async function imageZoom(page: Page) {
  return page
    .locator('.image-surface:visible, .comparison-left canvas:visible')
    .first()
    .evaluate((element) => new DOMMatrix(getComputedStyle(element).transform).a);
}

export async function zoomImage(page: Page, zoom: number) {
  const native = page.locator('.image-viewport:visible');
  const paired = !(await native.count());
  const viewport = paired ? page.locator('.comparison-canvas-viewport:visible').last() : native;
  await viewport.scrollIntoViewIfNeeded();
  const box = (await viewport.boundingBox())!;
  await page.mouse.move(box.x + box.width * (paired ? 0.75 : 0.5), box.y + box.height / 2);
  await page.mouse.wheel(0, -Math.log(zoom / (await imageZoom(page))) / 0.002);
  await expect.poll(() => imageZoom(page)).toBeCloseTo(zoom, 2);
}

export async function showStudies(page: Page) {
  const tools = page.getByRole('dialog', { name: 'Analysis controls', exact: true });
  if (await tools.isVisible())
    await tools.getByRole('button', { name: 'Close panel', exact: true }).click();
  const studies = page.getByRole('complementary', { name: 'Data & Studies', exact: true });
  await expect(studies).toBeVisible();
  return studies;
}

export async function showDataControls(page: Page) {
  const studies = await showStudies(page);
  return studies.getByRole('toolbar', { name: 'Workspace files', exact: true });
}

export async function clickDataControl(page: Page, name: string) {
  const files = await showDataControls(page);
  await files.getByRole('button', { name, exact: true }).click();
  if (name === 'New study') {
    const dialog = page.getByRole('dialog', { name: 'New study', exact: true });
    await dialog.getByRole('button', { name: 'Create study', exact: true }).click();
    await expect(dialog).not.toBeVisible();
  }
}

export async function openNewStudy(page: Page) {
  const files = await showDataControls(page);
  await files.getByRole('button', { name: 'New study', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'New study', exact: true });
  await expect(dialog).toBeVisible();
  return dialog;
}

export async function loadExample(page: Page, kind: 't2' | 'multimodal' = 't2') {
  const dialog = await openNewStudy(page);
  await dialog
    .getByRole('button', {
      name: kind === 't2' ? 'Load synthetic study' : 'Load multimodal study',
      exact: true,
    })
    .click();
  await expect(page.getByTestId('roi-surface')).toBeVisible();
  await expect(dialog).not.toBeVisible();
}

export async function openSettings(page: Page) {
  const dialog = page.getByRole('dialog', { name: 'Settings', exact: true });
  if (!(await dialog.isVisible())) {
    const studies = await showStudies(page);
    await studies.getByRole('button', { name: 'Settings', exact: true }).click();
  }
  await expect(dialog).toBeVisible();
  return dialog;
}

export async function openRoiCoordinates(page: Page) {
  const form = page.getByRole('form', { name: 'ROI coordinate editor', exact: true });
  if (!(await form.isVisible())) await page.getByText('Coordinates', { exact: true }).click();
  await expect(form).toBeVisible();
}

export async function selectDataFormat(page: Page, label: string, format: 'Table' | 'Raw') {
  const button = page
    .getByRole('group', { name: `${label} format`, exact: true })
    .getByRole('button', { name: format, exact: true });
  await button.click();
  await expect(button).toHaveAttribute('aria-pressed', 'true');
}

export async function selectWorkflow(page: Page, name: string, task?: string) {
  const tab = page.getByRole('tab', { name, exact: true });
  await tab.click();
  if ((await tab.getAttribute('aria-haspopup')) === 'menu') {
    const menu = page.getByRole('menu', { name: `${name} tasks`, exact: true });
    await expect(menu).toBeVisible();
    const item = task
      ? menu.getByRole('menuitemradio', { name: task, exact: true })
      : menu.locator('[role="menuitemradio"]:focus');
    await item.click();
    await expect(menu).not.toBeVisible();
  }
  await expect(page.getByRole('tab', { name, exact: true, includeHidden: true })).toHaveAttribute(
    'aria-selected',
    'true',
  );
}

export async function selectTheme(page: Page, theme: 'light' | 'dark') {
  const controls = page.getByRole('dialog', { name: 'Analysis controls', exact: true });
  const restoreControls = await controls.isVisible();
  await openSettings(page);
  await page.getByRole('tab', { name: 'Appearance', exact: true }).click();
  const names = { light: 'Light theme', dark: 'Dark theme' };
  const button = page
    .getByRole('group', { name: 'Theme', exact: true })
    .getByRole('button', { name: names[theme], exact: true });
  await button.click();
  await expect(button).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
  await page
    .getByRole('dialog', { name: 'Settings', exact: true })
    .getByRole('button', { name: 'Done', exact: true })
    .click();
  if (restoreControls) await page.getByRole('button', { name: 'Controls', exact: true }).click();
}

export async function closeMetadataBrowser(page: Page) {
  if (await page.getByRole('region', { name: 'Metadata browser', exact: true }).isVisible())
    await selectWorkflow(page, 'View', 'Image display');
  const drawer = page.getByRole('dialog', { name: 'Analysis controls', exact: true });
  if (await drawer.isVisible())
    await drawer.getByRole('button', { name: 'Close panel', exact: true }).click();
}

export async function openMetadataBrowser(page: Page, category = 'Study overview') {
  const browser = page.getByRole('region', { name: 'Metadata browser', exact: true });
  if (!(await browser.isVisible())) {
    await selectWorkflow(page, 'View', 'Metadata');
  }
  await expect(browser).toBeVisible();
  await browser
    .getByRole('group', { name: 'Metadata categories', exact: true })
    .getByRole('button', { name: category, exact: true })
    .click();
  return browser;
}

export async function selectExperiment(page: Page, name: string) {
  const group = page
    .locator('.study-group')
    .filter({ has: page.locator('.study-item[aria-current="true"]') });
  const expand = group.getByRole('button', { name: /^Expand study:/ });
  if (await expand.count()) await expand.click();
  const experiment = page
    .locator('.study-group')
    .filter({ has: page.locator('.study-item[aria-current="true"]') })
    .getByRole('region', { name: /^Experiments in / })
    .getByRole('button', { name, exact: true });
  await experiment.click();
  await expect(experiment).toHaveAttribute('aria-current', 'true');
}

export async function openExport(page: Page) {
  const dialog = page.getByRole('dialog', { name: 'Export data', exact: true });
  if (!(await dialog.isVisible())) {
    await clickDataControl(page, 'Export data');
  }
  await expect(dialog).toBeVisible();
  return dialog;
}

export async function closeExport(page: Page) {
  const dialog = page.getByRole('dialog', { name: 'Export data', exact: true });
  if (await dialog.isVisible())
    await dialog.getByRole('button', { name: 'Close export', exact: true }).click();
}

export async function chooseFilePath(
  page: Page,
  path: string,
  action: 'Open' | 'Save' | 'Select folder' = 'Open',
  replace = false,
  folder = false,
) {
  const dialog = page.locator('.file-system-dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Go', exact: true })).toBeEnabled();
  await dialog.getByLabel('Location', { exact: true }).fill(path);
  await dialog.getByRole('button', { name: 'Go', exact: true }).click();
  if (action === 'Save') {
    const filename = path.split(/[\\/]/).at(-1)!;
    await expect(
      dialog.locator('input[aria-label="File name"], input[aria-label="Folder name"]'),
    ).toHaveValue(filename);
  } else if (action === 'Open' && !folder)
    await expect(dialog.getByLabel('File name', { exact: true })).toHaveValue(
      path.split(/[\\/]/).at(-1)!,
    );
  else await expect(dialog.getByLabel('Location', { exact: true })).toHaveValue(path);
  await dialog.getByRole('button', { name: action, exact: true }).click();
  if (replace) await dialog.getByRole('button', { name: 'Replace', exact: true }).click();
  await expect(dialog).not.toBeVisible();
}

export async function chooseExportFolder(page: Page, path: string, create = false) {
  const dialog = page.locator('.file-system-dialog');
  const parent = path.slice(0, path.lastIndexOf('/'));
  const name = path.split('/').at(-1)!;
  await expect(dialog.getByRole('button', { name: 'Go', exact: true })).toBeEnabled();
  await dialog.getByLabel('Location', { exact: true }).fill(create ? parent : path);
  await dialog.getByRole('button', { name: 'Go', exact: true }).click();
  await expect(dialog.getByLabel('Location', { exact: true })).toHaveValue(create ? parent : path);
  if (create) {
    await dialog.getByRole('button', { name: 'New folder', exact: true }).click();
    await dialog.getByLabel('New folder name', { exact: true }).fill(name);
    await dialog.getByRole('button', { name: 'Create folder', exact: true }).click();
    await dialog.getByRole('button', { name: name + ' Folder', exact: true }).dblclick();
    await expect(dialog.getByLabel('Location', { exact: true })).toHaveValue(path);
  }
  await expect(dialog.getByRole('button', { name: 'Replace', exact: true })).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Select folder', exact: true }).click();
  await expect(dialog).not.toBeVisible();
}

export async function saveWorkspace(page: Page, path: string) {
  await clickDataControl(page, 'Save workspace');
  await chooseFilePath(page, path, 'Save');
  await expect(page.locator('[aria-label="Workspace status"]')).toHaveText('Workspace saved');
}

export async function openWorkspace(page: Page, path: string) {
  await clickDataControl(page, 'Open workspace');
  await chooseFilePath(page, path);
  await expect(page.getByRole('status', { name: 'Workspace status' })).toHaveText(
    'Workspace opened',
  );
}
