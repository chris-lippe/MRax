import { execFileSync } from 'node:child_process';
import { mkdir, readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { test, expect, type APIRequestContext, type Page, type TestInfo } from '@playwright/test';
import { chooseFilePath, openWorkspace, selectDataFormat, selectWorkflow } from './ui';
import type { Workspace } from '../src/api';

const opened = async (request: APIRequestContext): Promise<Workspace[]> =>
  (await request.get('/api/workspaces')).json();
const pipeline = (page: Page) =>
  page.getByRole('region', { name: 'Processing pipeline', exact: true });

async function fixture(page: Page, info: TestInfo, raw = true) {
  const suffix = raw ? 'raw' : 'images';
  const path = info.outputPath('independent-composite-' + suffix + '.mrax');
  await mkdir(dirname(path), { recursive: true });
  const fit = info.outputPath('independent-fit-' + suffix + '.mfit');
  const incompatible = info.outputPath('independent-nufft-' + suffix + '.mrec');
  execFileSync('python', [
    '-c',
    `
import sys
import numpy as np
from mrax.data_struct import AcquisitionData, Quantity
from mrax.gui.session import GuiSession
from mrax.gui.services.workspace import StudyWorkspace
from mrax.gui.services.pipeline_blocks import block_options
from mrax.gui.services.persistence import save_workspace_bundle
from mrax.reco import fftnc
from mrax.reco.plans import reconstruction_presets
from mrax.workflows import WorkflowDocument, save_workflow
times = np.linspace(10,120,8,dtype=np.float32)
x,y = np.meshgrid(np.linspace(-1,1,4),np.linspace(-1,1,5),indexing='ij')
image = (1e6*(.6+.4*np.exp(-3*(x*x+y*y)))[None]*np.exp(-times[:,None,None]/80)).astype(np.float32)
session=GuiSession();session.create_empty_study('Independent composite workflow',field_strength_mhz=100)
session.add_experiment('A',image,dims={'frame':8,'x':4,'y':5},model_name='t2_se',observables={'TE':Quantity(times,'ms')})
session.fit_settings.num_steps=600
session.fit_settings.param_overrides={'bias':{'trainable':False,'value':0}}
if sys.argv[4]=='raw':
 samples=np.asarray(fftnc(image[:,None].astype(np.complex64)*2,axes=(-2,-1)))
 acquisition=AcquisitionData(samples,(4,5),('x','y'))
 session.current_experiment.update(acquisition=acquisition,raw_kspace=samples)
 session.study.datasets[0].acquisition=acquisition;session.study.datasets[0].kspace=samples
work=StudyWorkspace('independent',session.study_name,session)
document=block_options(work,'fit',{'use_roi':False})['workflow'];document['name']='T2 signal fit'
save_workflow(WorkflowDocument.model_validate(document),sys.argv[2])
save_workflow(WorkflowDocument.model_validate(next(p['document'] for p in reconstruction_presets() if p['id']=='nufft-adjoint')),sys.argv[3])
save_workspace_bundle(work,sys.argv[1])
`,
    path,
    fit,
    incompatible,
    raw ? 'raw' : 'images',
  ]);
  await page.goto('/');
  await openWorkspace(page, path);
  await pipeline(page).getByRole('button', { name: 'Show processing steps', exact: true }).click();
  return { fit, incompatible };
}

async function pixels(request: APIRequestContext) {
  const [w] = await opened(request);
  const response = await request.get('/api/workspaces/' + w.id + '/slice');
  const data = await response.body();
  const width = response.headers()['x-dtype'] === 'float64-le' ? 8 : 4;
  return Array.from({ length: data.length / width }, (_, index) =>
    width === 8 ? data.readDoubleLE(index * width) : data.readFloatLE(index * width),
  );
}

test.afterEach(async ({ request }) => {
  const headers = { 'x-mrax-token': (await (await request.get('/api/session')).json()).token };
  for (const w of await opened(request))
    await request.delete('/api/workspaces/' + w.id, {
      headers,
      data: { revision: w.revision, discard: true },
    });
  for (const entry of await (await request.get('/api/workflows')).json())
    await request.delete('/api/workflows/' + entry.id + '?revision=' + entry.revision, { headers });
});

test('reconstruction and fitting blocks compose, save as mpip and keep maps beside images', async ({
  page,
  request,
}, info) => {
  const { fit } = await fixture(page, info);
  const chain = pipeline(page);
  const before = await pixels(request);
  await expect(chain).not.toContainText('Protected');
  const input = chain.getByRole('listitem', { name: 'Pipeline input', exact: true });
  await expect(input).toContainText('Image data');
  await expect(input).toContainText('In-memory data');
  await expect(chain.getByLabel('Pipeline source', { exact: true })).toHaveValue('vendor');
  await chain.getByRole('button', { name: 'Insert step', exact: true }).click();
  const insertion = chain.getByRole('form', { name: 'Insert pipeline step', exact: true });
  await insertion.getByLabel('New pipeline step', { exact: true }).selectOption('reconstruct');
  await insertion
    .getByLabel('reconstruction workflow', { exact: true })
    .selectOption('preset:cartesian-fourier');
  await insertion.getByRole('button', { name: 'Insert into draft', exact: true }).click();
  await expect(input).toContainText('Acquired k-space');
  await expect(
    chain.getByRole('listitem', { name: 'Cartesian Fourier', exact: true }),
  ).toBeVisible();
  await chain.getByRole('button', { name: 'Close settings', exact: true }).click();
  await chain.getByRole('button', { name: 'Insert step', exact: true }).click();
  await insertion.getByLabel('New pipeline step', { exact: true }).selectOption('fit');
  await insertion.getByRole('button', { name: 'Open .mfit', exact: true }).click();
  await chooseFilePath(page, fit, 'Open');
  await expect(insertion.getByLabel('Workflow block name', { exact: true })).toHaveValue(
    'T2 signal fit',
  );
  await insertion.getByRole('button', { name: 'Insert into draft', exact: true }).click();
  expect((await opened(request))[0].pipeline.steps).toHaveLength(0);
  expect(await pixels(request)).toEqual(before);
  await chain.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(2);
  const [w] = await opened(request);
  expect(w.pipeline.steps.map((step) => step.kind)).toEqual(['reconstruct', 'fit']);
  expect(w.maps.some((map) => map.name === 'T2 signal fit / t2')).toBe(true);
  (await pixels(request)).forEach((value, index) =>
    expect(Math.abs(value / before[index] - 2)).toBeLessThan(0.00001),
  );
  await expect(chain.getByLabel('Pipeline source', { exact: true })).toHaveValue('mrax');
  await page.screenshot({ path: info.outputPath('composite-pipeline-applied.png') });
  await selectWorkflow(page, 'Preprocess', 'Denoising');
  await page.getByRole('button', { name: 'Processing workflows', exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'Reusable workflows', exact: true });
  await modal.getByRole('button', { name: 'Capture current pipeline', exact: true }).click();
  const file = info.outputPath('complete.mpip');
  await modal.getByRole('button', { name: 'Save workflow file', exact: true }).click();
  await chooseFilePath(page, file, 'Save');
  await expect(modal.getByRole('status')).toHaveText('Workflow saved.');
  const document = JSON.parse(await readFile(file, 'utf8'));
  expect(document.format_version).toBe(2);
  expect(document.steps[0].settings.workflow.kind).toBe('reconstruction');
  expect(document.steps[1].settings.workflow.kind).toBe('fitting');
  await modal.getByRole('button', { name: 'Done', exact: true }).click();
  await chain.getByRole('button', { name: 'Remove all processing', exact: true }).click();
  await chain.getByRole('button', { name: 'Apply pipeline', exact: true }).click();
  await expect.poll(async () => (await opened(request))[0].pipeline.steps.length).toBe(0);
  expect(await pixels(request)).toEqual(before);
  expect((await opened(request))[0].maps.some((map) => map.name === 'T2 signal fit / t2')).toBe(
    true,
  );
  await expect(chain.getByLabel('Pipeline source', { exact: true })).toHaveValue('vendor');
  await expect(input).toContainText('Image data');
  await page.screenshot({ path: info.outputPath('composite-pipeline.png') });
});

test('reconstruction requires valid raw data and incompatible mrec blocks cannot be inserted', async ({
  page,
  request,
}, info) => {
  const { incompatible } = await fixture(page, info, false);
  const chain = pipeline(page);
  await expect(
    chain.getByLabel('Pipeline source', { exact: true }).locator('option[value="mrax"]'),
  ).toHaveJSProperty('disabled', true);
  await chain.getByRole('button', { name: 'Insert step', exact: true }).click();
  const insertion = chain.getByRole('form', { name: 'Insert pipeline step', exact: true });
  await expect(
    insertion
      .getByLabel('New pipeline step', { exact: true })
      .locator('option[value="reconstruct"]'),
  ).toHaveJSProperty('disabled', true);
  await insertion.getByRole('button', { name: 'Cancel insertion', exact: true }).click();
  const [old] = await opened(request);
  await page.getByRole('button', { name: /^Independent composite workflow/ }).hover();
  await page.getByRole('button', { name: /^Close study:/ }).click();
  await fixture(page, info, true);
  await expect.poll(async () => (await opened(request))[0].id).not.toBe(old.id);
  await chain.getByRole('button', { name: 'Insert step', exact: true }).click();
  await insertion.getByLabel('New pipeline step', { exact: true }).selectOption('reconstruct');
  await insertion.getByRole('button', { name: 'Open .mrec', exact: true }).click();
  await chooseFilePath(page, incompatible, 'Open');
  await insertion.getByRole('button', { name: 'Insert into draft', exact: true }).click();
  await expect(chain.getByRole('alert')).toContainText('incompatible');
  expect((await opened(request))[0].pipeline.steps).toHaveLength(0);
  await expect(chain.locator('.pipeline-step')).toHaveCount(0);
});

test('editing an embedded workflow does not submit or apply the surrounding insertion controls', async ({
  page,
  request,
}, info) => {
  const formErrors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && /descendant|nested.*form/i.test(message.text()))
      formErrors.push(message.text());
  });
  await fixture(page, info, false);
  const chain = pipeline(page);
  await chain.getByRole('button', { name: 'Insert step', exact: true }).click();
  const insertion = chain.getByRole('form', { name: 'Insert pipeline step', exact: true });
  await insertion.getByLabel('New pipeline step', { exact: true }).selectOption('fit');
  await expect(insertion.getByLabel('Workflow block name', { exact: true })).toBeEnabled();
  await insertion.locator('summary').filter({ hasText: 'Workflow definition' }).click();
  await selectDataFormat(page, 'Embedded workflow definition', 'Raw');
  const text = insertion.getByLabel('Embedded workflow definition', { exact: true });
  const definition = JSON.parse(await text.inputValue());
  definition.name = 'Edited fitting block';
  definition.steps[0].settings.settings.num_steps = 333;
  await text.fill(JSON.stringify(definition));
  await insertion
    .getByRole('button', { name: 'Apply embedded workflow definition', exact: true })
    .click();
  await expect(insertion.getByLabel('Workflow block name', { exact: true })).toHaveValue(
    'Edited fitting block',
  );
  await expect(chain.locator('.pipeline-step')).toHaveCount(0);
  expect((await opened(request))[0].pipeline.steps).toHaveLength(0);
  await insertion.getByRole('button', { name: 'Insert into draft', exact: true }).click();
  await expect(
    chain.getByRole('listitem', { name: 'Edited fitting block', exact: true }),
  ).toBeVisible();
  expect((await opened(request))[0].pipeline.steps).toHaveLength(0);
  expect(formErrors).toEqual([]);
});
