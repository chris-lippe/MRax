import type { components } from './api.generated';
import type { Options } from './api';

export type RecoStage = {
  kind: string;
  enabled?: boolean;
  when?: 'always' | 'configured' | 'cartesian' | 'sampled';
  options: Options;
};
export type RecoPlan = { schema_version: 1; name: string; encoding: string; stages: RecoStage[] };
export type RecoDefinition = {
  kind: string;
  label: string;
  order: number;
  solver: boolean;
  fields: components['schemas']['ProcessingField'][];
  defaults: Options;
  insert_options: Options;
  available: boolean;
  reason: string | null;
  note?: string | null;
  configured: boolean;
};
export type RecoCatalog = {
  prepared: boolean;
  encoding: string | null;
  encoding_dimensions: number;
  stages: RecoDefinition[];
  presets: {
    id: string;
    filename: string;
    name: string;
    description: string;
    plan: RecoPlan;
    compatible: boolean;
  }[];
  plan: RecoPlan;
  component: string;
  label: string;
};

export function recoProblem(
  plan: RecoPlan,
  definitions: RecoDefinition[],
  encoding: string | null,
) {
  if (!plan.name.trim()) return 'Give the reconstruction pipeline a name.';
  if (encoding && plan.encoding !== 'any' && plan.encoding !== encoding)
    return 'This preset requires ' + plan.encoding + ' acquisitions.';
  const active = plan.stages.filter((stage) => stage.enabled !== false);
  const kinds = active.map((stage) => stage.kind);
  if (kinds.filter((kind) => ['fft', 'variational', 'nufft'].includes(kind)).length !== 1)
    return 'Keep exactly one Fourier, variational or NUFFT solver.';
  if (!kinds.includes('format') || !kinds.includes('combine'))
    return 'Keep output geometry and coil combination stages.';
  const order = active.map(
    (stage) => definitions.find((item) => item.kind === stage.kind)?.order ?? 0,
  );
  if (order.some((value, index) => index > 0 && value < order[index - 1]))
    return 'K-space corrections precede the solver. Geometry and coil combination follow it.';
  if (kinds.includes('nufft') && kinds.includes('gridding'))
    return 'NUFFT has its own interpolation. Remove the gridding stage.';
  if (encoding === 'sampled' && !kinds.includes('nufft') && !kinds.includes('gridding'))
    return 'Add Kaiser–Bessel gridding before Fourier or variational reconstruction of sampled data.';
  return '';
}

export function insertRecoStage(
  plan: RecoPlan,
  definition: RecoDefinition,
  definitions: RecoDefinition[],
  encoding?: string | null,
) {
  const stage: RecoStage = {
    kind: definition.kind,
    enabled: true,
    when: 'always',
    options: structuredClone(definition.insert_options),
  };
  const stages = plan.stages.filter(
    (item) =>
      item.kind !== stage.kind &&
      !(definition.solver && definitions.find((other) => other.kind === item.kind)?.solver) &&
      !(stage.kind === 'nufft' && item.kind === 'gridding'),
  );
  if (
    definition.solver &&
    stage.kind !== 'nufft' &&
    encoding === 'sampled' &&
    !stages.some((item) => item.kind === 'gridding')
  ) {
    const gridding = definitions.find((item) => item.kind === 'gridding')!;
    const beforeSolver = stages.findIndex(
      (item) =>
        (definitions.find((other) => other.kind === item.kind)?.order ?? 0) > gridding.order,
    );
    stages.splice(beforeSolver < 0 ? stages.length : beforeSolver, 0, {
      kind: 'gridding',
      enabled: true,
      when: 'sampled',
      options: {},
    });
  }
  const index = stages.findIndex(
    (item) =>
      (definitions.find((other) => other.kind === item.kind)?.order ?? 0) > definition.order,
  );
  stages.splice(index < 0 ? stages.length : index, 0, stage);
  return { ...plan, stages };
}

export function recoFieldValue(stage: RecoStage, definition: RecoDefinition, key: string) {
  const [base, index] = key.split('.');
  const value = Object.hasOwn(stage.options, base)
    ? stage.options[base]
    : definition.defaults[base];
  return index === undefined ? value : ((Array.isArray(value) ? value[Number(index)] : 0) ?? 0);
}

export function recoFieldOptions(
  stage: RecoStage,
  definition: RecoDefinition,
  key: string,
  value: unknown,
  dimensions: number,
): Options {
  const [base, index] = key.split('.');
  if (index !== undefined) {
    const previous = Object.hasOwn(stage.options, base)
      ? stage.options[base]
      : definition.defaults[base];
    const values = Array.from({ length: base === 'image_shift' ? dimensions : 2 }, (_, i) =>
      Array.isArray(previous) ? (previous[i] ?? 0) : 0,
    );
    values[Number(index)] = value;
    return { ...stage.options, [base]: values };
  }
  if (base === 'kspace_window' || base === 'window_parameters')
    return { ...stage.options, [base]: Array(dimensions).fill(value) };
  return { ...stage.options, [key]: value };
}
