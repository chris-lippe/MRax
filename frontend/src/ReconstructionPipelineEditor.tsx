import { useEffect, useRef, useState } from 'react';
import { WorkbenchIcon } from './WorkbenchIcon';
import {
  insertRecoStage,
  recoFieldOptions,
  recoFieldValue,
  type RecoCatalog,
  type RecoPlan,
  type RecoStage,
} from './reconstructionPlan';

const names: Record<string, string> = {
  rss: 'Root sum of squares',
  sum: 'Complex sum',
  tv: 'Total variation',
  l1: 'L1',
  l2: 'L2',
  none: 'None',
  fista: 'FISTA',
  adam: 'Adam',
  split_bregman: 'Split Bregman',
  pocs: 'POCS',
  homodyne: 'Homodyne',
  hann: 'Hann',
  hamming: 'Hamming',
  sine: 'Sine bell',
  sine_squared: 'Squared sine bell',
  image: 'Image',
  frequency_hz: 'Frequency (Hz)',
  adjoint: 'Adjoint',
  nearest: 'Nearest neighbour',
  linear: 'Linear',
  sinc: 'Sinc',
  lanczos: 'Lanczos',
  entropy: 'Entropy',
  fixed: 'Fixed phase',
  navigator: 'Navigators',
  boxcar: 'Boxcar',
};

export function ReconstructionPipelineEditor({
  plan,
  catalog,
  change,
  disabled,
}: {
  plan: RecoPlan;
  catalog: RecoCatalog;
  change: (plan: RecoPlan) => void;
  disabled: boolean;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [insertion, setInsertion] = useState('window');
  const [dragging, setDragging] = useState<string | null>(null);
  const list = useRef<HTMLOListElement>(null);
  useEffect(() => {
    if (expanded)
      list.current?.querySelector('#reco-stage-' + expanded)?.scrollIntoView({ block: 'nearest' });
  }, [expanded, plan.stages.length]);
  const update = (index: number, patch: Partial<RecoStage>) =>
    change({
      ...plan,
      stages: plan.stages.map((stage, i) => (i === index ? { ...stage, ...patch } : stage)),
    });
  const move = (from: number, to: number) => {
    if (to < 0 || to >= plan.stages.length) return;
    const stages = [...plan.stages];
    stages.splice(to, 0, stages.splice(from, 1)[0]);
    change({ ...plan, stages });
  };
  return (
    <section className="reconstruction-editor" aria-label="Reconstruction stages">
      <div className="reconstruction-input">
        <WorkbenchIcon name="reconstruct" />
        <span>Acquired k-space</span>
        <small>
          {catalog.encoding === 'sampled'
            ? 'Non-Cartesian'
            : catalog.encoding === 'cartesian'
              ? 'Cartesian'
              : 'Load raw data to inspect encoding'}
        </small>
      </div>
      <ol className="reconstruction-stages" ref={list}>
        {plan.stages.map((stage, index) => {
          const definition = catalog.stages.find((row) => row.kind === stage.kind);
          if (!definition) return null;
          const open = expanded === stage.kind;
          const conditional = stage.when && stage.when !== 'always';
          const configured = (() => {
            const value = (key: string) => recoFieldValue(stage, definition, key);
            if (stage.kind === 'window')
              return Boolean((value('kspace_window') as unknown[])?.length);
            if (stage.kind === 'shift')
              return Boolean(
                (value('image_shift') as unknown[])?.length ||
                (value('frame_image_shifts') as unknown[])?.length,
              );
            if (stage.kind === 'partial_fourier')
              return catalog.encoding !== 'sampled' && value('partial_fourier') !== 'none';
            if (stage.kind === 'epi_phase') return value('epi_phase_correction') !== 'none';
            if (stage.kind === 'product') return value('output_product') !== 'image';
            if (stage.kind === 'profile_inversion')
              return definition.available && Boolean(value('profile_inversion'));
            if (stage.kind === 'readout_regrid')
              return definition.available && Boolean(value('readout_regrid'));
            return definition.configured;
          })();
          const inactive =
            stage.enabled === false ||
            (stage.when === 'configured' && catalog.prepared && !configured) ||
            (stage.when === 'sampled' && catalog.encoding === 'cartesian');
          return (
            <li
              key={stage.kind}
              className={'reconstruction-stage' + (inactive ? ' is-inactive' : '')}
              aria-label={definition.label}
              onDragStart={(event) => {
                setDragging(stage.kind);
                event.dataTransfer.setData('text/plain', stage.kind);
              }}
              onDragEnd={() => setDragging(null)}
              onDragOver={(event) => {
                if (dragging) event.preventDefault();
              }}
              onDrop={(event) => {
                if (!dragging) return;
                event.preventDefault();
                const from = plan.stages.findIndex((item) => item.kind === dragging);
                if (from >= 0) move(from, index);
                setDragging(null);
              }}
            >
              <button
                type="button"
                className="reconstruction-stage-heading"
                draggable={!disabled}
                aria-label={definition.label + ' settings'}
                aria-expanded={open}
                aria-controls={open ? 'reco-stage-' + stage.kind : undefined}
                onClick={() => setExpanded(open ? null : stage.kind)}
              >
                <span className="reconstruction-stage-number">{index + 1}</span>
                <WorkbenchIcon
                  name={
                    definition.solver
                      ? 'reconstruct'
                      : stage.kind === 'combine'
                        ? 'layers'
                        : 'settings'
                  }
                />
                <span>
                  {definition.label}
                  {conditional && (
                    <small>
                      {stage.when === 'configured'
                        ? 'Acquisition correction'
                        : stage.when === 'sampled'
                          ? 'Non-Cartesian only'
                          : 'Cartesian only'}
                    </small>
                  )}
                </span>
                <WorkbenchIcon name="chevron" />
              </button>
              {open && (
                <div className="reconstruction-stage-settings" id={'reco-stage-' + stage.kind}>
                  <label className="check-label">
                    <input
                      type="checkbox"
                      checked={stage.enabled !== false}
                      onChange={(event) => update(index, { enabled: event.target.checked })}
                    />
                    Enable stage
                  </label>
                  {conditional && (
                    <p className="control-note">
                      {stage.when === 'configured'
                        ? 'Runs when the acquisition configures this correction, or its options below enable it.'
                        : `Runs for ${stage.when} acquisitions.`}
                    </p>
                  )}
                  {definition.reason && <p className="control-note">{definition.reason}</p>}
                  {definition.note && <p className="control-note">{definition.note}</p>}
                  {definition.solver && (
                    <label>
                      Reconstruction method
                      <select
                        aria-label="Reconstruction method"
                        value={stage.kind}
                        onChange={(event) => {
                          const replacement = catalog.stages.find(
                            (row) => row.kind === event.target.value,
                          )!;
                          const next = insertRecoStage(
                            plan,
                            replacement,
                            catalog.stages,
                            catalog.encoding,
                          );
                          if (catalog.presets.some((preset) => preset.name === plan.name))
                            next.name = replacement.label;
                          change(next);
                          setExpanded(replacement.kind);
                        }}
                      >
                        {catalog.stages
                          .filter((row) => row.solver)
                          .map((row) => (
                            <option value={row.kind} key={row.kind} disabled={!row.available}>
                              {row.label}
                            </option>
                          ))}
                      </select>
                    </label>
                  )}
                  <div className="field-grid">
                    {definition.fields.map((field) => {
                      let value = recoFieldValue(stage, definition, field.key);
                      const perAxis =
                        Array.isArray(value) && value.some((entry) => entry !== value[0]);
                      if (field.key === 'kspace_window')
                        value = perAxis
                          ? '__per_axis'
                          : Array.isArray(value)
                            ? (value[0] ?? 'none')
                            : 'none';
                      if (field.key === 'window_parameters')
                        value = perAxis ? null : Array.isArray(value) ? (value[0] ?? 0) : 0;
                      const set = (value: unknown) =>
                        update(index, {
                          options: recoFieldOptions(
                            stage,
                            definition,
                            field.key,
                            value,
                            catalog.encoding_dimensions,
                          ),
                        });
                      return (
                        <label
                          className={field.kind === 'boolean' ? 'check-label' : ''}
                          key={field.key}
                        >
                          {field.kind !== 'boolean' && field.label}
                          {field.kind === 'boolean' ? (
                            <>
                              <input
                                aria-label={field.label}
                                type="checkbox"
                                checked={Boolean(value)}
                                onChange={(event) => set(event.target.checked)}
                              />
                              {field.label}
                            </>
                          ) : field.kind === 'choice' ? (
                            <select
                              aria-label={field.label}
                              value={value == null ? '' : String(value)}
                              onChange={(event) =>
                                set(event.target.value === '' ? null : event.target.value)
                              }
                            >
                              {value === '__per_axis' && (
                                <option value="__per_axis" disabled>
                                  Per-axis acquisition settings
                                </option>
                              )}
                              {(field.nullable || definition.defaults[field.key] == null) && (
                                <option value="">Automatic</option>
                              )}
                              {field.choices.map((choice) => (
                                <option value={choice} key={choice}>
                                  {names[choice] ?? choice}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <input
                              aria-label={field.label}
                              type="number"
                              step={field.kind === 'integer' ? 1 : 'any'}
                              min={field.minimum ?? undefined}
                              max={field.maximum ?? undefined}
                              required={!field.nullable && !perAxis}
                              placeholder={perAxis ? 'Per-axis acquisition settings' : undefined}
                              value={value == null ? '' : Number(value)}
                              onChange={(event) =>
                                set(event.target.value === '' ? null : Number(event.target.value))
                              }
                            />
                          )}
                        </label>
                      );
                    })}
                  </div>
                  {!definition.fields.length && (
                    <p className="control-note">Uses the acquisition's readout-profile model.</p>
                  )}
                  <div className="reconstruction-stage-actions">
                    <button type="button" onClick={() => update(index, { options: {} })}>
                      {definition.solver ? 'Default settings' : 'Acquisition settings'}
                    </button>
                    <button
                      type="button"
                      className="icon-button"
                      title="Move stage up"
                      aria-label={`Move ${definition.label} up`}
                      disabled={index === 0}
                      onClick={() => move(index, index - 1)}
                    >
                      <WorkbenchIcon name="up" />
                    </button>
                    <button
                      type="button"
                      className="icon-button"
                      title="Move stage down"
                      aria-label={`Move ${definition.label} down`}
                      disabled={index === plan.stages.length - 1}
                      onClick={() => move(index, index + 1)}
                    >
                      <WorkbenchIcon name="chevron" />
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        change({ ...plan, stages: plan.stages.filter((_row, i) => i !== index) });
                        setExpanded(null);
                      }}
                    >
                      <WorkbenchIcon name="close" />
                      Remove stage
                    </button>
                  </div>
                </div>
              )}
            </li>
          );
        })}
      </ol>
      <div className="reconstruction-insert">
        <label>
          Add reconstruction stage
          <select
            aria-label="Add reconstruction stage"
            value={insertion}
            onChange={(event) => setInsertion(event.target.value)}
          >
            {catalog.stages.map((stage) => (
              <option
                value={stage.kind}
                key={stage.kind}
                disabled={!stage.available || plan.stages.some((row) => row.kind === stage.kind)}
              >
                {stage.label}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          disabled={
            disabled ||
            !catalog.stages.find((row) => row.kind === insertion)?.available ||
            plan.stages.some((row) => row.kind === insertion)
          }
          onClick={() => {
            const item = catalog.stages.find((row) => row.kind === insertion)!;
            change(insertRecoStage(plan, item, catalog.stages, catalog.encoding));
            setExpanded(insertion);
          }}
        >
          <WorkbenchIcon name="add" />
          Insert stage
        </button>
      </div>
    </section>
  );
}
