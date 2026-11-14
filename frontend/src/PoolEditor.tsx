import { useState, type ReactNode } from 'react';
import { InspectorSection } from './InspectorSection';
import { WorkbenchIcon } from './WorkbenchIcon';

export type PoolEdit = {
  operation: 'add' | 'remove' | 'rename';
  name: string;
  template?: string;
  new_name?: string;
};

const fixedNames = new Set(['water', 'mt']);
const validName = (name: string) => /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(name);

export function PoolEditor({
  pools,
  templates,
  busy,
  onEdit,
  renderParameters,
}: {
  pools: string[];
  templates: string[];
  busy: boolean;
  onEdit: (edit: PoolEdit) => Promise<void>;
  renderParameters: (pool: string) => ReactNode;
}) {
  const [selectedTemplate, setTemplate] = useState('');
  const [newName, setNewName] = useState<string | null>(null);
  const [names, setNames] = useState<Record<string, string>>({});
  const available = templates.filter((pool) => !fixedNames.has(pool) || !pools.includes(pool));
  const template = available.includes(selectedTemplate)
    ? selectedTemplate
    : (available.find((name) => !fixedNames.has(name)) ?? available[0] ?? '');
  let suggested = template;
  for (let index = 2; pools.includes(suggested); index++) {
    const suffix = '_' + index;
    suggested = template.slice(0, 64 - suffix.length) + suffix;
  }
  const name = newName ?? suggested;
  const edit = async (change: PoolEdit) => {
    await onEdit(change);
    setNames((previous) => {
      const next = { ...previous };
      delete next[change.name];
      if (change.new_name) delete next[change.new_name];
      return next;
    });
    if (change.operation === 'add') setNewName(null);
  };
  return (
    <InspectorSection title="Pools" icon="layers" className="control-wide">
      <ul className="pool-list" aria-label="Included pools">
        {pools.map((pool) => {
          const draftName = Object.hasOwn(names, pool) ? names[pool] : pool;
          const renamed = draftName.trim();
          return (
            <li className="pool-card" key={pool}>
              <div className="pool-row">
                <input
                  aria-label={'Name for pool ' + pool}
                  value={draftName}
                  disabled={busy || fixedNames.has(pool)}
                  maxLength={64}
                  onChange={(event) => setNames((old) => ({ ...old, [pool]: event.target.value }))}
                />
                <button
                  type="button"
                  className="secondary small"
                  aria-label={'Rename pool ' + pool}
                  disabled={busy || fixedNames.has(pool) || !validName(renamed) || renamed === pool}
                  onClick={() => void edit({ operation: 'rename', name: pool, new_name: renamed })}
                >
                  Rename
                </button>
                <button
                  type="button"
                  className="secondary small"
                  aria-label={'Remove pool ' + pool}
                  title={'Remove pool ' + pool}
                  disabled={busy || pools.length <= 1}
                  onClick={() => void edit({ operation: 'remove', name: pool })}
                >
                  <WorkbenchIcon name="close" />
                </button>
              </div>
              <details className="pool-parameters" aria-label={'Parameters for pool ' + pool}>
                <summary>Parameters</summary>
                <div className="parameter-table">{renderParameters(pool)}</div>
              </details>
            </li>
          );
        })}
      </ul>
      <div className="field-grid">
        <label>
          Pool template
          <select
            aria-label="Pool template"
            value={template}
            disabled={busy}
            onChange={(event) => {
              setTemplate(event.target.value);
              setNewName(null);
            }}
          >
            {templates.map((pool) => (
              <option key={pool} disabled={!available.includes(pool)}>
                {pool}
              </option>
            ))}
          </select>
        </label>
        <label>
          New pool name
          <input
            aria-label="New pool name"
            aria-describedby="pool-name-help"
            aria-invalid={!validName(name.trim())}
            value={name}
            disabled={busy}
            maxLength={64}
            onChange={(event) => setNewName(event.target.value)}
          />
        </label>
      </div>
      <p id="pool-name-help" className="control-note">
        Use letters, digits and underscores.
      </p>
      <button
        type="button"
        className="secondary full"
        disabled={busy || !template || !validName(name.trim()) || pools.includes(name.trim())}
        onClick={() => void edit({ operation: 'add', name: name.trim(), template })}
      >
        <WorkbenchIcon name="new" /> Add pool
      </button>
    </InspectorSection>
  );
}
