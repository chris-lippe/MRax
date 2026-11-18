import { useState } from 'react';
import type { Options, Workspace } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';

export function PipelineSelector({
  workspace: w,
  disabled,
  run,
}: {
  workspace: Workspace;
  disabled: boolean;
  run: (operation: string, options?: Options) => Promise<void>;
}) {
  const [manage, setManage] = useState(false);
  const current = w.pipeline.choices.find((choice) => choice.id === w.pipeline.id);
  const [name, setName] = useState(current?.name ?? 'Pipeline 1');
  return (
    <div className="pipeline-selector">
      <WorkbenchIcon name="pipeline" />
      <label>
        <span className="visually-hidden">Active pipeline</span>
        <select
          aria-label="Active pipeline"
          disabled={disabled}
          value={w.pipeline.id}
          onChange={(event) => {
            setManage(false);
            void run('pipeline', { action: 'select', pipeline_id: event.target.value });
          }}
        >
          {w.pipeline.choices.map((choice) => (
            <option key={choice.id} value={choice.id}>
              {choice.name}
              {choice.cached ? ' · Results' : ''}
            </option>
          ))}
        </select>
      </label>
      <button
        className="icon-button"
        aria-label="New pipeline"
        title="New pipeline"
        disabled={disabled || w.pipeline.choices.length >= 16}
        onClick={() => {
          setManage(false);
          void run('pipeline', { action: 'create' });
        }}
      >
        <WorkbenchIcon name="add" />
      </button>
      <button
        className="icon-button"
        aria-label="Duplicate pipeline"
        title="Duplicate pipeline"
        disabled={disabled || w.pipeline.choices.length >= 16}
        onClick={() => {
          setManage(false);
          void run('pipeline', { action: 'duplicate' });
        }}
      >
        <WorkbenchIcon name="copy" />
      </button>
      <button
        className="icon-button"
        aria-label="Pipeline name and removal"
        title="Pipeline name and removal"
        aria-expanded={manage}
        disabled={disabled}
        onClick={() => {
          setName(current?.name ?? 'Pipeline 1');
          setManage(!manage);
        }}
      >
        <WorkbenchIcon name="settings" />
      </button>
      {manage && (
        <div className="pipeline-manage" role="group" aria-label="Pipeline name and removal">
          <label>
            Name
            <input
              aria-label="Pipeline name"
              value={name}
              maxLength={80}
              disabled={disabled}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <button
            className="secondary"
            disabled={disabled || !name.trim()}
            onClick={() => {
              void run('pipeline', { action: 'rename', name });
              setManage(false);
            }}
          >
            Rename
          </button>
          <button
            className="secondary"
            disabled={disabled || w.pipeline.choices.length < 2}
            onClick={() => {
              if (window.confirm('Delete ' + current?.name + ' and its cached results?')) {
                void run('pipeline', { action: 'delete' });
                setManage(false);
              }
            }}
          >
            Delete pipeline
          </button>
        </div>
      )}
    </div>
  );
}
