import { useState } from 'react';
import { comparisonNavigation, type ComparisonSettings, type Workspace } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';

export function PipelineComparison({
  workspace: w,
  disabled,
  compare,
}: {
  workspace: Workspace;
  disabled: boolean;
  compare: (settings: ComparisonSettings) => void;
}) {
  const [target, setTarget] = useState('');
  const [result, setResult] = useState('image');
  const current = w.pipeline.choices.find((choice) => choice.active);
  const others = w.pipeline.choices.filter(
    (choice) => !choice.active && (choice.has_image || choice.maps.length),
  );
  if (!current || !others.length || (!current.has_image && !current.maps.length)) return null;
  const other = others.find((choice) => choice.id === target) ?? others[0];
  const maps = current.maps.flatMap((name) => {
    const matches = other.maps.includes(name)
      ? [name]
      : other.maps.filter(
          (candidate) => candidate.split(' / ').at(-1) === name.split(' / ').at(-1),
        );
    return matches.map((candidate) => [
      JSON.stringify([name, candidate]),
      name === candidate ? name : name + ' / ' + candidate,
    ]);
  });
  const options = [
    ...(current.has_image && other.has_image ? [['image', 'Processed images']] : []),
    ...maps.map(([id, label]) => ['map:' + id, label]),
  ];
  const selected = options.some(([id]) => id === result) ? result : options[0]?.[0];
  function open(mode: 'split' | 'overlay') {
    if (!selected) return;
    const isMap = selected.startsWith('map:');
    const mapNames = isMap ? (JSON.parse(selected.slice(4)) as [string, string]) : [null, null];
    const settings: ComparisonSettings = {
      ...w.scene.comparison,
      enabled: true,
      mode,
      frame: null,
      plane: null,
      indices: null,
      left: {
        experiment: w.experiment!,
        pipeline_id: current!.id,
        per_frame: false,
        kind: isMap ? 'map' : 'processed',
        map_name: mapNames[0],
        frame: null,
        component: 'auto',
        colormap: isMap ? 'viridis' : 'gray',
      },
      right: {
        experiment: w.experiment!,
        pipeline_id: other.id,
        per_frame: false,
        kind: isMap ? 'map' : 'processed',
        map_name: mapNames[1],
        frame: null,
        component: 'auto',
        colormap: isMap ? 'viridis' : 'gray',
      },
    };
    const navigation = comparisonNavigation(w, settings);
    compare({
      ...settings,
      plane: navigation.plane,
      indices: navigation.indices,
      frame: navigation.frame,
    });
  }
  return (
    <div className="pipeline-comparison" role="group" aria-label="Compare pipeline results">
      <span>Compare {current.name} with</span>
      <select
        aria-label="Comparison pipeline"
        disabled={disabled}
        value={other.id}
        onChange={(event) => setTarget(event.target.value)}
      >
        {others.map((choice) => (
          <option key={choice.id} value={choice.id}>
            {choice.name}
          </option>
        ))}
      </select>
      <select
        aria-label="Pipeline comparison result"
        disabled={disabled || !options.length}
        value={selected ?? ''}
        onChange={(event) => setResult(event.target.value)}
      >
        {!options.length && <option value="">No matching results</option>}
        {options.map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
      <button className="secondary" disabled={disabled || !selected} onClick={() => open('split')}>
        <WorkbenchIcon name="split" />
        Split
      </button>
      <button
        className="secondary"
        disabled={disabled || !selected}
        onClick={() => open('overlay')}
      >
        <WorkbenchIcon name="layers" />
        Overlay
      </button>
    </div>
  );
}
