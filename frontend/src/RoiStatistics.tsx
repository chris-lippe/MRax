import { useEffect, useState } from 'react';
import { api, type Workspace } from './api';
import { InspectorSection } from './InspectorSection';

type Row = { roi: string; stat: string; value: number | null; unit?: string };
const stats = [
  ['mean', 'Mean'],
  ['std', 'SD'],
  ['median', 'Median'],
  ['count', 'Valid n'],
] as const;
const format = (value: number | null | undefined) =>
  value == null ? '—' : Number(value.toPrecision(5)).toString();

export function RoiStatistics({
  workspace: w,
  active,
}: {
  workspace: Workspace | null;
  active: boolean;
}) {
  const [map, setMap] = useState('');
  const [loaded, setLoaded] = useState<{ key: string; rows: Row[] } | null>(null);
  const [error, setError] = useState('');
  const selected = w?.maps.some((m) => m.name === map) ? map : (w?.active_map ?? '');
  const key = JSON.stringify([w?.id, w?.experiment, w?.revision, selected]);
  useEffect(() => {
    let stopped = false;
    setError('');
    if (active && w && selected)
      void api<Row[]>(
        `/api/workspaces/${w.id}/roi-statistics?map_name=${encodeURIComponent(selected)}`,
      )
        .then((rows) => {
          if (!stopped) setLoaded({ key, rows });
        })
        .catch((failure) => {
          if (!stopped) setError(String(failure));
        });
    return () => {
      stopped = true;
    };
  }, [w?.id, w?.experiment, w?.revision, selected, key, active]);
  const rows = loaded?.key === key ? loaded.rows : [];
  return (
    <section className="control-layout roi-statistics" aria-label="ROI map statistics">
      <InspectorSection
        title="Map selection"
        icon="maps"
        help="Statistics use enabled masks across the entire map, independently of display visibility."
      >
        <label>
          Statistics map
          <select
            aria-label="Statistics map"
            value={selected}
            onChange={(event) => setMap(event.target.value)}
            disabled={!w?.maps.length}
          >
            {!w?.maps.length && <option value="">Fit or compute a map first</option>}
            {w?.maps.map((m) => (
              <option key={String(m.name)} value={String(m.name)}>
                {String(m.name)}
                {m.unit ? ` (${String(m.unit)})` : ''}
              </option>
            ))}
          </select>
        </label>
      </InspectorSection>
      {error && <p role="alert">{error}</p>}
      <InspectorSection title="Statistics" icon="statistics" className="control-wide">
        {rows.length > 0 ? (
          <div className="statistics-table">
            <table aria-label="ROI map statistics">
              <thead>
                <tr>
                  <th>ROI</th>
                  {stats.map(([id, label]) => (
                    <th key={id}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from(new Set(rows.map((row) => row.roi))).map((roi) => (
                  <tr key={roi}>
                    <th scope="row">{roi}</th>
                    {stats.map(([stat]) => (
                      <td key={stat}>
                        {format(rows.find((row) => row.roi === roi && row.stat === stat)?.value)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="control-note">Enabled ROIs and a scalar map are needed.</p>
        )}
      </InspectorSection>
    </section>
  );
}
