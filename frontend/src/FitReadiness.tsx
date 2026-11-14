import { useEffect, useRef, useState } from 'react';
import { api, type Options, type Workspace } from './api';
import type { components } from './api.generated';
import { WorkbenchIcon } from './WorkbenchIcon';

type Result = components['schemas']['FitReadinessResult'];

export function FitReadiness({
  workspace,
  options,
  disabled,
}: {
  workspace: Workspace | null;
  options: Options;
  disabled: boolean;
}) {
  const context = JSON.stringify([
    workspace?.id,
    workspace?.revision,
    workspace?.view_revision,
    options,
  ]);
  const request = useRef<AbortController | null>(null);
  const [state, setState] = useState<{ context: string; pending: boolean; result?: Result } | null>(
    null,
  );
  useEffect(() => () => request.current?.abort(), []);
  const current = state?.context === context ? state : null;
  const check = async () => {
    if (!workspace) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setState({ context, pending: true });
    try {
      const result = await api<Result>(
        '/api/workspaces/' + workspace.id + '/fit-check',
        'POST',
        { revision: workspace.revision, view_revision: workspace.view_revision, ...options },
        controller.signal,
      );
      if (request.current === controller) setState({ context, pending: false, result });
    } catch (error) {
      if (controller.signal.aborted || request.current !== controller) return;
      const message = error instanceof Error ? error.message : String(error);
      setState({
        context,
        pending: false,
        result: {
          name: '',
          model: workspace.model ?? null,
          ok: false,
          errors: [message],
          warnings: [],
          notes: [],
          report: message,
        },
      });
    }
  };
  return (
    <>
      <button
        type="button"
        className="secondary full"
        disabled={disabled || current?.pending}
        onClick={() => void check()}
      >
        <WorkbenchIcon name="info" />
        {current?.pending ? 'Checking fit readiness…' : 'Check fit readiness'}
      </button>
      {current?.result && (
        <div className="fit-readiness" role="status" aria-label="Fit readiness">
          <strong>{current.result.ok ? 'Ready to fit' : 'Needs attention'}</strong>
          {!!current.result.errors.length && (
            <ul>
              {current.result.errors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          )}
          {!!current.result.warnings.length && (
            <details>
              <summary>Warnings ({current.result.warnings.length})</summary>
              <ul>
                {current.result.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </details>
          )}
          <details>
            <summary>Check details</summary>
            <p className="fit-readiness-report">{current.result.report}</p>
          </details>
        </div>
      )}
    </>
  );
}
