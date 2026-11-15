import type { ButtonHTMLAttributes } from 'react';
import { WorkbenchIcon } from './WorkbenchIcon';

export function WorkflowButton(
  props: Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children' | 'className' | 'type'>,
) {
  return (
    <button {...props} type="button" className="secondary control-wide workflow-library-trigger">
      <WorkbenchIcon name="pipeline" />
      Workflows
    </button>
  );
}
