import { WorkbenchIcon } from './WorkbenchIcon';

export function StudyHeader() {
  return (
    <div className="header-data-controls">
      <h2>
        <WorkbenchIcon name="folder" />
        <span>Data &amp; Studies</span>
      </h2>
      <span className="header-panel-divider" aria-hidden="true" />
    </div>
  );
}
