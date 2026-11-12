import { useEffect, useState, type ComponentProps, type ReactNode } from 'react';
import { WorkbenchIcon } from './WorkbenchIcon';

export function InspectorChoice({
  label,
  caption = label,
  value,
  choices,
  onChange,
}: {
  label: string;
  caption?: string;
  value: string;
  choices: readonly (readonly [string, string])[];
  onChange: (value: string) => void;
}) {
  return (
    <label>
      {caption}
      <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>
        {!choices.some(([key]) => key === value) && (
          <option value={value}>{value ? 'Custom (see advanced settings)' : 'Loading…'}</option>
        )}
        {choices.map(([key, title]) => (
          <option key={key} value={key}>
            {title}
          </option>
        ))}
      </select>
    </label>
  );
}

export function InspectorGroup({
  title,
  icon,
  children,
  revealKey,
  onOpenChange,
  wide = true,
}: {
  title: string;
  icon: ComponentProps<typeof WorkbenchIcon>['name'];
  children: ReactNode;
  revealKey?: string;
  onOpenChange?: (open: boolean) => void;
  wide?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    if (revealKey) setExpanded(true);
  }, [revealKey]);
  return (
    <details
      className={'control-advanced control-group' + (wide ? ' control-wide' : '')}
      open={expanded}
      onToggle={(event) => {
        setExpanded(event.currentTarget.open);
        onOpenChange?.(event.currentTarget.open);
      }}
    >
      <summary>
        <WorkbenchIcon name={icon} />
        {title}
      </summary>
      <div className="control-group-body">{children}</div>
    </details>
  );
}

export function InspectorSection({
  title,
  icon,
  help,
  children,
  className = '',
}: {
  title: string;
  icon: ComponentProps<typeof WorkbenchIcon>['name'];
  help?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={'control-section ' + className} aria-label={title}>
      <header className="control-section-heading">
        <h3>
          <WorkbenchIcon name={icon} />
          {title}
        </h3>
        {help && (
          <details className="control-help">
            <summary aria-label={'About ' + title} title={'About ' + title}>
              <WorkbenchIcon name="info" />
            </summary>
            <p>{help}</p>
          </details>
        )}
      </header>
      <div className="control-section-body">{children}</div>
    </section>
  );
}
