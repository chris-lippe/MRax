import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { WorkbenchIcon } from './WorkbenchIcon';

export const processingGroups = {
  denoise: 'Denoise',
  series: 'Time series',
  normalize: 'Normalize',
  analyze: 'Derived maps',
} as const;

export type ProcessingSection = keyof typeof processingGroups;

export const workflows = [
  {
    id: 'image',
    label: 'View',
    tasks: [
      { id: 'display', label: 'Image display', icon: 'image' },
      { id: 'overlays', label: 'Image and ROI overlays', icon: 'layers' },
      { id: 'signal', label: 'Signal curves', icon: 'timecourse' },
      { id: 'metadata', label: 'Metadata', icon: 'metadata' },
    ],
  },
  {
    id: 'reconstruct',
    label: 'Reconstruct',
    tasks: [{ id: 'reconstruction', label: 'Image reconstruction', icon: 'reconstruct' }],
  },
  {
    id: 'preprocess',
    label: 'Preprocess',
    tasks: [
      { id: 'denoising', label: 'Denoising', icon: 'denoise' },
      { id: 'normalization', label: 'Signal normalization', icon: 'normalize' },
    ],
  },
  {
    id: 'registration',
    label: 'Register',
    tasks: [
      { id: 'alignment', label: 'Image alignment', icon: 'registration' },
      { id: 'drift', label: 'Drift correction', icon: 'drift' },
    ],
  },
  {
    id: 'rois',
    label: 'Segment',
    tasks: [
      { id: 'roi-edit', label: 'Create and edit ROIs', icon: 'rois' },
      { id: 'label-edit', label: 'Edit label masks', icon: 'brush' },
      { id: 'roi-volume', label: '3D expansion', icon: 'layers' },
      { id: 'roi-combine', label: 'Combine ROIs', icon: 'layers' },
      { id: 'mask-import', label: 'Import masks and ROI sets', icon: 'import' },
    ],
  },
  { id: 'fit', label: 'Fit', tasks: [{ id: 'model-fit', label: 'Model fitting', icon: 'fit' }] },
  {
    id: 'analyze',
    label: 'Analyze',
    tasks: [
      { id: 'timecourse', label: 'ROI timecourses', icon: 'timecourse' },
      { id: 'series-processing', label: 'Time-series corrections', icon: 'series' },
      { id: 'statistics', label: 'ROI map statistics', icon: 'statistics' },
      { id: 'derived-maps', label: 'Derived maps', icon: 'maps' },
    ],
  },
] as const;

export type Workflow = (typeof workflows)[number]['id'];
export type WorkflowTask = (typeof workflows)[number]['tasks'][number]['id'];
export type WorkflowTasks = Record<Workflow, WorkflowTask>;
export const initialWorkflowTasks = Object.fromEntries(
  workflows.map((workflow) => [workflow.id, workflow.tasks[0].id]),
) as WorkflowTasks;

function menuPosition(anchor: HTMLButtonElement) {
  const bounds = anchor.getBoundingClientRect();
  return {
    left: Math.max(8, Math.min(bounds.left, window.innerWidth - 268)),
    top: bounds.bottom + 7,
  };
}

function WorkflowMenu({
  workflow,
  selected,
  anchor,
  position,
  initialIndex,
  onSelect,
  onClose,
}: {
  workflow: (typeof workflows)[number];
  selected: WorkflowTask | null;
  anchor: HTMLButtonElement;
  position: { left: number; top: number };
  initialIndex: number;
  onSelect: (task: WorkflowTask) => void;
  onClose: (restoreFocus?: boolean) => void;
}) {
  const menu = useRef<HTMLDivElement>(null);
  const items = useRef<(HTMLButtonElement | null)[]>([]);
  const search = useRef({ text: '', time: 0 });
  useEffect(() => {
    items.current[initialIndex]?.focus();
  }, [initialIndex]);
  useEffect(() => {
    const outside = (event: PointerEvent | FocusEvent) => {
      if (
        event.target instanceof Node &&
        !menu.current?.contains(event.target) &&
        !anchor.contains(event.target)
      )
        onClose(false);
    };
    const resize = () => onClose(false);
    document.addEventListener('pointerdown', outside);
    document.addEventListener('focusin', outside);
    window.addEventListener('resize', resize);
    return () => {
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('focusin', outside);
      window.removeEventListener('resize', resize);
    };
  }, [anchor, onClose]);
  return createPortal(
    <div
      ref={menu}
      className="workflow-menu"
      role="menu"
      aria-label={`${workflow.label} tasks`}
      style={position}
      onKeyDown={(event) => {
        const index = items.current.indexOf(document.activeElement as HTMLButtonElement);
        const count = workflow.tasks.length;
        let next: number;
        if (event.key === 'Escape' || event.key === 'Tab') {
          if (event.key === 'Escape') event.preventDefault();
          onClose();
          return;
        }
        if (event.key === 'ArrowDown') next = (index + 1) % count;
        else if (event.key === 'ArrowUp') next = (index + count - 1) % count;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = count - 1;
        else if (
          event.key.length === 1 &&
          event.key !== ' ' &&
          !event.ctrlKey &&
          !event.metaKey &&
          !event.altKey
        ) {
          const text =
            (event.timeStamp - search.current.time < 700 ? search.current.text : '') +
            event.key.toLowerCase();
          search.current = { text, time: event.timeStamp };
          const prefix = new Set(text).size === 1 ? text[0] : text;
          next =
            Array.from({ length: count }, (_, offset) => (index + offset + 1) % count).find(
              (candidate) => workflow.tasks[candidate].label.toLowerCase().startsWith(prefix),
            ) ?? index;
        } else return;
        event.preventDefault();
        items.current[next]?.focus();
      }}
    >
      {workflow.tasks.map((task, index) => (
        <button
          key={task.id}
          ref={(element) => {
            items.current[index] = element;
          }}
          type="button"
          role="menuitemradio"
          aria-checked={selected === task.id}
          tabIndex={-1}
          onClick={() => onSelect(task.id)}
        >
          <WorkbenchIcon name={task.icon} />
          {task.label}
        </button>
      ))}
    </div>,
    document.body,
  );
}

export function WorkflowNav({
  value,
  tasks,
  onChange,
}: {
  value: Workflow;
  tasks: WorkflowTasks;
  onChange: (value: Workflow, task?: WorkflowTask) => void;
}) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const [menu, setMenu] = useState<{
    index: number;
    initialIndex: number;
    left: number;
    top: number;
  } | null>(null);
  const openMenu = (index: number, initialIndex?: number) => {
    const anchor = buttons.current[index];
    if (!anchor) return;
    const workflow = workflows[index];
    setMenu({
      index,
      initialIndex:
        initialIndex ??
        Math.max(
          0,
          workflow.tasks.findIndex((task) => task.id === tasks[workflow.id]),
        ),
      ...menuPosition(anchor),
    });
  };
  return (
    <nav
      className="workflow-nav"
      aria-label="Workflow"
      onScroll={() =>
        setMenu((current) => {
          const anchor = current && buttons.current[current.index];
          return current && anchor ? { ...current, ...menuPosition(anchor) } : null;
        })
      }
    >
      <div role="tablist" aria-label="Workflow sections">
        {workflows.map((item, index) => (
          <button
            key={item.id}
            ref={(element) => {
              buttons.current[index] = element;
            }}
            id={`workflow-${item.id}`}
            role="tab"
            type="button"
            aria-selected={value === item.id}
            aria-controls="workflow-panel"
            aria-haspopup={item.tasks.length > 1 ? 'menu' : undefined}
            aria-expanded={item.tasks.length > 1 ? menu?.index === index : undefined}
            tabIndex={value === item.id ? 0 : -1}
            onClick={() => {
              if (item.tasks.length === 1) {
                setMenu(null);
                onChange(item.id);
              } else if (menu?.index === index) {
                setMenu(null);
                buttons.current[index]?.focus();
              } else openMenu(index);
            }}
            onKeyDown={(event) => {
              if (item.tasks.length > 1 && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
                event.preventDefault();
                openMenu(index, event.key === 'ArrowDown' ? 0 : item.tasks.length - 1);
                return;
              }
              const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
              let next: number;
              if (step) next = (index + step + workflows.length) % workflows.length;
              else if (event.key === 'Home') next = 0;
              else if (event.key === 'End') next = workflows.length - 1;
              else return;
              event.preventDefault();
              setMenu(null);
              onChange(workflows[next].id);
              buttons.current[next]?.focus();
            }}
          >
            <WorkbenchIcon name={item.id} />
            <span>{item.label}</span>
            {item.tasks.length > 1 && (
              <svg className="workflow-chevron" viewBox="0 0 16 16" aria-hidden="true">
                <path d="m4 6 4 4 4-4" />
              </svg>
            )}
          </button>
        ))}
      </div>
      {menu && buttons.current[menu.index] && (
        <WorkflowMenu
          key={menu.index}
          workflow={workflows[menu.index]}
          selected={value === workflows[menu.index].id ? tasks[value] : null}
          anchor={buttons.current[menu.index]!}
          position={{ left: menu.left, top: menu.top }}
          initialIndex={menu.initialIndex}
          onClose={(restoreFocus = true) => {
            setMenu(null);
            if (restoreFocus) buttons.current[menu.index]?.focus();
          }}
          onSelect={(task) => {
            onChange(workflows[menu.index].id, task);
            setMenu(null);
            buttons.current[menu.index]?.focus();
          }}
        />
      )}
    </nav>
  );
}
