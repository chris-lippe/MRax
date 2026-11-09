import { useState, type RefObject } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { WorkbenchIcon } from './WorkbenchIcon';

import { accessibilityOptions, type Accessibility, type Theme } from './preferences';
import {
  bindingError,
  defaultBindings,
  shortcutChoices,
  shortcutDefinitions,
  type Bindings,
  type ShortcutAction,
} from './shortcuts';

export function SettingsDialog({
  open,
  onOpenChange,
  trigger,
  theme,
  setTheme,
  resetPanels,
  accessibility,
  setAccessibility,
  bindings,
  setBindings,
  tutorialActive,
  startTutorial,
  endTutorial,
  disabled,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  trigger: RefObject<HTMLButtonElement | null>;
  theme: Theme;
  setTheme: (theme: Theme) => void;
  resetPanels: () => void;
  accessibility: Accessibility;
  setAccessibility: (preferences: Accessibility) => void;
  bindings: Bindings;
  setBindings: (bindings: Bindings) => void;
  tutorialActive: boolean;
  startTutorial: () => void;
  endTutorial: () => void;
  disabled: boolean;
}) {
  const [section, setSection] = useState('appearance');
  const [shortcutError, setShortcutError] = useState('');
  const sections = [
    ['appearance', 'Appearance', 'desktop'],
    ['keyboard', 'Keyboard shortcuts', 'keyboard'],
    ['accessibility', 'Accessibility', 'accessibility'],
    ['tutorial', 'Tutorial', 'tutorial'],
  ] as const;
  const changeBinding = (id: ShortcutAction, value: string) => {
    const error = bindingError(bindings, id, value);
    setShortcutError(error ?? '');
    if (!error) setBindings({ ...bindings, [id]: value });
  };
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content
          className="settings-dialog"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            const button = trigger.current;
            if (button?.getClientRects().length) button.focus();
            else document.getElementById('browse-studies')?.focus();
          }}
        >
          <div className="settings-heading">
            <Dialog.Title>
              <WorkbenchIcon name="gear" />
              Settings
            </Dialog.Title>
            <Dialog.Close className="icon-button" aria-label="Close settings">
              <WorkbenchIcon name="close" />
            </Dialog.Close>
          </div>
          <Dialog.Description>
            Appearance, keyboard access and workflow guidance.
          </Dialog.Description>
          <div className="settings-layout">
            <div
              className="settings-nav"
              role="tablist"
              aria-label="Settings sections"
              aria-orientation="vertical"
              onKeyDown={(event) => {
                const buttons = [
                  ...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]'),
                ];
                const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
                let next: number;
                if (event.key === 'ArrowDown' || event.key === 'ArrowRight')
                  next = (index + 1) % buttons.length;
                else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft')
                  next = (index - 1 + buttons.length) % buttons.length;
                else if (event.key === 'Home') next = 0;
                else if (event.key === 'End') next = buttons.length - 1;
                else return;
                event.preventDefault();
                buttons[next].focus();
              }}
            >
              {sections.map(([id, label, icon]) => (
                <button
                  key={id}
                  id={'settings-tab-' + id}
                  role="tab"
                  aria-selected={section === id}
                  aria-controls={'settings-panel-' + id}
                  tabIndex={section === id ? 0 : -1}
                  onFocus={() => setSection(id)}
                  onClick={() => setSection(id)}
                >
                  <WorkbenchIcon name={icon} />
                  {label}
                </button>
              ))}
            </div>
            <div className="settings-content">
              {sections.map(([id]) => (
                <div
                  key={id}
                  id={'settings-panel-' + id}
                  role="tabpanel"
                  aria-labelledby={'settings-tab-' + id}
                  hidden={section !== id}
                  tabIndex={0}
                >
                  {id === 'appearance' && (
                    <>
                      <section aria-label="Appearance preferences">
                        <h3>Appearance</h3>
                        <div className="theme-control" role="group" aria-label="Theme">
                          {(['system', 'light', 'dark'] as const).map((value) => (
                            <button
                              key={value}
                              type="button"
                              className="theme-option"
                              aria-label={
                                (value === 'system'
                                  ? 'System'
                                  : value === 'light'
                                    ? 'Light'
                                    : 'Dark') + ' theme'
                              }
                              aria-pressed={theme === value}
                              onClick={() => setTheme(value)}
                            >
                              <WorkbenchIcon name={`theme-${value}`} />
                              {value === 'system' ? 'System' : value === 'light' ? 'Light' : 'Dark'}
                            </button>
                          ))}
                        </div>
                      </section>
                      <section aria-label="Workspace layout preferences">
                        <h3>Workspace layout</h3>
                        <p className="hint">Drag the panel dividers to adjust their widths.</p>
                        <button className="secondary" onClick={resetPanels}>
                          Reset panel widths
                        </button>
                      </section>
                    </>
                  )}
                  {id === 'keyboard' && (
                    <section aria-label="Keyboard shortcut preferences">
                      <h3>Keyboard shortcuts</h3>
                      <p className="hint">
                        App shortcuts use Ctrl+Shift+1–9. Image bindings work only with a focused
                        image. Shortcuts do not run while typing or while a dialog is open. Unassign
                        a binding before using it for another action.
                      </p>
                      <table className="shortcut-table">
                        <caption className="visually-hidden">Editable keyboard bindings</caption>
                        <thead>
                          <tr>
                            <th>Action</th>
                            <th>Context</th>
                            <th>Binding</th>
                          </tr>
                        </thead>
                        <tbody>
                          {shortcutDefinitions.map((definition) => (
                            <tr key={definition.id}>
                              <th scope="row">{definition.label}</th>
                              <td>
                                {definition.context === 'app'
                                  ? 'Workspace'
                                  : definition.context === 'image'
                                    ? 'Focused image'
                                    : 'Segment editor'}
                              </td>
                              <td>
                                <select
                                  aria-label={definition.label + ' shortcut'}
                                  value={bindings[definition.id]}
                                  onChange={(event) =>
                                    changeBinding(definition.id, event.target.value)
                                  }
                                >
                                  <option value="">Not assigned</option>
                                  {shortcutChoices(definition.id).map((value) => (
                                    <option
                                      key={value}
                                      value={value}
                                      disabled={Boolean(
                                        bindingError(bindings, definition.id, value),
                                      )}
                                    >
                                      {value}
                                    </option>
                                  ))}
                                </select>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {shortcutError && <p role="alert">{shortcutError}</p>}
                      <p className="hint">
                        Tab, Enter, Escape, arrow keys, function keys and browser/system key
                        combinations are reserved. AltGr combinations cannot be assigned.
                      </p>
                      <details className="control-advanced">
                        <summary>Fixed interaction keys</summary>
                        <dl className="fixed-shortcuts">
                          <dt>Tab / Shift+Tab</dt>
                          <dd>Move keyboard focus.</dd>
                          <dt>Arrow keys / Home / End</dt>
                          <dd>Navigate menus and adjust sliders or panel dividers.</dd>
                          <dt>Enter</dt>
                          <dd>Activate a control or finish a polygon.</dd>
                          <dt>Escape</dt>
                          <dd>Cancel drawing or close a menu or dialog.</dd>
                          <dt>Alt-drag / middle-drag</dt>
                          <dd>Pan the image while drawing.</dd>
                        </dl>
                      </details>
                      <button
                        className="secondary"
                        onClick={() => {
                          setBindings(defaultBindings());
                          setShortcutError('');
                        }}
                      >
                        Reset keyboard bindings
                      </button>
                    </section>
                  )}
                  {id === 'accessibility' && (
                    <section aria-label="Accessibility preferences">
                      <h3>Accessibility</h3>
                      <p className="hint">
                        System preferences for reduced motion and transparency are also respected.
                      </p>
                      <div className="accessibility-options">
                        {accessibilityOptions.map(([key, title, help]) => (
                          <label className="check-label" key={key}>
                            <input
                              type="checkbox"
                              aria-label={title}
                              aria-describedby={'a11y-help-' + key}
                              checked={accessibility[key]}
                              onChange={(event) =>
                                setAccessibility({ ...accessibility, [key]: event.target.checked })
                              }
                            />
                            <span>
                              <strong>{title}</strong>
                              <small id={'a11y-help-' + key}>{help}</small>
                            </span>
                          </label>
                        ))}
                      </div>
                    </section>
                  )}
                  {id === 'tutorial' && (
                    <section aria-label="Workflow tutorial preferences">
                      <h3>Guided workflow</h3>
                      <p>
                        Work through viewing, denoising, comparison, ROI drawing, fitting and
                        analysis on a synthetic T2 study.
                      </p>
                      <p className="hint">
                        The tutorial creates a separate study. Your other studies stay open.
                        Practice results are retained when you finish.
                      </p>
                      <button className="primary" disabled={disabled} onClick={startTutorial}>
                        {tutorialActive ? 'Resume tutorial' : 'Start tutorial'}
                      </button>
                      {tutorialActive && (
                        <button className="secondary" disabled={disabled} onClick={endTutorial}>
                          End tutorial
                        </button>
                      )}
                    </section>
                  )}
                </div>
              ))}
            </div>
          </div>
          <div className="settings-footer">
            <Dialog.Close asChild>
              <button className="secondary">Done</button>
            </Dialog.Close>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
