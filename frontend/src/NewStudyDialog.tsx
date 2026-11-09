import { useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { WorkbenchIcon } from './WorkbenchIcon';

export type NewStudyKind = 'empty' | 't2' | 'multimodal';

export function NewStudyDialog({
  open,
  onOpenChange,
  onAfterClose,
  create,
  disabled,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAfterClose: () => void;
  create: (kind: NewStudyKind) => Promise<void>;
  disabled: boolean;
}) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  async function start(kind: NewStudyKind) {
    setPending(true);
    setError('');
    try {
      await create(kind);
      onOpenChange(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not create the study.');
    } finally {
      setPending(false);
    }
  }
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        if (!pending) {
          setError('');
          onOpenChange(next);
        }
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content
          className="settings-dialog new-study-dialog"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            onAfterClose();
          }}
        >
          <Dialog.Title>New study</Dialog.Title>
          <Dialog.Description>Create an empty study or open example data.</Dialog.Description>
          <form
            aria-label="New study form"
            onSubmit={(event) => {
              event.preventDefault();
              void start('empty');
            }}
          >
            <fieldset disabled={disabled || pending}>
              <button type="submit" className="secondary full">
                <WorkbenchIcon name="new" />
                Create study
              </button>
              <section aria-label="Example studies">
                <h3>Example studies</h3>
                <div className="new-study-examples">
                  <button type="button" className="secondary" onClick={() => void start('t2')}>
                    <WorkbenchIcon name="image" />
                    Load synthetic study
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => void start('multimodal')}
                  >
                    <WorkbenchIcon name="layers" />
                    Load multimodal study
                  </button>
                </div>
              </section>
            </fieldset>
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            <div className="settings-footer">
              <Dialog.Close asChild>
                <button type="button" className="secondary" disabled={pending}>
                  Cancel
                </button>
              </Dialog.Close>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
