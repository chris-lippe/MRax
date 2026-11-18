import type { ReactNode } from 'react';
import { ResizablePreview } from './ResizablePreview';

const defaultHeight = () => Math.round(Math.max(160, Math.min(200, window.innerHeight * 0.18)));

export function SignalPreview({ hidden, children }: { hidden: boolean; children: ReactNode }) {
  return (
    <ResizablePreview
      className="signal-preview"
      label="Resize signal preview"
      bodyId="signal-preview-body"
      storageKey="mrax.signalHeight"
      minimum={140}
      maximum={800}
      defaultHeight={defaultHeight}
      hidden={hidden}
    >
      <div id="signal-preview-body" className="signal-preview-body">
        {children}
      </div>
    </ResizablePreview>
  );
}
