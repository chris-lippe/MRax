import { useEffect, useRef, type RefObject } from 'react';
import { MIN_ZOOM, MAX_ZOOM, type ImageDisplay } from './useImageDisplay';

export const imageNavigationHelp =
  'Scroll to zoom. Drag to pan. Alt-drag or middle-drag pans while drawing. Double-click to fit the image. Keyboard bindings are available in Settings.';

export function useImageNavigation(
  viewport: RefObject<HTMLDivElement | null>,
  display: ImageDisplay,
) {
  const current = useRef(display);
  useEffect(() => {
    current.current = display;
  });
  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      const view = current.current;
      const delta =
        event.deltaY *
        (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? element.clientHeight : 1);
      const zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, view.zoom * Math.exp(-delta * 0.002)));
      const box = element.getBoundingClientRect();
      const anchor = [
        event.clientX - box.left - box.width / 2,
        event.clientY - box.top - box.height / 2,
      ];
      view.setZoom(zoom);
      view.setPan(
        zoom === MIN_ZOOM
          ? [0, 0]
          : [
              anchor[0] - ((anchor[0] - view.pan[0]) * zoom) / view.zoom,
              anchor[1] - ((anchor[1] - view.pan[1]) * zoom) / view.zoom,
            ],
      );
    };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => {
      element.removeEventListener('wheel', wheel);
    };
  }, [viewport]);
}
