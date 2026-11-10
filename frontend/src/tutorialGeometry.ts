type Box = { left: number; top: number; width: number; height: number };
export type ScrollCue = { direction: 'up' | 'down' | 'left' | 'right'; left: number; top: number };

export function findTutorialTarget(selector: string) {
  if (!selector) return null;
  for (const element of document.querySelectorAll<HTMLElement>(selector)) {
    const closed = element.closest('details:not([open])');
    const target = closed?.querySelector<HTMLElement>('summary') ?? element;
    if (target.getClientRects().length) return target;
  }
  return null;
}

export function measureTutorialTarget(target: HTMLElement | null): {
  target: Box | null;
  cue: ScrollCue | null;
  anchor: Box | null;
} {
  if (!target) return { target: null, cue: null, anchor: null };
  const rect = target.getBoundingClientRect();
  const clip = { left: 0, top: 0, right: window.innerWidth, bottom: window.innerHeight };
  for (let parent = target.parentElement; parent; parent = parent.parentElement) {
    const style = getComputedStyle(parent);
    const bounds = parent.getBoundingClientRect();
    if (/auto|scroll|hidden|clip/.test(style.overflowX)) {
      clip.left = Math.max(clip.left, bounds.left + parent.clientLeft);
      clip.right = Math.min(clip.right, bounds.left + parent.clientLeft + parent.clientWidth);
    }
    if (/auto|scroll|hidden|clip/.test(style.overflowY)) {
      clip.top = Math.max(clip.top, bounds.top + parent.clientTop);
      clip.bottom = Math.min(clip.bottom, bounds.top + parent.clientTop + parent.clientHeight);
    }
  }
  const left = Math.max(rect.left, clip.left);
  const top = Math.max(rect.top, clip.top);
  const width = Math.min(rect.right, clip.right) - left;
  const height = Math.min(rect.bottom, clip.bottom) - top;
  if (clip.right <= clip.left || clip.bottom <= clip.top)
    return { target: null, cue: null, anchor: null };
  // Large image surfaces may fill a viewport without fitting inside it in full.
  if (
    width > 0 &&
    height > 0 &&
    width >= Math.min(rect.width, clip.right - clip.left) - 1 &&
    height >= Math.min(rect.height, clip.bottom - clip.top) - 1
  ) {
    const paddedLeft = Math.max(clip.left, left - 3);
    const paddedTop = Math.max(clip.top, top - 3);
    return {
      target: {
        left: paddedLeft,
        top: paddedTop,
        width: Math.min(clip.right, rect.right + 3) - paddedLeft,
        height: Math.min(clip.bottom, rect.bottom + 3) - paddedTop,
      },
      cue: null,
      anchor: { left, top, width, height },
    };
  }
  const verticalMissing = height < Math.min(rect.height, clip.bottom - clip.top) - 1;
  const direction = verticalMissing
    ? rect.bottom > clip.bottom
      ? 'down'
      : 'up'
    : rect.right > clip.right
      ? 'right'
      : 'left';
  const center = (value: number, start: number, end: number) =>
    Math.max(
      start + Math.min(24, (end - start) / 2),
      Math.min(end - Math.min(24, (end - start) / 2), value),
    );
  const x =
    direction === 'left'
      ? clip.left + 24
      : direction === 'right'
        ? clip.right - 24
        : center(rect.left + rect.width / 2, clip.left, clip.right);
  const y =
    direction === 'up'
      ? clip.top + 24
      : direction === 'down'
        ? clip.bottom - 24
        : center(rect.top + rect.height / 2, clip.top, clip.bottom);
  return {
    target: null,
    cue: { direction, left: x - 18, top: y - 18 },
    anchor: { left: x - 18, top: y - 18, width: 36, height: 36 },
  };
}
