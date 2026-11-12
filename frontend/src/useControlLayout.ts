import { useEffect, type RefObject } from 'react';

/** Pack inspector cards into independent columns. */
export function useControlLayout(body: RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    const element = body.current;
    if (!element) return;
    const observed = new Set<HTMLElement>();
    const assigned = new Set<HTMLElement>();
    let frame = 0;
    const items = (parent: HTMLElement): HTMLElement[] =>
      [...parent.children].flatMap((child) =>
        child instanceof HTMLElement
          ? getComputedStyle(child).display === 'contents'
            ? items(child)
            : [child]
          : [],
      );
    const arrange = () => {
      frame = 0;
      const targets = new Set<HTMLElement>();
      const updates: [HTMLElement, string][] = [];
      for (const layout of element.querySelectorAll<HTMLElement>('.control-layout')) {
        const packed =
          getComputedStyle(layout).getPropertyValue('--control-columns').trim() === '2';
        layout.classList.toggle('controls-packed', packed);
        const metrics = getComputedStyle(layout);
        const row = Number.parseFloat(metrics.gridAutoRows) || 4;
        const gap = Number.parseFloat(metrics.columnGap) || 0;
        targets.add(layout);
        for (const card of items(layout)) {
          targets.add(card);
          const height = card.getBoundingClientRect().height;
          updates.push([
            card,
            packed && height > 0 ? 'span ' + Math.ceil((height + gap) / row) : '',
          ]);
        }
      }
      for (const [card, span] of updates) {
        if (card.style.gridRowEnd !== span) card.style.gridRowEnd = span;
        assigned.add(card);
      }
      for (const target of observed) {
        if (!targets.has(target)) {
          resize.unobserve(target);
          observed.delete(target);
          target.style.removeProperty('grid-row-end');
          assigned.delete(target);
        }
      }
      for (const target of targets) {
        if (!observed.has(target)) {
          resize.observe(target);
          observed.add(target);
        }
      }
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(arrange);
    };
    const resize = new ResizeObserver(schedule);
    const mutations = new MutationObserver(schedule);
    mutations.observe(element, { childList: true, subtree: true });
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      mutations.disconnect();
      assigned.forEach((card) => card.style.removeProperty('grid-row-end'));
      element
        .querySelectorAll('.controls-packed')
        .forEach((layout) => layout.classList.remove('controls-packed'));
    };
  }, [body]);
}
