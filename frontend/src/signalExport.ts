export function serializeSignalFigure(svg: SVGSVGElement | null) {
  if (!svg) throw Error('Open View → Signal curves to save a figure.');
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const originals = [svg, ...svg.querySelectorAll('*')],
    copies = [clone, ...clone.querySelectorAll('*')];
  const properties = [
    'fill',
    'fill-opacity',
    'stroke',
    'stroke-width',
    'stroke-opacity',
    'stroke-dasharray',
    'stroke-linecap',
    'stroke-linejoin',
    'font-family',
    'font-size',
    'font-weight',
    'text-anchor',
    'opacity',
  ];
  originals.forEach((element, i) => {
    const style = getComputedStyle(element);
    for (const name of properties) {
      (copies[i] as SVGElement).style.setProperty(name, style.getPropertyValue(name));
      if (copies[i].hasAttribute(name)) copies[i].setAttribute(name, style.getPropertyValue(name));
    }
  });
  clone.querySelectorAll('[data-plot-inspection]').forEach((e) => e.remove());
  clone.removeAttribute('tabindex');
  clone.removeAttribute('aria-hidden');
  clone.removeAttribute('data-aria-hidden');
  clone.removeAttribute('aria-describedby');
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('width', String(svg.viewBox.baseVal.width));
  clone.setAttribute('height', String(svg.viewBox.baseVal.height));
  const background = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  background.setAttribute('width', '100%');
  background.setAttribute('height', '100%');
  background.setAttribute('fill', getComputedStyle(svg).backgroundColor);
  clone.insertBefore(background, clone.firstChild);
  return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(clone);
}
