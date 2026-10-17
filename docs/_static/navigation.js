/* Fit the top-level links to the header, retaining the theme's dropdown. */
(() => {
  const initialize = () => {
    const nav = document.querySelector(".bd-header .bd-navbar-elements");
    if (!nav) return;

    let more = nav.querySelector(":scope > .dropdown");
    if (!more) {
      more = document.createElement("li");
      more.className = "nav-item dropdown";
      more.innerHTML = '<button class="btn dropdown-toggle nav-item" type="button" '
        + 'data-bs-toggle="dropdown" aria-expanded="false" aria-controls="mrax-nav-more">'
        + 'More</button><ul id="mrax-nav-more" class="dropdown-menu"></ul>';
      nav.append(more);
    }
    more.classList.add("mrax-nav-overflow");
    const menu = more.querySelector(".dropdown-menu");
    const toggle = more.querySelector("button");
    const items = [...nav.children].filter((item) => item !== more).concat([...menu.children]);
    const desktop = window.matchMedia("(min-width: 960px)");
    const width = (item) => {
      const style = getComputedStyle(item);
      return item.getBoundingClientRect().width
        + (parseFloat(style.marginLeft) || 0) + (parseFloat(style.marginRight) || 0);
    };

    const layout = () => {
      if (!desktop.matches) return;
      const focused = document.activeElement;
      const firstOverflowLink = menu.querySelector("a");
      if (toggle.getAttribute("aria-expanded") === "true") toggle.click();
      for (const item of items) {
        item.classList.add("nav-item");
        item.querySelector("a").classList.remove("dropdown-item");
        nav.insertBefore(item, more);
      }
      more.hidden = false;
      const widths = items.map(width);
      const gap = parseFloat(getComputedStyle(nav).columnGap) || 0;
      const available = nav.getBoundingClientRect().width;
      const total = widths.reduce((sum, value) => sum + value, 0)
        + gap * Math.max(0, items.length - 1);
      let visible = items.length;
      if (total > available) {
        let used = width(more);
        visible = 0;
        while (visible < items.length && used + widths[visible] + gap <= available) {
          used += widths[visible] + gap;
          visible += 1;
        }
      }
      for (const item of items.slice(visible)) {
        item.classList.remove("nav-item");
        item.querySelector("a").classList.add("dropdown-item");
        menu.append(item);
      }
      more.hidden = visible === items.length;
      const active = [...menu.children].some((item) => item.classList.contains("active"));
      more.classList.toggle("active", active);
      more.classList.toggle("current", active);
      if (focused === toggle && more.hidden && firstOverflowLink) {
        firstOverflowLink.focus({ preventScroll: true });
      } else if (focused && items.some((item) => item.contains(focused))) {
        (menu.contains(focused) ? toggle : focused).focus({ preventScroll: true });
      }
    };

    let scheduled = false;
    const schedule = () => {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        layout();
      });
    };
    if (window.ResizeObserver) new ResizeObserver(schedule).observe(nav.parentElement);
    window.addEventListener("resize", schedule);
    document.fonts?.ready.then(schedule);
    schedule();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
})();
