import { useEffect, useState } from 'react';

export type Theme = 'system' | 'light' | 'dark';
export const accessibilityOptions = [
  ['contrast', 'Higher contrast', 'Stronger text, borders and control outlines.'],
  ['largeText', 'Larger text', 'Increase small interface text to at least 16 px.'],
  ['largeTargets', 'Larger controls', 'Increase pointer targets for buttons and form controls.'],
  ['strongFocus', 'Stronger keyboard focus', 'Make the focused control easier to locate.'],
  ['reduceMotion', 'Reduce motion', 'Disable interface transitions and animations.'],
  ['opaquePanels', 'Opaque panels', 'Remove transparency and background blur.'],
  ['alwaysClose', 'Always show study close buttons', 'Keep hover controls visible.'],
] as const;
export type Accessibility = Record<(typeof accessibilityOptions)[number][0], boolean>;
const defaults = Object.fromEntries(
  accessibilityOptions.map(([key]) => [key, false]),
) as Accessibility;

function readAccessibility(): Accessibility {
  try {
    const saved = JSON.parse(localStorage.getItem('mrax.accessibility') ?? '{}');
    return Object.fromEntries(
      accessibilityOptions.map(([key]) => [key, saved?.[key] === true]),
    ) as Accessibility;
  } catch {
    return { ...defaults };
  }
}

function save(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Preferences remain usable when browser storage is unavailable.
  }
}

export function usePreferences() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = localStorage.getItem('mrax.theme');
      return saved === 'light' || saved === 'dark' ? saved : 'system';
    } catch {
      return 'system';
    }
  });
  const [accessibility, setAccessibility] = useState<Accessibility>(readAccessibility);
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      document.documentElement.dataset.theme =
        theme === 'system' ? (media.matches ? 'dark' : 'light') : theme;
      document.documentElement.dataset.themePreference = theme;
    };
    apply();
    media.addEventListener('change', apply);
    save('mrax.theme', theme);
    return () => media.removeEventListener('change', apply);
  }, [theme]);
  useEffect(() => {
    for (const [key] of accessibilityOptions)
      document.documentElement.dataset[key] = String(accessibility[key]);
    save('mrax.accessibility', JSON.stringify(accessibility));
  }, [accessibility]);
  return { theme, setTheme, accessibility, setAccessibility };
}
