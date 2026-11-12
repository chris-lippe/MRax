export type ExportDraft = {
  selected: string;
  format: string;
  filename?: string;
};

export function exportSuffix(format: string) {
  return (
    ({ nifti: '.nii', dicom: '.dcm', roi_set: '.npz' } as Record<string, string>)[format] ??
    '.' + format
  );
}

export function roiExportName(name: string, format: string) {
  let safe =
    [...name]
      .map((character) =>
        character.charCodeAt(0) < 32 || /[<>:"/\\|?*]/.test(character) ? '_' : character,
      )
      .join('')
      .trim()
      .replace(/[. ]+$/, '') || 'ROI';
  if (/^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(safe)) safe = '_' + safe;
  return safe.slice(0, 180) + exportSuffix(format);
}

export function exportNameError(name: string) {
  return !name.trim() ||
    name.length > 200 ||
    /[<>:"/\\|?*]/.test(name) ||
    [...name].some((character) => character.charCodeAt(0) < 32) ||
    /[. ]$/.test(name) ||
    /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(name)
    ? 'Enter a file name without folder paths or reserved characters.'
    : '';
}

export function changeExportSuffix(name: string, previous: string, next: string) {
  if (name.toLowerCase().endsWith(exportSuffix(next))) return name;
  const suffix = exportSuffix(previous);
  return (
    (name.toLowerCase().endsWith(suffix) ? name.slice(0, -suffix.length) : name) +
    exportSuffix(next)
  );
}

export const defaultExportDraft: ExportDraft = {
  selected: '',
  format: 'npz',
};
