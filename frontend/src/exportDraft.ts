export type ExportDraft = {
  selected: string;
  format: string;
};

export const defaultExportDraft: ExportDraft = {
  selected: '',
  format: 'npz',
};
