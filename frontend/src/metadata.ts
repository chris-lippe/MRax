import { compileTable, parseData, printData, type DataNode, type DataPath } from './structuredData';
import type { Options } from './api';

export const metadataCategories = [
  { id: 'experiment', label: 'Experiment metadata', editable: true },
  { id: 'constants', label: 'Study constants', editable: true },
  { id: 'overview', label: 'Study overview', editable: false },
  { id: 'arrays', label: 'Arrays', editable: false },
  { id: 'diagnostics', label: 'Diagnostics', editable: false },
] as const;
export type MetadataCategory = (typeof metadataCategories)[number]['id'];
export type ExportObject = {
  id: string;
  kind: string;
  label: string;
  shape: number[];
  unit: string;
  formats: { id: string; label: string }[];
};
export type Inspection = {
  metadata: Options;
  study_globals: Options;
  summary: unknown;
  objects: ExportObject[];
  diagnostics: unknown;
  fit: unknown;
  registrations: unknown;
  drifts: unknown;
};
export const metadataSource = (data: Inspection | null, category: MetadataCategory): unknown => {
  if (!data) return {};
  if (category === 'experiment') return data.metadata;
  if (category === 'constants') return data.study_globals;
  if (category === 'overview') return data.summary;
  if (category === 'arrays') return data.objects;
  return {
    messages: data.diagnostics,
    fit: data.fit,
    registrations: data.registrations,
    drift: data.drifts,
  };
};
export const metadataPointer = (path: DataPath) =>
  path.length
    ? '/' + path.map((key) => String(key).replaceAll('~', '~0').replaceAll('/', '~1')).join('/')
    : '';

export function searchMetadata(node: DataNode, query: string) {
  const matches: { path: DataPath; preview: string }[] = [];
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return matches;
  const visit = (value: DataNode, path: DataPath) => {
    const text =
      value.kind === 'object'
        ? `${value.entries.length} fields`
        : value.kind === 'array'
          ? `${value.items.length} values`
          : value.kind === 'null'
            ? 'null'
            : String(value.value);
    const index = text.toLocaleLowerCase().indexOf(needle);
    if (path.length && (metadataPointer(path).toLocaleLowerCase().includes(needle) || index >= 0)) {
      const start = Math.max(0, index - 30);
      matches.push({
        path,
        preview: `${start ? '…' : ''}${text.slice(start, start + 150)}${text.length > start + 150 ? '…' : ''}`,
      });
    }
    if (value.kind === 'object')
      value.entries.forEach(([key, child]) => visit(child, [...path, key]));
    else if (value.kind === 'array')
      value.items.forEach((child, index) => visit(child, [...path, index]));
  };
  visit(node, []);
  return matches;
}

export type ImportMode = 'merge' | 'add';
export type MetadataImport = { path: string[]; value: DataNode }[];

function csvRows(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [],
    cell = '',
    quoted = false,
    endedQuote = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') {
          cell += '"';
          index++;
        } else {
          quoted = false;
          endedQuote = true;
        }
      } else cell += char;
    } else if (char === delimiter || char === '\n' || char === '\r') {
      row.push(cell);
      cell = '';
      endedQuote = false;
      if (char !== delimiter) {
        if (row.some((value) => value.trim())) rows.push(row);
        row = [];
        if (char === '\r' && text[index + 1] === '\n') index++;
      }
    } else if (char === '"' && !cell && !endedQuote) quoted = true;
    else {
      if (endedQuote || char === '"') throw Error('Invalid quoting in the metadata table.');
      cell += char;
    }
  }
  if (quoted) throw Error('An imported table cell has an unclosed quote.');
  row.push(cell);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

export function parseMetadataImport(text: string, format: 'json' | 'csv' | 'tsv'): MetadataImport {
  text = text.replace(/^\uFEFF/, '');
  if (format === 'json') return [{ path: [], value: compileTable(text) }];
  const [header, ...rows] = csvRows(text, format === 'csv' ? ',' : '\t');
  if (!header) throw Error('The metadata table is empty.');
  const columns = header.map((name) => name.trim().toLowerCase());
  const key = columns.includes('path') ? 'path' : columns.includes('name') ? 'name' : 'key';
  if (!columns.includes(key) || !columns.includes('value'))
    throw Error('A metadata table needs a name (or path) column and a value column.');
  if (new Set(columns).size !== columns.length) throw Error('Table headers must be unique.');
  const seen = new Set<string>();
  const paths: string[][] = [];
  return rows.map((row, index) => {
    if (row.length !== columns.length)
      throw Error(`Row ${index + 2} has the wrong number of cells.`);
    const field = (name: string) => row[columns.indexOf(name)] ?? '';
    const name = field(key).trim();
    if (!name) throw Error(`Row ${index + 2} needs a field name.`);
    if (key === 'path' && (!name.startsWith('/') || /~(?![01])/u.test(name)))
      throw Error(`Row ${index + 2}: paths must be JSON pointers such as /sequence/TE.`);
    const path =
      key === 'path'
        ? name
            .slice(1)
            .split('/')
            .map((part) => part.replaceAll('~1', '/').replaceAll('~0', '~'))
        : [name];
    const identity = JSON.stringify(path);
    if (seen.has(identity)) throw Error(`Row ${index + 2} repeats a metadata path.`);
    if (
      paths.some((previous) =>
        previous
          .slice(0, Math.min(previous.length, path.length))
          .every((key, i) => key === path[i]),
      )
    )
      throw Error(`Row ${index + 2} overlaps another metadata path.`);
    seen.add(identity);
    paths.push(path);
    const type = field('type').trim().toLowerCase();
    const raw = field('value');
    let value: unknown;
    if (type === 'string' || type === 'text') value = raw;
    else if (type === 'boolean') {
      if (!['true', 'false'].includes(raw.trim().toLowerCase()))
        throw Error(`Row ${index + 2} needs true or false.`);
      value = raw.trim().toLowerCase() === 'true';
    } else if (type === 'null') {
      if (raw.trim() && raw.trim() !== 'null')
        throw Error(`Row ${index + 2} needs null or an empty value.`);
      value = null;
    } else if (type) {
      if (!['number', 'array', 'object', 'json'].includes(type))
        throw Error(`Row ${index + 2} has an unknown type.`);
      value = parseData(raw);
      if (
        (type === 'number' && typeof value !== 'number') ||
        (type === 'array' && !Array.isArray(value)) ||
        (type === 'object' && (!value || typeof value !== 'object' || Array.isArray(value)))
      )
        throw Error(`Row ${index + 2} does not contain the requested type.`);
    } else {
      try {
        value = parseData(raw);
      } catch {
        value = raw;
      }
    }
    const unit = field('unit').trim();
    if (unit) value = { values: value, unit };
    return { path, value: compileTable(JSON.stringify(value)) };
  });
}

function mergeNode(base: DataNode, patch: DataNode, mode: ImportMode): DataNode {
  if (base.kind !== 'object' || patch.kind !== 'object') return mode === 'add' ? base : patch;
  const entries: [string, DataNode][] = [...base.entries];
  for (const [key, value] of patch.entries) {
    const index = entries.findIndex(([name]) => name === key);
    if (index < 0) entries.push([key, value]);
    else entries[index] = [key, mergeNode(entries[index][1], value, mode)];
  }
  return { kind: 'object', entries };
}

export function importMetadata(
  base: DataNode,
  updates: MetadataImport,
  mode: ImportMode,
): DataNode {
  const assign = (current: DataNode, path: string[], value: DataNode): DataNode => {
    if (!path.length) return mergeNode(current, value, mode);
    const [key, ...rest] = path;
    if (current.kind === 'object') {
      const index = current.entries.findIndex(([name]) => name === key);
      const child =
        index >= 0
          ? current.entries[index][1]
          : rest.length && /^\d+$/u.test(rest[0])
            ? { kind: 'array' as const, items: [] }
            : { kind: 'object' as const, entries: [] };
      const result = rest.length
        ? assign(child, rest, value)
        : index < 0
          ? value
          : mergeNode(child, value, mode);
      const entries: [string, DataNode][] = [...current.entries];
      if (index < 0) entries.push([key, result]);
      else entries[index] = [key, result];
      return { kind: 'object', entries };
    }
    if (
      current.kind === 'array' &&
      /^(0|[1-9]\d*)$/u.test(key) &&
      Number(key) <= current.items.length
    ) {
      const index = Number(key),
        items = [...current.items];
      const child =
        items[index] ??
        (rest.length && /^\d+$/u.test(rest[0])
          ? { kind: 'array' as const, items: [] }
          : { kind: 'object' as const, entries: [] });
      items[index] = rest.length
        ? assign(child, rest, value)
        : index === items.length
          ? value
          : mergeNode(child, value, mode);
      return { kind: 'array', items };
    }
    throw Error(
      `Cannot import through ${metadataPointer(path)}. Select an object or a valid list index.`,
    );
  };
  let result = base;
  for (const update of updates) result = assign(result, update.path, update.value);
  // Validate the combined document against the same limits as the table editor.
  compileTable(printData(result));
  return result;
}
