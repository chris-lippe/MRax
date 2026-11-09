export type DataNode =
  | { kind: 'string'; value: string }
  | { kind: 'number'; value: string }
  | { kind: 'boolean'; value: boolean }
  | { kind: 'null' }
  | { kind: 'object'; entries: [string, DataNode][] }
  | { kind: 'array'; items: DataNode[] };

export type DataPath = (string | number)[];

export function readData(node: DataNode, path: DataPath): DataNode | undefined {
  let current: DataNode | undefined = node;
  for (const key of path) {
    if (current?.kind === 'object') current = current.entries.find(([name]) => name === key)?.[1];
    else if (current?.kind === 'array') current = current.items[Number(key)];
    else return undefined;
  }
  return current;
}

export function parseData(text: string): unknown {
  return JSON.parse(text, (_key, value: unknown) => {
    if (typeof value === 'number' && !Number.isFinite(value))
      throw Error('Numbers must be finite.');
    return value;
  });
}

export function compileTable(text: string): DataNode {
  if (text.length > 1_000_000) throw Error('This document is too large for the table view.');
  let count = 0;
  const visit = (value: unknown, depth: number): DataNode => {
    if (++count > 10_000 || depth > 16)
      throw Error('This document is too large or deeply nested for the table view.');
    if (value === null) return { kind: 'null' };
    if (typeof value === 'number') return { kind: 'number', value: String(value) };
    if (typeof value === 'string') return { kind: 'string', value };
    if (typeof value === 'boolean') return { kind: 'boolean', value };
    if (Array.isArray(value))
      return { kind: 'array', items: value.map((v) => visit(v, depth + 1)) };
    return {
      kind: 'object',
      entries: Object.entries(value as Record<string, unknown>).map(([key, v]) => [
        key,
        visit(v, depth + 1),
      ]),
    };
  };
  return visit(parseData(text), 0);
}

export function printData(node: DataNode, depth = 0): string {
  if (node.kind === 'string') return JSON.stringify(node.value);
  if (node.kind === 'number') return node.value;
  if (node.kind === 'boolean') return String(node.value);
  if (node.kind === 'null') return 'null';
  const indent = '  '.repeat(depth);
  const children =
    node.kind === 'array'
      ? node.items.map((item) => printData(item, depth + 1))
      : node.entries.map(([key, item]) => `${JSON.stringify(key)}: ${printData(item, depth + 1)}`);
  const [open, close] = node.kind === 'array' ? ['[', ']'] : ['{', '}'];
  return children.length
    ? `${open}\n${children.map((line) => `${indent}  ${line}`).join(',\n')}\n${indent}${close}`
    : open + close;
}

export function replaceData(node: DataNode, path: DataPath, replacement: DataNode): DataNode {
  if (!path.length) return replacement;
  const [key, ...rest] = path;
  if (node.kind === 'array')
    return {
      ...node,
      items: node.items.map((item, index) =>
        index === key ? replaceData(item, rest, replacement) : item,
      ),
    };
  if (node.kind === 'object')
    return {
      ...node,
      entries: node.entries.map(([name, item]) => [
        name,
        name === key ? replaceData(item, rest, replacement) : item,
      ]),
    };
  throw Error('The selected field is no longer available.');
}

export function serializeData(value: unknown): string {
  try {
    return JSON.stringify(value ?? null, null, 2) ?? String(value);
  } catch {
    return 'This value could not be serialized as JSON.';
  }
}

export function numberError(value: string): boolean {
  try {
    return typeof parseData(value) !== 'number';
  } catch {
    return true;
  }
}
