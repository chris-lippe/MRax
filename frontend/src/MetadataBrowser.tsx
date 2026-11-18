import {
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from 'react';
import { api, type Options, type Workspace } from './api';
import { DataEditor, Report, dataController, dataDraft, type DataDraft } from './Controls';
import { MetadataActions } from './MetadataActions';
import {
  metadataCategories,
  metadataPointer,
  metadataSource,
  searchMetadata,
  type Inspection,
  type MetadataCategory,
} from './metadata';
import {
  compileTable,
  parseData,
  printData,
  readData,
  replaceData,
  serializeData,
  type DataNode,
  type DataPath,
} from './structuredData';

type DraftEntry = { base: string; draft: DataDraft };
const draftText = (draft: DataDraft) =>
  !draft.raw && draft.node ? printData(draft.node) : draft.text;
const dirtyDraft = (entry: DraftEntry) => draftText(entry.draft) !== entry.base;
const snapshotKey = (workspace: Workspace | null) =>
  JSON.stringify([workspace?.id, workspace?.experiment, workspace?.revision]);
const categoryKey = (workspace: Workspace | null, category: MetadataCategory) =>
  JSON.stringify([workspace?.id, category === 'constants' ? '' : workspace?.experiment, category]);

function MetadataGroups({
  node,
  path,
  selected,
  onSelect,
}: {
  node: DataNode;
  path: DataPath;
  selected: DataPath;
  onSelect: (path: DataPath) => void;
}) {
  const children =
    node.kind === 'object'
      ? node.entries
      : node.kind === 'array'
        ? node.items.map((value, index) => [index, value] as const)
        : [];
  return (
    <ul>
      {children
        .filter(([, value]) => value.kind === 'object' || value.kind === 'array')
        .map(([key, value]) => {
          const child = [...path, key];
          const active = child.every((part, i) => part === selected[i]);
          return (
            <li key={key}>
              <button
                type="button"
                className={active ? 'metadata-group active' : 'metadata-group'}
                aria-current={
                  metadataPointer(selected) === metadataPointer(child) ? 'location' : undefined
                }
                aria-label={`Browse ${metadataPointer(child)}`}
                onClick={() => onSelect(child)}
              >
                {key}
                <span>
                  {value.kind === 'object'
                    ? value.entries.length
                    : value.kind === 'array'
                      ? value.items.length
                      : ''}
                </span>
              </button>
              {active && (
                <MetadataGroups node={value} path={child} selected={selected} onSelect={onSelect} />
              )}
            </li>
          );
        })}
    </ul>
  );
}

function Names({
  workspace,
  disabled,
  apply,
  onRenamed,
}: {
  workspace: Workspace;
  disabled: boolean;
  apply: (action: string, values: Options) => Promise<Workspace>;
  onRenamed: (previous: string, next: string) => void;
}) {
  const [study, setStudy] = useState(workspace.label);
  const [experiment, setExperiment] = useState(workspace.experiment);
  const [error, setError] = useState('');
  useEffect(() => setStudy(workspace.label), [workspace.label]);
  useEffect(() => setExperiment(workspace.experiment), [workspace.experiment]);
  return (
    <section className="metadata-names" aria-label="Study and experiment names">
      <h3>Names</h3>
      {(['study', 'experiment'] as const).map((kind) => (
        <form
          key={kind}
          className="name-editor"
          onSubmit={(event) => {
            event.preventDefault();
            setError('');
            void apply(`${kind}.rename`, { name: kind === 'study' ? study : experiment })
              .then((next) => {
                if (kind === 'experiment') onRenamed(workspace.experiment, next.experiment);
              })
              .catch((failure) => setError(String(failure)));
          }}
        >
          <label>
            {kind === 'study' ? 'Study name' : 'Active experiment name'}
            <input
              aria-label={kind === 'study' ? 'Study name' : 'Active experiment name'}
              value={kind === 'study' ? study : experiment}
              required
              disabled={disabled || (kind === 'experiment' && !workspace.loaded)}
              onChange={(event) =>
                kind === 'study' ? setStudy(event.target.value) : setExperiment(event.target.value)
              }
            />
          </label>
          <button
            className="secondary"
            disabled={disabled || (kind === 'experiment' && !workspace.loaded)}
          >
            Rename {kind}
          </button>
        </form>
      ))}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}

export function MetadataBrowser({
  workspace,
  active,
  busy,
  category,
  onCategoryChange,
  apply,
}: {
  workspace: Workspace | null;
  active: boolean;
  busy: boolean;
  category: MetadataCategory;
  onCategoryChange: (category: MetadataCategory) => void;
  apply: (action: string, values: Options) => Promise<Workspace>;
}) {
  const [loaded, setLoaded] = useState<{ key: string; context: string; value: Inspection } | null>(
    null,
  );
  const [loadError, setLoadError] = useState('');
  const [documents, setDocuments] = useState<Record<string, DraftEntry>>({});
  const [locations, setLocations] = useState<Record<string, DataPath>>({});
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query);
  const [searchPage, setSearchPage] = useState({ query: '', page: 0 });
  const [notice, setNotice] = useState('');
  const [applying, setApplying] = useState(false);
  const key = snapshotKey(workspace);
  const currentSnapshot = useRef(key);
  const context = JSON.stringify([workspace?.id, workspace?.experiment]);
  const documentKey = categoryKey(workspace, category);
  const info = metadataCategories.find((item) => item.id === category)!;
  useEffect(() => {
    let cancelled = false;
    currentSnapshot.current = key;
    if (!active || !workspace) return;
    setNotice('');
    setLoadError('');
    void api<Inspection>(`/api/workspaces/${workspace.id}/inspect`)
      .then((value) => {
        if (!cancelled) setLoaded({ key, context, value });
      })
      .catch((error) => {
        if (!cancelled) setLoadError(String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [active, workspace?.id, workspace?.experiment, workspace?.revision, key, context]);
  const data = loaded?.context === context ? loaded.value : null;
  const ready = Boolean(workspace && loaded?.key === key);
  const source = serializeData(metadataSource(data, category));
  const initial = useMemo(() => dataDraft(source), [source]);
  const cached = documents[documentKey];
  const entry =
    cached && (!ready || dirtyDraft(cached) || cached.base === source)
      ? cached
      : { base: source, draft: { ...initial, ...(cached?.draft.raw ? { raw: true } : {}) } };
  const setDraft: Dispatch<SetStateAction<DataDraft>> = (update) =>
    setDocuments((previous) => {
      const old = previous[documentKey];
      const current = old && (!ready || dirtyDraft(old) || old.base === source) ? old : entry;
      return {
        ...previous,
        [documentKey]: {
          base: current.base,
          draft: typeof update === 'function' ? update(current.draft) : update,
        },
      };
    });
  const controller = dataController(entry.draft, setDraft);
  const node = useMemo(() => {
    if (entry.draft.node && !entry.draft.raw) return entry.draft.node;
    try {
      return compileTable(controller.text);
    } catch {
      return undefined;
    }
  }, [entry.draft.node, entry.draft.raw, controller.text]);
  const storedPath = locations[documentKey] ?? [];
  const path = node && readData(node, storedPath) ? storedPath : [];
  const selected = node ? readData(node, path) : undefined;
  const conflict = ready && dirtyDraft(entry) && entry.base !== source;
  const disabled = busy || applying || !ready || (category === 'experiment' && !workspace?.loaded);
  const search = useMemo(() => {
    if (!active || !deferredQuery.trim() || !data) return [];
    return metadataCategories.flatMap((item) => {
      const id = categoryKey(workspace, item.id);
      const pending = documents[id];
      try {
        const root = compileTable(
          pending && dirtyDraft(pending)
            ? draftText(pending.draft)
            : serializeData(metadataSource(data, item.id)),
        );
        return searchMetadata(root, deferredQuery).map((result) => ({
          ...result,
          category: item.id,
          label: item.label,
        }));
      } catch {
        return [];
      }
    });
  }, [active, data, deferredQuery, workspace, documents]);
  const page =
    searchPage.query === deferredQuery
      ? Math.min(searchPage.page, Math.max(0, Math.ceil(search.length / 50) - 1))
      : 0;
  const navigate = (next: DataPath, nextCategory = category) => {
    onCategoryChange(nextCategory);
    setLocations((previous) => ({ ...previous, [categoryKey(workspace, nextCategory)]: next }));
    if (nextCategory === category && entry.draft.raw && node) controller.chooseFormat(false);
    setNotice('');
  };
  // New custom fields belong to metadata, not the experiment document's schema.
  const targetPath = category === 'experiment' && !path.length ? ['metadata'] : path;
  const target = node ? readData(node, targetPath) : undefined;
  const readonlyValue =
    !info.editable && selected ? parseData(printData(selected)) : metadataSource(data, category);
  return (
    <section
      className="metadata-browser"
      aria-label="Metadata browser"
      hidden={!active}
      aria-busy={busy || applying || Boolean(workspace && !ready)}
    >
      <header className="metadata-toolbar">
        <h2>Metadata</h2>
        <label className="metadata-search">
          Search metadata
          <input
            type="search"
            aria-label="Search metadata"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Field, value or unit"
            disabled={!workspace}
          />
        </label>
        {info.editable && (
          <MetadataActions
            key={documentKey}
            label={info.label}
            node={target}
            path={targetPath}
            disabled={disabled}
            quantityDefault={category === 'constants' || targetPath[0] === 'observables'}
            onChange={(value, nextPath) => {
              if (!node) return;
              setDraft({
                raw: false,
                text: controller.text,
                node: replaceData(node, targetPath, value),
                error: '',
              });
              if (nextPath) setLocations((previous) => ({ ...previous, [documentKey]: nextPath }));
              setNotice('Draft updated. Apply to save these changes.');
            }}
          />
        )}
      </header>
      {loadError && <p role="alert">{loadError}</p>}
      {!workspace ? (
        <p className="metadata-empty">Open or create a study to browse its metadata.</p>
      ) : (
        <div className="metadata-explorer">
          <nav className="metadata-navigation" aria-label="Metadata navigation">
            <div className="metadata-categories" role="group" aria-label="Metadata categories">
              {metadataCategories.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="metadata-category"
                  aria-pressed={category === item.id}
                  onClick={() => {
                    onCategoryChange(item.id);
                    setNotice('');
                  }}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {deferredQuery.trim() ? (
              <div className="metadata-results">
                <p role="status">
                  {search.length} {search.length === 1 ? 'match' : 'matches'}
                </p>
                <ul>
                  {search.slice(page * 50, (page + 1) * 50).map((result) => (
                    <li key={`${result.category}/${metadataPointer(result.path)}`}>
                      <button
                        type="button"
                        aria-label={`Open ${result.label} ${metadataPointer(result.path)}`}
                        onClick={() => navigate(result.path, result.category)}
                      >
                        <strong>
                          {result.label} · {metadataPointer(result.path)}
                        </strong>
                        <span>{result.preview}</span>
                      </button>
                    </li>
                  ))}
                </ul>
                {search.length > 50 && (
                  <div className="data-pages">
                    <button
                      className="secondary"
                      disabled={!page}
                      onClick={() => setSearchPage({ query: deferredQuery, page: page - 1 })}
                    >
                      Previous matches
                    </button>
                    <span>
                      {page + 1} / {Math.ceil(search.length / 50)}
                    </span>
                    <button
                      className="secondary"
                      disabled={(page + 1) * 50 >= search.length}
                      onClick={() => setSearchPage({ query: deferredQuery, page: page + 1 })}
                    >
                      Next matches
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <>
                <button
                  type="button"
                  className="metadata-group"
                  aria-current={!path.length ? 'location' : undefined}
                  onClick={() => navigate([])}
                >
                  All fields
                </button>
                {node && (
                  <MetadataGroups node={node} path={[]} selected={path} onSelect={navigate} />
                )}
              </>
            )}
          </nav>
          <div className="metadata-content">
            <nav className="metadata-breadcrumbs" aria-label="Metadata path">
              <ol>
                {[[], ...path.map((_, index) => path.slice(0, index + 1))].map((prefix) => (
                  <li key={metadataPointer(prefix)}>
                    <button
                      type="button"
                      aria-current={prefix.length === path.length ? 'location' : undefined}
                      onClick={() => navigate(prefix)}
                    >
                      {prefix.length ? prefix.at(-1) : info.label}
                    </button>
                  </li>
                ))}
              </ol>
            </nav>
            {!ready && !loadError && <p role="status">Loading metadata…</p>}
            {notice && <p role="status">{notice}</p>}
            {conflict && (
              <div className="metadata-conflict" role="alert">
                <p>
                  Metadata changed elsewhere. Reload the latest values or keep this draft before
                  applying it.
                </p>
                <button
                  className="secondary"
                  onClick={() =>
                    setDocuments((previous) => ({
                      ...previous,
                      [documentKey]: { base: source, draft: initial },
                    }))
                  }
                >
                  Reload latest
                </button>
                <button
                  className="secondary"
                  onClick={() =>
                    setDocuments((previous) => ({
                      ...previous,
                      [documentKey]: { ...entry, base: source },
                    }))
                  }
                >
                  Keep draft
                </button>
              </div>
            )}
            {category === 'overview' && (
              <Names
                workspace={workspace}
                disabled={busy || applying}
                apply={apply}
                onRenamed={(old, next) => {
                  const oldKey = JSON.stringify([workspace.id, old, 'experiment']),
                    nextKey = JSON.stringify([workspace.id, next, 'experiment']);
                  if (oldKey === nextKey) return;
                  setDocuments((previous) => {
                    if (!previous[oldKey]) return previous;
                    const moved = { ...previous, [nextKey]: previous[oldKey] };
                    delete moved[oldKey];
                    return moved;
                  });
                  setLocations((previous) => {
                    if (!previous[oldKey]) return previous;
                    const moved = { ...previous, [nextKey]: previous[oldKey] };
                    delete moved[oldKey];
                    return moved;
                  });
                }}
              />
            )}
            {info.editable ? (
              <>
                <div className="metadata-draft-status">
                  <span role="status">
                    {dirtyDraft(entry) ? 'Unapplied changes' : 'Up to date'}
                  </span>
                  <button
                    className="secondary"
                    disabled={disabled || !dirtyDraft(entry)}
                    onClick={() => {
                      setDocuments((previous) => ({
                        ...previous,
                        [documentKey]: { base: source, draft: initial },
                      }));
                      setNotice('Draft reverted.');
                    }}
                  >
                    Revert draft
                  </button>
                </div>
                <DataEditor
                  label={info.label}
                  controller={controller}
                  path={path}
                  disabled={disabled || conflict}
                  onApply={async (values) => {
                    if (category === 'experiment') {
                      if (
                        Object.keys(values).some(
                          (key) =>
                            !['metadata', 'observables', 'exp_type', 'model_name'].includes(key),
                        )
                      )
                        throw Error('Put custom fields inside the metadata group.');
                      for (const key of ['metadata', 'observables'])
                        if (
                          Object.hasOwn(values, key) &&
                          (!values[key] ||
                            typeof values[key] !== 'object' ||
                            Array.isArray(values[key]))
                        )
                          throw Error(`${key} must be an object.`);
                    }
                    setApplying(true);
                    try {
                      const next = await apply(
                        category === 'constants' ? 'study.globals' : 'metadata.update',
                        values,
                      );
                      const fresh = await api<Inspection>(`/api/workspaces/${next.id}/inspect`);
                      const updated = serializeData(metadataSource(fresh, category));
                      if (currentSnapshot.current === snapshotKey(next))
                        setLoaded({
                          key: snapshotKey(next),
                          context: JSON.stringify([next.id, next.experiment]),
                          value: fresh,
                        });
                      setDocuments((previous) => ({
                        ...previous,
                        [documentKey]: {
                          base: updated,
                          draft: {
                            ...dataDraft(updated),
                            raw: previous[documentKey]?.draft.raw ?? entry.draft.raw,
                          },
                        },
                      }));
                      if (currentSnapshot.current === snapshotKey(next))
                        setNotice('Metadata applied. Save the workspace to retain it on disk.');
                    } finally {
                      setApplying(false);
                    }
                  }}
                />
              </>
            ) : (
              <Report
                key={metadataPointer(path)}
                value={readonlyValue}
                label={
                  category === 'overview'
                    ? 'Study summary'
                    : category === 'arrays'
                      ? 'Array details'
                      : 'Diagnostics and provenance'
                }
              />
            )}
          </div>
        </div>
      )}
    </section>
  );
}
