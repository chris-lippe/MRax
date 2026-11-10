import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { api } from './api';
import { WorkbenchIcon } from './WorkbenchIcon';

export type FileSelection = {
  root: string;
  path: string;
  name: string;
  parent: string;
  directory: boolean;
  exists: boolean;
  writable: boolean;
  separator: string;
  overwrite?: boolean;
};
type Place = { id: string; label: string; path: string; writable: boolean; mounted?: boolean };
type Entry = { name: string; path: string; directory: boolean; bytes: number | null };
type Listing = FileSelection & { entries: Entry[]; more: boolean };
type Favorite = { place: string; path: string; label: string };
type FolderLocation = { root: string; path: string };
type ChooseOptions = {
  title: string;
  mode: 'open' | 'file' | 'folder' | 'save-file';
  suggestedName?: string;
  extensions?: string[];
  allowOverwrite?: boolean;
  writable?: boolean;
  initial?: { root: string; path: string };
};
type Choose = (options: ChooseOptions) => Promise<FileSelection | null>;
const FileSystemContext = createContext<Choose | null>(null);
const query = (root: string, path: string) => new URLSearchParams({ root, path }).toString();
const placeKey = (place: Place) => JSON.stringify([place.path, place.label]);
const favoritesKey = 'mrax.fileFavorites';

function readFavorites(): Favorite[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(favoritesKey) ?? '[]');
    return Array.isArray(value)
      ? value
          .filter(
            (item): item is Favorite =>
              item &&
              typeof item.place === 'string' &&
              typeof item.path === 'string' &&
              typeof item.label === 'string',
          )
          .slice(0, 100)
      : [];
  } catch {
    return [];
  }
}

function placeIcon(place: Place) {
  if (place.mounted || /^(drive|volume)/.test(place.id)) return 'drive';
  switch (place.id) {
    case 'home':
    case 'desktop':
    case 'documents':
    case 'downloads':
    case 'music':
    case 'videos':
      return place.id;
    case 'pictures':
      return 'image';
    default:
      return 'folder';
  }
}

function fileSize(bytes: number | null) {
  if (bytes === null) return '';
  if (bytes < 1024) return new Intl.NumberFormat('en').format(bytes) + ' B';
  const unit = Math.min(3, Math.floor(Math.log(bytes) / Math.log(1024)));
  return (
    new Intl.NumberFormat('en', { maximumFractionDigits: 1 }).format(bytes / 1024 ** unit) +
    ' ' +
    ['B', 'KB', 'MB', 'GB'][unit]
  );
}

export function useFileSystem() {
  const choose = useContext(FileSystemContext);
  if (!choose) throw new Error('File chooser is unavailable');
  return choose;
}

export function FileSystemProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<ChooseOptions | null>(null);
  const [places, setPlaces] = useState<Place[]>([]);
  const [root, setRoot] = useState('');
  const [listing, setListing] = useState<Listing | null>(null);
  const [location, setLocation] = useState('');
  const [selected, setSelected] = useState<Entry | null>(null);
  const [name, setName] = useState('');
  const [filter, setFilter] = useState('');
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [replacement, setReplacement] = useState<FileSelection | null>(null);
  const [newFolder, setNewFolder] = useState<string | null>(null);
  const [favorites, setFavorites] = useState(readFavorites);
  const [history, setHistory] = useState<{ entries: FolderLocation[]; index: number }>({
    entries: [],
    index: -1,
  });
  const address = useRef<HTMLInputElement | null>(null);
  const resolve = useRef<((value: FileSelection | null) => void) | null>(null);
  const origin = useRef<HTMLElement | null>(null);
  const focusWait = useRef<MutationObserver | null>(null);
  const last = useRef<{ root: string; path: string } | null>(null);
  const generation = useRef(0);
  const saving = request?.mode.startsWith('save-');
  const folderMode = request?.mode === 'folder';
  const currentPlace = places.find((place) => place.id === root);
  const favorite = favorites.find(
    (item) =>
      item.path === listing?.path &&
      (currentPlace?.path || item.place === (currentPlace && placeKey(currentPlace))),
  );
  const visibleEntries =
    listing?.entries.filter(
      (entry) =>
        entry.directory ||
        (!folderMode &&
          (!request?.extensions?.length ||
            request.extensions.some((ext) => entry.name.toLowerCase().endsWith(ext)))),
    ) ?? [];

  function saveFavorites(next: Favorite[]) {
    setFavorites(next);
    try {
      localStorage.setItem(favoritesKey, JSON.stringify(next));
    } catch {
      /* Keep session shortcuts when storage is unavailable. */
    }
  }
  function toggleFavorite() {
    if (!listing || !currentPlace) return;
    saveFavorites(
      favorite
        ? favorites.filter((item) => item !== favorite)
        : [
            ...favorites,
            {
              place: placeKey(currentPlace),
              path: listing.path,
              label: listing.name || currentPlace.label,
            },
          ],
    );
  }

  async function browse(key: string, path: string, page = 0, search = filter) {
    const serial = ++generation.current;
    setBusy(true);
    setError('');
    setListing(null);
    setSelected(null);
    try {
      const result = await api<Listing>(
        '/api/files/list?' +
          query(key, path) +
          '&' +
          new URLSearchParams({
            offset: String(page),
            search,
          }).toString(),
      );
      if (serial !== generation.current) return null;
      setRoot(key);
      setListing(result);
      setLocation(result.path);
      setOffset(page);
      setSelected(null);
      last.current = { root: key, path: result.path };
      return result;
    } catch (failure) {
      if (serial === generation.current)
        setError(failure instanceof Error ? failure.message : 'Cannot open this folder');
      return null;
    } finally {
      if (serial === generation.current) setBusy(false);
    }
  }
  async function navigate(key: string, path: string, historyIndex?: number) {
    setFilter('');
    setNewFolder(null);
    if (!saving) setName('');
    const result = await browse(key, path, 0, '');
    if (result)
      setHistory((current) => {
        if (historyIndex !== undefined) return { ...current, index: historyIndex };
        const previous = current.entries[current.index];
        if (previous?.root === key && previous.path === result.path) return current;
        const entries = [
          ...current.entries.slice(0, current.index + 1),
          { root: key, path: result.path },
        ];
        return { entries, index: entries.length - 1 };
      });
    return result;
  }
  function moveHistory(step: number) {
    const index = history.index + step;
    const target = history.entries[index];
    if (target && !busy) void navigate(target.root, target.path, index);
  }
  function selectEntry(entry: Entry) {
    setSelected(entry);
    if (!saving || entry.directory === folderMode) setName(entry.name);
  }
  const choose: Choose = (options) => {
    focusWait.current?.disconnect();
    if (resolve.current) return Promise.reject(new Error('Close the current file dialog first'));
    const opening = ++generation.current;
    origin.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setRequest(options);
    setListing(null);
    setSelected(null);
    setName(options.suggestedName ?? '');
    setError('');
    setFilter('');
    setReplacement(null);
    setNewFolder(null);
    setHistory({ entries: [], index: -1 });
    setBusy(true);
    void api<Place[]>('/api/files/roots')
      .then(async (found) => {
        if (opening !== generation.current) return;
        setPlaces(found);
        const initial = options.initial ?? last.current;
        const place = found.find((item) => item.id === initial?.root) ?? found[0];
        if (!place) {
          setError('No file locations are available');
          setBusy(false);
          return;
        }
        const path = initial?.root === place.id ? initial.path : place.path;
        const result = await browse(place.id, path, 0, '');
        if (result) setHistory({ entries: [{ root: place.id, path: result.path }], index: 0 });
      })
      .catch((failure) => {
        if (opening !== generation.current) return;
        setError(String(failure));
        setBusy(false);
      });
    return new Promise((done) => {
      resolve.current = done;
    });
  };
  function finish(value: FileSelection | null) {
    generation.current++;
    resolve.current?.(value);
    resolve.current = null;
    setRequest(null);
    setReplacement(null);
  }
  // Invalidate pending directory reads if the provider is removed.
  useEffect(
    () => () => {
      generation.current++;
      focusWait.current?.disconnect();
      resolve.current?.(null);
    },
    [],
  );
  const childPath = (value: string) =>
    (listing?.path ? listing.path.replace(/[\\/]$/, '') + listing.separator : '') + value;
  async function go() {
    const serial = ++generation.current;
    setError('');
    setBusy(true);
    setListing(null);
    setSelected(null);
    try {
      const result = await api<FileSelection>('/api/files/stat?' + query(root, location));
      if (serial !== generation.current) return;
      if (result.directory) await navigate(root, result.path);
      else if (result.exists || saving) {
        if (await navigate(root, result.parent)) {
          setName(result.name);
          if (result.exists) setSelected({ ...result, bytes: null });
        }
      } else setError('The selected file or folder does not exist');
    } catch (failure) {
      if (serial === generation.current) setError(String(failure));
    } finally {
      if (serial === generation.current) setBusy(false);
    }
  }
  async function confirm(entry?: Entry) {
    if (!request || !listing) return;
    const serial = ++generation.current;
    setBusy(true);
    setError('');
    try {
      let filename = name.trim();
      if (
        saving &&
        !folderMode &&
        request.extensions?.length &&
        !request.extensions.some((ext) => filename.toLowerCase().endsWith(ext))
      )
        filename += request.extensions[0];
      const target = saving
        ? childPath(filename)
        : (entry?.path ?? (filename ? childPath(filename) : (selected?.path ?? listing.path)));
      if (saving && (!name.trim() || /[\\/]/.test(name) || ['.', '..'].includes(name))) {
        setError('Enter a file or folder name without path separators');
        return;
      }
      const result = await api<FileSelection>('/api/files/stat?' + query(root, target));
      if (serial !== generation.current) return;
      if (request.writable && !result.writable) throw new Error('This location is read-only');
      if (!saving) {
        if (!result.exists) throw new Error('The selected file or folder no longer exists');
        if (request.mode === 'file' && result.directory) {
          await navigate(root, result.path);
          return;
        }
        if (request.mode === 'folder' && !result.directory) throw new Error('Choose a folder');
        if (
          !result.directory &&
          request.extensions?.length &&
          !request.extensions.some((ext) => result.name.toLowerCase().endsWith(ext))
        ) {
          throw new Error(
            'Choose a file in one of these formats: ' + request.extensions.join(', '),
          );
        }
      } else {
        if (!result.writable) throw new Error('This location is read-only');
        if (result.exists && result.directory !== folderMode)
          throw new Error(
            folderMode ? 'A file already uses this name' : 'A folder already uses this name',
          );
        if (result.exists) {
          if (request.allowOverwrite === false)
            throw new Error(
              'Choose a new name. Existing metadata and text files are not replaced.',
            );
          setReplacement(result);
          return;
        }
      }
      finish(result);
    } catch (failure) {
      if (serial === generation.current)
        setError(failure instanceof Error ? failure.message : 'Cannot select this path');
    } finally {
      if (serial === generation.current) setBusy(false);
    }
  }
  return (
    <FileSystemContext.Provider value={choose}>
      {children}
      <Dialog.Root
        open={request !== null}
        onOpenChange={(open) => {
          if (!open) finish(null);
        }}
      >
        <Dialog.Portal>
          <Dialog.Overlay className="drawer-overlay filesystem-overlay" />
          <Dialog.Content
            className="file-system-dialog"
            onKeyDown={(event) => {
              if (event.altKey && ['ArrowLeft', 'ArrowRight', 'ArrowUp'].includes(event.key)) {
                event.preventDefault();
                if (event.key === 'ArrowUp') {
                  if (listing && !busy) void navigate(root, listing.parent);
                } else moveHistory(event.key === 'ArrowLeft' ? -1 : 1);
              } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'l') {
                event.preventDefault();
                address.current?.focus();
                address.current?.select();
              }
            }}
            onCloseAutoFocus={(event) => {
              event.preventDefault();
              const trigger = origin.current;
              const serial = generation.current;
              requestAnimationFrame(() => {
                if (!trigger?.isConnected || serial !== generation.current) return;
                // Loading a selected workspace temporarily disables its initiating button.
                const restore = () => {
                  if (serial !== generation.current || !trigger.isConnected) {
                    focusWait.current?.disconnect();
                    return;
                  }
                  if (trigger.matches(':disabled')) return;
                  focusWait.current?.disconnect();
                  focusWait.current = null;
                  if (
                    document.activeElement === document.body ||
                    document.activeElement === trigger
                  )
                    trigger.focus({ preventScroll: true });
                };
                focusWait.current = new MutationObserver(restore);
                focusWait.current.observe(trigger, {
                  attributes: true,
                  attributeFilter: ['disabled'],
                });
                restore();
              });
            }}
          >
            <Dialog.Title>{request?.title ?? 'Choose a file'}</Dialog.Title>
            <Dialog.Description className="visually-hidden">
              Select a file or folder. Double-click a folder or press Enter to open it. Use the
              address bar, standard folders or favorites to navigate.
            </Dialog.Description>
            <Dialog.Close className="drawer-close" aria-label="Close file dialog">
              <WorkbenchIcon name="close" />
            </Dialog.Close>
            <div className="file-navigation">
              <div className="file-history" role="group" aria-label="Folder navigation">
                <button
                  className="secondary file-icon-button"
                  aria-label="Back"
                  title="Back (Alt+Left)"
                  disabled={busy || history.index <= 0}
                  onClick={() => moveHistory(-1)}
                >
                  <WorkbenchIcon name="back" />
                </button>
                <button
                  className="secondary file-icon-button"
                  aria-label="Forward"
                  title="Forward (Alt+Right)"
                  disabled={busy || history.index >= history.entries.length - 1}
                  onClick={() => moveHistory(1)}
                >
                  <WorkbenchIcon name="forward" />
                </button>
                <button
                  className="secondary file-icon-button"
                  aria-label="Parent folder"
                  title="Up one folder (Alt+Up)"
                  disabled={busy || !listing || listing.path === listing.parent}
                  onClick={() => void navigate(root, listing!.parent)}
                >
                  <WorkbenchIcon name="up" />
                </button>
              </div>
              <form
                className="file-location"
                onSubmit={(event) => {
                  event.preventDefault();
                  void go();
                }}
              >
                <WorkbenchIcon name="folder" />
                <label className="visually-hidden" htmlFor="file-location">
                  Location
                </label>
                <input
                  ref={address}
                  id="file-location"
                  aria-label="Location"
                  value={location}
                  onChange={(event) => setLocation(event.target.value)}
                  spellCheck={false}
                />
                <button
                  className="secondary file-icon-button"
                  aria-label="Go"
                  title="Go to address"
                  disabled={!root || busy}
                >
                  <WorkbenchIcon name="forward" />
                </button>
              </form>
              <form
                className="file-search"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (listing) void browse(root, listing.path, 0);
                }}
              >
                <label className="visually-hidden" htmlFor="file-search">
                  Find in folder
                </label>
                <input
                  id="file-search"
                  aria-label="Find in folder"
                  placeholder="Search this folder"
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                />
                <button
                  className="secondary file-icon-button"
                  aria-label="Find"
                  title="Search this folder"
                  disabled={!listing || busy}
                >
                  <WorkbenchIcon name="search" />
                </button>
              </form>
            </div>
            <div className="file-system-layout">
              <nav aria-label="File locations" className="file-places">
                <section aria-label="Favorite folders">
                  <h3 className="file-places-heading">Favorites</h3>
                  {favorites.map((item, index) => {
                    const place = places.find((place) => placeKey(place) === item.place);
                    if (!place) return null;
                    return (
                      <div className="file-favorite" key={item.place + item.path}>
                        <button
                          className={
                            root === place.id && listing?.path === item.path ? 'selected' : ''
                          }
                          aria-current={
                            root === place.id && listing?.path === item.path
                              ? 'location'
                              : undefined
                          }
                          title={item.path || place.label}
                          disabled={busy}
                          onClick={() => void navigate(place.id, item.path)}
                        >
                          <WorkbenchIcon name="star" />
                          <span>{item.label}</span>
                        </button>
                        <button
                          className="file-remove-favorite"
                          aria-label={`Remove ${item.label} from favorites`}
                          title="Remove shortcut"
                          onClick={() => saveFavorites(favorites.filter((_, i) => i !== index))}
                        >
                          <WorkbenchIcon name="close" />
                        </button>
                      </div>
                    );
                  })}
                  {!favorites.some((item) =>
                    places.some((place) => placeKey(place) === item.place),
                  ) && <p className="file-favorites-empty">Use the star to add a folder.</p>}
                </section>
                {[
                  {
                    label: 'Folders',
                    items: places.filter(
                      (place) => !place.mounted && !/^(drive|volume)/.test(place.id),
                    ),
                  },
                  {
                    label: 'This PC',
                    items: places.filter(
                      (place) => !place.mounted && /^(drive|volume)/.test(place.id),
                    ),
                  },
                  { label: 'Mounted folders', items: places.filter((place) => place.mounted) },
                ]
                  .filter((group) => group.items.length)
                  .map((group) => (
                    <section key={group.label} aria-label={group.label}>
                      <h3 className="file-places-heading">{group.label}</h3>
                      {group.items.map((place) => (
                        <button
                          key={place.id}
                          title={place.path || undefined}
                          className={
                            root === place.id && listing?.path === place.path ? 'selected' : ''
                          }
                          aria-current={
                            root === place.id && listing?.path === place.path
                              ? 'location'
                              : undefined
                          }
                          disabled={busy}
                          onClick={() => void navigate(place.id, place.path)}
                        >
                          <WorkbenchIcon name={placeIcon(place)} />
                          <span>{place.label}</span>
                        </button>
                      ))}
                    </section>
                  ))}
              </nav>
              <section className="file-browser" aria-label="Folder contents">
                <div className="file-command-bar">
                  <button
                    className="secondary"
                    disabled={busy || !listing?.writable}
                    onClick={() => setNewFolder('')}
                  >
                    <WorkbenchIcon name="new-folder" />
                    New folder
                  </button>
                  <button
                    className="secondary"
                    disabled={busy || !listing}
                    aria-pressed={Boolean(favorite)}
                    onClick={toggleFavorite}
                  >
                    <WorkbenchIcon name="star" />
                    {favorite ? 'Remove favorite' : 'Add to favorites'}
                  </button>
                  <button
                    className="secondary file-icon-button"
                    aria-label="Refresh folder"
                    title="Refresh folder"
                    disabled={busy || !listing}
                    onClick={() => void browse(root, listing!.path, offset)}
                  >
                    <WorkbenchIcon name="refresh" />
                  </button>
                </div>
                <div className="file-column-headings" aria-hidden="true">
                  <span>Name</span>
                  <span>Type</span>
                  <span>Size</span>
                </div>
                <div
                  className="file-entries"
                  role="list"
                  aria-label="Files and folders"
                  aria-busy={busy}
                  onKeyDown={(event) => {
                    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
                    const buttons = Array.from(
                      event.currentTarget.querySelectorAll<HTMLButtonElement>('button'),
                    );
                    const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
                    const index =
                      event.key === 'Home'
                        ? 0
                        : event.key === 'End'
                          ? buttons.length - 1
                          : Math.max(
                              0,
                              Math.min(
                                buttons.length - 1,
                                current + (event.key === 'ArrowDown' ? 1 : -1),
                              ),
                            );
                    if (!buttons[index]) return;
                    event.preventDefault();
                    buttons[index].focus();
                    selectEntry(visibleEntries[index]);
                  }}
                >
                  {busy && <p role="status">Opening folder…</p>}
                  {visibleEntries.map((entry) => (
                    <div role="listitem" key={entry.path}>
                      <button
                        className={
                          selected?.path === entry.path ? 'file-entry selected' : 'file-entry'
                        }
                        aria-pressed={selected?.path === entry.path}
                        aria-label={`${entry.name} ${entry.directory ? 'Folder' : fileSize(entry.bytes)}`}
                        onClick={() => selectEntry(entry)}
                        onKeyDown={(event) => {
                          if (event.key !== 'Enter') return;
                          event.preventDefault();
                          if (entry.directory) void navigate(root, entry.path);
                          else if (!saving) void confirm(entry);
                        }}
                        onDoubleClick={() => {
                          if (entry.directory) void navigate(root, entry.path);
                          else if (!saving) void confirm(entry);
                        }}
                      >
                        <span
                          className={entry.directory ? 'file-folder-icon' : 'file-document-icon'}
                        >
                          <WorkbenchIcon name={entry.directory ? 'folder' : 'file'} />
                        </span>
                        <span className="file-entry-name">{entry.name}</span>
                        <span className="file-type">
                          {entry.directory
                            ? 'File folder'
                            : entry.name.includes('.') && !entry.name.startsWith('.')
                              ? entry.name.split('.').at(-1)!.toUpperCase() + ' file'
                              : 'File'}
                        </span>
                        <span className="file-size">
                          {entry.directory ? '' : fileSize(entry.bytes)}
                        </span>
                      </button>
                    </div>
                  ))}
                  {!busy && listing && !visibleEntries.length && (
                    <p className="muted">
                      {filter || request?.extensions?.length
                        ? 'No matching files or folders.'
                        : 'This folder is empty.'}
                    </p>
                  )}
                </div>
                {(offset > 0 || listing?.more) && (
                  <div className="button-row file-paging">
                    <button
                      className="secondary"
                      disabled={busy || !offset}
                      onClick={() => void browse(root, listing!.path, offset - 500)}
                    >
                      Previous
                    </button>
                    <button
                      className="secondary"
                      disabled={busy || !listing?.more}
                      onClick={() => void browse(root, listing!.path, offset + 500)}
                    >
                      More
                    </button>
                  </div>
                )}
                {newFolder !== null && (
                  <form
                    className="file-new-folder"
                    onSubmit={(event) => {
                      event.preventDefault();
                      if (
                        !newFolder.trim() ||
                        /[\\/]/.test(newFolder) ||
                        ['.', '..'].includes(newFolder)
                      ) {
                        setError('Enter a folder name without path separators');
                        return;
                      }
                      void api('/api/files/folder', 'POST', {
                        root,
                        path: childPath(newFolder.trim()),
                      })
                        .then(() => {
                          setNewFolder(null);
                          return browse(root, listing!.path);
                        })
                        .catch((failure) => setError(String(failure)));
                    }}
                  >
                    <label>
                      New folder name
                      <input
                        aria-label="New folder name"
                        value={newFolder}
                        onChange={(event) => setNewFolder(event.target.value)}
                      />
                    </label>
                    <button className="secondary">Create folder</button>
                    <button type="button" className="secondary" onClick={() => setNewFolder(null)}>
                      Cancel new folder
                    </button>
                  </form>
                )}
              </section>
            </div>
            <div className="file-system-footer">
              <label className="file-name-field">
                <span>{folderMode ? 'Folder name' : 'File name'}</span>
                <input
                  aria-label={folderMode ? 'Folder name' : 'File name'}
                  value={name}
                  onChange={(event) => {
                    setName(event.target.value);
                    setSelected(null);
                    setReplacement(null);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') void confirm();
                  }}
                  spellCheck={false}
                />
              </label>
              <div className="file-format">
                <span>File type</span>
                <span>
                  {folderMode
                    ? 'Folders'
                    : request?.extensions?.length
                      ? request.extensions.join(', ')
                      : 'All files and folders'}
                </span>
              </div>
            </div>
            {error && <p role="alert">{error}</p>}
            {replacement ? (
              <div className="file-replacement" role="alert">
                <p>
                  <strong>{replacement.name}</strong> already exists. Replace it?
                </p>
                <p>The existing file will be overwritten. This cannot be undone.</p>
                <div className="button-row">
                  <button className="secondary" onClick={() => setReplacement(null)}>
                    Keep existing
                  </button>
                  <button
                    className="primary"
                    onClick={() => finish({ ...replacement, overwrite: true })}
                  >
                    Replace
                  </button>
                </div>
              </div>
            ) : (
              <div className="dialog-actions">
                <button
                  className="primary"
                  disabled={
                    busy ||
                    !listing ||
                    (saving
                      ? !name.trim() || !listing.writable
                      : request?.mode === 'file' && !selected && !name.trim())
                  }
                  onClick={() => void confirm()}
                >
                  {saving ? 'Save' : folderMode ? 'Select folder' : 'Open'}
                </button>
                <button className="secondary" onClick={() => finish(null)}>
                  Cancel
                </button>
              </div>
            )}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </FileSystemContext.Provider>
  );
}
