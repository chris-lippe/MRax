"""Application filesystem access, with optional shared-folder boundaries."""

from __future__ import annotations

import os
import re
from pathlib import Path

from mrax.io.output_guard import SourceProtection

_USER_FOLDERS = (
    ("desktop", "Desktop", "DESKTOP", "Desktop"),
    ("documents", "Documents", "DOCUMENTS", "Personal"),
    ("downloads", "Downloads", "DOWNLOAD", "{374DE290-123F-4565-9164-39C4925E467B}"),
    ("pictures", "Pictures", "PICTURES", "My Pictures"),
    ("music", "Music", "MUSIC", "My Music"),
    ("videos", "Videos", "VIDEOS", "My Video"),
)


def _windows_user_folders():
    """Read Explorer's user folders, including redirected and OneDrive locations."""
    if os.name != "nt":
        return {}
    import winreg

    folders = {}
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as registry:
            for key, _, _, value_name in _USER_FOLDERS:
                try:
                    value, _ = winreg.QueryValueEx(registry, value_name)
                    if isinstance(value, str):
                        folders[key] = Path(os.path.expandvars(value))
                except OSError:
                    continue
    except OSError:
        pass
    return folders


def _standard_folders(home=None):
    """Find existing standard folders without creating directories or running shell code."""
    home = Path.home() if home is None else Path(home)
    overrides = _windows_user_folders()
    if os.name != "nt":
        config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "user-dirs.dirs"
        try:
            lines = config.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            lines = []
        names = {xdg: key for key, _, xdg, _ in _USER_FOLDERS}
        for line in lines:
            match = re.fullmatch(r'\s*XDG_(\w+)_DIR="((?:\\.|[^"\\])*)"\s*(?:#.*)?', line)
            if not match or match[1] not in names:
                continue
            value = re.sub(r'\\([\\"$`])', r"\1", match[2])
            for prefix in ("$HOME", "${HOME}"):
                if value == prefix or value.startswith(prefix + "/"):
                    value = str(home) + value[len(prefix) :]
                    break
            path = Path(value)
            if path.is_absolute():
                overrides[names[match[1]]] = path
    result = []
    for key, label, _, _ in _USER_FOLDERS:
        path = overrides.get(key, home / label)
        try:
            if path != home and path.is_dir():
                result.append((key, label, path))
        except OSError:
            continue
    return result


def _mounted_folders(mountinfo=Path("/proc/self/mountinfo")):
    """Discover directory mounts without exposing host-side source paths."""
    try:
        lines = mountinfo.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    except OSError:
        return []
    folders = set()
    for line in lines:
        fields, separator, _ = line.partition(" - ")
        columns = fields.split()
        if not separator or len(columns) < 6:
            continue
        target = re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match[1], 8)), columns[4])
        path = Path(target)
        if not path.is_absolute() or any(
            path.is_relative_to(system) for system in (Path("/proc"), Path("/sys"), Path("/dev"))
        ):
            continue
        try:
            if path.is_dir():
                folders.add(path.resolve())
        except OSError:
            continue
    return sorted(folders, key=lambda path: str(path).casefold())


class FileRoots:
    def __init__(self, read_roots=(), write_roots=(), *, scope="shared"):
        if scope not in {"system", "shared"}:
            raise ValueError("Unknown filesystem scope")
        self.system = scope == "system"
        self.sources = SourceProtection()
        writable = {Path(p).expanduser().resolve() for p in write_roots}
        paths = list(
            dict.fromkeys([*(Path(p).expanduser().resolve() for p in read_roots), *writable])
        )
        if any(not p.is_dir() for p in paths):
            raise ValueError("File locations must already exist")
        self.items = {
            f"root{index}": {"path": path, "writable": self.system or path in writable}
            for index, path in enumerate(paths)
        }
        if self.system:
            places = [
                ("home", "Home", Path.home()),
                *_standard_folders(),
                ("working", "Working folder", Path.cwd()),
            ]
            drives = (
                [Path(p) for p in os.listdrives()] if hasattr(os, "listdrives") else [Path("/")]
            )
            places.extend(
                (f"drive{i}", str(path) if os.name == "nt" else "File system", path)
                for i, path in enumerate(drives)
            )
            volumes = Path("/Volumes")
            try:
                mounted_volumes = list(volumes.iterdir()) if volumes.is_dir() else []
            except OSError:
                mounted_volumes = []
            places.extend((f"volume{i}", path.name, path) for i, path in enumerate(mounted_volumes))
            for key, label, path in places:
                try:
                    if path.is_dir():
                        self.items[key] = {"path": path.resolve(), "writable": True, "label": label}
                except OSError:
                    continue
            known = {row["path"] for row in self.items.values()}
            for index, path in enumerate(_mounted_folders()):
                if path in known:
                    continue
                self.items[f"mount{index}"] = {
                    "path": path,
                    "writable": True,
                    "label": path.name,
                    "mounted": True,
                }
                known.add(path)

    def catalog(self):
        return [
            {
                "id": key,
                "label": row.get("label", row["path"].name or str(row["path"])),
                "path": str(row["path"]) if self.system else "",
                "writable": row["writable"],
                "mounted": row.get("mounted", False),
            }
            for key, row in self.items.items()
        ]

    def resolve(self, key, relative="", *, write=False):
        if key not in self.items:
            raise ValueError("Choose a file location")
        row = self.items[key]
        value = (
            Path(str(relative)).expanduser()
            if self.system
            else Path(str(relative).replace("\\", "/"))
        )
        if not self.system and (
            value.is_absolute() or ".." in value.parts or (value.parts and ":" in value.parts[0])
        ):
            raise ValueError("The path must stay within its shared folder")
        candidate = (row["path"] / value).resolve()
        if not self.system and not candidate.is_relative_to(row["path"]):
            raise ValueError("The selected path leaves the shared folder")
        if write and not row["writable"]:
            raise ValueError("This folder is read-only")
        return candidate

    def location(self, key, path):
        if self.system:
            return str(path)
        relative = path.relative_to(self.items[key]["path"]).as_posix()
        return "" if relative == "." else relative

    def describe(self, key, relative=""):
        path = self.resolve(key, relative)
        parent = path.parent
        if not self.system and not parent.is_relative_to(self.items[key]["path"]):
            parent = path
        return {
            "root": key,
            "path": self.location(key, path),
            "name": path.name or str(path),
            "parent": self.location(key, parent),
            "directory": path.is_dir(),
            "exists": path.exists(),
            "writable": self.items[key]["writable"],
            "separator": os.sep,
        }

    def listing(self, key, relative="", offset=0, *, search=""):
        root = self.resolve(key, relative)
        if not root.is_dir():
            raise ValueError("Choose an existing folder")
        items = []
        try:
            for path in root.iterdir():
                try:
                    if search.casefold() not in path.name.casefold():
                        continue
                    if not self.system and (
                        path.is_symlink()
                        or not path.resolve().is_relative_to(self.items[key]["path"])
                    ):
                        continue
                    folder = path.is_dir()
                    items.append(
                        {
                            "name": path.name,
                            "path": self.location(key, path),
                            "directory": folder,
                            "bytes": None if folder else path.stat().st_size,
                        }
                    )
                except OSError:
                    continue
        except PermissionError as exc:
            raise ValueError("You do not have permission to open this folder") from exc
        items.sort(key=lambda item: (not item["directory"], item["name"].casefold()))
        return {
            **self.describe(key, relative),
            "entries": items[offset : offset + 500],
            "more": len(items) > offset + 500,
            "offset": offset,
        }

    def checked_tree(self, key, relative):
        path = self.resolve(key, relative)
        if not path.exists():
            raise ValueError("The selected file or folder no longer exists")
        if not self.system and path.is_dir():
            boundary = self.items[key]["path"]
            for index, child in enumerate(path.rglob("*")):
                if index > 200000:
                    raise ValueError("Choose a smaller source folder")
                if child.is_symlink() and not child.resolve().is_relative_to(boundary):
                    raise ValueError("Source folders cannot contain links outside a shared folder")
        return path
