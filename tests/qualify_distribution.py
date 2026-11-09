"""Inspect release archives without importing the application or extracting files."""

import argparse
import ast
import configparser
import json
import re
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
    "data",
    "tmp",
    "node_modules",
    "test-results",
    "playwright-report",
    "__pycache__",
    ".git",
    ".venv",
    ".wheel-env",
    "_build",
}
MEASUREMENT_SUFFIXES = {
    ".mrax",
    ".mrd",
    ".nii",
    ".dcm",
    ".ima",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".mat",
    ".dat",
    ".pkl",
    ".pickle",
}
RETIRED_MODULES = {"mrax/gui/app.py", "mrax/gui/_deps.py", "mrax/gui/__main__.py"}
SOURCE_FILES = {
    "pyproject.toml",
    "README.md",
    "MANIFEST.in",
    "MRax_logo.svg",
    "MRax_icon.svg",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/openapi.json",
    "frontend/index.html",
    "frontend/tsconfig.json",
    "frontend/vite.config.ts",
    "frontend/playwright.config.ts",
    "frontend/eslint.config.js",
    "frontend/.prettierrc.json",
    "frontend/.prettierignore",
    "frontend/src/main.tsx",
    "frontend/src/api.generated.ts",
    "frontend/scripts/export_schema.py",
    "tests/qualify_distribution.py",
    "tests/qualify_browser_install.py",
}


def inspect_artifact(path, expected_version):
    path = Path(path)
    wheel = path.suffix == ".whl"
    if wheel:
        archive = zipfile.ZipFile(path)
        members = {item.filename: item for item in archive.infolist()}
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in members.values()):
            archive.close()
            raise ValueError("Distribution links are not permitted")
        read = archive.read
        prefix = ""
    else:
        archive = tarfile.open(path, "r:gz")
        members = {item.name: item for item in archive.getmembers()}
        if any(item.issym() or item.islnk() for item in members.values()):
            archive.close()
            raise ValueError("Distribution links are not permitted")

        def read(name):
            with archive.extractfile(members[name]) as stream:
                return stream.read()

        roots = {PurePosixPath(name).parts[0] for name in members}
        if len(roots) != 1:
            archive.close()
            raise ValueError("A source archive must have one project root")
        prefix = next(iter(roots)) + "/"
    try:
        names = set()
        for full in members:
            name = full.removeprefix(prefix)
            item = PurePosixPath(name)
            if item.is_absolute() or ".." in item.parts or "\\" in name:
                raise ValueError("Unsafe path in distribution")
            if (
                any(part.casefold() in FORBIDDEN_PARTS for part in item.parts)
                or item.suffix.casefold() in MEASUREMENT_SUFFIXES
                or name.casefold().endswith((".nii.gz", ".pyc", ".pyo", ".partial"))
                or item.name.casefold() in {"fid", "2dseq", "acqp", "visu_pars"}
            ):
                raise ValueError("Measurement data or development artifacts are present")
            if name in RETIRED_MODULES:
                raise ValueError("Retired GUI implementation is present")
            names.add(name)
        metadata_names = (
            [name for name in members if name.endswith(".dist-info/METADATA")]
            if wheel
            else [prefix + "PKG-INFO"]
        )
        if len(metadata_names) != 1 or metadata_names[0] not in members:
            raise ValueError("Distribution metadata is missing or ambiguous")
        metadata = BytesParser().parsebytes(read(metadata_names[0]))
        if (metadata["Name"] or "").casefold() != "mrax" or metadata["Version"] != expected_version:
            raise ValueError("Distribution identity does not match the intended release")
        if any(
            value.casefold().startswith(("pyside", "pyqt", "qtpy", "superqt"))
            for value in metadata.get_all("Requires-Dist", [])
        ):
            raise ValueError("A retired native-widget dependency is present")
        if wheel:
            entries = configparser.ConfigParser()
            entry_path = metadata_names[0].replace("METADATA", "entry_points.txt")
            entries.read_string(read(entry_path).decode())
            if dict(entries["console_scripts"]) != {"mrax-gui": "mrax.web.cli:main"}:
                raise ValueError("The distribution must expose only the mrax-gui launcher")
        else:
            if not SOURCE_FILES <= names:
                raise ValueError(
                    "The source archive lacks frontend sources, configuration or icons"
                )
            for file in ("frontend/package.json", "frontend/package-lock.json"):
                if json.loads(read(prefix + file))["version"] != expected_version:
                    raise ValueError("Frontend and Python release versions differ")
        base = "mrax/web/static/"
        index = read(prefix + base + "index.html").decode()
        assets = re.findall(r'(?:src|href)="([^"]+)"', index)
        if not assets:
            raise ValueError("The GUI index contains no assets")
        for asset in assets:
            if not asset.startswith("/") or asset.startswith("//"):
                raise ValueError("GUI assets must be bundled locally")
            name = base + asset.lstrip("/")
            if name not in names or not read(prefix + name):
                raise ValueError("A bundled GUI asset is missing")
        return {
            "artifact": path.name,
            "kind": "wheel" if wheel else "sdist",
            "version": metadata["Version"],
            "files": len(names),
            "assets": len(assets),
        }
    finally:
        archive.close()


def source_version():
    source = Path(__file__).parents[1] / "mrax" / "_version.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MRAX_VERSION" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("The source version is not defined")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--expected-version", default=None)
    args = parser.parse_args()
    expected = args.expected_version or source_version()
    files = []
    for path in args.archives:
        files.extend(
            sorted([*path.glob("*.whl"), *path.glob("*.tar.gz")]) if path.is_dir() else [path]
        )
    if not files:
        parser.error("Choose built wheel or source archives")
    for path in files:
        print(json.dumps(inspect_artifact(path, expected)))


if __name__ == "__main__":
    main()
