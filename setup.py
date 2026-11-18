"""Materialize the notebook gallery in source distributions on every platform."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.sdist import sdist


class NotebookSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        root = Path(base_dir)
        for page in (root / "docs" / "notebooks").glob("*.ipynb"):
            # Git can check out these links as path text. Replace the staging
            # entry first because setuptools may have hard-linked it to the checkout.
            page.unlink()
            shutil.copyfile(root / "notebooks" / page.name, page)


setup(cmdclass={"sdist": NotebookSdist})
