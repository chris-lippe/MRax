"""Installation contracts for scripts, scientific extras and the browser GUI."""

import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).parents[1]


def test_runtime_profiles_keep_the_web_server_optional_and_workflows_shared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    base = {Requirement(value).name.lower() for value in project["dependencies"]}
    assert {"jax", "optax", "numpy", "scipy", "pydantic", "nibabel", "pydicom"} <= base
    assert not {"fastapi", "uvicorn", "simpleitk", "dipy", "matplotlib", "pypulseq"} & base
    assert (ROOT / "requirements.txt").read_text().splitlines() == project["dependencies"]
    extras = project["optional-dependencies"]
    assert {"fastapi", "uvicorn", "mrax"} == {
        Requirement(value).name.lower() for value in extras["gui"]
    }
    assert Requirement(extras["gui"][0]).extras == {"full"}
    assert {"simpleitk", "dipy", "matplotlib", "pypulseq", "pymapvbvd"} == {
        Requirement(value).name.lower() for value in extras["full"]
    }


def test_public_toolbox_imports_do_not_load_specialized_or_web_dependencies():
    script = """
import importlib.abc, sys
blocked = {"fastapi", "uvicorn", "SimpleITK", "dipy", "pypulseq", "mapvbvd", "matplotlib"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError("Optional dependency blocked by installation contract: " + fullname)
sys.meta_path.insert(0, Block())
import mrax.core, mrax.data_struct, mrax.analysis, mrax.reco, mrax.preprocessing, mrax.io
from mrax.workflows import WorkflowDocument
assert WorkflowDocument.model_json_schema()
assert not blocked.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, timeout=40)
