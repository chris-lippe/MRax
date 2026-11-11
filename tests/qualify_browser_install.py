"""Qualify an installed wheel from any OS, outside the source tree.

Run with the target environment's Python. ``--device gpu`` requires a real
accelerator and fails if none is present; CPU execution cannot satisfy it.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def qualify(device):
    import mrax

    installed = Path(mrax.__file__).resolve()
    assert "site-packages" in installed.parts, "Run this check against an installed wheel"
    assert importlib.util.find_spec("mrax.gui.app") is None
    retired = ("pyside", "pyqt", "qtpy", "superqt", "shiboken")
    assert not any(
        requirement.lower().startswith(retired)
        for requirement in importlib.metadata.requires("mrax") or []
    )
    scripts = {
        entry.name: entry.value
        for entry in importlib.metadata.distribution("mrax").entry_points
        if entry.group == "console_scripts"
    }
    assert scripts == {"mrax-gui": "mrax.web.cli:main"}
    launcher = Path(sysconfig.get_path("scripts")) / (
        "mrax-gui.exe" if os.name == "nt" else "mrax-gui"
    )
    assert launcher.is_file(), "The mrax-gui command is missing from the installation"
    with tempfile.TemporaryDirectory(prefix="mrax_qualification_") as temporary:
        root = Path(temporary)
        log_path = root / "server.log"
        env = {**os.environ, "PYTHONPATH": "", "JAX_PLATFORMS": device}
        with log_path.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [str(launcher), "--no-browser", "--port", "0"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                url = ""
                token = ""

                def request(path, method="GET", payload=None, raw=None):
                    data = (
                        raw
                        if raw is not None
                        else (None if payload is None else json.dumps(payload).encode())
                    )
                    req = urllib.request.Request(
                        url + path,
                        method=method,
                        data=data,
                        headers={"Content-Type": "application/json", "x-mrax-token": token},
                    )
                    with urllib.request.urlopen(req, timeout=30) as response:
                        body = response.read()
                        if "application/json" in response.headers.get("Content-Type", ""):
                            return json.loads(body)
                        return body

                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError("The installed browser launcher exited during startup")
                    match = re.search(
                        r"MRax GUI: (http://127\.0\.0\.1:\d+)", log_path.read_text(encoding="utf-8")
                    )
                    if match:
                        url = match[1]
                        try:
                            if request("/api/health")["status"] == "ok":
                                break
                        except (OSError, urllib.error.URLError):
                            pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("The installed browser launcher did not become ready")
                token = request("/api/session")["token"]
                assert request("/openapi.json")["info"]["version"] == mrax.__version__
                index = request("/").decode()
                assets = re.findall(r'(?:src|href)="([^"]+\.(?:js|css|svg))"', index)
                assert assets
                assert all(request(asset) for asset in assets)
                catalog = request("/api/catalog")
                target = next(
                    (d["id"] for d in catalog["devices"] if d["id"].startswith(device + ":")), None
                )
                assert target is not None, f"No {device} device; qualification failed"
                workspace = request("/api/workspaces/synthetic", "POST")
                path = f"/api/workspaces/{workspace['id']}"
                workspace = request(
                    path + "/actions",
                    "POST",
                    {
                        "revision": workspace["revision"],
                        "view_revision": workspace["view_revision"],
                        "action": "roi.create",
                        "options": {
                            "name": "Qualification ROI",
                            "tool": "rect",
                            "bounds": [4, 3, 8, 6],
                            "label": 7,
                        },
                    },
                )
                job = request(
                    path + "/preview-fit",
                    "POST",
                    {
                        "revision": workspace["revision"],
                        "roi_name": "Qualification ROI",
                        "steps": 600,
                        "device": target,
                    },
                )
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    job = request("/api/jobs/" + job["id"])
                    if job["status"] not in {"queued", "running", "cancelling"}:
                        break
                    time.sleep(0.1)
                assert job["status"] == "completed", job["detail"]
                result = job["result"]
                assert abs(result["parameters"]["t2"] - 80) < 0.05
                assert result["relative_rmse"] < 1e-4
                bundle = request(path + "/save", "POST")
                restored = request("/api/workspaces/upload", "POST", raw=bundle)
                assert restored["rois"] == workspace["rois"]
                assert restored["preview"]["parameters"]["t2"] == result["parameters"]["t2"]
                return {
                    "os": platform.system(),
                    "python": platform.python_version(),
                    "installed_wheel": True,
                    "compiled_assets": True,
                    "browser_launcher": True,
                    "device": target,
                    "t2_ms": result["parameters"]["t2"],
                    "relative_rmse": result["relative_rmse"],
                    "bundle_round_trip": True,
                }
            except Exception:
                print(log_path.read_text(encoding="utf-8")[-6000:], file=sys.stderr)
                raise
            finally:
                if os.name == "nt":
                    # The Windows venv redirector can own a Python child. Stop
                    # this test's complete process tree so it cannot hold files.
                    subprocess.run(
                        ["taskkill", "/PID", str(server.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        check=False,
                    )
                else:
                    server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    print(json.dumps(qualify(parser.parse_args().device), indent=2))
