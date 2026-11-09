"""Browser-only launch and packaging contracts, using independent synthetic data."""

import socket
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from mrax.gui.services.persistence import save_workspace_bundle
from mrax.gui.services.synthetic import synthetic_t2_session
from mrax.gui.services.workspace import StudyWorkspace
from mrax.web.app import create_app
from mrax.web.cli import _open_when_ready, main
from mrax.web.job_runtime import Jobs


def test_default_distribution_has_no_native_widget_stack():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    project = metadata["project"]
    dependencies = project["dependencies"] + [
        item for group in project["optional-dependencies"].values() for item in group
    ]
    retired = ("pyside", "pyqt", "qtpy", "superqt")
    assert not any(dep.lower().startswith(retired) for dep in dependencies)
    assert project["scripts"] == {"mrax-gui": "mrax.web.cli:main"}
    assert not (Path(__file__).parents[1] / "mrax/gui/app.py").exists()
    assert not (Path(__file__).parents[1] / "mrax/gui/_deps.py").exists()
    assert not (Path(__file__).parents[1] / "mrax/gui/__main__.py").exists()


def test_command_help_uses_the_gui_command(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert capsys.readouterr().out.startswith("usage: mrax-gui ")


def test_browser_opens_after_startup_and_cancels_on_failure(monkeypatch):
    opened = threading.Event()
    monkeypatch.setattr("webbrowser.open", lambda url: opened.set() or True)
    server = SimpleNamespace(started=False)
    stopped = threading.Event()
    opener = threading.Thread(target=_open_when_ready, args=(server, "http://localhost", stopped))
    opener.start()
    try:
        assert not opened.wait(0.15)
        server.started = True
        assert opened.wait(2)
    finally:
        stopped.set()
        opener.join(timeout=2)
    opened.clear()
    server.started = False
    stopped.clear()
    opener = threading.Thread(target=_open_when_ready, args=(server, "http://localhost", stopped))
    opener.start()
    stopped.set()
    opener.join(timeout=2)
    assert not opened.is_set()
    assert not opener.is_alive()


def test_occupied_port_never_opens_browser_or_starts_server(monkeypatch, capsys):
    opened, run = Mock(), Mock()
    monkeypatch.setattr("webbrowser.open", opened)
    monkeypatch.setattr("uvicorn.Server.run", run)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(SystemExit) as failure:
            main(["--port", str(occupied.getsockname()[1])])
    assert failure.value.code == 2
    assert "--port 0" in capsys.readouterr().err
    opened.assert_not_called()
    run.assert_not_called()


def test_startup_bundle_does_not_expand_shared_folder_access(tmp_path):
    bundle = tmp_path / "Independent Studie Ω.mrax"
    save_workspace_bundle(
        StudyWorkspace(
            key="startup", label="Independent synthetic", session=synthetic_t2_session()
        ),
        bundle,
    )
    with TestClient(
        create_app(initial_paths=[bundle], filesystem_scope="shared"), base_url="http://127.0.0.1"
    ) as client:
        opened = client.get("/api/workspaces").json()
        assert len(opened) == 1
        assert opened[0]["label"] == "Synthetic T2 phantom"
        assert opened[0]["loaded"]
        assert client.get("/api/files/roots").json() == []


def test_startup_failure_does_not_leave_worker_threads(tmp_path):
    before = {thread.ident for thread in threading.enumerate()}
    with pytest.raises((ValueError, FileNotFoundError)):
        with TestClient(
            create_app(initial_paths=[tmp_path / "missing.mrax"]), base_url="http://127.0.0.1"
        ):
            pytest.fail("A failed input must not start serving")
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.ident not in before and thread.name.startswith(("mrax-", "MRax"))
    ]


def test_progress_is_in_memory_without_progress_file_reads_or_writes(tmp_path, monkeypatch):
    import mrax.web.job_runtime as runtime

    started, release = threading.Event(), threading.Event()

    def worker(_workspace, _root, _operation, _options, progress, **_kwargs):
        progress(1, 2, "Synthetic operation")
        started.set()
        assert release.wait(5)
        return {"operation": "export", "value": {"saved": True}}

    def locked(*_args, **_kwargs):
        raise PermissionError("Reader holds the Windows file")

    monkeypatch.setattr(runtime, "_workflow_worker", worker)
    monkeypatch.setattr(Path, "read_text", locked)
    jobs = Jobs(tmp_path / "jobs", lambda *_: True)
    try:
        job = jobs.submit("synthetic", 0, object(), operation="export")
        assert started.wait(5)
        assert jobs.get(job["id"])["progress"] == 0.5
        assert jobs.get(job["id"])["status"] == "running"
        assert not list(tmp_path.rglob("*.json"))
    finally:
        release.set()
        jobs.close()
