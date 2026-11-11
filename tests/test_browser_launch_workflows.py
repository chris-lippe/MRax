"""Browser-only launch and packaging contracts, using independent synthetic data."""

import threading
from types import SimpleNamespace

import pytest

from mrax.web.cli import _open_when_ready, main


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
