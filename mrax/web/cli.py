"""Desktop launcher for the local MRax browser application."""

import argparse
import socket
import threading
import webbrowser
from pathlib import Path


def _open_when_ready(server, url, stopped):
    while not stopped.wait(0.05):
        if server.started:
            if not webbrowser.open(url):
                print(f"Open {url} in your browser.", flush=True)
            return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mrax-gui", description="Start the MRax GUI")
    parser.add_argument(
        "workspace",
        nargs="?",
        type=Path,
        help="MRax bundle, package directory, study manifest or saved state to open",
    )
    parser.add_argument("--host", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Local port (0 selects a free port)")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--read-root",
        type=Path,
        action="append",
        default=[],
        help="Add a file-browser location (read-only in shared mode; repeatable)",
    )
    parser.add_argument(
        "--write-root",
        type=Path,
        action="append",
        default=[],
        help="Add a writable file-browser location (repeatable)",
    )
    parser.add_argument(
        "--filesystem-scope",
        choices=["system", "shared"],
        default="system",
        help="Browse system files (default), or restrict access to configured read/write roots",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Keep automatic recovery bundles in this private local folder",
    )
    args = parser.parse_args(argv)
    try:
        import uvicorn

        from .app import create_app
    except ImportError as exc:
        parser.exit(2, f"MRax installation is incomplete; reinstall mrax: {exc}\n")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    if args.workspace is not None and not args.workspace.exists():
        parser.error("The startup workspace does not exist")
    if not (Path(__file__).with_name("static") / "index.html").is_file():
        parser.exit(
            2,
            "Browser assets are missing. Reinstall the built MRax wheel, or run "
            "npm ci && npm run build in frontend/ for a source checkout.\n",
        )
    app = create_app(
        filesystem_scope=args.filesystem_scope,
        read_roots=args.read_root,
        write_roots=args.write_root,
        state_dir=args.state_dir,
        initial_paths=[] if args.workspace is None else [args.workspace.resolve()],
    )
    stopped = threading.Event()
    opener = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # Reserve the port before launching a browser; never navigate to another
        # application's occupied port. Windows has different reuse semantics.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((args.host, args.port))
        except OSError:
            parser.exit(
                2,
                f"Cannot bind MRax on {args.host}:{args.port}. "
                "Close the other server or select --port 0.\n",
            )
        port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=port, access_log=False))
        print(f"MRax GUI: {url}", flush=True)
        if not args.no_browser:
            opener = threading.Thread(
                target=_open_when_ready, args=(server, url, stopped), daemon=True
            )
            opener.start()
        try:
            server.run(sockets=[listener])
        finally:
            stopped.set()
            if opener is not None:
                opener.join(timeout=2)
    return 0 if server.started else 1
