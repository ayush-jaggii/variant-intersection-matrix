import os
import sys
import webbrowser
import time
import subprocess
import socket
import tempfile
import threading


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _claim_browser_open_token() -> bool:
    """Allow only one process to open a browser tab during startup bursts."""
    token_path = os.path.join(tempfile.gettempdir(), "vim_analyzer_browser_open.lock")
    now = time.time()

    # Expire stale lock files so future launches can open a tab again.
    if os.path.exists(token_path):
        try:
            mtime = os.path.getmtime(token_path)
            if now - mtime > 30:
                os.remove(token_path)
        except OSError:
            return False

    try:
        fd = os.open(token_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(now))
        return True
    except FileExistsError:
        return False
    except OSError:
        return False


def _open_browser_once(url: str) -> None:
    if not _claim_browser_open_token():
        return

    # In bundled/no-console mode, webbrowser.open() may fail silently on some
    # macOS setups. Retry and fall back to the native "open" command.
    for _ in range(3):
        try:
            if webbrowser.open(url, new=2):
                return
        except Exception:
            pass

        if sys.platform == "darwin":
            try:
                subprocess.Popen(
                    ["open", url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
                return
            except Exception:
                pass

        time.sleep(1)


def _wait_and_open_browser(port: int, timeout_seconds: int = 90) -> None:
    """Open browser after server becomes reachable."""
    url = f"http://localhost:{port}"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if port_open("127.0.0.1", port):
            _open_browser_once(url)
            return
        time.sleep(1)


def main():

    port = 8501

    # PyInstaller runtime path
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)

    app_path = os.path.join(base_path, "interface", "app.py")
    log_root = tempfile.gettempdir() if getattr(sys, "frozen", False) else base_path
    log_path = os.path.join(log_root, "data", "cache", "streamlit_launch.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"🚀 Starting Variant Intersection Matrix Analyzer on port {port}...")
    print(f"App Path: {app_path}")

    if port_open("127.0.0.1", port):
        _open_browser_once(f"http://localhost:{port}")
        return

    # In frozen onefile mode, spawning `sys.executable -m streamlit` re-enters
    # this launcher binary and the server never comes up. Run Streamlit in-proc.
    if getattr(sys, "frozen", False):
        from streamlit.web import cli as stcli

        threading.Thread(target=_wait_and_open_browser, args=(port,), daemon=True).start()

        sys.argv = [
            "streamlit",
            "run",
            app_path,
            "--global.developmentMode=false",
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            "--server.maxUploadSize=200",
            "--theme.primaryColor=#1a237e",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f8f9fa",
            "--theme.textColor=#212121",
        ]
        sys.exit(stcli.main())

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.maxUploadSize=200",
        "--theme.primaryColor=#1a237e",
        "--theme.backgroundColor=#ffffff",
        "--theme.secondaryBackgroundColor=#f8f9fa",
        "--theme.textColor=#212121",
    ]

    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            cmd,
            cwd=base_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    for _ in range(30):
        if port_open("127.0.0.1", port):
            _open_browser_once(f"http://localhost:{port}")
            return
        time.sleep(1)

    print(f"Started background process, but the server did not respond on http://localhost:{port} within 30 seconds.")
    sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
