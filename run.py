#!/usr/bin/env python3
"""
Variant Intersection Matrix Analyzer — Entry Point
====================================================

Usage:
    python run.py              # Launch Streamlit app
    python run.py --help       # Show help
"""

import subprocess
import sys
import socket
import time
import webbrowser
from pathlib import Path
from typing import List


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _launch_detached(command: List[str], cwd: Path) -> None:
    log_path = cwd / "data" / "cache" / "streamlit_launch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def main():
    """Launch the Streamlit application."""
    app_path = Path(__file__).parent / "interface" / "app.py"
    port = 8501
    project_root = Path(__file__).parent

    if not app_path.exists():
        print(f"Error: Application file not found at {app_path}")
        sys.exit(1)

    if _port_open("127.0.0.1", port):
        print(f"🚀 Variant Intersection Matrix Analyzer is already running on http://localhost:{port}")
        webbrowser.open(f"http://localhost:{port}")
        return

    print("🚀 Launching Variant Intersection Matrix Analyzer...")
    print(f"   App: {app_path}")
    print(f"   URL: http://localhost:{port}")
    print()

    command = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        "--server.maxUploadSize=200",
        "--theme.primaryColor=#1a237e",
        "--theme.backgroundColor=#ffffff",
        "--theme.secondaryBackgroundColor=#f8f9fa",
        "--theme.textColor=#212121",
    ]

    _launch_detached(command, project_root)

    for _ in range(30):
        if _port_open("127.0.0.1", port):
            webbrowser.open(f"http://localhost:{port}")
            return
        time.sleep(1)

    print(f"Started background process, but the server did not respond on http://localhost:{port} within 30 seconds.")


if __name__ == "__main__":
    main()
