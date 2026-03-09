import os
import sys
import webbrowser
import threading
import time

from streamlit.web import cli as stcli

def main():

    port = "8501"

    # PyInstaller runtime path
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)

    app_path = os.path.join(base_path, "interface", "app.py")

    print(f"🚀 Starting Variant Intersection Matrix Analyzer on port {port}...")
    print(f"App Path: {app_path}")

    # open browser automatically
    def open_browser():
        time.sleep(2)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser).start()

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
        "--theme.textColor=#212121"
    ]

    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
