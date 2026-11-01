"""Run the API and the Vite dev server together, and stop both when either ends.

Run ``make dev`` or ``python scripts/dev.py``. Needs the api extra, a saved model and ``npm ci`` in
web/. Open http://localhost:5173. Add sample incidents first with ``python scripts/seed_demo.py``.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WEB = Path("web")
VITE = Path("node_modules/vite/bin/vite.js")
ORIGIN = "http://localhost:5173"


def main() -> int:
    """Start both servers and wait."""
    node = shutil.which("node")
    if node is None or not (WEB / VITE).is_file():
        print("Install Node, then run `make web-install`.", file=sys.stderr)
        return 1
    api = subprocess.Popen(
        [sys.executable, "-c", "from cascadelens.cli import app; app()", "serve"],
        env={**os.environ, "CASCADELENS_CORS_ORIGINS": ORIGIN},
    )
    # Vite runs directly, because stopping npm can leave its child running on Windows.
    web = subprocess.Popen([node, str(VITE)], cwd=WEB)
    try:
        while api.poll() is None and web.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (api, web):
            process.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
