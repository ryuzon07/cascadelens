"""Start the Streamlit demo on this machine only, with its usage statistics switched off."""

import sys
from pathlib import Path

from streamlit.web import cli


def main() -> None:
    """Run the app with the safe flags."""
    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).with_name("app.py")),
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    cli.main()


if __name__ == "__main__":
    main()
